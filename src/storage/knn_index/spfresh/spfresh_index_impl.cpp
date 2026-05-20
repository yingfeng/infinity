// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

module infinity_core:spfresh_index.impl;

import :spfresh_index;
import :spfresh_defs;
import :index_spfresh;
import :kmeans_partition;
import :infinity_exception;
import :local_file_handle;
import :logger;

import std;

import internal_types;
import embedding_info;

namespace infinity {

// ========== Construction / Destruction ==========

SPFreshIndexInMem::SPFreshIndexInMem(RowID begin_row_id, const IndexSPFresh *index_def, u32 embedding_dim, u32 max_vectors)
    : begin_row_id_(begin_row_id), rabitq_data_(nullptr),
      num_vectors_(0), max_vectors_(max_vectors), dim_(embedding_dim),
      rot_matrix_(nullptr), centroid_hnsw_(nullptr),
      num_centroids_(index_def ? index_def->num_centroids_ : 1000),
      replica_count_(index_def ? index_def->replica_count_ : 1),
      delta_a_(nullptr), delta_b_(nullptr), active_delta_idx_(0),
      bucket_locks_(nullptr), bucket_locks_count_(0) {

    size_t code_size = RabitQVecSize(dim_);
    size_t data_size = static_cast<size_t>(max_vectors) * code_size;
    rabitq_data_ = new char[data_size];
    std::memset(rabitq_data_, 0, data_size);
    mem_used_ += data_size;

    // Rotation matrix: dim_ x dim_ float matrix (Givens rotation)
    rot_matrix_ = new f32[static_cast<size_t>(dim_) * dim_]();
    GenerateRotationMatrix();
    mem_used_ += static_cast<size_t>(dim_) * dim_ * sizeof(f32);

    // Centroids
    centroids_.resize(static_cast<size_t>(num_centroids_) * dim_);
    bucket_metas_.resize(num_centroids_);
    bucket_assignments_.resize(static_cast<size_t>(max_vectors) * replica_count_, u32(-1));
    running_means_.resize(num_centroids_);

    // Per-bucket locks
    bucket_locks_count_ = num_centroids_ > 0 ? num_centroids_ : 1;
    bucket_locks_ = new std::mutex[bucket_locks_count_];

    // Delta double-buffer
    delta_a_ = new SPFreshDeltaBuffer(code_size);
    delta_b_ = new SPFreshDeltaBuffer(code_size);
}

SPFreshIndexInMem::~SPFreshIndexInMem() {
    delete[] rabitq_data_;
    delete[] rot_matrix_;
    delete[] bucket_locks_;
    delete delta_a_;
    delete delta_b_;
}

// ========== Rotation Matrix Generation (Phase A) ==========

void SPFreshIndexInMem::GenerateRotationMatrix() {
    // Random orthogonal matrix via Givens rotations (dim_ x dim_)
    for (u32 i = 0; i < dim_; ++i) {
        rot_matrix_[i * dim_ + i] = 1.0f;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<f32> angle_dist(0.0f, 2.0f * std::numbers::pi_v<f32>);

    for (u32 iter = 0; iter < dim_ * 2; ++iter) {
        u32 i = static_cast<u32>(rng() % dim_);
        u32 j = static_cast<u32>(rng() % dim_);
        if (i == j) continue;
        if (j < i) std::swap(i, j);

        f32 angle = angle_dist(rng);
        f32 c = std::cos(angle);
        f32 s = std::sin(angle);

        for (u32 k = 0; k < dim_; ++k) {
            f32 a = rot_matrix_[i * dim_ + k];
            f32 b = rot_matrix_[j * dim_ + k];
            rot_matrix_[i * dim_ + k] = c * a - s * b;
            rot_matrix_[j * dim_ + k] = s * a + c * b;
        }
    }
}

void SPFreshIndexInMem::ApplyRotation(const f32 *vec, f32 *out) const {
    std::fill(out, out + dim_, 0.0f);
    for (u32 i = 0; i < dim_; ++i) {
        f32 sum = 0.0f;
        for (u32 j = 0; j < dim_; ++j) {
            sum += rot_matrix_[i * dim_ + j] * vec[j];
        }
        out[i] = sum;
    }
}

void SPFreshIndexInMem::ApplyInverseRotation(const f32 *vec, f32 *out) const {
    std::fill(out, out + dim_, 0.0f);
    for (u32 i = 0; i < dim_; ++i) {
        f32 sum = 0.0f;
        for (u32 j = 0; j < dim_; ++j) {
            sum += rot_matrix_[j * dim_ + i] * vec[j];
        }
        out[i] = sum;
    }
}

// ========== RaBitQ Encode/Decode with Rotation (Phase A) ==========

size_t SPFreshIndexInMem::EncodeWithRotation(f32 *code_buf, const f32 *vec) const {
    auto *code = reinterpret_cast<RabitQVec *>(code_buf);
    std::memset(code_buf, 0, RabitQVecSize(dim_));

    // Step 1: Compute raw norm (||o||^2)
    f64 raw_norm = 0;
    for (u32 d = 0; d < dim_; ++d) {
        raw_norm += static_cast<f64>(vec[d]) * vec[d];
    }
    code->raw_norm_ = static_cast<f32>(raw_norm);
    f64 inv_raw_norm = (raw_norm > 1e-30) ? (1.0 / std::sqrt(raw_norm)) : 1.0;

    // Step 2: Normalize vector → unit sphere
    std::vector<f32> normalized(dim_);
    for (u32 d = 0; d < dim_; ++d) {
        normalized[d] = static_cast<f32>(static_cast<f64>(vec[d]) * inv_raw_norm);
    }

    // Step 3: Apply rotation
    std::vector<f32> rotated(dim_);
    ApplyRotation(normalized.data(), rotated.data());

    // Step 4: Sign encoding (1-bit per dimension)
    f32 sum_pos = 0.0f;
    for (u32 d = 0; d < dim_; ++d) {
        if (rotated[d] >= 0) {
            code->compress_[d / 8] |= (1 << (d % 8));
            sum_pos += 1.0f;
        }
    }
    code->sum_ = sum_pos;
    code->norm_ = 1.0f; // unit norm after normalization
    code->error_ = 0.95f;

    return RabitQVecSize(dim_);
}

void SPFreshIndexInMem::DecodeWithRotation(const RabitQVec *code, f32 *out_vec) const {
    f32 mag = 1.0f / std::sqrt(static_cast<f32>(dim_));
    std::vector<f32> rotated(dim_);
    for (u32 d = 0; d < dim_; ++d) {
        bool bit = (code->compress_[d / 8] >> (d % 8)) & 1;
        rotated[d] = bit ? mag : -mag;
    }

    ApplyInverseRotation(rotated.data(), out_vec);

    f32 raw_norm_sqrt = std::sqrt(code->raw_norm_);
    for (u32 d = 0; d < dim_; ++d) {
        out_vec[d] *= raw_norm_sqrt;
    }
}

f32 SPFreshIndexInMem::RabitQDistWithRotation(const RabitQVec *code, const f32 *rotated_query_normed,
                                               u32 dim, f32 inv_sqrt_d) {
    // Both code and query are in normalized-rotated space (unit norm = 1.0).
    // Code is sign-encoded: x̃_d = sign(x_d) / sqrt(dim), so ||x̃|| = 1.
    // Inner product <q, x̃> = sum(q_d * sign(x_d)) / sqrt(dim).
    // cos θ = <q, x̃> / (||q|| * ||x̃||) = <q, x̃> (since both unit norm).
    // ≈ ip / sqrt(dim) where ip = sum(q_d * sign(x_d))

    f32 ip = 0.0f;
    for (u32 d = 0; d < dim; ++d) {
        bool bit = (code->compress_[d / 8] >> (d % 8)) & 1;
        ip += bit ? rotated_query_normed[d] : -rotated_query_normed[d];
    }

    // cos_theta = ip / sqrt(dim)
    f32 cos_theta = ip * inv_sqrt_d;
    cos_theta = std::clamp(cos_theta, -1.0f, 1.0f);

    // L2^2 in unit sphere = 2 - 2*cos_theta
    return 2.0f * (1.0f - cos_theta);
}

// ========== Nearest Centroid (brute-force, Phase A) ==========

u32 SPFreshIndexInMem::FindNearestCentroid(const f32 *vec) const {
    u32 best = 0;
    f32 best_dist = std::numeric_limits<f32>::max();
    for (u32 c = 0; c < num_centroids_; ++c) {
        f32 dist = 0.0f;
        for (u32 d = 0; d < dim_; ++d) {
            f32 diff = vec[d] - centroids_[static_cast<size_t>(c) * dim_ + d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }
    return best;
}

// ========== Centroid HNSW Routing (Phase A) ==========

void SPFreshIndexInMem::BuildCentroidIndex() {
    if (num_centroids_ == 0 || dim_ == 0) return;

    // HNSW is only valuable for >32 centroids; below that brute-force is faster and more reliable
    if (num_centroids_ > 32) {
        u32 chunk_size = 1;
        while (chunk_size < num_centroids_ + 16) chunk_size <<= 1;
        auto hnsw = HnswType::Make(chunk_size, 1, dim_, 16, 200);
        if (hnsw) {
            auto iter = DenseVectorIter<f32, u32>(centroids_.data(), dim_, num_centroids_);
            HnswInsertConfig config{true};
            hnsw->InsertVecs(iter, config);
            centroid_hnsw_ = std::move(hnsw);
            LOG_TRACE(fmt::format("SPFresh BuildCentroidIndex: HNSW built for {} centroids, dim={}", num_centroids_, dim_));
            return;
        }
    }
    centroid_hnsw_ = nullptr;
    LOG_TRACE(fmt::format("SPFresh BuildCentroidIndex: brute-force for {} centroids, dim={}", num_centroids_, dim_));
}

void SPFreshIndexInMem::FindTopKCentroids(const f32 *query, u32 top_k,
                                           std::vector<u32> &out_ids,
                                           std::vector<f32> &out_dists) const {
    out_ids.clear();
    out_dists.clear();
    if (num_centroids_ == 0) return;

    u32 k = std::min(top_k, num_centroids_);

    if (centroid_hnsw_ && k > 0) {
        KnnSearchOption opt;
        opt.ef_ = k * 4;
        auto [cnt, dists_ptr, labels_ptr] = centroid_hnsw_->template KnnSearch<>(
            static_cast<const f32 *>(query), k, std::nullopt, opt);
        i64 n = std::min(static_cast<i64>(cnt), static_cast<i64>(k));
        for (i64 i = 0; i < n; ++i) {
            out_ids.push_back(labels_ptr[i]);
            out_dists.push_back(dists_ptr[i]);
        }
    } else {
        // Fallback: brute-force for all centroids
        std::vector<std::pair<f32, u32>> heap;
        heap.reserve(k + 1);
        for (u32 c = 0; c < num_centroids_; ++c) {
            f32 dist = 0.0f;
            for (u32 d = 0; d < dim_; ++d) {
                f32 diff = query[d] - centroids_[static_cast<size_t>(c) * dim_ + d];
                dist += diff * diff;
            }
            if (heap.size() < k) {
                heap.emplace_back(-dist, c);
                std::push_heap(heap.begin(), heap.end());
            } else if (-dist > heap[0].first) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {-dist, c};
                std::push_heap(heap.begin(), heap.end());
            }
        }
        std::sort(heap.begin(), heap.end());
        for (auto &[neg_dist, id] : heap) {
            out_ids.push_back(id);
            out_dists.push_back(-neg_dist);
        }
    }
}

// ========== Build / Insert Vector (Phase A) ==========

void SPFreshIndexInMem::Build(const f32 *vectors, u32 count) {
    if (count == 0 || num_centroids_ == 0) return;

    // Step 1: Run K-Means on ORIGINAL vectors (not rotated)
    std::vector<f32> centroid_vecs;
    u32 actual_centroids = GetKMeansCentroids<f32, f32>(
        MetricType::kMetricL2, dim_, count, vectors,
        centroid_vecs, num_centroids_, 10, 32, 256, 1.0f);

    if (centroid_vecs.empty() || actual_centroids == 0) {
        LOG_WARN("SPFresh Build: K-Means returned no centroids");
        return;
    }
    num_centroids_ = actual_centroids;
    centroids_.assign(centroid_vecs.begin(), centroid_vecs.end());

    bucket_metas_.resize(num_centroids_);
    delete[] bucket_locks_;
    bucket_locks_count_ = num_centroids_;
    bucket_locks_ = new std::mutex[bucket_locks_count_];
    running_means_.resize(num_centroids_);

    BuildCentroidIndex();

    for (u32 i = 0; i < count; ++i) {
        InsertVector(vectors, i);
        vectors += dim_;
    }

    LOG_INFO(fmt::format("SPFresh Build: {} vectors, {} centroids", count, num_centroids_));
}

void SPFreshIndexInMem::InsertVector(const f32 *vec, u32 local_id) {
    if (local_id >= max_vectors_) return;

    // Find nearest centroid in ORIGINAL space
    u32 centroid_id = FindNearestCentroid(vec);
    for (u32 r = 0; r < replica_count_; ++r) {
        bucket_assignments_[local_id * replica_count_ + r] = centroid_id;
    }
    bucket_metas_[centroid_id].base_count_++;

    // Encode with rotation (normalize → rotate → sign)
    size_t code_size = RabitQVecSize(dim_);
    char *code_ptr = rabitq_data_ + static_cast<size_t>(local_id) * code_size;
    EncodeWithRotation(reinterpret_cast<f32 *>(code_ptr), vec);

    running_means_[centroid_id].Update(vec, dim_);

    ++num_vectors_;
}

// ========== Incremental Insert with RNGSelection (Phase B) ==========

void SPFreshIndexInMem::InsertDelta(const f32 *vec, u32 row_id) {
    if (!delta_a_ || !delta_b_ || num_centroids_ == 0) return;

    // RNGSelection: find top replica_count_ centroids using ORIGINAL vector space
    std::vector<u32> centroid_ids;
    std::vector<f32> centroid_dists;
    FindTopKCentroids(vec, replica_count_, centroid_ids, centroid_dists);

    if (centroid_ids.empty()) {
        centroid_ids.push_back(0);
    }

    // Encode with rotation
    size_t code_size = RabitQVecSize(dim_);
    std::vector<char> code_buf(code_size);
    EncodeWithRotation(reinterpret_cast<f32 *>(code_buf.data()), vec);

    // Append to active delta buffer (store bucket_id + original row_id + code)
    {
        std::shared_lock lock(compact_mtx_);
        u32 idx = active_delta_idx_.load(std::memory_order_acquire);
        auto *active = (idx == 0) ? delta_a_ : delta_b_;

        for (u32 cid : centroid_ids) {
            active->Append(cid, row_id, code_buf.data());
            {
                std::lock_guard<std::mutex> blk(bucket_locks_[cid % bucket_locks_count_]);
                bucket_metas_[cid].delta_count_++;
            }
        }
    }

    LOG_TRACE(fmt::format("SPFresh InsertDelta: row={}, centroids={}", row_id, centroid_ids.size()));
}

// ========== Search with Centroid Pruning + Delta Merge (Phase A+B) ==========

void SPFreshIndexInMem::Search(const f32 *query, u32 dim,
                                const SearchCallback &callback) const {
    if (dim != dim_ || num_vectors_ == 0) return;

    f32 inv_sqrt_d = 1.0f / std::sqrt(static_cast<f32>(dim_));
    size_t code_size = RabitQVecSize(dim_);

    // Step 1: Normalize query (to unit sphere)
    f64 query_raw_norm = 0;
    for (u32 d = 0; d < dim_; ++d) {
        query_raw_norm += static_cast<f64>(query[d]) * query[d];
    }
    f64 inv_query_norm = (query_raw_norm > 1e-30) ? (1.0 / std::sqrt(query_raw_norm)) : 1.0;
    std::vector<f32> query_normed(dim_);
    for (u32 d = 0; d < dim_; ++d) {
        query_normed[d] = static_cast<f32>(static_cast<f64>(query[d]) * inv_query_norm);
    }

    // Step 2: Rotate normalized query
    std::vector<f32> rotated_query(dim_);
    ApplyRotation(query_normed.data(), rotated_query.data());

    // Step 3: Find top-K centroids using ORIGINAL (unrotated, normalized) query
    u32 search_centroid_k = std::min(static_cast<u32>(64), num_centroids_);
    std::vector<u32> candidate_centroids;
    std::vector<f32> centroid_dists;
    FindTopKCentroids(query_normed.data(), search_centroid_k, candidate_centroids, centroid_dists);

    if (candidate_centroids.empty()) return;

    // Build centroid filter mask
    std::vector<bool> centroid_mask(num_centroids_, false);
    for (u32 cid : candidate_centroids) {
        centroid_mask[cid] = true;
    }

    // Read deleted set snapshot
    std::unordered_set<u32> deleted_snapshot;
    {
        std::shared_lock lock(delete_mtx_);
        deleted_snapshot = deleted_set_;
    }

    // Step 4: Scan base array — only vectors assigned to candidate centroids
    // Base vector row_id = internal index (matches column-store offset within segment)
    for (u32 i = 0; i < num_vectors_; ++i) {
        if (deleted_snapshot.find(i) != deleted_snapshot.end()) continue;

        bool in_candidate = false;
        for (u32 r = 0; r < replica_count_; ++r) {
            u32 c = bucket_assignments_[i * replica_count_ + r];
            if (c < num_centroids_ && centroid_mask[c]) {
                in_candidate = true;
                break;
            }
        }
        if (!in_candidate) continue;

        const auto *code = reinterpret_cast<const RabitQVec *>(rabitq_data_ + static_cast<size_t>(i) * code_size);
        f32 dist = RabitQDistWithRotation(code, rotated_query.data(), dim_, inv_sqrt_d);
        callback(i, dist);
    }

    // Step 5: Scan delta entries — use stored ORIGINAL row_id in callback
    auto scan_delta = [&](const SPFreshDeltaBuffer *delta) {
        if (!delta || delta->entry_count_ == 0) return;
        for (u32 di = 0; di < delta->entry_count_; ++di) {
            u32 bucket_id = delta->GetBucketId(di);
            if (bucket_id >= num_centroids_ || !centroid_mask[bucket_id]) continue;

            u32 original_row_id = delta->GetRowId(di);
            if (deleted_snapshot.find(original_row_id) != deleted_snapshot.end()) continue;

            const auto *code = reinterpret_cast<const RabitQVec *>(delta->GetCode(di));
            f32 dist = RabitQDistWithRotation(code, rotated_query.data(), dim_, inv_sqrt_d);
            callback(original_row_id, dist);
        }
    };

    scan_delta(delta_a_);
    scan_delta(delta_b_);
}

// ========== Delete (Phase B) ==========

void SPFreshIndexInMem::MarkDeleted(u32 row_id) {
    std::lock_guard lock(delete_mtx_);
    deleted_set_.insert(row_id);
    LOG_TRACE(fmt::format("SPFresh MarkDeleted: row={}, total={}", row_id, deleted_set_.size()));
}

// ========== Compact with COW (Phase B+C) ==========

void SPFreshIndexInMem::Compact() {
    if (!delta_a_ || !delta_b_) return;
    std::lock_guard lock(compact_mtx_);
    size_t code_size = RabitQVecSize(dim_);

    // Snapshot deleted set
    std::unordered_set<u32> deleted;
    {
        std::shared_lock dlock(delete_mtx_);
        deleted = deleted_set_;
    }

    // Freeze the active delta
    u32 old_idx = active_delta_idx_.load(std::memory_order_acquire);
    u32 new_idx = 1 - old_idx;
    auto *old_delta = (old_idx == 0) ? delta_a_ : delta_b_;

    // Build list of (original_row_id, code_data) for all live vectors:
    // base vectors that are not deleted + delta entries that are not deleted
    struct LiveEntry {
        u32 original_row_id;
        const char *code;
    };
    std::vector<LiveEntry> live;
    live.reserve(num_vectors_ + old_delta->entry_count_);

    for (u32 i = 0; i < num_vectors_; ++i) {
        if (deleted.find(i) != deleted.end()) continue;
        live.push_back({i, static_cast<const char *>(rabitq_data_) + static_cast<size_t>(i) * code_size});
    }
    for (u32 di = 0; di < old_delta->entry_count_; ++di) {
        u32 orig_id = old_delta->GetRowId(di);
        if (deleted.find(orig_id) != deleted.end()) continue;
        live.push_back({orig_id, old_delta->GetCode(di)});
    }

    u32 new_count = static_cast<u32>(live.size());
    if (new_count == 0) {
        delete[] static_cast<char *>(rabitq_data_);
        rabitq_data_ = new char[code_size];
        std::memset(rabitq_data_, 0, code_size);
        num_vectors_ = 0;
        max_vectors_ = 1;
        old_delta->Clear();
        active_delta_idx_.store(new_idx, std::memory_order_release);
        return;
    }

    // Build new base data + remap bucket assignments
    auto *new_data = new char[static_cast<size_t>(new_count) * code_size];
    std::vector<u32> new_assignments(static_cast<size_t>(new_count) * replica_count_, u32(-1));
    // Rebuild original_row_id → new_index mapping
    std::unordered_map<u32, u32> old_to_new;
    old_to_new.reserve(new_count);

    for (u32 j = 0; j < new_count; ++j) {
        std::memcpy(new_data + static_cast<size_t>(j) * code_size, live[j].code, code_size);
        old_to_new[live[j].original_row_id] = j;
        // Copy bucket assignments from original if available
        u32 orig = live[j].original_row_id;
        if (orig < bucket_assignments_.size() / replica_count_) {
            for (u32 r = 0; r < replica_count_; ++r) {
                new_assignments[static_cast<size_t>(j) * replica_count_ + r] =
                    bucket_assignments_[static_cast<size_t>(orig) * replica_count_ + r];
            }
        }
    }

    // Atomic swap
    delete[] static_cast<char *>(rabitq_data_);
    rabitq_data_ = new_data;
    num_vectors_ = new_count;
    max_vectors_ = new_count;
    bucket_assignments_ = std::move(new_assignments);

    // Rebuild deleted_set_ with new indices
    std::unordered_set<u32> new_deleted;
    for (u32 old_id : deleted) {
        auto it = old_to_new.find(old_id);
        if (it != old_to_new.end()) {
            new_deleted.insert(it->second);
        }
    }
    {
        std::lock_guard dlock(delete_mtx_);
        deleted_set_ = std::move(new_deleted);
    }

    old_delta->Clear();
    active_delta_idx_.store(new_idx, std::memory_order_release);

    LOG_TRACE(fmt::format("SPFresh Compact: merged {} live vectors, {} deleted remapped",
                          new_count, new_deleted.size()));
}

// ========== Auto-Compact (Phase B) ==========

bool SPFreshIndexInMem::TryAutoCompact(u32 delta_threshold) {
    if (!delta_a_ || !delta_b_) return false;
    u32 idx = active_delta_idx_.load(std::memory_order_acquire);
    auto *active = (idx == 0) ? delta_a_ : delta_b_;
    if (active->entry_count_ >= delta_threshold) {
        Compact();
        return true;
    }
    return false;
}

// ========== SplitBucket (Phase C) ==========

std::vector<u32> SPFreshIndexInMem::SplitBucket(u32 bucket_id) {
    LOG_TRACE(fmt::format("SPFresh SplitBucket: bucket {}", bucket_id));

    std::vector<f32> bucket_vectors;
    std::vector<u32> bucket_local_ids;
    size_t code_size = RabitQVecSize(dim_);

    for (u32 i = 0; i < num_vectors_; ++i) {
        bool in_bucket = false;
        for (u32 r = 0; r < replica_count_; ++r) {
            u32 assn = bucket_assignments_[i * replica_count_ + r];
            if (assn == bucket_id) {
                in_bucket = true;
                break;
            }
        }
        if (in_bucket) {
            const auto *code = reinterpret_cast<const RabitQVec *>(rabitq_data_ + static_cast<size_t>(i) * code_size);
            size_t offset = bucket_vectors.size();
            bucket_vectors.resize(offset + dim_);
            DecodeWithRotation(code, bucket_vectors.data() + offset);
            bucket_local_ids.push_back(i);
        }
    }

    u32 n_in_bucket = static_cast<u32>(bucket_vectors.size() / dim_);
    if (n_in_bucket < 4) return {};

    std::vector<f32> split_centroids;
    u32 actual_k = GetKMeansCentroids<f32, f32>(
        MetricType::kMetricL2, dim_, n_in_bucket, bucket_vectors.data(),
        split_centroids, 2, 10, 4, 256, 1.0f);

    if (split_centroids.empty() || actual_k < 2) return {};

    bucket_metas_[bucket_id].is_retired_ = true;

    u32 new_c1 = num_centroids_++;
    u32 new_c2 = num_centroids_++;

    centroids_.resize(static_cast<size_t>(num_centroids_) * dim_);
    std::memcpy(centroids_.data() + static_cast<size_t>(new_c1) * dim_,
                split_centroids.data(), dim_ * sizeof(f32));
    std::memcpy(centroids_.data() + static_cast<size_t>(new_c2) * dim_,
                split_centroids.data() + dim_, dim_ * sizeof(f32));

    bucket_metas_.resize(num_centroids_);
    delete[] bucket_locks_;
    bucket_locks_count_ = num_centroids_;
    bucket_locks_ = new std::mutex[bucket_locks_count_];
    running_means_.resize(num_centroids_);

    for (u32 vi = 0; vi < n_in_bucket; ++vi) {
        u32 lid = bucket_local_ids[vi];
        const f32 *v = bucket_vectors.data() + static_cast<size_t>(vi) * dim_;

        f32 d1 = 0, d2 = 0;
        for (u32 d = 0; d < dim_; ++d) {
            f32 diff1 = v[d] - split_centroids[d];
            f32 diff2 = v[d] - split_centroids[dim_ + d];
            d1 += diff1 * diff1;
            d2 += diff2 * diff2;
        }
        u32 new_c = (d1 <= d2) ? new_c1 : new_c2;

        for (u32 r = 0; r < replica_count_; ++r) {
            if (bucket_assignments_[lid * replica_count_ + r] == bucket_id) {
                bucket_assignments_[lid * replica_count_ + r] = new_c;
            }
        }
    }

    BuildCentroidIndex();

    LOG_INFO(fmt::format("SPFresh SplitBucket: {} → {} (c1={}, c2={})",
                         bucket_id, num_centroids_, new_c1, new_c2));
    return {new_c1, new_c2};
}

// ========== Rebalance (Phase C) ==========

void SPFreshIndexInMem::Rebalance(u32 bucket_size_limit) {
    if (GetDeltaCount() > 0) Compact();
    if (num_centroids_ == 0) return;

    // WAL note: bucket split operations are driven by the rebalancer.
    // If a crash occurs during split, the split state is lost.
    // On recovery, the last dumped index is loaded and rows are replayed via WAL.
    // The rebalancer will re-detect overloaded buckets and split again.
    // This means splits are eventually consistent — no correctness issue, just a temporary
    // imbalance until the next rebalance cycle.

    std::vector<u32> bucket_counts(num_centroids_, 0);
    for (u32 i = 0; i < num_vectors_; ++i) {
        for (u32 r = 0; r < replica_count_; ++r) {
            u32 c = bucket_assignments_[i * replica_count_ + r];
            if (c < num_centroids_) bucket_counts[c]++;
        }
    }

    bool any_split = false;
    for (u32 c = 0; c < num_centroids_; ++c) {
        if (bucket_counts[c] > bucket_size_limit && !bucket_metas_[c].is_retired_) {
            auto new_ids = SplitBucket(c);
            if (!new_ids.empty()) any_split = true;
        }
    }

    if (any_split) {
        BuildCentroidIndex();
        LOG_INFO(fmt::format("SPFresh Rebalance: done, now {} centroids", num_centroids_));
    }
}

// ========== Persistence (Phase A+B) ==========

void SPFreshIndexInMem::Save(LocalFileHandle &file_handle) const {
    u32 magic = 0x50504652;
    u32 version = 4;
    file_handle.Append(&magic, sizeof(magic));
    file_handle.Append(&version, sizeof(version));

    file_handle.Append(&num_vectors_, sizeof(num_vectors_));
    file_handle.Append(&dim_, sizeof(dim_));

    size_t code_size = RabitQVecSize(dim_);
    u64 bucket_bytes = static_cast<u64>(num_vectors_) * code_size;
    file_handle.Append(&bucket_bytes, sizeof(bucket_bytes));
    if (num_vectors_ > 0) {
        file_handle.Append(static_cast<const char *>(rabitq_data_), bucket_bytes);
    }

    u64 rot_bytes = static_cast<u64>(dim_) * dim_ * sizeof(f32);
    file_handle.Append(&rot_bytes, sizeof(rot_bytes));
    if (rot_matrix_) {
        file_handle.Append(rot_matrix_, rot_bytes);
    }

    file_handle.Append(&num_centroids_, sizeof(num_centroids_));
    for (auto &meta : bucket_metas_) {
        file_handle.Append(&meta, sizeof(meta));
    }
    if (num_centroids_ > 0) {
        file_handle.Append(centroids_.data(), static_cast<size_t>(num_centroids_) * dim_ * sizeof(f32));
    }

    file_handle.Append(&replica_count_, sizeof(replica_count_));

    u64 assign_bytes = static_cast<u64>(num_vectors_) * replica_count_ * sizeof(u32);
    file_handle.Append(&assign_bytes, sizeof(assign_bytes));
    if (num_vectors_ > 0) {
        file_handle.Append(bucket_assignments_.data(), assign_bytes);
    }
}

void SPFreshIndexInMem::Load(LocalFileHandle &file_handle, size_t file_size) {
    u32 magic = 0;
    file_handle.Read(&magic, sizeof(magic));
    if (magic != 0x50504652) {
        UnrecoverableError("SPFreshIndexInMem: Invalid magic number");
        return;
    }

    u32 version = 0;
    file_handle.Read(&version, sizeof(version));

    file_handle.Read(&num_vectors_, sizeof(num_vectors_));
    file_handle.Read(&dim_, sizeof(dim_));

    size_t code_size = RabitQVecSize(dim_);

    u64 bucket_bytes = 0;
    file_handle.Read(&bucket_bytes, sizeof(bucket_bytes));
    size_t expected = static_cast<size_t>(num_vectors_) * code_size;
    if (bucket_bytes != expected) {
        UnrecoverableError("SPFreshIndexInMem: bucket size mismatch");
        return;
    }

    max_vectors_ = std::max(num_vectors_, 1u);
    delete[] static_cast<char *>(rabitq_data_);
    rabitq_data_ = new char[static_cast<size_t>(max_vectors_) * code_size];
    if (num_vectors_ > 0) {
        file_handle.Read(static_cast<char *>(rabitq_data_), bucket_bytes);
    }

    if (version >= 3) {
        u64 rot_bytes = 0;
        file_handle.Read(&rot_bytes, sizeof(rot_bytes));
        delete[] rot_matrix_;
        rot_matrix_ = new f32[rot_bytes / sizeof(f32)];
        file_handle.Read(rot_matrix_, rot_bytes);
    } else {
        delete[] rot_matrix_;
        rot_matrix_ = new f32[static_cast<size_t>(dim_) * dim_]();
        const_cast<SPFreshIndexInMem *>(this)->GenerateRotationMatrix();
    }

    if (version >= 1) {
        file_handle.Read(&num_centroids_, sizeof(num_centroids_));
        bucket_metas_.resize(num_centroids_);
        for (auto &meta : bucket_metas_) {
            file_handle.Read(&meta, sizeof(meta));
        }
        centroids_.resize(static_cast<size_t>(num_centroids_) * dim_);
        if (num_centroids_ > 0) {
            file_handle.Read(centroids_.data(), static_cast<size_t>(num_centroids_) * dim_ * sizeof(f32));
        }
    }

    if (version >= 4) {
        file_handle.Read(&replica_count_, sizeof(replica_count_));

        u64 assign_bytes = 0;
        file_handle.Read(&assign_bytes, sizeof(assign_bytes));
        if (num_vectors_ > 0) {
            bucket_assignments_.resize(static_cast<size_t>(num_vectors_) * replica_count_);
            file_handle.Read(bucket_assignments_.data(), assign_bytes);
        }
    } else {
        replica_count_ = 1;
        bucket_assignments_.resize(num_vectors_);
        for (u32 i = 0; i < num_vectors_; ++i) {
            bucket_assignments_[i] = FindNearestCentroid(
                reinterpret_cast<const f32 *>(rabitq_data_ + static_cast<size_t>(i) * code_size));
        }
    }

    delete[] bucket_locks_;
    bucket_locks_count_ = std::max(num_centroids_, 1u);
    bucket_locks_ = new std::mutex[bucket_locks_count_];
    running_means_.resize(num_centroids_);

    delete delta_a_;
    delete delta_b_;
    delta_a_ = new SPFreshDeltaBuffer(code_size);
    delta_b_ = new SPFreshDeltaBuffer(code_size);
    active_delta_idx_.store(0, std::memory_order_release);

    LOG_INFO(fmt::format("SPFresh Load: {} vectors, {} centroids, dim={}, version={}",
                         num_vectors_, num_centroids_, dim_, version));
}

void SPFreshIndexInMem::Dump(BufferObj *buffer_obj, size_t *p_dump_size) {}

// ========== BaseMemIndex interface ==========

MemIndexTracerInfo SPFreshIndexInMem::GetInfo() const {
    u32 delta_count = GetDeltaCount();
    size_t delta_mem = delta_count * (sizeof(u32) * 2 + RabitQVecSize(dim_));
    return MemIndexTracerInfo(std::make_shared<std::string>(index_name_),
                              std::make_shared<std::string>(table_name_),
                              std::make_shared<std::string>(db_name_),
                              mem_used_ + delta_mem,
                              GetRowCount());
}

const ChunkIndexMetaInfo SPFreshIndexInMem::GetChunkIndexMetaInfo() const {
    return ChunkIndexMetaInfo{"spfresh_chunk", begin_row_id_, GetRowCount(), 0, mem_used_};
}

} // namespace infinity

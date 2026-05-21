// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

module;

#include <immintrin.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

module infinity_core:spfresh_index.impl;

import :spfresh_index;
import :spfresh_defs;
import :index_spfresh;
import :kmeans_partition;
import :simd_functions;
import :infinity_exception;
import :local_file_handle;
import :logger;

import std;

import internal_types;
import embedding_info;

namespace infinity {

// P3.1: Background maintenance thread function
static void SpfreshMaintenanceLoop(std::stop_token st, SPFreshIndexInMem *index) {
    u32 cycle = 0;
    while (!st.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::seconds(60));
        if (st.stop_requested()) break;
        ++cycle;
        index->TryAutoCompact(8192);
        if (cycle % 10 == 0) {
            index->Rebalance(10000);
        }
    }
}

// ========== Construction / Destruction ==========

SPFreshIndexInMem::SPFreshIndexInMem(RowID begin_row_id, const IndexSPFresh *index_def, u32 embedding_dim, u32 max_vectors,
                                      const std::string &base_path)
    : begin_row_id_(begin_row_id), rabitq_data_(nullptr),
      num_vectors_(0), max_vectors_(max_vectors), dim_(embedding_dim),
      pad_dim_(1),
      max_delta_bytes_(static_cast<u64>(index_def ? index_def->max_delta_mb_ : 512) * 1024 * 1024),
      base_file_path_(base_path), base_fd_(-1), base_file_size_(0),
      hadamard_flip_(nullptr),
      num_centroids_(index_def ? index_def->num_centroids_ : 1000), coarse_count_(0),
      centroids_(), centroid_to_coarse_(), coarse_centroids_(), coarse_hnsw_(nullptr),
      bucket_metas_(), replica_count_(index_def ? index_def->replica_count_ : 1),
      bucket_assignments_(),
      delta_a_(nullptr), delta_b_(nullptr), active_delta_idx_(0),
      bucket_locks_(nullptr), bucket_locks_count_(0),
      running_means_(), deleted_set_(), mem_used_(0) {

    while (pad_dim_ < dim_) pad_dim_ <<= 1;
    size_t code_size = RabitQVecSize(dim_);
    size_t data_size = static_cast<size_t>(max_vectors) * code_size;

    // P2.3: File-backed storage (mmap) or DRAM
    if (!base_file_path_.empty()) {
        base_fd_ = ::open(base_file_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (base_fd_ >= 0) {
            base_file_size_ = data_size;
            ::ftruncate(base_fd_, static_cast<off_t>(base_file_size_));
            rabitq_data_ = static_cast<char *>(::mmap(nullptr, base_file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, base_fd_, 0));
            if (rabitq_data_ == MAP_FAILED) { rabitq_data_ = nullptr; ::close(base_fd_); base_fd_ = -1; }
        }
    }
    if (rabitq_data_ == nullptr) {
        // DRAM fallback
        rabitq_data_ = new char[data_size]();
    }
    mem_used_ += data_size;

    hadamard_flip_ = new bool[pad_dim_];
    GenerateHadamardParams();
    mem_used_ += pad_dim_ * sizeof(bool);

    centroids_.resize(static_cast<size_t>(num_centroids_) * dim_);
    bucket_metas_.resize(num_centroids_);
    bucket_assignments_.resize(static_cast<size_t>(max_vectors) * replica_count_, u32(-1));
    centroid_to_coarse_.resize(num_centroids_, 0);
    running_means_.resize(num_centroids_);

    bucket_locks_count_ = num_centroids_ > 0 ? num_centroids_ : 1;
    bucket_locks_ = new std::mutex[bucket_locks_count_];

    delta_a_ = new SPFreshDeltaBuffer(code_size);
    delta_b_ = new SPFreshDeltaBuffer(code_size);
}

SPFreshIndexInMem::~SPFreshIndexInMem() {
    if (base_fd_ >= 0 && rabitq_data_ != nullptr) {
        ::munmap(rabitq_data_, base_file_size_);
        ::close(base_fd_);
        ::unlink(base_file_path_.c_str());
    } else {
        delete[] rabitq_data_;
    }
    delete[] hadamard_flip_;
    delete[] bucket_locks_;
    delete delta_a_;
    delete delta_b_;
}

// ========== P1.1: Hadamard Transform ==========

void SPFreshIndexInMem::GenerateHadamardParams() {
    std::mt19937 rng(42);
    std::bernoulli_distribution dist(0.5);
    for (u32 i = 0; i < pad_dim_; ++i) hadamard_flip_[i] = dist(rng);
}

void SPFreshIndexInMem::ApplyHadamard(f32 *vec, u32 n) const {
    // In-place Walsh-Hadamard transform O(n log n)
    // n must be power of 2
    for (u32 len = 1; len < n; len <<= 1) {
        for (u32 i = 0; i < n; i += len * 2) {
            for (u32 j = 0; j < len; ++j) {
                f32 a = vec[i + j];
                f32 b = vec[i + j + len];
                vec[i + j] = a + b;
                vec[i + j + len] = a - b;
            }
        }
    }
    // Apply random signs
    for (u32 i = 0; i < n; ++i) {
        if (hadamard_flip_[i]) vec[i] = -vec[i];
    }
    // Normalize
    f32 inv = 1.0f / std::sqrt(static_cast<f32>(n));
    for (u32 i = 0; i < n; ++i) vec[i] *= inv;
}

void SPFreshIndexInMem::ApplyRotation(const f32 *vec, f32 *out) const {
    // Pad to power of 2, apply Hadamard, truncate back
    thread_local std::vector<f32> buf;
    if (buf.size() < pad_dim_) buf.resize(pad_dim_);
    std::memcpy(buf.data(), vec, dim_ * sizeof(f32));
    std::memset(buf.data() + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));
    ApplyHadamard(buf.data(), pad_dim_);
    std::memcpy(out, buf.data(), dim_ * sizeof(f32));
}

void SPFreshIndexInMem::ApplyInverseRotation(const f32 *vec, f32 *out) const {
    // Hadamard is self-inverse up to scaling factor (already handled in forward by 1/sqrt(n))
    // So inverse = same as forward
    ApplyRotation(vec, out);
}

// ========== RaBitQ Encode/Decode (P1.1: Hadamard) ==========

size_t SPFreshIndexInMem::EncodeWithRotation(f32 *code_buf, const f32 *vec) const {
    auto *code = reinterpret_cast<RabitQVec *>(code_buf);
    std::memset(code_buf, 0, RabitQVecSize(dim_));

    thread_local std::vector<f32> normed_buf;
    thread_local std::vector<f32> rot_buf;
    if (normed_buf.size() < pad_dim_) normed_buf.resize(pad_dim_);
    if (rot_buf.size() < pad_dim_) rot_buf.resize(pad_dim_);

    f64 raw_norm = 0;
    for (u32 d = 0; d < dim_; ++d) raw_norm += static_cast<f64>(vec[d]) * vec[d];
    code->raw_norm_ = static_cast<f32>(raw_norm);
    f64 inv = (raw_norm > 1e-30) ? (1.0 / std::sqrt(raw_norm)) : 1.0;

    std::memset(normed_buf.data(), 0, pad_dim_ * sizeof(f32));
    for (u32 d = 0; d < dim_; ++d) normed_buf[d] = static_cast<f32>(static_cast<f64>(vec[d]) * inv);

    // Apply Hadamard on padded vector
    std::memcpy(rot_buf.data(), normed_buf.data(), pad_dim_ * sizeof(f32));
    ApplyHadamard(rot_buf.data(), pad_dim_);

    f32 sum_pos = 0.0f;
    for (u32 d = 0; d < dim_; ++d) {
        if (rot_buf[d] >= 0) {
            code->compress_[d / 8] |= (1 << (d % 8));
            sum_pos += 1.0f;
        }
    }
    code->sum_ = sum_pos;
    code->norm_ = 1.0f;
    code->error_ = 0.95f;

    return RabitQVecSize(dim_);
}

void SPFreshIndexInMem::DecodeWithRotation(const RabitQVec *code, f32 *out_vec) const {
    thread_local std::vector<f32> buf;
    if (buf.size() < pad_dim_) buf.resize(pad_dim_);
    f32 mag = 1.0f / std::sqrt(static_cast<f32>(dim_));
    for (u32 d = 0; d < dim_; ++d) {
        buf[d] = ((code->compress_[d / 8] >> (d % 8)) & 1) ? mag : -mag;
    }
    std::memset(buf.data() + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));
    ApplyHadamard(buf.data(), pad_dim_);
    for (u32 d = 0; d < dim_; ++d) out_vec[d] = buf[d];
    f32 s = std::sqrt(code->raw_norm_);
    for (u32 d = 0; d < dim_; ++d) out_vec[d] *= s;
}

// ========== P0.2: SIMD RaBitQ Distance ==========

f32 SPFreshIndexInMem::RabitQDistWithRotation(const RabitQVec *code, const f32 *rotated_query_normed,
                                               u32 dim, f32 inv_sqrt_d) {
    // AVX2 SIMD-accelerated inner product (P0.2)
    // Process 8 dimensions at a time: sign bits from compress_[d/8], 8 lanes in __m256
    f32 ip = 0.0f;
    u32 d = 0;
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    for (; d + 8 <= dim; d += 8) {
        __m256 q = _mm256_loadu_ps(rotated_query_normed + d);
        u8 byte = code->compress_[d / 8];
        // Create sign mask: bit0→lane0..bit7→lane7
        __m256i idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i shl = _mm256_sllv_epi32(_mm256_set1_epi32(byte), idx);
        __m256i sign = _mm256_srai_epi32(shl, 7); // 0 or -1
        // q * sign(bit) = q if bit=1, -q if bit=0
        q = _mm256_xor_ps(q, _mm256_castsi256_ps(sign));
        sum = _mm256_add_ps(sum, q);
    }
    // Horizontal sum across lanes
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    ip = _mm_cvtss_f32(lo);
#endif
    for (; d < dim; ++d) {
        ip += ((code->compress_[d / 8] >> (d % 8)) & 1) ? rotated_query_normed[d] : -rotated_query_normed[d];
    }

    f32 cos_theta = ip * inv_sqrt_d;
    cos_theta = std::clamp(cos_theta, -1.0f, 1.0f);
    return 2.0f * (1.0f - cos_theta);
}

// ========== Centroid helpers ==========

u32 SPFreshIndexInMem::FindNearestCentroid(const f32 *vec) const {
    u32 best = 0;
    f32 best_dist = std::numeric_limits<f32>::max();
    for (u32 c = 0; c < num_centroids_; ++c) {
        f32 dist = 0.0f;
        for (u32 d = 0; d < dim_; ++d) {
            f32 diff = vec[d] - centroids_[static_cast<size_t>(c) * dim_ + d];
            dist += diff * diff;
        }
        if (dist < best_dist) { best_dist = dist; best = c; }
    }
    return best;
}

// ========== P1.2: Hierarchical Centroid Index ==========

void SPFreshIndexInMem::BuildCentroidIndex() {
    if (num_centroids_ == 0 || dim_ == 0) return;

    // Build coarse level: ∼sqrt(num_centroids_) coarse centroids via K-Means on fine centroids
    coarse_count_ = std::max(1u, static_cast<u32>(std::sqrt(static_cast<f64>(num_centroids_))));
    if (coarse_count_ > num_centroids_) coarse_count_ = num_centroids_;

    std::vector<f32> coarse_vecs;
    u32 actual_coarse = GetKMeansCentroids<f32, f32>(
        MetricType::kMetricL2, dim_, num_centroids_, centroids_.data(),
        coarse_vecs, coarse_count_, 5, 4, 256, 1.0f);

    if (coarse_vecs.empty() || actual_coarse == 0) {
        coarse_count_ = 1;
        coarse_centroids_.assign(centroids_.begin(), centroids_.begin() + dim_);
    } else {
        coarse_count_ = actual_coarse;
        coarse_centroids_.assign(coarse_vecs.begin(), coarse_vecs.end());
    }

    // Assign each fine centroid to nearest coarse
    centroid_to_coarse_.resize(num_centroids_);
    for (u32 c = 0; c < num_centroids_; ++c) {
        const f32 *cv = centroids_.data() + static_cast<size_t>(c) * dim_;
        u32 best_coarse = 0;
        f32 best_dist = std::numeric_limits<f32>::max();
        for (u32 cc = 0; cc < coarse_count_; ++cc) {
            f32 dist = 0.0f;
            for (u32 d = 0; d < dim_; ++d) {
                f32 diff = cv[d] - coarse_centroids_[static_cast<size_t>(cc) * dim_ + d];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_coarse = cc; }
        }
        centroid_to_coarse_[c] = best_coarse;
    }

    // Build HNSW on coarse centroids (>32 coarse)
    if (coarse_count_ > 32) {
        u32 chunk = 1;
        while (chunk < coarse_count_ + 16) chunk <<= 1;
        auto hnsw = HnswType::Make(chunk, 1, dim_, 16, 200);
        if (hnsw) {
            auto iter = DenseVectorIter<f32, u32>(coarse_centroids_.data(), dim_, coarse_count_);
            HnswInsertConfig cfg{true};
            hnsw->InsertVecs(iter, cfg);
            coarse_hnsw_ = std::move(hnsw);
            LOG_TRACE(fmt::format("SPFresh BuildCentroidIndex: {} coarse (HNSW), {} fine", coarse_count_, num_centroids_));
            return;
        }
    }
    coarse_hnsw_ = nullptr;
    LOG_TRACE(fmt::format("SPFresh BuildCentroidIndex: {} coarse (brute), {} fine", coarse_count_, num_centroids_));
}

void SPFreshIndexInMem::FindTopKCentroids(const f32 *query, u32 top_k,
                                           std::vector<u32> &out_ids,
                                           std::vector<f32> &out_dists) const {
    out_ids.clear();
    out_dists.clear();
    if (num_centroids_ == 0) return;

    u32 k = std::min(top_k, num_centroids_);
    u32 search_coarse = std::min(static_cast<u32>(std::ceil(static_cast<f64>(k) * 4.0 / (num_centroids_ / std::max(1u, coarse_count_)))), coarse_count_);
    if (search_coarse == 0) search_coarse = std::min(8u, coarse_count_);

    // Step 1: Find top coarse centroids
    std::vector<u32> coarse_ids;
    if (coarse_hnsw_ && coarse_count_ > 0 && search_coarse > 0) {
        KnnSearchOption opt;
        opt.ef_ = search_coarse * 4;
        auto [cnt, dptr, lptr] = coarse_hnsw_->template KnnSearch<>(
            static_cast<const f32 *>(query), search_coarse, std::nullopt, opt);
        for (i64 i = 0; i < static_cast<i64>(cnt) && i < static_cast<i64>(search_coarse); ++i) {
            coarse_ids.push_back(lptr[i]);
        }
    } else {
        for (u32 cc = 0; cc < coarse_count_; ++cc) {
            coarse_ids.push_back(cc);
        }
    }

    if (coarse_ids.empty()) { coarse_ids.push_back(0); }

    // Build coarse filter
    std::vector<bool> coarse_mask(coarse_count_, false);
    for (u32 cid : coarse_ids) coarse_mask[cid] = true;

    // Step 2: Search fine centroids only in selected coarse groups
    std::vector<std::pair<f32, u32>> heap;
    heap.reserve(k + 1);

    for (u32 c = 0; c < num_centroids_; ++c) {
        if (!coarse_mask[centroid_to_coarse_[c]]) continue;
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
    for (auto &[nd, id] : heap) {
        out_ids.push_back(id);
        out_dists.push_back(-nd);
    }
}

// ========== Build ==========

void SPFreshIndexInMem::Build(const f32 *vectors, u32 count) {
    if (count == 0 || num_centroids_ == 0) return;

    std::lock_guard wlock(global_mtx_);

    // K-Means
    std::vector<f32> centroid_vecs;
    u32 actual = GetKMeansCentroids<f32, f32>(
        MetricType::kMetricL2, dim_, count, vectors,
        centroid_vecs, num_centroids_, 10, 32, 256, 1.0f);

    if (centroid_vecs.empty() || actual == 0) { LOG_WARN("SPFresh Build: K-Means empty"); return; }
    num_centroids_ = actual;
    centroids_.assign(centroid_vecs.begin(), centroid_vecs.end());
    centroid_to_coarse_.resize(num_centroids_);

    bucket_metas_.resize(num_centroids_);
    delete[] bucket_locks_;
    bucket_locks_count_ = num_centroids_;
    bucket_locks_ = new std::mutex[bucket_locks_count_];
    running_means_.resize(num_centroids_);

    BuildCentroidIndex();

    // SIMD batch centroid assignment
    size_t code_size = RabitQVecSize(dim_);
    std::vector<u32> assign_labels(count);
    std::vector<f32> assign_dists(count);
    auto search_fn = GetSIMD_FUNCTIONS().SearchTop1WithDisF32U32_func_ptr_;
    search_fn(dim_, count, vectors, num_centroids_, centroids_.data(),
              assign_labels.data(), assign_dists.data());

    // Batch normalize + rotate
    std::vector<f32> batch_buf(static_cast<size_t>(count) * pad_dim_);
    thread_local std::vector<f32> rot_tmp;
    if (rot_tmp.size() < pad_dim_) rot_tmp.resize(pad_dim_);

    // Normalize into padded buffer
    for (u32 i = 0; i < count; ++i) {
        const f32 *vec = vectors + static_cast<size_t>(i) * dim_;
        f32 *out = batch_buf.data() + static_cast<size_t>(i) * pad_dim_;
        f64 raw_norm = 0;
        for (u32 d = 0; d < dim_; ++d) raw_norm += static_cast<f64>(vec[d]) * vec[d];
        f64 inv = (raw_norm > 1e-30) ? (1.0 / std::sqrt(raw_norm)) : 1.0;
        for (u32 d = 0; d < dim_; ++d) out[d] = static_cast<f32>(static_cast<f64>(vec[d]) * inv);
        std::memset(out + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));
        ApplyHadamard(out, pad_dim_);
    }

    // Encode
    for (u32 i = 0; i < count; ++i) {
        const f32 *vec = vectors + static_cast<size_t>(i) * dim_;
        const f32 *rotated = batch_buf.data() + static_cast<size_t>(i) * pad_dim_;
        u32 cid = assign_labels[i];
        for (u32 r = 0; r < replica_count_; ++r) bucket_assignments_[static_cast<size_t>(i) * replica_count_ + r] = cid;
        bucket_metas_[cid].base_count_++;

        auto *code = reinterpret_cast<RabitQVec *>(rabitq_data_ + static_cast<size_t>(i) * code_size);
        std::memset(code, 0, code_size);
        f64 raw_norm = 0;
        for (u32 d = 0; d < dim_; ++d) raw_norm += static_cast<f64>(vec[d]) * vec[d];
        code->raw_norm_ = static_cast<f32>(raw_norm);
        code->norm_ = 1.0f;
        code->error_ = 0.95f;
        f32 sp = 0.0f;
        for (u32 d = 0; d < dim_; ++d) {
            if (rotated[d] >= 0) { code->compress_[d / 8] |= (1 << (d % 8)); sp += 1.0f; }
        }
        code->sum_ = sp;
        running_means_[cid].Update(vec, dim_);
    }

    num_vectors_ = count;
    LOG_INFO(fmt::format("SPFresh Build: {} vectors, {} fine, {} coarse", count, num_centroids_, coarse_count_));
}

void SPFreshIndexInMem::InsertVector(const f32 *vec, u32 local_id) {
    if (local_id >= max_vectors_) return;
    std::shared_lock rlock(global_mtx_);
    u32 cid = FindNearestCentroid(vec);
    for (u32 r = 0; r < replica_count_; ++r) bucket_assignments_[local_id * replica_count_ + r] = cid;
    bucket_metas_[cid].base_count_++;
    size_t code_size = RabitQVecSize(dim_);
    char *ptr = rabitq_data_ + static_cast<size_t>(local_id) * code_size;
    EncodeWithRotation(reinterpret_cast<f32 *>(ptr), vec);
    running_means_[cid].Update(vec, dim_);
    ++num_vectors_;
}

// ========== P3.1: Background Maintenance ==========

void SPFreshIndexInMem::StartBackgroundMaintenance() {
    if (maintenance_thread_.joinable()) return;
    maintenance_stop_.store(false, std::memory_order_relaxed);
    maintenance_thread_ = std::jthread(SpfreshMaintenanceLoop, this);
}

void SPFreshIndexInMem::StopBackgroundMaintenance() {
    if (maintenance_thread_.joinable()) {
        maintenance_thread_.request_stop();
        maintenance_thread_.join();
    }
}

// ========== Incremental Insert (P0.1: protected by global_mtx_) ==========

void SPFreshIndexInMem::InsertDelta(const f32 *vec, u32 row_id) {
    if (!delta_a_ || !delta_b_ || num_centroids_ == 0) return;

    // P2.2: Check memory budget — auto-compact if delta exceeds limit
    u32 idx_snap = active_delta_idx_.load(std::memory_order_acquire);
    auto *active_snap = (idx_snap == 0) ? delta_a_ : delta_b_;
    if (active_snap->data_.size() > max_delta_bytes_) {
        Compact();
    }

    std::vector<u32> centroid_ids;
    std::vector<f32> centroid_dists;
    {
        std::shared_lock rlock(global_mtx_);
        FindTopKCentroids(vec, replica_count_, centroid_ids, centroid_dists);
    }
    if (centroid_ids.empty()) centroid_ids.push_back(0);

    size_t code_size = RabitQVecSize(dim_);
    std::vector<char> code_buf(code_size);
    EncodeWithRotation(reinterpret_cast<f32 *>(code_buf.data()), vec);

    {
        std::shared_lock lock(compact_mtx_);
        u32 idx = active_delta_idx_.load(std::memory_order_acquire);
        auto *active = (idx == 0) ? delta_a_ : delta_b_;
        for (u32 cid : centroid_ids) {
            active->Append(cid, row_id, code_buf.data());
            std::lock_guard<std::mutex> blk(bucket_locks_[cid % bucket_locks_count_]);
            bucket_metas_[cid].delta_count_++;
        }
    }
}

// ========== P0.2+P0.3: SIMD Search + Atomic Compact ==========

void SPFreshIndexInMem::Search(const f32 *query, u32 dim,
                                const SearchCallback &callback) const {
    if (dim != dim_ || num_vectors_ == 0) return;

    thread_local std::vector<f32> qnorm;
    thread_local std::vector<f32> qrot;
    if (qnorm.size() < pad_dim_) qnorm.resize(pad_dim_);
    if (qrot.size() < pad_dim_) qrot.resize(pad_dim_);

    f32 inv_sqrt_d = 1.0f / std::sqrt(static_cast<f32>(dim_));
    size_t code_size = RabitQVecSize(dim_);

    // Normalize + rotate query via Hadamard
    f64 qn = 0;
    for (u32 d = 0; d < dim_; ++d) qn += static_cast<f64>(query[d]) * query[d];
    f64 inv = (qn > 1e-30) ? (1.0 / std::sqrt(qn)) : 1.0;
    for (u32 d = 0; d < dim_; ++d) qrot[d] = static_cast<f32>(static_cast<f64>(query[d]) * inv);
    std::memset(qrot.data() + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));
    ApplyHadamard(qrot.data(), pad_dim_);

    // P0.1: Snapshot under global read lock
    std::vector<u32> local_assignments;
    std::unordered_set<u32> deleted_snapshot;
    u32 local_num_centroids, local_replica;
    u32 local_num_vectors;
    {
        std::shared_lock rlock(global_mtx_);
        local_num_centroids = num_centroids_;
        local_replica = replica_count_;
        local_num_vectors = num_vectors_;
        local_assignments = bucket_assignments_;
    }
    {
        std::shared_lock dlock(delete_mtx_);
        deleted_snapshot = deleted_set_;
    }

    // P1.2: Find top centroids via hierarchy
    std::vector<u32> cand_centroids;
    std::vector<f32> cand_dists;
    {
        std::shared_lock rlock(global_mtx_);
        FindTopKCentroids(qrot.data(), std::min(64u, local_num_centroids), cand_centroids, cand_dists);
    }
    if (cand_centroids.empty()) return;

    std::vector<bool> cmask(local_num_centroids, false);
    for (u32 cid : cand_centroids) cmask[cid] = true;

    // Scan base
    for (u32 i = 0; i < local_num_vectors; ++i) {
        if (deleted_snapshot.find(i) != deleted_snapshot.end()) continue;
        bool in_c = false;
        for (u32 r = 0; r < local_replica; ++r) {
            u32 c = local_assignments[i * local_replica + r];
            if (c < local_num_centroids && cmask[c]) { in_c = true; break; }
        }
        if (!in_c) continue;
        const auto *code = reinterpret_cast<const RabitQVec *>(rabitq_data_ + static_cast<size_t>(i) * code_size);
        callback(i, RabitQDistWithRotation(code, qrot.data(), dim_, inv_sqrt_d));
    }

    // Scan delta
    auto scan_delta = [&](const SPFreshDeltaBuffer *delta) {
        if (!delta || delta->entry_count_ == 0) return;
        for (u32 di = 0; di < delta->entry_count_; ++di) {
            u32 bid = delta->GetBucketId(di);
            if (bid >= local_num_centroids || !cmask[bid]) continue;
            u32 rid = delta->GetRowId(di);
            if (deleted_snapshot.find(rid) != deleted_snapshot.end()) continue;
            const auto *code = reinterpret_cast<const RabitQVec *>(delta->GetCode(di));
            callback(rid, RabitQDistWithRotation(code, qrot.data(), dim_, inv_sqrt_d));
        }
    };
    scan_delta(delta_a_);
    scan_delta(delta_b_);
}

void SPFreshIndexInMem::MarkDeleted(u32 row_id) {
    std::lock_guard lock(delete_mtx_);
    deleted_set_.insert(row_id);
}

// ========== P0.3 + P2.3: Atomic Compact (mmap-aware) ==========

void SPFreshIndexInMem::Compact() {
    if (!delta_a_ || !delta_b_) return;
    std::lock_guard lock(compact_mtx_);
    size_t code_size = RabitQVecSize(dim_);

    std::unordered_set<u32> deleted;
    { std::shared_lock dlock(delete_mtx_); deleted = deleted_set_; }

    u32 old_idx = active_delta_idx_.load(std::memory_order_acquire);
    u32 new_idx = 1 - old_idx;
    auto *old_delta = (old_idx == 0) ? delta_a_ : delta_b_;

    // Collect live entries
    struct Entry { u32 rid; const char *code; };
    std::vector<Entry> live;
    live.reserve(num_vectors_ + old_delta->entry_count_);
    for (u32 i = 0; i < num_vectors_; ++i)
        if (deleted.find(i) == deleted.end())
            live.push_back({i, static_cast<const char *>(rabitq_data_) + static_cast<size_t>(i) * code_size});
    for (u32 di = 0; di < old_delta->entry_count_; ++di) {
        u32 orig = old_delta->GetRowId(di);
        if (deleted.find(orig) == deleted.end())
            live.push_back({orig, old_delta->GetCode(di)});
    }

    u32 new_count = static_cast<u32>(live.size());
    size_t new_data_size = static_cast<size_t>(new_count) * code_size;

    // Handle empty case
    if (new_count == 0) {
        char *new_empty = nullptr;
        if (base_fd_ >= 0) {
            ::ftruncate(base_fd_, static_cast<off_t>(code_size));
            new_empty = static_cast<char *>(::mmap(nullptr, code_size, PROT_READ | PROT_WRITE, MAP_SHARED, base_fd_, 0));
            ::munmap(rabitq_data_, base_file_size_);
            rabitq_data_ = (new_empty != MAP_FAILED) ? new_empty : new char[code_size]();
            base_file_size_ = code_size;
        } else {
            delete[] static_cast<char *>(rabitq_data_);
            rabitq_data_ = new char[code_size]();
        }
        std::memset(rabitq_data_, 0, code_size);
        num_vectors_ = 0; max_vectors_ = 1;
        old_delta->Clear(); active_delta_idx_.store(new_idx, std::memory_order_release);
        return;
    }

    // Build temp buffer with live data
    std::vector<char> tmp_buf(new_data_size);
    std::vector<u32> new_assign(static_cast<size_t>(new_count) * replica_count_, u32(-1));
    std::unordered_map<u32, u32> old2new;
    old2new.reserve(new_count);

    for (u32 j = 0; j < new_count; ++j) {
        std::memcpy(tmp_buf.data() + static_cast<size_t>(j) * code_size, live[j].code, code_size);
        old2new[live[j].rid] = j;
        u32 orig = live[j].rid;
        if (orig < bucket_assignments_.size() / replica_count_) {
            for (u32 r = 0; r < replica_count_; ++r)
                new_assign[static_cast<size_t>(j) * replica_count_ + r] =
                    bucket_assignments_[static_cast<size_t>(orig) * replica_count_ + r];
        }
    }

    // Atomic swap (P0.3) with mmap support (P2.3)
    {
        std::lock_guard wlock(global_mtx_);

        if (base_fd_ >= 0) {
            // Write temp file → rename → munmap old → ftruncate → mmap new
            std::string tmp_path = base_file_path_ + ".compact";
            int tmp_fd = ::open(tmp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
            if (tmp_fd >= 0) {
                ::write(tmp_fd, tmp_buf.data(), new_data_size);
                ::close(tmp_fd);
                if (::rename(tmp_path.c_str(), base_file_path_.c_str()) == 0) {
                    ::munmap(rabitq_data_, base_file_size_);
                    ::ftruncate(base_fd_, static_cast<off_t>(new_data_size));
                    char *m = static_cast<char *>(::mmap(nullptr, new_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, base_fd_, 0));
                    rabitq_data_ = (m != MAP_FAILED) ? m : new char[new_data_size]();
                    base_file_size_ = new_data_size;
                }
            }
        } else {
            // DRAM mode: allocate new, copy, delete old
            auto *new_data = new char[new_data_size];
            std::memcpy(new_data, tmp_buf.data(), new_data_size);
            delete[] static_cast<char *>(rabitq_data_);
            rabitq_data_ = new_data;
        }

        num_vectors_ = new_count;
        max_vectors_ = new_count;
        bucket_assignments_ = std::move(new_assign);
    }

    // Rebuild deleted_set_
    std::unordered_set<u32> nd;
    for (u32 oid : deleted) {
        auto it = old2new.find(oid);
        if (it != old2new.end()) nd.insert(it->second);
    }
    { std::lock_guard dlock(delete_mtx_); deleted_set_ = std::move(nd); }

    old_delta->Clear();
    active_delta_idx_.store(new_idx, std::memory_order_release);
    compact_count_.fetch_add(1, std::memory_order_relaxed);

    // P3.3: Track peak delta
    size_t da = delta_a_->data_.size() + delta_b_->data_.size();
    size_t prev = peak_delta_bytes_.load(std::memory_order_relaxed);
    while (da > prev && !peak_delta_bytes_.compare_exchange_weak(prev, da, std::memory_order_relaxed)) {}
}

// ========== Auto Compact ==========

bool SPFreshIndexInMem::TryAutoCompact(u32 threshold) {
    if (!delta_a_ || !delta_b_) return false;
    u32 idx = active_delta_idx_.load(std::memory_order_acquire);
    auto *active = (idx == 0) ? delta_a_ : delta_b_;
    if (active->entry_count_ >= threshold) { Compact(); return true; }
    return false;
}

// ========== P0.1 + P0.3: SplitBucket (protected by global_mtx_) ==========

std::vector<u32> SPFreshIndexInMem::SplitBucket(u32 bucket_id) {
    std::unique_lock wlock(global_mtx_);
    LOG_TRACE(fmt::format("SPFresh SplitBucket: bucket {}", bucket_id));

    std::vector<f32> bvecs;
    std::vector<u32> bids;
    size_t code_size = RabitQVecSize(dim_);
    for (u32 i = 0; i < num_vectors_; ++i) {
        bool in = false;
        for (u32 r = 0; r < replica_count_; ++r) {
            if (bucket_assignments_[i * replica_count_ + r] == bucket_id) { in = true; break; }
        }
        if (in) {
            const auto *code = reinterpret_cast<const RabitQVec *>(rabitq_data_ + static_cast<size_t>(i) * code_size);
            size_t off = bvecs.size(); bvecs.resize(off + dim_);
            DecodeWithRotation(code, bvecs.data() + off);
            bids.push_back(i);
        }
    }

    u32 n = static_cast<u32>(bvecs.size() / dim_);
    if (n < 4) return {};

    std::vector<f32> split_c;
    u32 ak = GetKMeansCentroids<f32, f32>(MetricType::kMetricL2, dim_, n, bvecs.data(),
                                           split_c, 2, 10, 4, 256, 1.0f);
    if (split_c.empty() || ak < 2) return {};

    bucket_metas_[bucket_id].is_retired_ = true;

    u32 nc1 = num_centroids_++;
    u32 nc2 = num_centroids_++;

    centroids_.resize(static_cast<size_t>(num_centroids_) * dim_);
    std::memcpy(centroids_.data() + static_cast<size_t>(nc1) * dim_, split_c.data(), dim_ * sizeof(f32));
    std::memcpy(centroids_.data() + static_cast<size_t>(nc2) * dim_, split_c.data() + dim_, dim_ * sizeof(f32));

    bucket_metas_.resize(num_centroids_);
    centroid_to_coarse_.resize(num_centroids_);
    centroid_to_coarse_[nc1] = centroid_to_coarse_[bucket_id];
    centroid_to_coarse_[nc2] = centroid_to_coarse_[bucket_id];

    // Reallocate bucket_locks_
    delete[] bucket_locks_;
    bucket_locks_count_ = num_centroids_;
    bucket_locks_ = new std::mutex[bucket_locks_count_];
    running_means_.resize(num_centroids_);

    for (u32 vi = 0; vi < n; ++vi) {
        u32 lid = bids[vi];
        const f32 *v = bvecs.data() + static_cast<size_t>(vi) * dim_;
        f32 d1 = 0, d2 = 0;
        for (u32 d = 0; d < dim_; ++d) {
            f32 df1 = v[d] - split_c[d], df2 = v[d] - split_c[dim_ + d];
            d1 += df1 * df1; d2 += df2 * df2;
        }
        u32 nc = (d1 <= d2) ? nc1 : nc2;
        for (u32 r = 0; r < replica_count_; ++r) {
            if (bucket_assignments_[lid * replica_count_ + r] == bucket_id)
                bucket_assignments_[lid * replica_count_ + r] = nc;
        }
    }

    split_count_.fetch_add(1, std::memory_order_relaxed);
    BuildCentroidIndex();
    LOG_INFO(fmt::format("SPFresh SplitBucket: {} → {} (c1={}, c2={})", bucket_id, num_centroids_, nc1, nc2));
    return {nc1, nc2};
}

void SPFreshIndexInMem::Rebalance(u32 bucket_size_limit) {
    if (GetDeltaCount() > 0) Compact();
    if (num_centroids_ == 0) return;
    {
        std::shared_lock rlock(global_mtx_);
        std::vector<u32> cnt(num_centroids_, 0);
        for (u32 i = 0; i < num_vectors_; ++i)
            for (u32 r = 0; r < replica_count_; ++r)
                if (bucket_assignments_[i * replica_count_ + r] < num_centroids_)
                    cnt[bucket_assignments_[i * replica_count_ + r]]++;

        bool any = false;
        for (u32 c = 0; c < num_centroids_; ++c) {
            if (cnt[c] > bucket_size_limit && !bucket_metas_[c].is_retired_) {
                auto ids = SplitBucket(c);
                if (!ids.empty()) any = true;
            }
        }
        if (any) { BuildCentroidIndex(); LOG_INFO(fmt::format("SPFresh Rebalance: done, {} centroids", num_centroids_)); }
    }
}

// ========== Persistence (Save v5: Hadamard + hierarchy) ==========

void SPFreshIndexInMem::Save(LocalFileHandle &fh) const {
    u32 magic = 0x50504652, version = 5;
    fh.Append(&magic, sizeof(magic));
    fh.Append(&version, sizeof(version));
    fh.Append(&num_vectors_, sizeof(num_vectors_));
    fh.Append(&dim_, sizeof(dim_));
    fh.Append(&pad_dim_, sizeof(pad_dim_));

    size_t cs = RabitQVecSize(dim_);
    u64 bb = static_cast<u64>(num_vectors_) * cs;
    fh.Append(&bb, sizeof(bb));
    if (num_vectors_ > 0) fh.Append(static_cast<const char *>(rabitq_data_), bb);

    // Hadamard flip signs
    u64 hb = static_cast<u64>(pad_dim_) * sizeof(bool);
    fh.Append(&hb, sizeof(hb));
    fh.Append(hadamard_flip_, hb);

    // Centroids
    fh.Append(&num_centroids_, sizeof(num_centroids_));
    fh.Append(&coarse_count_, sizeof(coarse_count_));
    for (auto &m : bucket_metas_) fh.Append(&m, sizeof(m));
    if (num_centroids_ > 0) fh.Append(centroids_.data(), static_cast<size_t>(num_centroids_) * dim_ * sizeof(f32));
    if (coarse_count_ > 0) fh.Append(coarse_centroids_.data(), static_cast<size_t>(coarse_count_) * dim_ * sizeof(f32));
    if (num_centroids_ > 0) fh.Append(centroid_to_coarse_.data(), num_centroids_ * sizeof(u32));

    fh.Append(&replica_count_, sizeof(replica_count_));
    u64 ab = static_cast<u64>(num_vectors_) * replica_count_ * sizeof(u32);
    fh.Append(&ab, sizeof(ab));
    if (num_vectors_ > 0) fh.Append(bucket_assignments_.data(), ab);
}

void SPFreshIndexInMem::Load(LocalFileHandle &fh, size_t file_size) {
    u32 magic = 0;
    fh.Read(&magic, sizeof(magic));
    if (magic != 0x50504652) { UnrecoverableError("SPFresh: bad magic"); return; }
    u32 version = 0;
    fh.Read(&version, sizeof(version));
    fh.Read(&num_vectors_, sizeof(num_vectors_));
    fh.Read(&dim_, sizeof(dim_));
    fh.Read(&pad_dim_, sizeof(pad_dim_));

    size_t cs = RabitQVecSize(dim_);
    u64 bb = 0;
    fh.Read(&bb, sizeof(bb));
    size_t exp = static_cast<size_t>(num_vectors_) * cs;
    if (bb != exp) { UnrecoverableError("SPFresh: bucket size mismatch"); return; }

    max_vectors_ = std::max(num_vectors_, 1u);
    delete[] rabitq_data_; rabitq_data_ = new char[static_cast<size_t>(max_vectors_) * cs];
    if (num_vectors_ > 0) fh.Read(static_cast<char *>(rabitq_data_), bb);

    delete[] hadamard_flip_;
    if (version >= 5) {
        u64 hb = 0;
        fh.Read(&hb, sizeof(hb));
        hadamard_flip_ = new bool[hb / sizeof(bool)];
        fh.Read(hadamard_flip_, hb);
    } else {
        hadamard_flip_ = new bool[pad_dim_];
        GenerateHadamardParams();
    }

    if (version >= 1) {
        fh.Read(&num_centroids_, sizeof(num_centroids_));
        if (version >= 5) fh.Read(&coarse_count_, sizeof(coarse_count_));
        bucket_metas_.resize(num_centroids_);
        for (auto &m : bucket_metas_) fh.Read(&m, sizeof(m));
        centroids_.resize(static_cast<size_t>(num_centroids_) * dim_);
        if (num_centroids_ > 0) fh.Read(centroids_.data(), static_cast<size_t>(num_centroids_) * dim_ * sizeof(f32));
        if (version >= 5 && coarse_count_ > 0) {
            coarse_centroids_.resize(static_cast<size_t>(coarse_count_) * dim_);
            fh.Read(coarse_centroids_.data(), static_cast<size_t>(coarse_count_) * dim_ * sizeof(f32));
            centroid_to_coarse_.resize(num_centroids_);
            fh.Read(centroid_to_coarse_.data(), num_centroids_ * sizeof(u32));
        } else {
            coarse_count_ = 1;
            coarse_centroids_.resize(dim_);
            std::memcpy(coarse_centroids_.data(), centroids_.data(), dim_ * sizeof(f32));
            centroid_to_coarse_.assign(num_centroids_, 0);
        }
    }

    if (version >= 4) {
        fh.Read(&replica_count_, sizeof(replica_count_));
        u64 ab = 0;
        fh.Read(&ab, sizeof(ab));
        if (num_vectors_ > 0) {
            bucket_assignments_.resize(static_cast<size_t>(num_vectors_) * replica_count_);
            fh.Read(bucket_assignments_.data(), ab);
        }
    } else {
        replica_count_ = 1;
        bucket_assignments_.resize(num_vectors_);
        for (u32 i = 0; i < num_vectors_; ++i)
            bucket_assignments_[i] = FindNearestCentroid(reinterpret_cast<const f32 *>(rabitq_data_ + static_cast<size_t>(i) * cs));
    }

    delete[] bucket_locks_;
    bucket_locks_count_ = std::max(num_centroids_, 1u);
    bucket_locks_ = new std::mutex[bucket_locks_count_];
    running_means_.resize(num_centroids_);

    delete delta_a_; delete delta_b_;
    delta_a_ = new SPFreshDeltaBuffer(cs);
    delta_b_ = new SPFreshDeltaBuffer(cs);
    active_delta_idx_.store(0, std::memory_order_release);

    LOG_INFO(fmt::format("SPFresh Load: {} vectors, {} fine, {} coarse, dim={}, v={}",
                         num_vectors_, num_centroids_, coarse_count_, dim_, version));
}

void SPFreshIndexInMem::Dump(BufferObj *, size_t *) {}

MemIndexTracerInfo SPFreshIndexInMem::GetInfo() const {
    u32 dc = GetDeltaCount();
    size_t dm = dc * (sizeof(u32) * 2 + RabitQVecSize(dim_));
    size_t total_mem = mem_used_ + dm;
    // P3.3: Include all components in memory estimate
    total_mem += centroids_.size() * sizeof(f32);
    total_mem += bucket_metas_.size() * sizeof(SPFreshBucketMeta);
    total_mem += bucket_assignments_.size() * sizeof(u32);
    total_mem += running_means_.size() * sizeof(SPFreshRunningMean);
    return MemIndexTracerInfo(std::make_shared<std::string>(index_name_),
                              std::make_shared<std::string>(table_name_),
                              std::make_shared<std::string>(db_name_),
                              total_mem, GetRowCount());
}

const ChunkIndexMetaInfo SPFreshIndexInMem::GetChunkIndexMetaInfo() const {
    return ChunkIndexMetaInfo{"spfresh_chunk", begin_row_id_, GetRowCount(), 0, mem_used_};
}

} // namespace infinity

// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

module;

#include <immintrin.h>

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

// Background maintenance
static void SpfreshMaintenanceLoop(std::stop_token st, SPFreshIndexInMem *idx) {
    u32 cycle = 0;
    while (!st.stop_requested()) {
        // Check stop_token every 1s to avoid long blocking in ~SPFreshIndexInMem::join()
        for (u32 i = 0; i < 60 && !st.stop_requested(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (st.stop_requested())
            break;
        idx->TryAutoCompact(8192);
        if (++cycle % 10 == 0)
            idx->Rebalance(10000);
    }
}

// ========== Construction ==========

SPFreshIndexInMem::SPFreshIndexInMem() : begin_row_id_(), mem_used_(0) {}

SPFreshIndexInMem::SPFreshIndexInMem(RowID begin_row_id,
                                     const IndexSPFresh *index_def,
                                     u32 embedding_dim,
                                     u32 max_vectors,
                                     const std::string &base_path)
    : begin_row_id_(begin_row_id), dim_(embedding_dim), pad_dim_(1), hadamard_flip_(nullptr), buckets_(), bucket_metas_(), num_vectors_(0),
      replica_count_(index_def ? index_def->replica_count_ : 1),
      max_delta_bytes_(static_cast<u64>(index_def ? index_def->max_delta_mb_ : 512) * 1024 * 1024),
      num_centroids_(index_def ? index_def->num_centroids_ : 1000), coarse_count_(0), centroids_(), centroid_to_coarse_(),
      coarse_centroids_(), coarse_hnsw_(nullptr), running_means_(), mem_used_(0) {
    while (pad_dim_ < dim_)
        pad_dim_ <<= 1;

    size_t cs = RabitQVecSize(dim_);

    // Init buckets for each centroid
    buckets_.resize(num_centroids_);
    bucket_metas_.resize(num_centroids_);
    centroid_to_coarse_.resize(num_centroids_, 0);
    running_means_.resize(num_centroids_);

    bucket_locks_count_ = num_centroids_ > 0 ? num_centroids_ : 1;
    bucket_locks_ = new std::mutex[bucket_locks_count_];

    hadamard_flip_ = new bool[pad_dim_];
    GenerateHadamardParams();
    mem_used_ += pad_dim_ * sizeof(bool);

    // Pre-allocate centroids_ to avoid UB in FindTopKCentroids before Build()
    if (num_centroids_ > 0) {
        centroids_.resize(static_cast<size_t>(num_centroids_) * dim_, 0.0f);
    }

    delta_a_ = new SPFreshDeltaBuffer(cs);
    delta_b_ = new SPFreshDeltaBuffer(cs);
}

SPFreshIndexInMem::~SPFreshIndexInMem() {
    StopBackgroundMaintenance();
    delete[] hadamard_flip_;
    delete[] bucket_locks_;
    delete delta_a_;
    delete delta_b_;
}

// ========== Hadamard ==========

void SPFreshIndexInMem::GenerateHadamardParams() {
    std::mt19937 rng(42);
    std::bernoulli_distribution d(0.5);
    for (u32 i = 0; i < pad_dim_; ++i)
        hadamard_flip_[i] = d(rng);
}

void SPFreshIndexInMem::ApplyHadamard(f32 *vec, u32 n) const {
    for (u32 len = 1; len < n; len <<= 1) {
        for (u32 i = 0; i < n; i += len * 2) {
            for (u32 j = 0; j < len; ++j) {
                f32 a = vec[i + j], b = vec[i + j + len];
                vec[i + j] = a + b;
                vec[i + j + len] = a - b;
            }
        }
    }
    for (u32 i = 0; i < n; ++i)
        if (hadamard_flip_[i])
            vec[i] = -vec[i];
    f32 inv = 1.0f / std::sqrt(static_cast<f32>(n));
    for (u32 i = 0; i < n; ++i)
        vec[i] *= inv;
}

void SPFreshIndexInMem::ApplyRotation(const f32 *vec, f32 *out) const {
    thread_local std::vector<f32> buf;
    if (buf.size() < pad_dim_)
        buf.resize(pad_dim_);
    std::memcpy(buf.data(), vec, dim_ * sizeof(f32));
    std::memset(buf.data() + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));
    ApplyHadamard(buf.data(), pad_dim_);
    std::memcpy(out, buf.data(), dim_ * sizeof(f32));
}

void SPFreshIndexInMem::ApplyInverseRotation(const f32 *vec, f32 *out) const { ApplyRotation(vec, out); }

// ========== RaBitQ Encode/Decode/Distance ==========

size_t SPFreshIndexInMem::EncodeWithRotation(f32 *code_buf, const f32 *vec) const {
    auto *code = reinterpret_cast<RabitQVec *>(code_buf);
    std::memset(code_buf, 0, RabitQVecSize(dim_));
    thread_local std::vector<f32> nb, rb;
    if (nb.size() < pad_dim_)
        nb.resize(pad_dim_);
    if (rb.size() < pad_dim_)
        rb.resize(pad_dim_);

    f64 rn = 0;
    for (u32 d = 0; d < dim_; ++d)
        rn += static_cast<f64>(vec[d]) * vec[d];
    code->raw_norm_ = static_cast<f32>(rn);
    f64 inv = (rn > 1e-30) ? (1.0 / std::sqrt(rn)) : 1.0;
    std::memset(nb.data(), 0, pad_dim_ * sizeof(f32));
    for (u32 d = 0; d < dim_; ++d)
        nb[d] = static_cast<f32>(static_cast<f64>(vec[d]) * inv);
    std::memcpy(rb.data(), nb.data(), pad_dim_ * sizeof(f32));
    ApplyHadamard(rb.data(), pad_dim_);

    f32 sp = 0;
    for (u32 d = 0; d < dim_; ++d) {
        if (rb[d] >= 0) {
            code->compress_[d / 8] |= (1 << (d % 8));
            sp += 1.0f;
        }
    }
    code->sum_ = sp;
    code->norm_ = 1.0f;
    code->error_ = 0.95f;
    return RabitQVecSize(dim_);
}

void SPFreshIndexInMem::DecodeWithRotation(const RabitQVec *code, f32 *out_vec) const {
    thread_local std::vector<f32> buf;
    if (buf.size() < pad_dim_)
        buf.resize(pad_dim_);
    f32 mag = 1.0f / std::sqrt(static_cast<f32>(dim_));
    for (u32 d = 0; d < dim_; ++d)
        buf[d] = ((code->compress_[d / 8] >> (d % 8)) & 1) ? mag : -mag;
    std::memset(buf.data() + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));
    ApplyHadamard(buf.data(), pad_dim_);
    for (u32 d = 0; d < dim_; ++d)
        out_vec[d] = buf[d];
    f32 s = std::sqrt(code->raw_norm_);
    for (u32 d = 0; d < dim_; ++d)
        out_vec[d] *= s;
}

f32 SPFreshIndexInMem::RabitQDistWithRotation(const RabitQVec *code, const f32 *rq, u32 dim, f32 isd) {
    // AVX2: expand byte sign bits → negate by XOR only sign bit (bit 31)
    // sign_mask = bit ? 0 : 0x80000000  → negate when bit=0
    __m256 sum = _mm256_setzero_ps();
    const __m256i sign_bit = _mm256_set1_epi32(0x80000000);
    u32 d = 0;
    for (; d + 8 <= dim; d += 8) {
        __m256 q = _mm256_loadu_ps(rq + d);
        u8 byte = code->compress_[d / 8];
        __m256i bcast = _mm256_set1_epi32(byte);
        __m256i shl = _mm256_sllv_epi32(bcast, _mm256_set_epi32(24, 25, 26, 27, 28, 29, 30, 31));
        __m256i extracted = _mm256_srai_epi32(shl, 31);
        __m256i neg_mask = _mm256_andnot_si256(extracted, sign_bit);
        q = _mm256_xor_ps(q, _mm256_castsi256_ps(neg_mask));
        sum = _mm256_add_ps(sum, q);
    }
    __m128 lo = _mm256_castps256_ps128(sum), hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    f32 ip = _mm_cvtss_f32(lo);
    for (; d < dim; ++d)
        ip += ((code->compress_[d / 8] >> (d % 8)) & 1) ? rq[d] : -rq[d];
    f32 ct = std::clamp(ip * isd, -1.0f, 1.0f);
    return 2.0f * (1.0f - ct);
}

// ========== Centroid ==========

void SPFreshIndexInMem::BuildCentroidIndex() {
    if (num_centroids_ == 0 || dim_ == 0)
        return;
    coarse_count_ = std::max(1u, static_cast<u32>(std::sqrt(static_cast<f64>(num_centroids_))));
    if (coarse_count_ > num_centroids_)
        coarse_count_ = num_centroids_;
    std::vector<f32> cv;
    u32 ac = GetKMeansCentroids<f32, f32>(MetricType::kMetricL2, dim_, num_centroids_, centroids_.data(), cv, coarse_count_, 5, 4, 256, 1.0f);
    if (cv.empty() || ac == 0) {
        coarse_count_ = 1;
        coarse_centroids_.assign(centroids_.begin(), centroids_.begin() + dim_);
    } else {
        coarse_count_ = ac;
        coarse_centroids_.assign(cv.begin(), cv.end());
    }
    centroid_to_coarse_.resize(num_centroids_);
    for (u32 c = 0; c < num_centroids_; ++c) {
        const f32 *cvp = centroids_.data() + static_cast<size_t>(c) * dim_;
        u32 bc = 0;
        f32 bd = std::numeric_limits<f32>::max();
        for (u32 cc = 0; cc < coarse_count_; ++cc) {
            f32 dd = 0;
            for (u32 d = 0; d < dim_; ++d) {
                f32 df = cvp[d] - coarse_centroids_[static_cast<size_t>(cc) * dim_ + d];
                dd += df * df;
            }
            if (dd < bd) {
                bd = dd;
                bc = cc;
            }
        }
        centroid_to_coarse_[c] = bc;
    }
    if (coarse_count_ > 32) {
        u32 ch = 1;
        while (ch < coarse_count_ + 16)
            ch <<= 1;
        auto hnsw = HnswType::Make(ch, 1, dim_, 16, 200);
        if (hnsw) {
            auto iter = DenseVectorIter<f32, u32>(coarse_centroids_.data(), dim_, coarse_count_);
            hnsw->InsertVecs(iter, HnswInsertConfig{true});
            coarse_hnsw_ = std::shared_ptr<const HnswType>(hnsw.release());
        }
    }
}

void SPFreshIndexInMem::FindTopKCentroids(const f32 *query, u32 top_k, std::vector<u32> &oids, std::vector<f32> &odists) const {
    oids.clear();
    odists.clear();
    if (num_centroids_ == 0)
        return;
    u32 k = std::min(top_k, num_centroids_);
    u32 sc = std::min(static_cast<u32>(std::ceil(static_cast<f64>(k) * 4.0 / (num_centroids_ / std::max(1u, coarse_count_)))), coarse_count_);
    if (sc == 0)
        sc = std::min(8u, coarse_count_);

    std::vector<u32> cids;
    if (coarse_hnsw_ && sc > 0) {
        KnnSearchOption opt;
        opt.ef_ = sc * 4;
        auto [cnt, dp, lp] = coarse_hnsw_->template KnnSearch<>(query, sc, std::nullopt, opt);
        for (i64 i = 0; i < static_cast<i64>(cnt) && i < static_cast<i64>(sc); ++i)
            cids.push_back(lp[i]);
    } else {
        for (u32 cc = 0; cc < coarse_count_; ++cc)
            cids.push_back(cc);
    }
    if (cids.empty())
        cids.push_back(0);

    std::vector<bool> cm(coarse_count_, false);
    for (u32 c : cids)
        cm[c] = true;

    std::vector<std::pair<f32, u32>> heap;
    heap.reserve(k + 1);
    for (u32 c = 0; c < num_centroids_; ++c) {
        if (!cm[centroid_to_coarse_[c]])
            continue;
        f32 dd = 0;
        for (u32 d = 0; d < dim_; ++d) {
            f32 df = query[d] - centroids_[static_cast<size_t>(c) * dim_ + d];
            dd += df * df;
        }
        if (heap.size() < k) {
            heap.emplace_back(-dd, c);
            std::push_heap(heap.begin(), heap.end());
        } else if (-dd > heap[0].first) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {-dd, c};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort(heap.begin(), heap.end());
    for (auto &[nd, id] : heap) {
        oids.push_back(id);
        odists.push_back(-nd);
    }
}

// ========== LIRE Build ==========

void SPFreshIndexInMem::Build(const f32 *vectors, u32 count) {
    if (count == 0 || num_centroids_ == 0)
        return;
    std::lock_guard wlock(global_mtx_);

    std::vector<f32> cv;
    u32 ac = GetKMeansCentroids<f32, f32>(MetricType::kMetricL2, dim_, count, vectors, cv, num_centroids_, 10, 32, 256, 1.0f);
    if (cv.empty() || ac == 0) {
        LOG_WARN("SPFresh Build: K-Means empty");
        return;
    }
    num_centroids_ = ac;
    centroids_.assign(cv.begin(), cv.end());
    centroid_to_coarse_.resize(num_centroids_);

    // Allocate LIRE buckets
    buckets_.resize(num_centroids_);
    bucket_metas_.resize(num_centroids_);
    delete[] bucket_locks_;
    bucket_locks_count_ = num_centroids_;
    bucket_locks_ = new std::mutex[bucket_locks_count_];
    running_means_.resize(num_centroids_);

    BuildCentroidIndex();

    size_t cs = RabitQVecSize(dim_);

    // LIRE: SIMD batch centroid assignment
    std::vector<u32> labels(count);
    std::vector<f32> dists(count);
    GetSIMD_FUNCTIONS().SearchTop1WithDisF32U32_func_ptr_(dim_, count, vectors, num_centroids_, centroids_.data(), labels.data(), dists.data());

    // Pre-count per-bucket sizes to reserve capacity (avoid repeated vector reallocation)
    std::vector<u32> bucket_sizes(num_centroids_, 0);
    for (u32 i = 0; i < count; ++i)
        ++bucket_sizes[labels[i]];
    for (u32 c = 0; c < num_centroids_; ++c) {
        if (bucket_sizes[c] > 0)
            buckets_[c].codes_.reserve(static_cast<size_t>(bucket_sizes[c]) * cs);
    }

    // LIRE: single-pass normalize + rotate + encode, store per-bucket
    // No batch buffer needed — process one vector at a time
    std::vector<f32> rot(pad_dim_);
    std::vector<char> tmp(cs);
    u32 base_offset = begin_row_id_.segment_offset_;
    for (u32 i = 0; i < count; ++i) {
        u32 cid = labels[i];
        const f32 *v = vectors + static_cast<size_t>(i) * dim_;
        u32 row_id = base_offset + i;

        // Normalize + Hadamard rotate (in one local buffer)
        f64 rn = 0;
        for (u32 d = 0; d < dim_; ++d)
            rn += static_cast<f64>(v[d]) * v[d];
        f64 inv = (rn > 1e-30) ? (1.0 / std::sqrt(rn)) : 1.0;
        for (u32 d = 0; d < dim_; ++d)
            rot[d] = static_cast<f32>(static_cast<f64>(v[d]) * inv);
        std::memset(rot.data() + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));
        ApplyHadamard(rot.data(), pad_dim_);

        // Encode to RaBitQ code
        auto *code = reinterpret_cast<RabitQVec *>(tmp.data());
        std::memset(code, 0, cs);
        code->raw_norm_ = static_cast<f32>(rn);
        code->norm_ = 1.0f;
        code->error_ = 0.95f;
        f32 sp = 0;
        for (u32 d = 0; d < dim_; ++d) {
            if (rot[d] >= 0) {
                code->compress_[d / 8] |= static_cast<u8>(1 << (d % 8));
                sp += 1.0f;
            }
        }
        code->sum_ = sp;

        buckets_[cid].codes_.insert(buckets_[cid].codes_.end(), tmp.data(), tmp.data() + cs);
        buckets_[cid].row_ids_.push_back(row_id);
        buckets_[cid].count_++;
        bucket_metas_[cid].base_count_++;
        running_means_[cid].Update(v, dim_);
        ++num_vectors_;
    }
    LOG_INFO(fmt::format("SPFresh Build: {} vectors, {} buckets", count, num_centroids_));
    PublishSnapshot();
}

// ========== LIRE: InsertDelta → global delta ==========

void SPFreshIndexInMem::InsertDelta(const f32 *vec, u32 row_id) {
    if (!delta_a_ || !delta_b_ || num_centroids_ == 0)
        return;

    u32 idx_snap = active_delta_idx_.load(std::memory_order_acquire);
    auto *active_snap = (idx_snap == 0) ? delta_a_ : delta_b_;
    if (active_snap->data_.size() > max_delta_bytes_)
        Compact();

    std::vector<u32> cids;
    std::vector<f32> cdists;
    {
        std::shared_lock rlock(global_mtx_);
        FindTopKCentroids(vec, replica_count_, cids, cdists);
    }
    if (cids.empty())
        cids.push_back(0);

    size_t cs = RabitQVecSize(dim_);
    std::vector<char> buf(cs);
    EncodeWithRotation(reinterpret_cast<f32 *>(buf.data()), vec);
    {
        std::shared_lock lock(compact_mtx_);
        u32 idx = active_delta_idx_.load(std::memory_order_acquire);
        auto *active = (idx == 0) ? delta_a_ : delta_b_;
        for (u32 cid : cids) {
            active->Append(cid, row_id, buf.data());
            {
                std::lock_guard<std::mutex> blk(bucket_locks_[cid % bucket_locks_count_]);
                bucket_metas_[cid].delta_count_++;
            }
            // Update running mean for centroid refinement
            running_means_[cid].Update(vec, dim_);
        }
    }
}

// ========== Running Mean Centroid Update ==========

void SPFreshIndexInMem::RefreshCentroidsFromRunningMeans() {
    // Update each centroid from its bucket's running mean (vector sum / count)
    for (u32 c = 0; c < num_centroids_; ++c) {
        if (running_means_[c].count_ > 0) {
            running_means_[c].GetCentroid(centroids_.data() + static_cast<size_t>(c) * dim_, dim_);
        }
    }
}

// ========== RCU Snapshot ==========

void SPFreshIndexInMem::PublishSnapshot() {
    auto snap = std::make_shared<SPFreshSnapshot>();
    snap->buckets_ = buckets_;
    snap->bucket_metas_ = bucket_metas_;
    snap->num_centroids_ = num_centroids_;
    snap->coarse_count_ = coarse_count_;
    snap->num_vectors_ = num_vectors_;
    snap->centroids_ = centroids_;
    snap->centroid_to_coarse_ = centroid_to_coarse_;
    snap->coarse_centroids_ = coarse_centroids_;
    snap->coarse_hnsw_ = coarse_hnsw_; // shared_ptr, no copy
    snap->replica_count_ = replica_count_;
    std::unique_lock wlock(snapshot_mtx_);
    snapshot_ = std::move(snap);
}

// ========== LIRE Search ==========

u32 SPFreshIndexInMem::GetRowCount() const { return num_vectors_ + GetDeltaCount() - static_cast<u32>(deleted_set_.size()); }
u32 SPFreshIndexInMem::GetBaseRowCount() const { return num_vectors_; }

void SPFreshIndexInMem::Search(const f32 *query, u32 dim, const SearchCallback &callback) const {
    if (dim != dim_)
        return;
    thread_local std::vector<f32> qrot;
    if (qrot.size() < pad_dim_)
        qrot.resize(pad_dim_);
    f32 isd = 1.0f / std::sqrt(static_cast<f32>(dim_));
    size_t cs = RabitQVecSize(dim_);

    f64 qn = 0;
    for (u32 d = 0; d < dim_; ++d)
        qn += static_cast<f64>(query[d]) * query[d];
    f64 inv = (qn > 1e-30) ? (1.0 / std::sqrt(qn)) : 1.0;
    for (u32 d = 0; d < dim_; ++d)
        qrot[d] = static_cast<f32>(static_cast<f64>(query[d]) * inv);
    std::memset(qrot.data() + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));
    ApplyHadamard(qrot.data(), pad_dim_);

    // RCU: grab latest snapshot (lock-free, no writer blocking)
    std::shared_ptr<const SPFreshSnapshot> snap;
    {
        std::shared_lock rlock(snapshot_mtx_);
        snap = snapshot_;
    }
    if (!snap || snap->num_vectors_ == 0)
        return;

    // Take deleted_set_ snapshot
    std::unordered_set<u32> ds;
    {
        std::shared_lock dlock(delete_mtx_);
        ds = deleted_set_;
    }

    // Find candidate centroids using snapshot data
    const auto &snap_centroids = snap->centroids_;
    const auto &snap_centroid_to_coarse = snap->centroid_to_coarse_;
    const auto &snap_coarse_hnsw = snap->coarse_hnsw_;
    u32 snap_nc = snap->num_centroids_;
    u32 snap_cc = snap->coarse_count_;

    // Inline FindTopKCentroids using snapshot (no lock needed)
    std::vector<u32> cand;
    {
        u32 k = std::min(64u, snap_nc);
        if (k == 0)
            return;
        u32 sc = std::min(static_cast<u32>(std::ceil(static_cast<f64>(k) * 4.0 / (snap_nc / std::max(1u, snap_cc)))), snap_cc);
        if (sc == 0)
            sc = std::min(8u, snap_cc);

        std::vector<u32> cids;
        if (snap_coarse_hnsw && sc > 0) {
            KnnSearchOption opt;
            opt.ef_ = sc * 4;
            auto [cnt, dp, lp] = snap_coarse_hnsw->template KnnSearch<>(qrot.data(), sc, std::nullopt, opt);
            for (i64 i = 0; i < static_cast<i64>(cnt) && i < static_cast<i64>(sc); ++i)
                cids.push_back(lp[i]);
        } else {
            for (u32 cc = 0; cc < snap_cc; ++cc)
                cids.push_back(cc);
        }
        if (cids.empty())
            cids.push_back(0);

        std::vector<bool> cm(snap_cc, false);
        for (u32 c : cids)
            cm[c] = true;

        std::vector<std::pair<f32, u32>> heap;
        heap.reserve(k + 1);
        for (u32 c = 0; c < snap_nc; ++c) {
            if (!cm[snap_centroid_to_coarse[c]])
                continue;
            f32 dd = 0;
            for (u32 d = 0; d < dim_; ++d) {
                f32 df = qrot[d] - snap_centroids[static_cast<size_t>(c) * dim_ + d];
                dd += df * df;
            }
            if (heap.size() < k) {
                heap.emplace_back(-dd, c);
                std::push_heap(heap.begin(), heap.end());
            } else if (-dd > heap[0].first) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {-dd, c};
                std::push_heap(heap.begin(), heap.end());
            }
        }
        std::sort(heap.begin(), heap.end());
        for (auto &[nd, id] : heap)
            cand.push_back(id);
    }
    if (cand.empty())
        return;

    std::vector<bool> cm(snap_nc, false);
    for (u32 c : cand)
        cm[c] = true;

    // Scan candidate buckets from snapshot
    for (u32 c = 0; c < snap_nc; ++c) {
        if (!cm[c] || c >= snap->buckets_.size())
            continue;
        auto &bkt = snap->buckets_[c];
        if (bkt.count_ == 0)
            continue;
        for (u32 j = 0; j < bkt.count_; ++j) {
            u32 rid = bkt.row_ids_[j];
            if (ds.find(rid) != ds.end())
                continue;
            const auto *code = reinterpret_cast<const RabitQVec *>(bkt.codes_.data() + static_cast<size_t>(j) * cs);
            callback(rid, RabitQDistWithRotation(code, qrot.data(), dim_, isd));
        }
    }

    // Scan global delta entries (read directly — they're atomic-swapped)
    auto sd = [&](const SPFreshDeltaBuffer *delta) {
        if (!delta || delta->entry_count_ == 0)
            return;
        for (u32 di = 0; di < delta->entry_count_; ++di) {
            u32 bid = delta->GetBucketId(di);
            if (bid >= snap_nc || !cm[bid])
                continue;
            u32 rid = delta->GetRowId(di);
            if (ds.find(rid) != ds.end())
                continue;
            const auto *code = reinterpret_cast<const RabitQVec *>(delta->GetCode(di));
            callback(rid, RabitQDistWithRotation(code, qrot.data(), dim_, isd));
        }
    };
    sd(delta_a_);
    sd(delta_b_);
}

// ========== Delete ==========

void SPFreshIndexInMem::MarkDeleted(u32 row_id) {
    std::lock_guard lock(delete_mtx_);
    deleted_set_.insert(row_id);
}

// ========== LIRE: Start/Stop Maintenance ==========

void SPFreshIndexInMem::StartBackgroundMaintenance() {
    if (maintenance_thread_.joinable())
        return;
    maintenance_thread_ = std::jthread(SpfreshMaintenanceLoop, this);
}
void SPFreshIndexInMem::StopBackgroundMaintenance() {
    if (maintenance_thread_.joinable()) {
        maintenance_thread_.request_stop();
        maintenance_thread_.join();
    }
}

// ========== LIRE Compact (per-bucket: merge delta entries into bucket's base codes_) ==========

bool SPFreshIndexInMem::TryAutoCompact(u32 threshold) {
    if (!delta_a_ || !delta_b_)
        return false;
    u32 idx = active_delta_idx_.load(std::memory_order_acquire);
    auto *active = (idx == 0) ? delta_a_ : delta_b_;
    if (active->entry_count_ >= threshold) {
        Compact();
        return true;
    }
    return false;
}

void SPFreshIndexInMem::Compact() {
    if (!delta_a_ || !delta_b_)
        return;
    std::lock_guard lock(compact_mtx_);
    size_t cs = RabitQVecSize(dim_);

    std::unordered_set<u32> deleted;
    {
        std::shared_lock dlock(delete_mtx_);
        deleted = deleted_set_;
    }

    u32 old_idx = active_delta_idx_.load(std::memory_order_acquire);
    u32 new_idx = 1 - old_idx;
    auto *old_delta = (old_idx == 0) ? delta_a_ : delta_b_;

    // LIRE: Group delta entries by bucket_id, filter deleted
    struct DeltaEntry {
        u32 bucket_id;
        u32 row_id;
        const char *code;
    };
    std::unordered_map<u32, std::vector<DeltaEntry>> delta_by_bucket;
    for (u32 di = 0; di < old_delta->entry_count_; ++di) {
        u32 bid = old_delta->GetBucketId(di);
        if (bid >= num_centroids_)
            continue;
        u32 rid = old_delta->GetRowId(di);
        if (deleted.find(rid) != deleted.end())
            continue;
        delta_by_bucket[bid].push_back({bid, rid, old_delta->GetCode(di)});
    }

    {
        std::lock_guard wlock(global_mtx_);

        // LIRE: Use row_ids_ from each bucket to match deleted_set_
        // Bucket has (codes_, row_ids_) in parallel; filter out rows whose row_id is in deleted_set_
        for (u32 bid = 0; bid < buckets_.size(); ++bid) {
            auto &bkt = buckets_[bid];
            if (bkt.count_ == 0)
                continue;
            std::vector<char> filtered_codes;
            std::vector<u32> filtered_ids;
            filtered_codes.reserve(bkt.codes_.size());
            filtered_ids.reserve(bkt.count_);
            u32 kept = 0;
            for (u32 j = 0; j < bkt.count_; ++j) {
                u32 rid = bkt.row_ids_[j];
                if (deleted.find(rid) != deleted.end())
                    continue;
                filtered_codes.insert(filtered_codes.end(),
                                      bkt.codes_.data() + static_cast<size_t>(j) * cs,
                                      bkt.codes_.data() + static_cast<size_t>(j + 1) * cs);
                filtered_ids.push_back(rid);
                ++kept;
            }
            u32 removed = bkt.count_ - kept;
            if (removed > 0) {
                bkt.codes_ = std::move(filtered_codes);
                bkt.row_ids_ = std::move(filtered_ids);
                bkt.count_ = kept;
                bucket_metas_[bid].base_count_ -= removed;
                num_vectors_ -= removed;
            }
        }

        // Append delta entries per bucket (delta already has correct row_ids)
        for (auto &[bid, entries] : delta_by_bucket) {
            if (bid >= buckets_.size())
                continue;
            auto &bkt = buckets_[bid];
            for (auto &e : entries) {
                bkt.AppendCode(e.code, cs, e.row_id);
                bucket_metas_[bid].base_count_++;
                bucket_metas_[bid].delta_count_--;
                ++num_vectors_;
            }
        }

        old_delta->Clear();
        active_delta_idx_.store(new_idx, std::memory_order_release);
    }

    // Clear deleted_set_ — the deleted rows have been physically removed from buckets by row_id match
    {
        std::lock_guard dlock(delete_mtx_);
        deleted_set_.clear();
    }

    compact_count_.fetch_add(1, std::memory_order_relaxed);
    RefreshCentroidsFromRunningMeans();
    PublishSnapshot();
}

// ========== LIRE SplitBucket ==========

std::vector<u32> SPFreshIndexInMem::SplitBucket(u32 bucket_id) {
    std::unique_lock wlock(global_mtx_);
    if (bucket_id >= buckets_.size() || buckets_[bucket_id].count_ < 4)
        return {};

    size_t cs = RabitQVecSize(dim_);
    auto &old_bkt = buckets_[bucket_id];
    u32 n = old_bkt.count_;

    // LIRE: Decode vectors from this bucket
    std::vector<f32> vecs(static_cast<size_t>(n) * dim_);
    for (u32 j = 0; j < n; ++j) {
        const auto *code = reinterpret_cast<const RabitQVec *>(old_bkt.codes_.data() + static_cast<size_t>(j) * cs);
        DecodeWithRotation(code, vecs.data() + static_cast<size_t>(j) * dim_);
    }

    std::vector<f32> spc;
    u32 ak = GetKMeansCentroids<f32, f32>(MetricType::kMetricL2, dim_, n, vecs.data(), spc, 2, 10, 4, 256, 1.0f);
    if (spc.empty() || ak < 2)
        return {};

    // LIRE: Mark old bucket retired, create two new buckets
    bucket_metas_[bucket_id].is_retired_ = true;
    old_bkt.Clear();
    old_bkt.count_ = 0; // retired

    u32 nc1 = num_centroids_;
    u32 nc2 = num_centroids_ + 1;
    num_centroids_ += 2;

    centroids_.resize(static_cast<size_t>(num_centroids_) * dim_);
    std::memcpy(centroids_.data() + static_cast<size_t>(nc1) * dim_, spc.data(), dim_ * sizeof(f32));
    std::memcpy(centroids_.data() + static_cast<size_t>(nc2) * dim_, spc.data() + dim_, dim_ * sizeof(f32));

    buckets_.resize(num_centroids_);
    bucket_metas_.resize(num_centroids_);
    centroid_to_coarse_.resize(num_centroids_);
    centroid_to_coarse_[nc1] = centroid_to_coarse_[bucket_id];
    centroid_to_coarse_[nc2] = centroid_to_coarse_[bucket_id];

    delete[] bucket_locks_;
    bucket_locks_count_ = num_centroids_;
    bucket_locks_ = new std::mutex[bucket_locks_count_];
    running_means_.resize(num_centroids_);

    // LIRE: Reassign vectors to new buckets (preserve original row_ids)
    std::vector<char> tmp(cs);
    for (u32 j = 0; j < n; ++j) {
        const f32 *v = vecs.data() + static_cast<size_t>(j) * dim_;
        u32 orig_row_id = old_bkt.row_ids_[j];
        f32 d1 = 0, d2 = 0;
        for (u32 d = 0; d < dim_; ++d) {
            f32 df1 = v[d] - spc[d], df2 = v[d] - spc[dim_ + d];
            d1 += df1 * df1;
            d2 += df2 * df2;
        }
        u32 nc = (d1 <= d2) ? nc1 : nc2;
        EncodeWithRotation(reinterpret_cast<f32 *>(tmp.data()), v);
        buckets_[nc].AppendCode(tmp.data(), cs, orig_row_id);
        bucket_metas_[nc].base_count_++;
        running_means_[nc].Update(v, dim_);
    }

    split_count_.fetch_add(1, std::memory_order_relaxed);
    BuildCentroidIndex();
    PublishSnapshot();
    LOG_INFO(fmt::format("SPFresh SplitBucket: {} → {} (c1={}, c2={})", bucket_id, num_centroids_, nc1, nc2));
    return {nc1, nc2};
}

// ========== LIRE Rebalance ==========

void SPFreshIndexInMem::Rebalance(u32 bucket_size_limit) {
    if (GetDeltaCount() > 0)
        Compact();
    if (num_centroids_ == 0)
        return;
    std::shared_lock rlock(global_mtx_);

    bool any = false;
    for (u32 c = 0; c < num_centroids_; ++c) {
        if (buckets_[c].count_ > bucket_size_limit && !bucket_metas_[c].is_retired_) {
            rlock.unlock();
            auto ids = SplitBucket(c);
            if (!ids.empty())
                any = true;
            rlock.lock();
        }
    }
    if (any) {
        BuildCentroidIndex();
        LOG_INFO(fmt::format("SPFresh Rebalance: done, {} buckets", num_centroids_));
    }
}

// ========== LIRE Persistence (v7 with row_ids_) ==========

void SPFreshIndexInMem::Save(LocalFileHandle &fh) const {
    u32 magic = 0x50504652, version = 7;
    fh.Append(&magic, sizeof(magic));
    fh.Append(&version, sizeof(version));
    fh.Append(&num_vectors_, sizeof(num_vectors_));
    fh.Append(&dim_, sizeof(dim_));
    fh.Append(&pad_dim_, sizeof(pad_dim_));
    fh.Append(&num_centroids_, sizeof(num_centroids_));
    fh.Append(&coarse_count_, sizeof(coarse_count_));

    size_t cs = RabitQVecSize(dim_);

    // LIRE: Save each bucket (codes + row_ids)
    for (auto &bkt : buckets_) {
        fh.Append(&bkt.count_, sizeof(bkt.count_));
        if (bkt.count_ > 0) {
            fh.Append(bkt.codes_.data(), static_cast<u64>(bkt.count_) * cs);
            fh.Append(bkt.row_ids_.data(), static_cast<u64>(bkt.count_) * sizeof(u32));
        }
    }
    for (auto &m : bucket_metas_)
        fh.Append(&m, sizeof(m));
    if (num_centroids_ > 0)
        fh.Append(centroids_.data(), static_cast<size_t>(num_centroids_) * dim_ * sizeof(f32));
    if (coarse_count_ > 0)
        fh.Append(coarse_centroids_.data(), static_cast<size_t>(coarse_count_) * dim_ * sizeof(f32));
    if (num_centroids_ > 0)
        fh.Append(centroid_to_coarse_.data(), num_centroids_ * sizeof(u32));

    u64 hb = static_cast<u64>(pad_dim_) * sizeof(bool);
    fh.Append(&hb, sizeof(hb));
    fh.Append(hadamard_flip_, hb);

    fh.Append(&replica_count_, sizeof(replica_count_));
}

void SPFreshIndexInMem::Load(LocalFileHandle &fh, size_t) {
    u32 magic = 0;
    fh.Read(&magic, sizeof(magic));
    if (magic != 0x50504652) {
        UnrecoverableError("SPFresh: bad magic");
        return;
    }
    u32 ver = 0;
    fh.Read(&ver, sizeof(ver));
    fh.Read(&num_vectors_, sizeof(num_vectors_));
    fh.Read(&dim_, sizeof(dim_));
    fh.Read(&pad_dim_, sizeof(pad_dim_));
    fh.Read(&num_centroids_, sizeof(num_centroids_));
    fh.Read(&coarse_count_, sizeof(coarse_count_));

    size_t cs = RabitQVecSize(dim_);

    // LIRE: Load buckets
    buckets_.resize(num_centroids_);
    for (auto &bkt : buckets_) {
        u32 cnt = 0;
        fh.Read(&cnt, sizeof(cnt));
        if (cnt > 0) {
            bkt.codes_.resize(static_cast<size_t>(cnt) * cs);
            fh.Read(bkt.codes_.data(), static_cast<u64>(cnt) * cs);
            if (ver >= 7) {
                // v7 stores row_ids_ alongside codes
                bkt.row_ids_.resize(cnt);
                fh.Read(bkt.row_ids_.data(), static_cast<u64>(cnt) * sizeof(u32));
            } else {
                // v6 and earlier: generate sequential row_ids for backward compatibility
                bkt.row_ids_.resize(cnt);
                u32 base_row = 0;
                for (u32 bc = 0; bc < buckets_.size(); ++bc) {
                    if (&buckets_[bc] == &bkt)
                        break;
                    base_row += buckets_[bc].count_;
                }
                for (u32 j = 0; j < cnt; ++j)
                    bkt.row_ids_[j] = base_row + j;
            }
            bkt.count_ = cnt;
        }
    }

    bucket_metas_.resize(num_centroids_);
    for (auto &m : bucket_metas_)
        fh.Read(&m, sizeof(m));
    centroids_.resize(static_cast<size_t>(num_centroids_) * dim_);
    if (num_centroids_ > 0)
        fh.Read(centroids_.data(), static_cast<size_t>(num_centroids_) * dim_ * sizeof(f32));
    if (coarse_count_ > 0) {
        coarse_centroids_.resize(static_cast<size_t>(coarse_count_) * dim_);
        fh.Read(coarse_centroids_.data(), static_cast<size_t>(coarse_count_) * dim_ * sizeof(f32));
    }
    centroid_to_coarse_.resize(num_centroids_);
    if (num_centroids_ > 0)
        fh.Read(centroid_to_coarse_.data(), num_centroids_ * sizeof(u32));

    u64 hb = 0;
    fh.Read(&hb, sizeof(hb));
    delete[] hadamard_flip_;
    hadamard_flip_ = new bool[hb / sizeof(bool)];
    fh.Read(hadamard_flip_, hb);

    if (ver >= 4) {
        fh.Read(&replica_count_, sizeof(replica_count_));
    } else {
        replica_count_ = 1;
    }

    delete[] bucket_locks_;
    bucket_locks_count_ = std::max(num_centroids_, 1u);
    bucket_locks_ = new std::mutex[bucket_locks_count_];
    running_means_.resize(num_centroids_);

    delete delta_a_;
    delete delta_b_;
    delta_a_ = new SPFreshDeltaBuffer(cs);
    delta_b_ = new SPFreshDeltaBuffer(cs);
    active_delta_idx_.store(0, std::memory_order_release);

    PublishSnapshot();
    LOG_INFO(fmt::format("SPFresh Load: {} vectors, {} buckets, v={}", num_vectors_, num_centroids_, ver));
}

// ========== LIRE Dump: move index data into BufferObj for persistence ==========

void SPFreshIndexInMem::TransferTo(SPFreshIndexInMem *target) {
    StopBackgroundMaintenance();

    // Move all base data
    target->begin_row_id_ = begin_row_id_;
    target->dim_ = dim_;
    target->pad_dim_ = pad_dim_;
    // Deep copy hadamard_flip_ so source retains its own copy for Search via mem_index
    if (hadamard_flip_ != nullptr) {
        target->hadamard_flip_ = new bool[pad_dim_];
        std::memcpy(target->hadamard_flip_, hadamard_flip_, pad_dim_ * sizeof(bool));
    }
    target->buckets_ = std::move(buckets_);
    target->bucket_metas_ = std::move(bucket_metas_);
    target->num_vectors_ = num_vectors_;
    target->replica_count_ = replica_count_;
    target->max_delta_bytes_ = max_delta_bytes_;
    target->num_centroids_ = num_centroids_;
    target->coarse_count_ = coarse_count_;
    target->centroids_ = std::move(centroids_);
    target->centroid_to_coarse_ = std::move(centroid_to_coarse_);
    target->coarse_centroids_ = std::move(coarse_centroids_);
    target->coarse_hnsw_ = coarse_hnsw_; // shared_ptr, no move needed
    target->running_means_ = std::move(running_means_);
    target->mem_used_ = mem_used_;

    // Allocate target's delta buffers if not already
    if (target->delta_a_ == nullptr) {
        size_t cs = RabitQVecSize(dim_);
        target->delta_a_ = new SPFreshDeltaBuffer(cs);
        target->delta_b_ = new SPFreshDeltaBuffer(cs);
    }
    if (target->bucket_locks_ == nullptr && num_centroids_ > 0) {
        target->bucket_locks_count_ = num_centroids_;
        target->bucket_locks_ = new std::mutex[num_centroids_];
    }
    // Publish snapshot on target so Search can find data immediately
    target->PublishSnapshot();
}

void SPFreshIndexInMem::Dump(BufferObj *buffer_obj, size_t *) {
    BufferHandle handle = buffer_obj->Load();
    auto *fw = handle.GetFileWorkerMut();
    auto *target = static_cast<SPFreshIndexInMem *>(fw->GetData());
    TransferTo(target);
    chunk_handle_ = std::move(handle);
}

// ========== BaseMemIndex interface ==========

MemIndexTracerInfo SPFreshIndexInMem::GetInfo() const {
    size_t tm = mem_used_;
    for (auto &b : buckets_)
        tm += b.codes_.size();
    tm += centroids_.size() * sizeof(f32);
    tm += bucket_metas_.size() * sizeof(SPFreshBucketMeta);
    return MemIndexTracerInfo(std::make_shared<std::string>(index_name_),
                              std::make_shared<std::string>(table_name_),
                              std::make_shared<std::string>(db_name_),
                              tm,
                              GetRowCount());
}
const ChunkIndexMetaInfo SPFreshIndexInMem::GetChunkIndexMetaInfo() const {
    return ChunkIndexMetaInfo{"spfresh_chunk", begin_row_id_, GetRowCount(), 0, mem_used_};
}

} // namespace infinity

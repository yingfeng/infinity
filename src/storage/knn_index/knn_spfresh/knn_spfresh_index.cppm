// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

export module infinity_core:spfresh_index;

import :spfresh_defs;
import :base_memindex;
import :memindex_tracer;
import :chunk_index_meta;
import :index_base;
import :index_spfresh;
import :local_file_handle;
import :buffer_handle;
import :hnsw_alg;
import :hnsw_common;
import :vec_store_type;
import :logger;

import std.compat;

import internal_types;
import column_def;
import row_id;

namespace infinity {

export class SPFreshIndexInMem : public BaseMemIndex {
public:
    SPFreshIndexInMem()
        : begin_row_id_(), rabitq_data_(nullptr), num_vectors_(0), max_vectors_(0), dim_(0),
          pad_dim_(0), hadamard_flip_(nullptr), num_centroids_(0), coarse_count_(0),
          centroids_(), centroid_to_coarse_(), coarse_centroids_(), coarse_hnsw_(nullptr),
          bucket_metas_(), replica_count_(0),
          bucket_assignments_(), delta_a_(nullptr), delta_b_(nullptr), active_delta_idx_(0),
          bucket_locks_(nullptr), bucket_locks_count_(0),
          running_means_(), deleted_set_(), mem_used_(0) {}

    SPFreshIndexInMem(RowID begin_row_id, const IndexSPFresh *index_def, u32 embedding_dim, u32 max_vectors,
                      const std::string &base_path = "");
    ~SPFreshIndexInMem() override;

    MemIndexTracerInfo GetInfo() const override;
    const ChunkIndexMetaInfo GetChunkIndexMetaInfo() const override;
    RowID GetBeginRowID() const override { return begin_row_id_; }

    // Bulk build
    void Build(const f32 *vectors, u32 count);
    void InsertVector(const f32 *vec, u32 local_id);

    // RaBitQ format
    struct RabitQVec {
        f32 raw_norm_;
        f32 norm_;
        f32 sum_;
        f32 error_;
        u8 compress_[];
    };
    static size_t RabitQVecSize(u32 dim) { return sizeof(RabitQVec) + dim / 8; }

    size_t EncodeWithRotation(f32 *code_buf, const f32 *vec) const;
    void DecodeWithRotation(const RabitQVec *code, f32 *out_vec) const;

    // SIMD RaBitQ distance (P0.2)
    static f32 RabitQDistWithRotation(const RabitQVec *code, const f32 *rotated_query_normed,
                                       u32 dim, f32 inv_sqrt_d);

    // Centroid routing: HNSW on coarse centroids (P1.2)
    using HnswType = KnnHnsw<PlainL2VecStoreType<f32>, u32>;

    void BuildCentroidIndex();
    void FindTopKCentroids(const f32 *query, u32 top_k,
                           std::vector<u32> &out_ids,
                           std::vector<f32> &out_dists) const;

    void InsertDelta(const f32 *vec, u32 row_id);

    // P0.3: Atomic Compact with temp buffer
    void Compact();

    using SearchCallback = std::function<void(SegmentOffset, f32)>;
    void Search(const f32 *query, u32 dim, const SearchCallback &callback) const;

    void MarkDeleted(u32 row_id);

    // P3.1: Background maintenance
    void StartBackgroundMaintenance();
    void StopBackgroundMaintenance();

    bool TryAutoCompact(u32 delta_threshold = 8192);
    void Rebalance(u32 bucket_size_limit = 10000);
    std::vector<u32> SplitBucket(u32 bucket_id);

    void Save(LocalFileHandle &file_handle) const;
    void Load(LocalFileHandle &file_handle, size_t file_size);
    void Dump(BufferObj *buffer_obj, size_t *p_dump_size = nullptr);

    u32 GetRowCount() const { return num_vectors_ + GetDeltaCount() - static_cast<u32>(deleted_set_.size()); }
    u32 GetBaseRowCount() const { return num_vectors_; }
    u32 GetDeltaCount() const {
        if (!delta_a_ || !delta_b_) return 0;
        u32 idx = active_delta_idx_.load(std::memory_order_acquire);
        return (idx == 0 ? delta_a_ : delta_b_)->entry_count_;
    }
    u32 dim() const { return dim_; }
    u32 GetDeletedCount() const { return static_cast<u32>(deleted_set_.size()); }
    u32 GetNumCentroids() const { return num_centroids_; }

    // P3.3: Monitoring metrics
    u64 GetCompactCount() const { return compact_count_; }
    u64 GetSplitCount() const { return split_count_; }
    u64 GetPeakDeltaBytes() const { return peak_delta_bytes_; }
    f64 GetAvgBucketSize() const {
        return (num_centroids_ > 0) ? static_cast<f64>(num_vectors_) / num_centroids_ : 0.0;
    }
    f64 GetImbalanceRatio() const {
        if (num_centroids_ == 0 || num_vectors_ == 0) return 0.0;
        f64 avg = static_cast<f64>(num_vectors_) / num_centroids_;
        if (avg == 0) return 0.0;
        u32 max_b = 0;
        for (auto &m : bucket_metas_) {
            u32 c = m.base_count_;
            if (c > max_b) max_b = c;
        }
        return static_cast<f64>(max_b) / avg;
    }

private:
    // P0.1: Global structure lock (protects centroids, bucket_locks, assignments)
    mutable std::shared_mutex global_mtx_;

    // P1.1: Hadamard rotation (flip_signs_ replaces rot_matrix_)
    void GenerateHadamardParams();       // generates flip_signs_
    void ApplyHadamard(f32 *vec, u32 n) const; // in-place, n must be power of 2
    void ApplyRotation(const f32 *vec, f32 *out) const; // pad + Hadamard
    void ApplyInverseRotation(const f32 *vec, f32 *out) const; // same as forward (Hadamard is self-inverse up to scale)
    // Find nearest centroid (brute-force, used during build)
    u32 FindNearestCentroid(const f32 *vec) const;

private:
    RowID begin_row_id_;

    char *rabitq_data_;
    u32 num_vectors_;
    u32 max_vectors_;
    u32 dim_;
    u32 pad_dim_;                 // next power of 2 for Hadamard (P1.1)

    // P2.2: Memory budget (max delta bytes before auto-compact)
    u64 max_delta_bytes_ = 512ULL * 1024 * 1024; // 512 MB default

    // P2.3: File-backed storage via mmap
    std::string base_file_path_;
    int base_fd_ = -1;            // file descriptor for mmap, -1 = DRAM mode
    size_t base_file_size_ = 0;   // current mmap size

    // P1.1: Hadamard sign flip vector (one bool per pad_dim)
    bool *hadamard_flip_;          // random ±1 signs for Hadamard

    // P1.2: Hierarchical coarse/fine centroids
    u32 num_centroids_;            // fine centroids
    u32 coarse_count_;             // ∼sqrt(num_centroids_)
    std::vector<f32> centroids_;                // fine centroids
    std::vector<u32> centroid_to_coarse_;       // fine→coarse mapping
    std::vector<f32> coarse_centroids_;         // coarse centroids
    std::unique_ptr<HnswType> coarse_hnsw_;     // HNSW on coarse

    std::vector<SPFreshBucketMeta> bucket_metas_;

    u32 replica_count_;
    std::vector<u32> bucket_assignments_;

    // P0.3: Delta double-buffer
    SPFreshDeltaBuffer *delta_a_;
    SPFreshDeltaBuffer *delta_b_;
    std::atomic<u32> active_delta_idx_;
    mutable std::shared_mutex compact_mtx_;

    // P0.1: Per-bucket locks (protected by global_mtx_)
    mutable std::mutex *bucket_locks_;
    u32 bucket_locks_count_;

    std::vector<SPFreshRunningMean> running_means_;
    std::unordered_set<u32> deleted_set_;
    mutable std::shared_mutex delete_mtx_;

    // P3.1: Background maintenance thread
    std::jthread maintenance_thread_;
    std::atomic<bool> maintenance_stop_{false};

    // P3.3: Metrics
    std::atomic<u64> compact_count_{0};
    std::atomic<u64> split_count_{0};
    std::atomic<size_t> peak_delta_bytes_{0};

    size_t mem_used_{0};
};

} // namespace infinity

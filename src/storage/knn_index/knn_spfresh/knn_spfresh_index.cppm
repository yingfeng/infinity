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
    SPFreshIndexInMem();
    SPFreshIndexInMem(RowID begin_row_id, const IndexSPFresh *index_def, u32 embedding_dim, u32 max_vectors, const std::string &base_path = "");
    ~SPFreshIndexInMem() override;

    MemIndexTracerInfo GetInfo() const override;
    const ChunkIndexMetaInfo GetChunkIndexMetaInfo() const override;
    RowID GetBeginRowID() const override { return begin_row_id_; }

    // LIRE: Bulk build
    void Build(const f32 *vectors, u32 count);

    // RaBitQ format
    struct RabitQVec {
        f32 raw_norm_, norm_, sum_, error_;
        u8 compress_[];
    };
    static size_t RabitQVecSize(u32 dim) { return sizeof(RabitQVec) + dim / 8; }
    size_t EncodeWithRotation(f32 *code_buf, const f32 *vec) const;
    void DecodeWithRotation(const RabitQVec *code, f32 *out_vec) const;
    static f32 RabitQDistWithRotation(const RabitQVec *code, const f32 *rotated_q, u32 dim, f32 inv_sqrt_d);

    // LIRE: centroid routing
    using HnswType = KnnHnsw<PlainL2VecStoreType<f32>, u32>;
    void BuildCentroidIndex();
    void FindTopKCentroids(const f32 *query, u32 top_k, std::vector<u32> &out_ids, std::vector<f32> &out_dists) const;

    // LIRE: InsertDelta → global delta
    void InsertDelta(const f32 *vec, u32 row_id);

    // LIRE: Compact per-bucket: merge its delta entries into its base data
    void Compact();

    // LIRE: Search via RCU snapshot (no lock held during callback)
    // Parameters:
    //   query, dim: query vector
    //   max_candidates: max number of candidates to emit (0 = unlimited)
    //   centroid_score_threshold: skip centroid if score < max_score * threshold (0.0 = no filter)
    //   callback: called for each candidate with (segment_offset, approx_dist, raw_f32_vec)
    //     raw_f32_vec points to the original f32 vector stored in the bucket (or nullptr if unavailable)
    using SearchCallback = std::function<void(SegmentOffset, f32)>;
    void Search(const f32 *query,
                u32 dim,
                const SearchCallback &callback,
                u32 max_candidates = 0,
                f32 centroid_score_threshold = 0.0f) const;

    void MarkDeleted(u32 row_id);

    void StartBackgroundMaintenance();
    void StopBackgroundMaintenance();
    bool TryAutoCompact(u32 threshold = 8192);
    void Rebalance(u32 bucket_size_limit = 10000);
    std::vector<u32> SplitBucket(u32 bucket_id);

    void Save(LocalFileHandle &fh) const;
    void Load(LocalFileHandle &fh, size_t file_size);
    void Dump(BufferObj *buffer_obj, size_t *dump_size_ptr = nullptr);
    void TransferTo(SPFreshIndexInMem *target);

    u32 GetRowCount() const;
    u32 GetBaseRowCount() const;
    u32 GetDeltaCount() const {
        if (!delta_a_ || !delta_b_)
            return 0;
        u32 idx = active_delta_idx_.load(std::memory_order_acquire);
        return (idx == 0 ? delta_a_ : delta_b_)->entry_count_;
    }
    u32 dim() const { return dim_; }
    u32 GetDeletedCount() const { return static_cast<u32>(deleted_set_.size()); }
    u32 GetNumCentroids() const { return static_cast<u32>(buckets_.size()); }
    u64 GetCompactCount() const { return compact_count_; }
    u64 GetSplitCount() const { return split_count_; }

private:
    void GenerateHadamardParams();
    void ApplyHadamard(f32 *vec, u32 n) const;
    void ApplyRotation(const f32 *vec, f32 *out) const;
    void ApplyInverseRotation(const f32 *vec, f32 *out) const;

    // LIRE: base data is per-bucket
    RowID begin_row_id_;
    u32 dim_{0};
    u32 pad_dim_{1};
    bool *hadamard_flip_{nullptr};

    // LIRE: per-bucket storage (replaces rabitq_data_ + bucket_assignments_)
    std::vector<SPFreshBucketData> buckets_;
    std::vector<SPFreshBucketMeta> bucket_metas_;

    u32 num_vectors_{0}; // total across all buckets
    u32 replica_count_{1};
    u64 max_delta_bytes_{512ULL * 1024 * 1024};

    // LIRE: RCU snapshot for lock-free search
    struct SPFreshSnapshot {
        std::vector<SPFreshBucketData> buckets_;
        std::vector<SPFreshBucketMeta> bucket_metas_;
        u32 num_centroids_{0};
        u32 coarse_count_{0};
        u32 num_vectors_{0};
        std::vector<f32> centroids_;
        std::vector<u32> centroid_to_coarse_;
        std::vector<f32> coarse_centroids_;
        std::shared_ptr<const HnswType> coarse_hnsw_;
        u32 replica_count_{1};
    };
    void PublishSnapshot();
    void RefreshCentroidsFromRunningMeans();
    std::shared_ptr<const SPFreshSnapshot> snapshot_;
    mutable std::shared_mutex snapshot_mtx_;

    // LIRE: per-bucket delta stored in global double-buffer
    SPFreshDeltaBuffer *delta_a_{nullptr};
    SPFreshDeltaBuffer *delta_b_{nullptr};
    std::atomic<u32> active_delta_idx_{0};

    // P1.2: Hierarchical centroids
    u32 num_centroids_{0};
    u32 coarse_count_{0};
    std::vector<f32> centroids_;
    std::vector<u32> centroid_to_coarse_;
    std::vector<f32> coarse_centroids_;
    std::shared_ptr<const HnswType> coarse_hnsw_;
    std::vector<SPFreshRunningMean> running_means_;

    // Concurrent access
    mutable std::shared_mutex global_mtx_;
    mutable std::mutex *bucket_locks_{nullptr};
    u32 bucket_locks_count_{0};
    mutable std::shared_mutex compact_mtx_;
    mutable std::shared_mutex delete_mtx_;
    std::unordered_set<u32> deleted_set_;

    // Background maintenance
    std::jthread maintenance_thread_;

    // Dump: hold BufferHandle to prevent GC
    BufferHandle chunk_handle_;

    // Metrics
    std::atomic<u64> compact_count_{0};
    std::atomic<u64> split_count_{0};

    size_t mem_used_{0};
};

} // namespace infinity

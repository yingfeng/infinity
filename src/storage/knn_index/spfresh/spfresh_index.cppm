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
          rot_matrix_(nullptr), centroid_hnsw_(nullptr), num_centroids_(0),
          centroids_(), bucket_metas_(), replica_count_(0),
          bucket_assignments_(), delta_a_(nullptr), delta_b_(nullptr), active_delta_idx_(0),
          compact_mtx_(), bucket_locks_(nullptr), bucket_locks_count_(0),
          running_means_(), deleted_set_(), delete_mtx_(), mem_used_(0) {}

    SPFreshIndexInMem(RowID begin_row_id, const IndexSPFresh *index_def, u32 embedding_dim, u32 max_vectors);
    ~SPFreshIndexInMem() override;

    // BaseMemIndex interface
    MemIndexTracerInfo GetInfo() const override;
    const ChunkIndexMetaInfo GetChunkIndexMetaInfo() const override;
    RowID GetBeginRowID() const override { return begin_row_id_; }

    // ── Phase A: Full RaBitQ with rotation ──

    // Bulk build: K-Means clustering + centroid HNSW + RaBitQ compress
    void Build(const f32 *vectors, u32 count);

    // Insert a single vector into base array (used during Build)
    void InsertVector(const f32 *vec, u32 local_id);

    // ── Phase A+B: RaBitQ encode/decode with rotation matrix ──

    // RaBitQ compressed vector format (with rotation)
    struct RabitQVec {
        f32 raw_norm_;   // ||o||^2
        f32 norm_;       // ||o_r|| (after rotation + normalization)
        f32 sum_;        // sum of positive bits after rotation
        f32 error_;      // encoding error
        u8 compress_[];  // 1-bit code [dim/8 bytes]
    };

    static size_t RabitQVecSize(u32 dim) { return sizeof(RabitQVec) + dim / 8; }

    // Encode with Givens rotation (Phase A)
    size_t EncodeWithRotation(f32 *code_buf, const f32 *vec) const;
    // Decode (inverse rotation + dequantize)
    void DecodeWithRotation(const RabitQVec *code, f32 *out_vec) const;

    // Distance with rotation-aware correction (Phase A)
    // Both code and query must be in normalized-rotated space (unit norm)
    static f32 RabitQDistWithRotation(const RabitQVec *code, const f32 *rotated_query_normed,
                                       u32 dim, f32 inv_sqrt_d);

    // ── Phase A: Centroid HNSW routing ──
    using HnswType = KnnHnsw<PlainL2VecStoreType<f32>, u32>;

    // Build centroid HNSW from centroids_ array
    void BuildCentroidIndex();

    // Find top-K centroids for a query
    void FindTopKCentroids(const f32 *rotated_query, u32 top_k,
                           std::vector<u32> &out_centroid_ids,
                           std::vector<f32> &out_dists) const;

    // ── Phase B: Incremental insert (RNGSelection + delta double-buffer) ──

    void InsertDelta(const f32 *vec, u32 row_id);

    // ── Phase B+C: Delta double-buffer ──

    // Compact: merge delta into base (COW)
    void Compact();

    // Search with centroid pruning (Phase A) + delta merge (Phase B) + rerank
    using SearchCallback = std::function<void(SegmentOffset, f32)>;
    void Search(const f32 *query, u32 dim, const SearchCallback &callback) const;

    // ── Phase B: Delete (lazy) ──

    void MarkDeleted(u32 row_id);

    // ── Phase C: Rebalancer ──

    bool TryAutoCompact(u32 delta_threshold = 8192);
    void Rebalance(u32 bucket_size_limit = 10000);
    std::vector<u32> SplitBucket(u32 bucket_id);

    // ── Persistence ──

    void Save(LocalFileHandle &file_handle) const;
    void Load(LocalFileHandle &file_handle, size_t file_size);
    void Dump(BufferObj *buffer_obj, size_t *p_dump_size = nullptr);

    // ── Accessors ──

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

private:
    // Generate random orthogonal matrix via Givens rotations (Phase A)
    void GenerateRotationMatrix();

    // Apply rotation: out = rot_matrix * vec
    void ApplyRotation(const f32 *vec, f32 *out) const;

    // Apply inverse rotation: out = rot_matrix^T * vec
    void ApplyInverseRotation(const f32 *vec, f32 *out) const;

    // Find nearest centroid (exhaustive, used during build)
    u32 FindNearestCentroid(const f32 *vec) const;

private:
    RowID begin_row_id_;

    // ── Base RaBitQ compressed data ──
    char *rabitq_data_;
    u32 num_vectors_;
    u32 max_vectors_;
    u32 dim_;

    // ── Phase A: Rotation matrix (dim_ x dim_) ──
    f32 *rot_matrix_;  // random orthogonal matrix

    // ── Phase A: Centroid HNSW index ──
    std::unique_ptr<HnswType> centroid_hnsw_;

    u32 num_centroids_;
    std::vector<f32> centroids_;
    std::vector<SPFreshBucketMeta> bucket_metas_;

    // ── Phase B: Replica assignments ──
    u32 replica_count_;
    // For each vector, which centroids it belongs to (flat array: vector_id * replica_count_ + replica_idx)
    std::vector<u32> bucket_assignments_;

    // ── Phase B: Delta double-buffer ──
    // Phase C: COW compaction
    SPFreshDeltaBuffer *delta_a_;
    SPFreshDeltaBuffer *delta_b_;
    std::atomic<u32> active_delta_idx_; // 0 = delta_a_, 1 = delta_b_
    mutable std::shared_mutex compact_mtx_; // lock during compaction

    // ── Phase B: Per-bucket locks ──
    // Use mutex array (non-copyable, allocated in constructor)
    mutable std::mutex *bucket_locks_;
    u32 bucket_locks_count_;

    // ── Phase C: Running means for centroids ──
    std::vector<SPFreshRunningMean> running_means_;

    // ── Phase B: Deleted set ──
    std::unordered_set<u32> deleted_set_;
    mutable std::shared_mutex delete_mtx_;

    // ── Memory tracking ──
    size_t mem_used_{0};
};

} // namespace infinity

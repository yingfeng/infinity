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
    SPFreshIndexInMem(RowID begin_row_id, const IndexSPFresh *index_def, u32 embedding_dim, u32 max_vectors,
                      const std::string &base_path = "");
    ~SPFreshIndexInMem() override;

    MemIndexTracerInfo GetInfo() const override;
    const ChunkIndexMetaInfo GetChunkIndexMetaInfo() const override;
    RowID GetBeginRowID() const override { return begin_row_id_; }

    // ── RaBitQ format ──
    struct RabitQVec {
        f32 raw_norm_, norm_, sum_, error_;
        u8 compress_[];
    };
    static size_t RabitQVecSize(u32 dim) { return sizeof(RabitQVec) + dim / 8; }
    size_t EncodeWithRotation(f32 *code_buf, const f32 *vec) const;
    void DecodeWithRotation(const RabitQVec *code, f32 *out_vec) const;
    static f32 RabitQDistWithRotation(const RabitQVec *code, const f32 *rotated_q, u32 dim, f32 inv_sqrt_d);

    // ── Storage mode ──
    enum class StorageMode { kOwned, kMmap };
    StorageMode GetStorageMode() const { return storage_mode_; }

    // ── Build (from column store, 3-level hierarchical clustering) ──
    void Build(const f32 *vectors, u32 count);

    // ── Centroid routing (3-level: coarse HNSW → fine brute-force) ──
    using HnswType = KnnHnsw<PlainL2VecStoreType<f32>, u32>;
    void BuildHierarchicalIndex();
    void FindTopKCentroids(const f32 *query, u32 top_k, std::vector<u32> &out_ids, std::vector<f32> &out_dists) const;

    // ── Incremental insert ──
    void InsertDelta(const f32 *vec, u32 row_id);

    // ── Compact: merge delta into base, resolve overflows ──
    void Compact();

    // ── Search via RCU snapshot ──
    using SearchCallback = std::function<void(SegmentOffset, f32)>;
    void Search(const f32 *query, u32 dim, const SearchCallback &callback, u32 max_candidates = 0,
                f32 centroid_score_threshold = 0.0f) const;

    // ── Persistence ──
    void Save(LocalFileHandle &fh) const;
    void Load(LocalFileHandle &fh, size_t file_size);
    void LoadFromMmap(const u8 *base, size_t size);
    void Dump(BufferObj *buffer_obj, size_t *dump_size_ptr = nullptr);
    void TransferTo(SPFreshIndexInMem *target);

    // Deletion: handled by column store, no-op in index
    void MarkDeleted(u32) {} // NOLINT

    // ── Accessors ──
    u32 GetRowCount() const;
    u32 GetBaseRowCount() const { return num_vectors_; }
    u32 GetDeltaCount() const {
        if (!delta_a_ || !delta_b_)
            return 0;
        u32 idx = active_delta_idx_.load(std::memory_order_acquire);
        return (idx == 0 ? delta_a_ : delta_b_)->entry_count_;
    }
    u32 dim() const { return dim_; }
    u32 GetNumCentroids() const { return num_centroids_; }
    u32 GetDeletedCount() const { return 0; } // deletion handled by column store

private:
    // ── Hadamard ──
    void GenerateHadamardParams();
    void ApplyHadamard(f32 *vec, u32 n) const;
    void ApplyRotation(const f32 *vec, f32 *out) const;
    void ApplyInverseRotation(const f32 *vec, f32 *out) const;

    // ── Snapshot management ──
    struct SPFreshSnapshot {
        // Shared data (always copied from main index)
        u32 num_centroids_{0};
        u32 coarse_count_{0};
        u32 num_vectors_{0};
        std::vector<f32> centroids_;
        std::vector<u32> centroid_to_coarse_;
        std::vector<f32> coarse_centroids_;
        std::shared_ptr<const HnswType> coarse_hnsw_;
        std::vector<SPFreshBucketMeta> bucket_metas_;
        std::vector<SPFreshOverflowRecord> overflow_records_;
        u32 replica_count_{1};

        // Storage mode
        StorageMode storage_mode_{StorageMode::kOwned};

        // kOwned mode: full bucket data copy
        std::vector<SPFreshBucketData> buckets_;

        // kMmap mode: lightweight references into mmap region
        // We keep a BufferHandle-style reference via owning the mmap_base
        // In practice, the SPFreshIndexInMem owns the mmap and is kept alive via MemIndex
        const u8 *mmap_base_{nullptr};
        const BucketOffsetEntry *mmap_offset_table_{nullptr};
        u32 mmap_bucket_count_{0};

        // Accessors for unified bucket access
        u32 GetBucketCount(u32 bucket_id) const;
        const u32 *GetBucketRowIDs(u32 bucket_id) const;
        const u8 *GetBucketCodes(u32 bucket_id, u32 code_size) const;
    };
    void PublishSnapshot();
    void RefreshCentroidsFromRunningMeans();

    // ── Mode transition ──
    // When in kMmap mode and InsertDelta/Compact is called,
    // load bucket data from mmap into mutable buckets_ first.
    void TransitionToOwned();

    // ── Overflow / Split ──
    static constexpr u32 kDefaultOverflowFactor = 3;

    // ── File I/O helpers ──
    void SerializeSection1(std::vector<char> &buf) const;
    void DeserializeSection1(const u8 *ptr, const SPFreshFileHeader &header);

    // ── Member data ──
    RowID begin_row_id_;
    StorageMode storage_mode_{StorageMode::kOwned};

    u32 dim_{0};
    u32 pad_dim_{1};
    bool *hadamard_flip_{nullptr};

    // Hierarchical centroids (always resident in DRAM)
    u32 num_centroids_{0};
    u32 coarse_count_{0};
    std::vector<f32> centroids_;
    std::vector<u32> centroid_to_coarse_;
    std::vector<f32> coarse_centroids_;
    std::shared_ptr<const HnswType> coarse_hnsw_;

    // Runtime metadata
    std::vector<SPFreshRunningMean> running_means_;
    std::vector<SPFreshBucketMeta> bucket_metas_;

    // kOwned mode: mutable bucket storage
    std::vector<SPFreshBucketData> buckets_;

    // kMmap mode: mmap region pointers
    const u8 *mmap_base_{nullptr};
    const BucketOffsetEntry *mmap_offset_table_{nullptr};
    SPFreshFileHeader mmap_header_{};

    // Delta double-buffer
    SPFreshDeltaBuffer *delta_a_{nullptr};
    SPFreshDeltaBuffer *delta_b_{nullptr};
    std::atomic<u32> active_delta_idx_{0};

    // Overflow tracking
    std::vector<SPFreshOverflowRecord> overflow_records_;
    mutable std::shared_mutex overflow_mtx_;

    // Base data tracking
    u32 num_vectors_{0};
    u32 replica_count_{1};
    u64 max_delta_bytes_{512ULL * 1024 * 1024};
    u32 bucket_size_limit_{10000};

    // RCU
    std::shared_ptr<const SPFreshSnapshot> snapshot_;
    mutable std::shared_mutex snapshot_mtx_;

    // Concurrency
    mutable std::shared_mutex global_mtx_;
    mutable std::mutex *bucket_locks_{nullptr};
    u32 bucket_locks_count_{0};
    mutable std::shared_mutex compact_mtx_;

    // Dump guard
    BufferHandle chunk_handle_;
    size_t mem_used_{0};
};

} // namespace infinity

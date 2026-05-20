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

import std.compat;

import internal_types;
import column_def;
import row_id;

namespace infinity {

export class SPFreshIndexInMem : public BaseMemIndex {
public:
    SPFreshIndexInMem()
        : begin_row_id_(), rabitq_meta_(nullptr), rabitq_data_(nullptr), num_vectors_(0), max_vectors_(0), dim_(0),
          delta_compacted_seq_(0) {}

    SPFreshIndexInMem(RowID begin_row_id, const IndexSPFresh *index_def, u32 embedding_dim, u32 max_vectors);
    ~SPFreshIndexInMem() override;

    // BaseMemIndex interface
    MemIndexTracerInfo GetInfo() const override;
    const ChunkIndexMetaInfo GetChunkIndexMetaInfo() const override;
    RowID GetBeginRowID() const override { return begin_row_id_; }

    // Bulk build: writes into base array
    void InsertBlockData(const f32 *vectors, u32 count);
    void InsertVector(const f32 *vec, u32 vector_id);

    // Incremental insert: writes into DRAM delta (append-only)
    // Raw vectors already persisted in column store; we only store RaBitQ code + bucket mapping
    void InsertDelta(const f32 *vec, u32 row_id);

    // Search: RaBitQ approximate distance, scans both base + delta
    using SearchCallback = std::function<void(SegmentOffset, f32)>;
    void Search(const f32 *query,
                u32 dim,
                const SearchCallback &callback) const;

    // Compaction: flush delta entries into base array, clear delta
    void Compact();

    // Persistence
    void Save(LocalFileHandle &file_handle) const;
    void Load(LocalFileHandle &file_handle, size_t file_size);
    void Dump(BufferObj *buffer_obj, size_t *p_dump_size = nullptr);

    // Phase 3: Rebalancer
    bool TryAutoCompact(u32 delta_threshold = 8192);
    void Rebalance(u32 bucket_size_limit = 10000);
    std::vector<u32> SplitBucket(u32 bucket_id);

    // Phase 4: Delete + GC
    // Mark a vector as deleted (lazy deletion - filtered during search, GC'd during compact)
    void MarkDeleted(u32 row_id);

    // Get the deleted set size
    u32 GetDeletedCount() const { return static_cast<u32>(deleted_set_.size()); }

    // Accessors
    u32 GetRowCount() const { return num_vectors_ + static_cast<u32>(delta_entries_.size()) - static_cast<u32>(deleted_set_.size()); }
    u32 GetBaseRowCount() const { return num_vectors_; }
    u32 GetDeltaCount() const { return static_cast<u32>(delta_entries_.size()); }
    u32 dim() const { return dim_; }

    // RaBitQ compressed vector format
    struct RabitQVec {
        f32 raw_norm_;
        f32 norm_;
        f32 sum_;
        f32 error_;
        u8 compress_[];
    };

    // Delta entry (DRAM, append-only)
    struct DeltaEntry {
        RabitQVec code_storage[1]; // flexible array: sizeof(RabitQVec) + dim_/8
    };

    // Encode a raw vector into a RabitQVec
    static size_t Encode(f32 *code_buf, const f32 *vec, u32 dim, f32 *out_norm);

private:
    static f32 RabitQDistance(const RabitQVec *code, const f32 *query, u32 dim, f32 inv_sqrt_d, f32 query_norm);

private:
    RowID begin_row_id_;

    // Base compressed data (from initial build / compaction)
    void *rabitq_meta_;  // unused placeholder
    void *rabitq_data_;  // base compressed vectors
    u32 num_vectors_;
    u32 max_vectors_;
    u32 dim_;

    // Delta (DRAM, append-only for incremental insert)
    std::vector<std::vector<char>> delta_entries_; // each entry: RabitQVec + compress bytes
    std::atomic<u64> delta_compacted_seq_;          // version to detect concurrent compact

    // Bucket metadata
    std::vector<u32> bucket_assignments_;
    u32 num_centroids_;
    std::vector<f32> centroids_;
    std::vector<SPFreshBucketMeta> bucket_metas_;

    // Deleted vector IDs (lazy, GC'd during compact)
    std::unordered_set<u32> deleted_set_;

    // Memory tracking
    size_t mem_used_{0};
};

} // namespace infinity

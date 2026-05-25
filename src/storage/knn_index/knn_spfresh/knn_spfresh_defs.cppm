// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

export module infinity_core:spfresh_defs;

import std.compat;

import :infinity_type;

namespace infinity {

// ── File Format Constants ──
export constexpr u32 SPFRESH_MAGIC = 0x50534652; // "PSFR"
export constexpr u32 SPFRESH_VERSION = 1;

// ── Bucket Offset Table Entry (in mmap file) ──
export struct BucketOffsetEntry {
    u64 file_offset_; // byte offset of this bucket's data in the file
    u32 count_;       // number of vectors in this bucket
    u32 flags_;       // bit 0: overflow, bits 1-15: reserved
};

// ── File Header (256 bytes, fixed size) ──
#pragma pack(1)
export struct SPFreshFileHeader {
    u32 magic_ = SPFRESH_MAGIC;
    u32 version_ = SPFRESH_VERSION;
    u32 dim_;
    u32 pad_dim_;
    u32 num_centroids_;   // fine centroids
    u32 coarse_count_;    // coarse centroids
    u32 num_vectors_;
    u32 bucket_count_;    // = num_centroids_
    u32 code_size_;       // RabitQVecSize(dim)
    u32 replica_count_;
    u32 bucket_size_limit_;
    u32 max_delta_mb_;
    u64 hadamard_seed_;
    u64 section1_offset_;        // = 0x100 (constant)
    u64 section1_size_;
    u64 offset_table_offset_;
    u64 offset_table_count_;
    u64 bucket_data_offset_;
    u64 total_bucket_data_size_;
    u8 reserved_[256 - 7 * 8 - 4 * 12]; // pad to 256 bytes

    bool Validate() const { return magic_ == SPFRESH_MAGIC && version_ == SPFRESH_VERSION; }
};
static_assert(sizeof(SPFreshFileHeader) == 256, "SPFreshFileHeader must be 256 bytes");
#pragma pack()

// ── Per-bucket metadata ──
export struct SPFreshBucketMeta {
    u32 base_count_{0};   // vectors in base file
    u32 delta_count_{0};  // vectors in delta buffer
    u64 version_{0};
    bool is_retired_{false}; // marked for GC after overflow split
    bool overflow_{false};   // exceeds overflow threshold

    u32 TotalCount() const { return base_count_ + delta_count_; }
    bool NeedsSplit(u32 limit) const { return overflow_ && base_count_ > limit * 2; }
};

// ── Running mean tracker for centroid updates ──
export struct SPFreshRunningMean {
    std::vector<f64> sum_;
    u64 count_{0};

    void Update(const f32 *vec, u32 dim) {
        if (sum_.empty())
            sum_.resize(dim, 0.0);
        for (u32 d = 0; d < dim; ++d)
            sum_[d] += static_cast<f64>(vec[d]);
        ++count_;
    }
    void GetCentroid(f32 *out, u32 dim) const {
        if (count_ == 0)
            return;
        f64 inv = 1.0 / static_cast<f64>(count_);
        for (u32 d = 0; d < dim; ++d)
            out[d] = static_cast<f32>(sum_[d] * inv);
    }
};

// ── Per-bucket RaBitQ code storage ──
export struct SPFreshBucketData {
    // Owned mode: mutable vectors for building/writing
    std::vector<char> codes_;
    std::vector<u32> row_ids_;
    u32 count_{0};

    // Mmap mode: read-only pointers into mmap region
    struct MmapView {
        const u8 *base_ptr_{nullptr};  // mmap base (keeps reference alive)
        const u8 *row_ids_ptr_{nullptr};
        const u8 *codes_ptr_{nullptr};
        u32 count_{0};
    };
    MmapView mmap_view_;

    // Accessors: unified interface for both modes
    bool IsMmap() const { return mmap_view_.base_ptr_ != nullptr && codes_.empty(); }
    u32 GetCount() const { return IsMmap() ? mmap_view_.count_ : count_; }
    const u32 *GetRowIDs() const { return IsMmap() ? reinterpret_cast<const u32 *>(mmap_view_.row_ids_ptr_) : row_ids_.data(); }
    const u8 *GetCodes() const { return IsMmap() ? mmap_view_.codes_ptr_ : reinterpret_cast<const u8 *>(codes_.data()); }

    // Non-owning access for reading from external buffer
    void SetMmapView(const u8 *base, u64 offset, u32 count, u32 code_size) {
        mmap_view_.base_ptr_ = base;
        mmap_view_.row_ids_ptr_ = base + offset;
        mmap_view_.codes_ptr_ = base + offset + static_cast<u64>(count) * sizeof(u32);
        mmap_view_.count_ = count;
    }
    void ClearMmapView() {
        mmap_view_ = MmapView{};
    }

    void AppendCode(const char *code, u32 entry_size, u32 row_id) {
        codes_.insert(codes_.end(), code, code + entry_size);
        row_ids_.push_back(row_id);
        ++count_;
    }

    void Clear() {
        codes_.clear();
        row_ids_.clear();
        count_ = 0;
        ClearMmapView();
    }
};

// ── Search result entry ──
export struct SPFreshSearchResult {
    u32 vector_id_{0};
    f32 distance_{0.0f};
    bool operator<(const SPFreshSearchResult &other) const { return distance_ < other.distance_; }
};

// ── Global delta buffer (entry = bucket_id + row_id + code) ──
export struct SPFreshDeltaBuffer {
    std::vector<char> data_;
    u32 entry_count_{0};
    u32 entry_size_; // sizeof(RabitQVec) + dim/8

    explicit SPFreshDeltaBuffer(u32 entry_size) : entry_size_(entry_size) {}

    void Append(u32 bucket_id, u32 row_id, const char *code) {
        size_t eb = sizeof(u32) * 2 + entry_size_;
        size_t pos = data_.size();
        data_.resize(pos + eb);
        std::memcpy(data_.data() + pos, &bucket_id, sizeof(u32));
        std::memcpy(data_.data() + pos + sizeof(u32), &row_id, sizeof(u32));
        std::memcpy(data_.data() + pos + sizeof(u32) * 2, code, entry_size_);
        ++entry_count_;
    }
    void Clear() {
        data_.clear();
        entry_count_ = 0;
    }
    size_t GetEntrySize() const { return sizeof(u32) * 2 + entry_size_; }
    u32 GetBucketId(size_t index) const {
        u32 id;
        std::memcpy(&id, data_.data() + index * GetEntrySize(), sizeof(u32));
        return id;
    }
    u32 GetRowId(size_t index) const {
        u32 id;
        std::memcpy(&id, data_.data() + index * GetEntrySize() + sizeof(u32), sizeof(u32));
        return id;
    }
    const char *GetCode(size_t index) const { return data_.data() + index * GetEntrySize() + sizeof(u32) * 2; }
};

// ── Overflow split tracking ──
export struct SPFreshOverflowRecord {
    u32 old_bucket_id_;
    u32 new_centroid_id_A_;
    u32 new_centroid_id_B_;
    bool resolved_{false}; // set to true after Compact resolves this record
};

} // namespace infinity

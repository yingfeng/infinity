// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

export module infinity_core:spfresh_defs;

import std.compat;

import :infinity_type;

namespace infinity {

// Per-bucket metadata
export struct SPFreshBucketMeta {
    u64 version_{0};
    u32 base_count_{0};
    u32 delta_count_{0};
    bool is_retired_{false};
};

// Running mean tracker for centroid updates
export struct SPFreshRunningMean {
    std::vector<f64> sum_;
    u64 count_{0};

    void Update(const f32 *vec, u32 dim) {
        if (sum_.empty()) {
            sum_.resize(dim, 0.0);
        }
        for (u32 i = 0; i < dim; ++i) {
            sum_[i] += static_cast<f64>(vec[i]);
        }
        ++count_;
    }

    void GetCentroid(f32 *out, u32 dim) const {
        if (count_ == 0) return;
        f64 inv = 1.0 / static_cast<f64>(count_);
        for (u32 i = 0; i < dim; ++i) {
            out[i] = static_cast<f32>(sum_[i] * inv);
        }
    }
};

// Search result entry
export struct SPFreshSearchResult {
    u32 vector_id_{0};
    f32 distance_{0.0f};
    bool operator<(const SPFreshSearchResult &other) const { return distance_ < other.distance_; }
};

// Delta buffer for double-buffering (stores bucket_id + original_row_id + compressed code)
export struct SPFreshDeltaBuffer {
    // Each delta entry: bucket_id (u32) + original_row_id (u32) + compressed code (RabitQVec + dim/8 bytes)
    std::vector<char> data_;
    u32 entry_count_{0};
    u32 entry_size_; // sizeof(RabitQVec) + dim_/8

    explicit SPFreshDeltaBuffer(u32 entry_size) : entry_size_(entry_size) {}

    void Append(u32 bucket_id, u32 row_id, const char *code) {
        size_t entry_bytes = sizeof(u32) * 2 + entry_size_;
        size_t pos = data_.size();
        data_.resize(pos + entry_bytes);
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

    const char *GetCode(size_t index) const {
        return data_.data() + index * GetEntrySize() + sizeof(u32) * 2;
    }
};

} // namespace infinity

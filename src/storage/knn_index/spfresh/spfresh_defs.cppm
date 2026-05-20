// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

export module infinity_core:spfresh_defs;

import std.compat;

import :infinity_type;

namespace infinity {

export struct SPFreshBucketMeta {
    u64 version_{0};
    u32 base_count_{0};
    u32 delta_count_{0};
    bool is_retired_{false};
};

export struct SPFreshRunningMean {
    std::vector<f64> sum_;
    u64 count_{0};

    void Update(const f32 *vec, u32 dim) {
        if (sum_.empty()) {
            sum_.resize(dim, 0.0);
        }
        for (u32 i = 0; i < dim; ++i) {
            sum_[i] += vec[i];
        }
        ++count_;
    }

    void GetCentroid(f32 *out, u32 dim) const {
        if (count_ == 0) return;
        for (u32 i = 0; i < dim; ++i) {
            out[i] = static_cast<f32>(sum_[i] / static_cast<f64>(count_));
        }
    }
};

export struct SPFreshSearchResult {
    u32 vector_id_{0};
    f32 distance_{0.0f};
    bool operator<(const SPFreshSearchResult &other) const { return distance_ < other.distance_; }
};

} // namespace infinity

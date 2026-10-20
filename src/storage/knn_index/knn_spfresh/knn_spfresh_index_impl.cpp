// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module;

#include <immintrin.h>

module infinity_core:spfresh_index.impl;

import :spfresh_index;
import :spfresh_defs;
import :index_spfresh;
import :local_file_handle;
import :infinity_exception;
import :logger;

import std;

import internal_types;
import embedding_info;

namespace infinity {

namespace {

inline u32 PopCount64(u64 x) {
#if defined(__AVX2__) || defined(__POPCNT__)
    return static_cast<u32>(__builtin_popcountll(x));
#else
    u32 c = 0;
    while (x) {
        x &= x - 1;
        ++c;
    }
    return c;
#endif
}

} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// Construction / destruction
// ═════════════════════════════════════════════════════════════════════════════

SPFreshIndexInMem::SPFreshIndexInMem() {
    mem_used_ = 0;
    // Default-constructed instances are used as Load() targets; the buffers must
    // exist before the entry stride is known.
    delta_a_ = new SPFreshDeltaBuffer(0);
    delta_b_ = new SPFreshDeltaBuffer(0);
}

SPFreshIndexInMem::SPFreshIndexInMem(RowID begin_row_id, const IndexSPFresh *index_def, u32 embedding_dim, u32 /*max_vectors*/,
                                     const std::string & /*base_path*/)
    : begin_row_id_(begin_row_id), storage_mode_(StorageMode::kOwned), dim_(embedding_dim) {
    pad_dim_ = 1;
    while (pad_dim_ < dim_) {
        pad_dim_ <<= 1;
    }
    code_width_ = QuantCodeWidth(dim_);
    rot_seed_ = 0x9E3779B97F4A7C15ULL;
    GenerateRotationParams(rot_seed_);

    if (index_def != nullptr) {
        // `bucket_size_limit` is reused as the partition capacity (fan-out). Legacy
        // indexes store 10000 which would produce a single huge partition, so clamp.
        u32 limit = index_def->bucket_size_limit_;
        max_partition_size_ = (limit == 0 || limit > 2048) ? kDefaultMaxPartitionSize : limit;
        max_partition_size_ = std::max(max_partition_size_, 16u);
        min_partition_size_ = std::max(1u, max_partition_size_ / 8);
        max_delta_bytes_ = static_cast<u64>(index_def->max_delta_mb_ == 0 ? 512 : index_def->max_delta_mb_) * 1024 * 1024;
    }

    partitions_ = std::make_shared<std::vector<PartitionMeta>>();
    centroids_ = std::make_shared<std::vector<f32>>();
    slot_table_ = std::make_shared<std::vector<DataSlot>>();
    data_region_ = std::make_shared<std::vector<char>>();
    deleted_bits_ = std::make_shared<std::vector<u64>>();

    delta_a_ = new SPFreshDeltaBuffer(QuantVecStride(dim_));
    delta_b_ = new SPFreshDeltaBuffer(QuantVecStride(dim_));

    EnsureCapacity(1024);
}

SPFreshIndexInMem::~SPFreshIndexInMem() {
    StopFixupWorker();
    delete delta_a_;
    delete delta_b_;
    delta_a_ = nullptr;
    delta_b_ = nullptr;
}

// ═════════════════════════════════════════════════════════════════════════════
// Rotation
// ═════════════════════════════════════════════════════════════════════════════

void SPFreshIndexInMem::GenerateRotationParams(u64 seed) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution dist(0.5);
    rotation_flip_.assign(pad_dim_, false);
    for (u32 i = 0; i < pad_dim_; ++i) {
        rotation_flip_[i] = dist(rng);
    }
}

void SPFreshIndexInMem::ApplyRotation(f32 *vec) const {
    // Fast Walsh-Hadamard transform (self-inverse up to the 1/sqrt(n) scaling).
    for (u32 len = 1; len < pad_dim_; len <<= 1) {
        for (u32 i = 0; i < pad_dim_; i += len << 1) {
            for (u32 j = 0; j < len; ++j) {
                const f32 a = vec[i + j];
                const f32 b = vec[i + j + len];
                vec[i + j] = a + b;
                vec[i + j + len] = a - b;
            }
        }
    }
    for (u32 i = 0; i < pad_dim_; ++i) {
        if (rotation_flip_[i]) {
            vec[i] = -vec[i];
        }
    }
    const f32 scale = 1.0f / std::sqrt(static_cast<f32>(pad_dim_));
    for (u32 i = 0; i < pad_dim_; ++i) {
        vec[i] *= scale;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Quantization
// ═════════════════════════════════════════════════════════════════════════════

void SPFreshIndexInMem::EncodeRelative(const f32 *vec, const f32 *centroid, QuantizedVectorMeta &meta, u8 *codes) const {
    std::memset(codes, 0, static_cast<size_t>(code_width_) * 8);
    thread_local std::vector<f32> buffer;
    if (buffer.size() < pad_dim_) {
        buffer.resize(pad_dim_);
    }

    f64 sq = 0;
    for (u32 d = 0; d < dim_; ++d) {
        const f64 diff = static_cast<f64>(vec[d]) - centroid[d];
        buffer[d] = static_cast<f32>(diff);
        sq += diff * diff;
    }
    std::memset(buffer.data() + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));

    const f32 centroid_distance = static_cast<f32>(std::sqrt(sq));
    meta.centroid_distance_ = centroid_distance;
    const f32 inv = centroid_distance > 1e-20f ? 1.0f / centroid_distance : 1.0f;
    for (u32 d = 0; d < dim_; ++d) {
        buffer[d] *= inv;
    }

    ApplyRotation(buffer.data());

    u32 ones = 0;
    for (u32 d = 0; d < dim_; ++d) {
        if (buffer[d] > 0.0f) {
            codes[d >> 3] |= static_cast<u8>(1u << (d & 7u));
            ++ones;
        }
    }
    meta.code_count_ = ones;

    // ⟨ō, o⟩ where ō is the ±1/√dim code vector and o is the rotated unit vector.
    const f32 per_dim = 1.0f / std::sqrt(static_cast<f32>(dim_));
    f64 dot = 0;
    for (u32 d = 0; d < dim_; ++d) {
        const bool bit = (codes[d >> 3] >> (d & 7u)) & 1u;
        dot += static_cast<f64>(bit ? per_dim : -per_dim) * buffer[d];
    }
    const f32 safe = std::fabs(static_cast<f32>(dot)) > 1e-20f ? static_cast<f32>(dot) : 1e-20f;
    meta.quantized_dot_product_ = 1.0f / safe;
}

void SPFreshIndexInMem::DecodeRelative(const QuantizedVectorMeta &meta, const u8 *codes, const f32 *centroid, f32 *out_vec) const {
    thread_local std::vector<f32> buffer;
    if (buffer.size() < pad_dim_) {
        buffer.resize(pad_dim_);
    }
    const f32 per_dim = 1.0f / std::sqrt(static_cast<f32>(dim_));
    for (u32 d = 0; d < dim_; ++d) {
        const bool bit = (codes[d >> 3] >> (d & 7u)) & 1u;
        buffer[d] = bit ? per_dim : -per_dim;
    }
    std::memset(buffer.data() + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));
    ApplyRotation(buffer.data());
    for (u32 d = 0; d < dim_; ++d) {
        out_vec[d] = centroid[d] + buffer[d] * meta.centroid_distance_;
    }
}

void SPFreshIndexInMem::QuantizeQuery(QueryQuantized &qq, const f32 *query, const f32 *centroid) const {
    thread_local std::vector<f32> buffer;
    if (buffer.size() < pad_dim_) {
        buffer.resize(pad_dim_);
    }

    f64 sq = 0;
    for (u32 d = 0; d < dim_; ++d) {
        const f64 diff = static_cast<f64>(query[d]) - centroid[d];
        buffer[d] = static_cast<f32>(diff);
        sq += diff * diff;
    }
    std::memset(buffer.data() + dim_, 0, (pad_dim_ - dim_) * sizeof(f32));

    const f32 centroid_distance = static_cast<f32>(std::sqrt(sq));
    qq.centroid_distance_ = centroid_distance;
    const f32 inv = centroid_distance > 1e-20f ? 1.0f / centroid_distance : 1.0f;
    for (u32 d = 0; d < dim_; ++d) {
        buffer[d] *= inv;
    }

    ApplyRotation(buffer.data());

    f32 min_val = std::numeric_limits<f32>::max();
    f32 max_val = std::numeric_limits<f32>::lowest();
    for (u32 d = 0; d < dim_; ++d) {
        min_val = std::min(min_val, buffer[d]);
        max_val = std::max(max_val, buffer[d]);
    }
    if (!(max_val > min_val)) {
        max_val = min_val + 1e-6f;
    }

    constexpr f32 kQuantRange = 15.0f; // 4 bits
    const f32 delta = (max_val - min_val) / kQuantRange;
    qq.min_val_ = min_val;
    qq.delta_ = delta;
    qq.sqrt_dim_ = std::sqrt(static_cast<f32>(dim_));
    qq.sqrt_dim_inv_ = 1.0f / qq.sqrt_dim_;

    for (u32 p = 0; p < 4; ++p) {
        qq.planes_[p].assign(code_width_, 0ull);
        qq.plane_popcount_[p] = 0;
    }
    qq.quantized_sum_ = 0;
    const f32 inv_delta = 1.0f / delta;
    for (u32 d = 0; d < dim_; ++d) {
        // Round-to-nearest: keeps the estimator unbiased for ranking purposes
        // (stochastic dither increases variance, which hurts small-dimensional data).
        f32 scaled = (buffer[d] - min_val) * inv_delta;
        if (scaled < 0.0f) {
            scaled = 0.0f;
        }
        u32 q = static_cast<u32>(scaled + 0.5f);
        if (q > 15u) {
            q = 15u;
        }
        qq.quantized_sum_ += q;
        for (u32 p = 0; p < 4; ++p) {
            if ((q >> p) & 1u) {
                qq.planes_[p][d >> 6] |= (1ull << (d & 63u));
                ++qq.plane_popcount_[p];
            }
        }
    }
}

f32 SPFreshIndexInMem::EstimateDistance(const QuantizedVectorMeta &meta, const u8 *codes, const QueryQuantized &qq) const {
    const auto *code = reinterpret_cast<const u64 *>(codes);

    // sum over dimensions where the stored sign bit is set, weighted by 2^p,
    // which reconstructs sum_{bit=1} quantized_value (NOT the Hamming weight).
    f32 bit_product = 0.0f;
    for (u32 p = 0; p < 4; ++p) {
        const u64 *plane = qq.planes_[p].data();
        u32 count = 0;
        for (u32 w = 0; w < code_width_; ++w) {
            count += PopCount64(code[w] & plane[w]);
        }
        bit_product += static_cast<f32>(count) * static_cast<f32>(1u << p);
    }
    const f32 quantized_sum = static_cast<f32>(qq.quantized_sum_);

    const f32 term1 = 2.0f * qq.delta_ * qq.sqrt_dim_inv_ * bit_product;
    const f32 term2 = 2.0f * qq.min_val_ * qq.sqrt_dim_inv_ * static_cast<f32>(meta.code_count_);
    const f32 term3 = qq.delta_ * qq.sqrt_dim_inv_ * quantized_sum;
    const f32 term4 = qq.sqrt_dim_ * qq.min_val_;
    f32 est = (term1 + term2 - term3 - term4) * meta.quantized_dot_product_;
    est = std::clamp(est, -1.0f, 1.0f);

    const f32 cd = meta.centroid_distance_;
    const f32 qd = qq.centroid_distance_;
    return cd * cd + qd * qd - 2.0f * cd * qd * est;
}

f32 SPFreshIndexInMem::ErrorBound(const QuantizedVectorMeta &meta, const QueryQuantized &qq) const {
    const f32 bound = 2.0f * meta.centroid_distance_ * qq.centroid_distance_ / std::sqrt(static_cast<f32>(dim_));
    return std::isfinite(bound) ? std::fabs(bound) : 0.0f;
}

// ═════════════════════════════════════════════════════════════════════════════
// K-means
// ═════════════════════════════════════════════════════════════════════════════

u32 SPFreshIndexInMem::KMeans(const f32 *vectors, u32 count, u32 k, u32 max_iter, std::vector<f32> &out_centroids,
                              std::vector<u32> &out_labels) const {
    out_centroids.clear();
    out_labels.clear();
    if (count == 0 || k == 0) {
        return 0;
    }
    k = std::min(k, count);
    out_centroids.assign(static_cast<size_t>(k) * dim_, 0.0f);
    out_labels.assign(count, 0);

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<u32> pick_index(0, count - 1);

    // ── k-means++ seeding ──
    const u32 first = pick_index(rng);
    std::memcpy(out_centroids.data(), vectors + static_cast<size_t>(first) * dim_, dim_ * sizeof(f32));

    std::vector<f32> min_dist(count, std::numeric_limits<f32>::max());
    for (u32 c = 1; c < k; ++c) {
        f64 total = 0;
        const f32 *prev = out_centroids.data() + static_cast<size_t>(c - 1) * dim_;
        for (u32 i = 0; i < count; ++i) {
            const f32 *v = vectors + static_cast<size_t>(i) * dim_;
            f64 sq = 0;
            for (u32 d = 0; d < dim_; ++d) {
                const f64 diff = static_cast<f64>(v[d]) - prev[d];
                sq += diff * diff;
            }
            const f32 dist = static_cast<f32>(sq);
            min_dist[i] = std::min(min_dist[i], dist);
            total += min_dist[i];
        }
        if (total <= 0) {
            std::memcpy(out_centroids.data() + static_cast<size_t>(c) * dim_, vectors + static_cast<size_t>(pick_index(rng)) * dim_,
                        dim_ * sizeof(f32));
            continue;
        }
        std::uniform_real_distribution<f64> uniform(0.0, total);
        f64 target = uniform(rng);
        u32 chosen = count - 1;
        for (u32 i = 0; i < count; ++i) {
            target -= min_dist[i];
            if (target <= 0) {
                chosen = i;
                break;
            }
        }
        std::memcpy(out_centroids.data() + static_cast<size_t>(c) * dim_, vectors + static_cast<size_t>(chosen) * dim_, dim_ * sizeof(f32));
    }

    // ── Lloyd iterations ──
    std::vector<f64> sums(static_cast<size_t>(k) * dim_, 0.0);
    std::vector<u32> counts(k, 0);
    std::vector<f32> new_centroids(static_cast<size_t>(k) * dim_, 0.0f);

    for (u32 iter = 0; iter < max_iter; ++iter) {
        // assign
        for (u32 i = 0; i < count; ++i) {
            const f32 *v = vectors + static_cast<size_t>(i) * dim_;
            u32 best = 0;
            f64 best_dist = std::numeric_limits<f64>::max();
            for (u32 c = 0; c < k; ++c) {
                const f32 *cent = out_centroids.data() + static_cast<size_t>(c) * dim_;
                f64 sq = 0;
                for (u32 d = 0; d < dim_; ++d) {
                    const f64 diff = static_cast<f64>(v[d]) - cent[d];
                    sq += diff * diff;
                }
                if (sq < best_dist) {
                    best_dist = sq;
                    best = c;
                }
            }
            out_labels[i] = best;
        }

        // update
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (u32 i = 0; i < count; ++i) {
            const u32 c = out_labels[i];
            const f32 *v = vectors + static_cast<size_t>(i) * dim_;
            f64 *s = sums.data() + static_cast<size_t>(c) * dim_;
            for (u32 d = 0; d < dim_; ++d) {
                s[d] += v[d];
            }
            ++counts[c];
        }
        f64 shift = 0;
        for (u32 c = 0; c < k; ++c) {
            f32 *dst = new_centroids.data() + static_cast<size_t>(c) * dim_;
            if (counts[c] == 0) {
                // Re-seed empty cluster to break degeneracy.
                std::memcpy(dst, vectors + static_cast<size_t>(pick_index(rng)) * dim_, dim_ * sizeof(f32));
            } else {
                const f64 inv = 1.0 / static_cast<f64>(counts[c]);
                const f64 *s = sums.data() + static_cast<size_t>(c) * dim_;
                for (u32 d = 0; d < dim_; ++d) {
                    dst[d] = static_cast<f32>(s[d] * inv);
                }
            }
            const f32 *old = out_centroids.data() + static_cast<size_t>(c) * dim_;
            for (u32 d = 0; d < dim_; ++d) {
                const f64 diff = static_cast<f64>(dst[d]) - old[d];
                shift += diff * diff;
            }
        }
        out_centroids.swap(new_centroids);
        if (shift < 1e-10) {
            break;
        }
    }

    // Drop empty clusters (compact the centroid list and remap labels).
    std::vector<u32> remap(k, 0);
    u32 alive = 0;
    for (u32 c = 0; c < k; ++c) {
        remap[c] = alive;
        if (counts[c] > 0) {
            if (alive != c) {
                std::memcpy(out_centroids.data() + static_cast<size_t>(alive) * dim_,
                            out_centroids.data() + static_cast<size_t>(c) * dim_, dim_ * sizeof(f32));
            }
            ++alive;
        }
    }
    if (alive == 0) {
        return 0;
    }
    out_centroids.resize(static_cast<size_t>(alive) * dim_);
    for (u32 i = 0; i < count; ++i) {
        out_labels[i] = remap[out_labels[i]];
    }
    return alive;
}

void SPFreshIndexInMem::BalancedKMeans2(const f32 *vectors, u32 count, std::vector<f32> &out_a, std::vector<f32> &out_b,
                                        std::vector<u8> &out_label) const {
    out_a.assign(dim_, 0.0f);
    out_b.assign(dim_, 0.0f);
    out_label.assign(count, 0);
    if (count == 0) {
        return;
    }
    if (count == 1) {
        std::memcpy(out_a.data(), vectors, dim_ * sizeof(f32));
        std::memcpy(out_b.data(), vectors, dim_ * sizeof(f32));
        return;
    }

    // k-means++ style seeding: first centroid random, second weighted by distance.
    std::mt19937 rng(0xBEEFu);
    std::uniform_int_distribution<u32> pick(0, count - 1);
    const u32 first = pick(rng);
    std::memcpy(out_a.data(), vectors + static_cast<size_t>(first) * dim_, dim_ * sizeof(f32));

    f64 total = 0;
    std::vector<f64> weights(count, 0.0);
    for (u32 i = 0; i < count; ++i) {
        const f32 *v = vectors + static_cast<size_t>(i) * dim_;
        f64 sq = 0;
        for (u32 d = 0; d < dim_; ++d) {
            const f64 diff = static_cast<f64>(v[d]) - out_a[d];
            sq += diff * diff;
        }
        weights[i] = sq;
        total += sq;
    }
    u32 second = pick(rng);
    if (total > 0) {
        std::uniform_real_distribution<f64> uniform(0.0, total);
        f64 target = uniform(rng);
        for (u32 i = 0; i < count; ++i) {
            target -= weights[i];
            if (target <= 0) {
                second = i;
                break;
            }
        }
    }
    std::memcpy(out_b.data(), vectors + static_cast<size_t>(second) * dim_, dim_ * sizeof(f32));

    const u32 min_count = (count * kDefaultAllowImbalancePct + 99) / 100;
    std::vector<std::pair<f64, u32>> order(count);

    std::vector<f64> sum_a(dim_, 0.0), sum_b(dim_, 0.0);
    for (u32 iter = 0; iter < kmeans_max_iter_; ++iter) {
        // Distance difference: < 0 prefers A, >= 0 prefers B.
        for (u32 i = 0; i < count; ++i) {
            const f32 *v = vectors + static_cast<size_t>(i) * dim_;
            f64 da = 0, db = 0;
            for (u32 d = 0; d < dim_; ++d) {
                const f64 diff_a = static_cast<f64>(v[d]) - out_a[d];
                const f64 diff_b = static_cast<f64>(v[d]) - out_b[d];
                da += diff_a * diff_a;
                db += diff_b * diff_b;
            }
            order[i] = {da - db, i};
        }

        // Start from the natural split, then enforce balance if needed.
        u32 count_a = 0;
        for (u32 i = 0; i < count; ++i) {
            if (order[i].first < 0) {
                ++count_a;
            }
        }
        if (count_a < min_count || (count - count_a) < min_count) {
            std::sort(order.begin(), order.end(), [](const auto &l, const auto &r) { return l.first < r.first; });
            count_a = std::clamp(count_a, min_count, count - min_count);
        } else {
            for (u32 i = 0; i < count; ++i) {
                out_label[order[i].second] = order[i].first < 0 ? 0 : 1;
            }
            // recompute sums with natural split
            std::fill(sum_a.begin(), sum_a.end(), 0.0);
            std::fill(sum_b.begin(), sum_b.end(), 0.0);
            u32 ca = 0, cb = 0;
            for (u32 i = 0; i < count; ++i) {
                const f32 *v = vectors + static_cast<size_t>(i) * dim_;
                f64 *s = out_label[i] == 0 ? sum_a.data() : sum_b.data();
                for (u32 d = 0; d < dim_; ++d) {
                    s[d] += v[d];
                }
                out_label[i] == 0 ? ++ca : ++cb;
            }
            if (ca == 0 || cb == 0) {
                continue;
            }
            for (u32 d = 0; d < dim_; ++d) {
                out_a[d] = static_cast<f32>(sum_a[d] / ca);
                out_b[d] = static_cast<f32>(sum_b[d] / cb);
            }
            continue;
        }

        // Forced balance: first `count_a` items in sorted order go to A.
        std::fill(sum_a.begin(), sum_a.end(), 0.0);
        std::fill(sum_b.begin(), sum_b.end(), 0.0);
        for (u32 r = 0; r < count; ++r) {
            const u32 i = order[r].second;
            const u32 label = r < count_a ? 0 : 1;
            out_label[i] = static_cast<u8>(label);
            const f32 *v = vectors + static_cast<size_t>(i) * dim_;
            f64 *s = label == 0 ? sum_a.data() : sum_b.data();
            for (u32 d = 0; d < dim_; ++d) {
                s[d] += v[d];
            }
        }
        const u32 cb = count - count_a;
        for (u32 d = 0; d < dim_; ++d) {
            out_a[d] = static_cast<f32>(sum_a[d] / count_a);
            out_b[d] = static_cast<f32>(sum_b[d] / cb);
        }
    }

    // Final assignment pass.
    for (u32 i = 0; i < count; ++i) {
        const f32 *v = vectors + static_cast<size_t>(i) * dim_;
        f64 da = 0, db = 0;
        for (u32 d = 0; d < dim_; ++d) {
            const f64 diff_a = static_cast<f64>(v[d]) - out_a[d];
            const f64 diff_b = static_cast<f64>(v[d]) - out_b[d];
            da += diff_a * diff_a;
            db += diff_b * diff_b;
        }
        out_label[i] = da <= db ? 0 : 1;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tree construction
// ═════════════════════════════════════════════════════════════════════════════

void SPFreshIndexInMem::EnsureCapacity(u32 required_partitions) {
    if (required_partitions <= partition_capacity_) {
        return;
    }
    u32 new_cap = std::max(required_partitions, std::max(partition_capacity_ * 2, 1024u));

    auto new_partitions = std::make_shared<std::vector<PartitionMeta>>(*partitions_);
    new_partitions->resize(new_cap);
    auto new_centroids = std::make_shared<std::vector<f32>>(*centroids_);
    new_centroids->resize(static_cast<size_t>(new_cap) * dim_, 0.0f);

    partitions_ = std::move(new_partitions);
    centroids_ = std::move(new_centroids);
    partition_capacity_ = new_cap;
    if (children_.size() < new_cap) {
        children_.resize(new_cap);
    }
    if (slot_table_->size() < new_cap) {
        slot_table_->resize(new_cap);
    }
}

PartitionKey SPFreshIndexInMem::AllocPartition(PartitionKey parent, u32 level, const f32 *centroid, u32 data_slot, u32 base_count) {
    EnsureCapacity(partition_count_ + 1);
    const PartitionKey key = next_partition_key_++;
    PartitionMeta meta;
    meta.parent_ = parent;
    meta.level_ = level;
    meta.state_ = PartitionState::kReady;
    meta.data_slot_ = data_slot;
    meta.base_count_ = base_count;
    meta.delta_count_ = 0;
    meta.version_ = 0;
    (*partitions_)[key] = meta;
    std::memcpy(centroids_->data() + static_cast<size_t>(key) * dim_, centroid, dim_ * sizeof(f32));
    if (children_.size() <= key) {
        children_.resize(key + 1);
    }
    children_[key].clear();
    partition_count_ = std::max(partition_count_, key + 1);
    return key;
}

void SPFreshIndexInMem::AppendChild(PartitionKey parent, PartitionKey child) {
    if (parent == kInvalidPartitionKey) {
        return;
    }
    if (children_.size() <= parent) {
        children_.resize(parent + 1);
    }
    children_[parent].push_back(child);
}

void SPFreshIndexInMem::ReplaceChild(PartitionKey parent, PartitionKey old_child, const std::vector<PartitionKey> &new_children) {
    if (parent == kInvalidPartitionKey || children_.size() <= parent) {
        return;
    }
    auto &list = children_[parent];
    auto it = std::find(list.begin(), list.end(), old_child);
    if (it == list.end()) {
        return;
    }
    it = list.erase(it);
    list.insert(it, new_children.begin(), new_children.end());
}

void SPFreshIndexInMem::RemoveChild(PartitionKey parent, PartitionKey child) {
    if (parent == kInvalidPartitionKey || children_.size() <= parent) {
        return;
    }
    auto &list = children_[parent];
    list.erase(std::remove(list.begin(), list.end(), child), list.end());
}

u32 SPFreshIndexInMem::AllocSlot(u32 count, u32 flags) {
    const u32 stride = 16u + code_width_ * 8u;
    const u64 bytes = static_cast<u64>(count) * stride;
    const u32 slot_id = next_slot_++;
    if (slot_table_->size() < next_slot_) {
        slot_table_->resize(std::max<size_t>(next_slot_, slot_table_->size() * 2 + 16));
    }
    (*slot_table_)[slot_id].offset_ = data_region_->size();
    (*slot_table_)[slot_id].count_ = count;
    (*slot_table_)[slot_id].flags_ = flags;
    data_region_->resize(data_region_->size() + bytes);
    return slot_id;
}

void SPFreshIndexInMem::WriteSlotVectors(u32 slot_id, const f32 *vectors, const std::vector<u32> &indices, const f32 *centroid) {
    const DataSlot &slot = (*slot_table_)[slot_id];
    const u32 count = slot.count_;
    char *base = data_region_->data() + slot.offset_;
    auto *row_ids = reinterpret_cast<u32 *>(base);
    auto *centroid_dists = reinterpret_cast<f32 *>(base + static_cast<size_t>(count) * 4);
    auto *qdots = reinterpret_cast<f32 *>(base + static_cast<size_t>(count) * 8);
    auto *code_counts = reinterpret_cast<u32 *>(base + static_cast<size_t>(count) * 12);
    u8 *codes = reinterpret_cast<u8 *>(base + static_cast<size_t>(count) * 16);

    const size_t code_bytes = static_cast<size_t>(code_width_) * 8;
    for (u32 i = 0; i < count; ++i) {
        const u32 src_index = indices[i];
        const f32 *v = vectors + static_cast<size_t>(src_index) * dim_;
        row_ids[i] = src_index;
        QuantizedVectorMeta meta;
        EncodeRelative(v, centroid, meta, codes + static_cast<size_t>(i) * code_bytes);
        centroid_dists[i] = meta.centroid_distance_;
        qdots[i] = meta.quantized_dot_product_;
        code_counts[i] = meta.code_count_;
    }
}

PartitionKey SPFreshIndexInMem::BuildSubtree(const f32 *vectors, const std::vector<u32> &indices, PartitionKey parent) {
    if (indices.empty()) {
        return kInvalidPartitionKey;
    }

    // Centroid == mean of all member vectors.
    std::vector<f32> centroid(dim_, 0.0f);
    for (u32 idx : indices) {
        const f32 *v = vectors + static_cast<size_t>(idx) * dim_;
        for (u32 d = 0; d < dim_; ++d) {
            centroid[d] += v[d];
        }
    }
    const f32 inv_n = 1.0f / static_cast<f32>(indices.size());
    for (u32 d = 0; d < dim_; ++d) {
        centroid[d] *= inv_n;
    }

    // ── Leaf ──
    if (indices.size() <= max_partition_size_) {
        const u32 slot = AllocSlot(static_cast<u32>(indices.size()), 0);
        WriteSlotVectors(slot, vectors, indices, centroid.data());
        return AllocPartition(parent, kLeafLevel, centroid.data(), slot, static_cast<u32>(indices.size()));
    }

    // ── Internal node ──
    // Aim for partitions of ~max_partition_size vectors.
    u32 k = static_cast<u32>((indices.size() + max_partition_size_ - 1) / max_partition_size_);
    k = std::clamp(k, 2u, max_partition_size_);

    std::vector<f32> sub_vectors(static_cast<size_t>(indices.size()) * dim_);
    for (size_t i = 0; i < indices.size(); ++i) {
        std::memcpy(sub_vectors.data() + i * dim_, vectors + static_cast<size_t>(indices[i]) * dim_, dim_ * sizeof(f32));
    }

    std::vector<f32> clusters;
    std::vector<u32> labels;
    const u32 actual_k = KMeans(sub_vectors.data(), static_cast<u32>(indices.size()), k, kmeans_max_iter_, clusters, labels);
    if (actual_k <= 1) {
        const u32 slot = AllocSlot(static_cast<u32>(indices.size()), 0);
        WriteSlotVectors(slot, vectors, indices, centroid.data());
        return AllocPartition(parent, kLeafLevel, centroid.data(), slot, static_cast<u32>(indices.size()));
    }

    const PartitionKey key = AllocPartition(parent, 0, centroid.data(), 0, 0);
    std::vector<std::vector<u32>> buckets(actual_k);
    for (size_t i = 0; i < indices.size(); ++i) {
        buckets[labels[i]].push_back(indices[i]);
    }
    u32 child_level = 0;
    for (u32 c = 0; c < actual_k; ++c) {
        if (buckets[c].empty()) {
            continue;
        }
        const PartitionKey child = BuildSubtree(vectors, buckets[c], key);
        if (child != kInvalidPartitionKey) {
            AppendChild(key, child);
            child_level = std::max(child_level, (*partitions_)[child].level_);
        }
    }
    (*partitions_)[key].level_ = child_level + 1;
    (*partitions_)[key].base_count_ = 0; // vectors only live in leaves
    height_ = std::max(height_, (*partitions_)[key].level_);
    return key;
}

void SPFreshIndexInMem::Build(const f32 *vectors, u32 count) {
    if (count == 0) {
        return;
    }
    StopFixupWorker();

    std::unique_lock tree_lock(tree_mtx_);
    std::lock_guard compact_lock(compact_mtx_);

    // Reset state.
    partition_count_ = 0;
    next_partition_key_ = kRootPartitionKey;
    next_slot_ = 0;
    num_vectors_ = 0;
    deleted_count_ = 0;
    height_ = kLeafLevel;

    partitions_ = std::make_shared<std::vector<PartitionMeta>>();
    centroids_ = std::make_shared<std::vector<f32>>();
    children_.clear();
    slot_table_ = std::make_shared<std::vector<DataSlot>>();
    data_region_ = std::make_shared<std::vector<char>>();
    partition_capacity_ = 0;

    {
        std::lock_guard deleted_lock(deleted_mtx_);
        deleted_bits_ = std::make_shared<std::vector<u64>>((count + 63u) / 64u + 1u, 0ull);
    }

    // Data region for 8M x 140B is ~1.1GB; reserve a bit to limit reallocation.
    data_region_->reserve(static_cast<size_t>(count) * (16u + code_width_ * 8u));

    const u32 est_capacity = static_cast<u32>(count / std::max(1u, max_partition_size_)) + 1024u;
    EnsureCapacity(est_capacity);

    std::vector<u32> indices(count);
    std::iota(indices.begin(), indices.end(), 0u);

    root_key_ = BuildSubtree(vectors, indices, kInvalidPartitionKey);
    num_vectors_ = count;

    std::lock_guard delta_lock(delta_mtx_);
    delta_a_->Clear();
    delta_b_->Clear();
    active_delta_idx_.store(0, std::memory_order_release);
    delta_bytes_.store(0, std::memory_order_release);

    tree_lock.unlock();
    PublishSnapshot();

    LOG_INFO(fmt::format("SPFresh Build: {} vectors, {} partitions, height {}, max_partition_size {}",
                         count, partition_count_, height_, max_partition_size_));
}

// ═════════════════════════════════════════════════════════════════════════════
// Snapshot
// ═════════════════════════════════════════════════════════════════════════════

bool SPFreshIndexInMem::TreeSnapshot::IsDeleted(u32 row_id) const {
    if (!deleted_bits_ || deleted_bits_->empty()) {
        return false;
    }
    const u32 word = row_id >> 6;
    if (word >= deleted_bits_->size()) {
        return false;
    }
    return ((*deleted_bits_)[word] >> (row_id & 63u)) & 1ull;
}

void SPFreshIndexInMem::PublishSnapshot() {
    auto snap = std::make_shared<TreeSnapshot>();
    snap->mode_ = storage_mode_;
    snap->partition_count_ = partition_count_;
    snap->root_key_ = root_key_;
    snap->height_ = height_;
    snap->num_vectors_ = num_vectors_;
    snap->dim_ = dim_;
    snap->code_width_ = code_width_;
    snap->partitions_ = partitions_;
    snap->centroids_ = centroids_;

    // Rebuild the children CSR (small, O(#partitions)).
    auto offsets = std::make_shared<std::vector<u32>>(partition_count_ + 1, 0);
    std::vector<u32> ids;
    for (u32 i = 0; i < partition_count_; ++i) {
        (*offsets)[i] = static_cast<u32>(ids.size());
        if (i < children_.size()) {
            for (PartitionKey child : children_[i]) {
                ids.push_back(child);
            }
        }
    }
    (*offsets)[partition_count_] = static_cast<u32>(ids.size());
    auto id_vec = std::make_shared<std::vector<u32>>(std::move(ids));

    snap->child_offsets_ = std::move(offsets);
    snap->child_ids_ = std::move(id_vec);
    snap->slots_ = slot_table_;
    snap->data_region_ = data_region_;
    snap->mmap_data_region_ = mmap_data_region_;
    snap->mmap_base_ = mmap_base_;
    snap->mmap_centroid_base_ = (storage_mode_ == StorageMode::kMmap) ? mmap_centroid_base_ : nullptr;
    snap->mmap_centroid_stride_ = mmap_centroid_stride_;
    snap->deleted_bits_ = deleted_bits_;

    {
        std::unique_lock lock(snapshot_mtx_);
        snapshot_ = std::move(snap);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Search
// ═════════════════════════════════════════════════════════════════════════════

void SPFreshIndexInMem::SearchTree(const TreeSnapshot &snap, const f32 *query, u32 beam_size, std::vector<PartitionKey> &out_leaves) const {
    out_leaves.clear();
    if (snap.partition_count_ == 0 || snap.root_key_ >= snap.partition_count_) {
        return;
    }

    std::vector<PartitionKey> frontier{snap.root_key_};
    std::vector<std::pair<f32, PartitionKey>> scored;
    std::vector<PartitionKey> pending_leaves;

    // The K-means tree is not perfectly balanced: partitions reached before the
    // maximum depth are already leaves. They must be carried over to the final
    // result instead of being dropped when their siblings expand.
    for (;;) {
        scored.clear();
        pending_leaves.clear();
        for (PartitionKey key : frontier) {
            const u32 begin = snap.ChildBegin(key);
            const u32 end = snap.ChildEnd(key);
            if (end <= begin) {
                pending_leaves.push_back(key);
                continue;
            }
            for (u32 i = begin; i < end; ++i) {
                const PartitionKey child = (*snap.child_ids_)[i];
                if (child >= snap.partition_count_) {
                    continue;
                }
                if (!(*snap.partitions_)[child].IsLive()) {
                    continue;
                }
                const f32 *centroid = snap.Centroid(child);
                f64 sq = 0;
                for (u32 d = 0; d < dim_; ++d) {
                    const f64 diff = static_cast<f64>(query[d]) - centroid[d];
                    sq += diff * diff;
                }
                scored.emplace_back(static_cast<f32>(sq), child);
            }
        }
        if (scored.empty()) {
            break;
        }
        const u32 take = std::min<u32>(beam_size == 0 ? base_beam_size_ : beam_size, static_cast<u32>(scored.size()));
        std::partial_sort(scored.begin(), scored.begin() + take, scored.end());
        frontier.clear();
        for (PartitionKey leaf : pending_leaves) {
            frontier.push_back(leaf);
        }
        for (u32 i = 0; i < take; ++i) {
            frontier.push_back(scored[i].second);
        }
    }
    out_leaves = std::move(frontier);
}

void SPFreshIndexInMem::ScanLeafPartition(const TreeSnapshot &snap, PartitionKey key, const QueryQuantized &qq,
                                          const SearchCallback &callback) const {
    if (!snap.partitions_ || key >= snap.partitions_->size()) {
        return;
    }
    const PartitionMeta &part_meta = (*snap.partitions_)[key];
    if (part_meta.base_count_ == 0) {
        return; // internal node or empty partition: no payload of its own
    }
    // `slot_table_` is indexed by slot id, not by partition key.
    const u32 slot_id = part_meta.data_slot_;
    if (!snap.slots_ || slot_id >= snap.slots_->size()) {
        return;
    }
    const DataSlot &slot = snap.Slot(slot_id);
    if (slot.count_ == 0 || (slot.flags_ & kSlotFlagTombstone)) {
        return;
    }
    const u32 count = slot.count_;
    const u8 *base = snap.RegionBase() + slot.offset_;
    const auto *row_ids = reinterpret_cast<const u32 *>(base);
    const auto *centroid_dists = reinterpret_cast<const f32 *>(base + static_cast<size_t>(count) * 4);
    const auto *qdots = reinterpret_cast<const f32 *>(base + static_cast<size_t>(count) * 8);
    const auto *code_counts = reinterpret_cast<const u32 *>(base + static_cast<size_t>(count) * 12);
    const u8 *codes = base + static_cast<size_t>(count) * 16;
    const size_t code_bytes = static_cast<size_t>(code_width_) * 8;

    for (u32 i = 0; i < count; ++i) {
        const u32 row_id = row_ids[i];
        if (snap.IsDeleted(row_id)) {
            continue;
        }
        QuantizedVectorMeta meta;
        meta.centroid_distance_ = centroid_dists[i];
        meta.quantized_dot_product_ = qdots[i];
        meta.code_count_ = code_counts[i];
        const f32 est = EstimateDistance(meta, codes + static_cast<size_t>(i) * code_bytes, qq);
        const f32 bound = ErrorBound(meta, qq);
        callback(static_cast<SegmentOffset>(row_id), est, bound);
    }
}

void SPFreshIndexInMem::Search(const f32 *query, u32 dim, const SearchCallback &callback, u32 max_candidates, u32 beam_size) const {
    if (dim != dim_ || callback == nullptr) {
        return;
    }
    std::shared_ptr<const TreeSnapshot> snap;
    {
        std::shared_lock lock(snapshot_mtx_);
        snap = snapshot_;
    }
    if (!snap || snap->partition_count_ == 0) {
        return;
    }

    std::vector<PartitionKey> leaves;
    SearchTree(*snap, query, beam_size, leaves);
    if (leaves.empty()) {
        return;
    }

    u32 emitted = 0;
    const bool limited = max_candidates > 0;

    for (PartitionKey leaf : leaves) {
        if (limited && emitted >= max_candidates) {
            break;
        }
        if (leaf >= snap->partition_count_) {
            continue;
        }
        // Query quantization is relative to the *partition* centroid, so it must
        // be recomputed for every candidate leaf.
        QueryQuantized qq;
        QuantizeQuery(qq, query, snap->Centroid(leaf));
        ScanLeafPartition(*snap, leaf, qq, [&](SegmentOffset offset, f32 est, f32 bound) {
            if (limited && emitted >= max_candidates) {
                return;
            }
            callback(offset, est, bound);
            ++emitted;
        });
    }

    // Delta entries are not associated with a single query quantization, so they
    // are quantized per owning partition on demand.
    {
        std::vector<PartitionKey> unique_leaves = leaves;
        std::sort(unique_leaves.begin(), unique_leaves.end());
        unique_leaves.erase(std::unique(unique_leaves.begin(), unique_leaves.end()), unique_leaves.end());

        auto scan_delta = [&](const SPFreshDeltaBuffer *delta) {
            if (delta == nullptr || delta->entry_count_ == 0) {
                return;
            }
            std::unordered_map<PartitionKey, QueryQuantized> qq_cache;
            for (u32 i = 0; i < delta->entry_count_; ++i) {
                if (limited && emitted >= max_candidates) {
                    return;
                }
                const PartitionKey key = delta->GetKey(i);
                if (!std::binary_search(unique_leaves.begin(), unique_leaves.end(), key)) {
                    continue;
                }
                const u32 row_id = delta->GetRowId(i);
                if (snap->IsDeleted(row_id)) {
                    continue;
                }
                auto it = qq_cache.find(key);
                if (it == qq_cache.end()) {
                    QueryQuantized qq;
                    QuantizeQuery(qq, query, snap->Centroid(key));
                    it = qq_cache.emplace(key, std::move(qq)).first;
                }
                const QuantizedVectorMeta meta = delta->GetMeta(i);
                const f32 est = EstimateDistance(meta, delta->GetCodes(i), it->second);
                const f32 bound = ErrorBound(meta, it->second);
                callback(static_cast<SegmentOffset>(row_id), est, bound);
                ++emitted;
            }
        };
        scan_delta(delta_a_);
        scan_delta(delta_b_);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Insert / delete
// ═════════════════════════════════════════════════════════════════════════════

PartitionKey SPFreshIndexInMem::RouteToLeaf(const f32 *vec) const {
    std::shared_ptr<const TreeSnapshot> snap;
    {
        std::shared_lock lock(snapshot_mtx_);
        snap = snapshot_;
    }
    if (!snap || snap->partition_count_ == 0 || snap->root_key_ >= snap->partition_count_) {
        return kInvalidPartitionKey;
    }
    PartitionKey current = snap->root_key_;
    for (;;) {
        const u32 begin = snap->ChildBegin(current);
        const u32 end = snap->ChildEnd(current);
        if (end <= begin) {
            return current;
        }
        PartitionKey best = kInvalidPartitionKey;
        f64 best_dist = std::numeric_limits<f64>::max();
        for (u32 i = begin; i < end; ++i) {
            const PartitionKey child = (*snap->child_ids_)[i];
            if (child >= snap->partition_count_ || !(*snap->partitions_)[child].IsLive()) {
                continue;
            }
            const f32 *centroid = snap->Centroid(child);
            f64 sq = 0;
            for (u32 d = 0; d < dim_; ++d) {
                const f64 diff = static_cast<f64>(vec[d]) - centroid[d];
                sq += diff * diff;
            }
            if (sq < best_dist) {
                best_dist = sq;
                best = child;
            }
        }
        if (best == kInvalidPartitionKey) {
            return current;
        }
        current = best;
    }
}

void SPFreshIndexInMem::MigrateVectorToPartition(const f32 *vec, u32 row_id, PartitionKey dst, SPFreshDeltaBuffer &delta) {
    std::vector<u8> codes(static_cast<size_t>(code_width_) * 8);
    QuantizedVectorMeta meta;
    const f32 *centroid = centroids_->data() + static_cast<size_t>(dst) * dim_;
    EncodeRelative(vec, centroid, meta, codes.data());
    delta.Append(dst, row_id, meta, codes.data());
}

void SPFreshIndexInMem::InsertDelta(const f32 *vec, u32 row_id) {
    if (dim_ == 0) {
        return;
    }
    if (storage_mode_ == StorageMode::kMmap) {
        TransitionToOwned();
    }
    if (delta_bytes_.load(std::memory_order_acquire) > max_delta_bytes_) {
        Compact(nullptr);
    }

    // Keep routing + encoding + appending atomic w.r.t. structural fixups so a
    // vector can never be appended to a partition that is concurrently retired.
    // tree_mtx_ (shared) -> delta_mtx_ matches the lock order used by Compact.
    bool need_split = false;
    {
        std::shared_lock tree_lock(tree_mtx_);

        const PartitionKey leaf = RouteToLeaf(vec);
        if (leaf == kInvalidPartitionKey || leaf >= partition_count_) {
            return;
        }
        if (!(*partitions_)[leaf].IsReady()) {
            return;
        }

        std::vector<u8> codes(static_cast<size_t>(code_width_) * 8);
        QuantizedVectorMeta meta;
        EncodeRelative(vec, centroids_->data() + static_cast<size_t>(leaf) * dim_, meta, codes.data());

        {
            std::lock_guard lock(delta_mtx_);
            const u32 idx = active_delta_idx_.load(std::memory_order_acquire);
            SPFreshDeltaBuffer *active = (idx == 0) ? delta_a_ : delta_b_;
            active->Append(leaf, row_id, meta, codes.data());
            delta_bytes_.fetch_add(active->Stride(), std::memory_order_release);
        }
        {
            std::lock_guard lock(counter_mtx_);
            ++(*partitions_)[leaf].delta_count_;
            ++num_vectors_;
        }

        // Ensure the deleted bitmap can address this row.
        {
            std::lock_guard lock(deleted_mtx_);
            const size_t needed = (static_cast<size_t>(row_id) >> 6) + 1;
            if (deleted_bits_->size() < needed) {
                deleted_bits_->resize(needed + 1, 0ull);
            }
        }

        need_split = (*partitions_)[leaf].TotalCount() > max_partition_size_;
        if (need_split) {
            const PartitionKey split_key = leaf;
            tree_lock.unlock();
            EnqueueFixup(FixupRequest::Type::kSplit, split_key);
            return;
        }
    }
}

void SPFreshIndexInMem::MarkDeleted(SegmentOffset offset) {
    std::lock_guard lock(deleted_mtx_);
    const size_t needed = (static_cast<size_t>(offset) >> 6) + 1;
    if (deleted_bits_->size() < needed) {
        deleted_bits_->resize(needed + 1, 0ull);
    }
    u64 &word = (*deleted_bits_)[static_cast<size_t>(offset) >> 6];
    const u64 mask = 1ull << (offset & 63u);
    if ((word & mask) == 0) {
        word |= mask;
        ++deleted_count_;
    }
}

u32 SPFreshIndexInMem::GetRowCount() const {
    std::lock_guard lock(counter_mtx_);
    const u64 deleted = deleted_count_;
    return deleted >= num_vectors_ ? 0 : static_cast<u32>(num_vectors_ - deleted);
}

u32 SPFreshIndexInMem::GetBaseRowCount() const {
    const u32 delta = GetDeltaCount();
    std::lock_guard lock(counter_mtx_);
    return num_vectors_ > delta ? num_vectors_ - delta : 0;
}

u32 SPFreshIndexInMem::GetDeltaCount() const {
    if (delta_a_ == nullptr || delta_b_ == nullptr) {
        return 0;
    }
    const u32 idx = active_delta_idx_.load(std::memory_order_acquire);
    return (idx == 0 ? delta_a_ : delta_b_)->entry_count_;
}

u32 SPFreshIndexInMem::GetDeletedCount() const {
    std::lock_guard lock(counter_mtx_);
    return static_cast<u32>(deleted_count_);
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixup queue
// ═════════════════════════════════════════════════════════════════════════════

void SPFreshIndexInMem::EnqueueFixup(FixupRequest::Type type, PartitionKey key) {
    {
        std::lock_guard lock(fixup_mtx_);
        for (const auto &req : fixup_queue_) {
            if (req.type_ == type && req.key_ == key) {
                return;
            }
        }
        if (fixup_queue_.size() >= kDefaultFixupQueueCapacity) {
            return; // backpressure: drop, will be retried on next insert
        }
        fixup_queue_.push_back(FixupRequest{type, key, (*partitions_)[key].version_});
    }
    EnsureFixupWorker();
    fixup_cv_.notify_one();
}

void SPFreshIndexInMem::EnsureFixupWorker() {
    bool expected = false;
    if (!fixup_running_.compare_exchange_strong(expected, true)) {
        return;
    }
    fixup_stop_.store(false, std::memory_order_release);
    fixup_thread_ = std::thread([this]() { FixupWorkerLoop(); });
}

void SPFreshIndexInMem::StopFixupWorker() {
    if (!fixup_running_.load(std::memory_order_acquire)) {
        return;
    }
    fixup_stop_.store(true, std::memory_order_release);
    fixup_cv_.notify_all();
    if (fixup_thread_.joinable()) {
        fixup_thread_.join();
    }
    fixup_running_.store(false, std::memory_order_release);
}

void SPFreshIndexInMem::FixupWorkerLoop() {
    size_t executed = 0;
    while (!fixup_stop_.load(std::memory_order_acquire)) {
        FixupRequest req;
        {
            std::unique_lock lock(fixup_mtx_);
            fixup_cv_.wait(lock, [this]() {
                return fixup_stop_.load(std::memory_order_acquire) || !fixup_queue_.empty();
            });
            if (fixup_stop_.load(std::memory_order_acquire)) {
                break;
            }
            req = fixup_queue_.front();
            fixup_queue_.pop_front();
        }

        bool changed = false;
        {
            std::unique_lock tree_lock(tree_mtx_);
            if (req.type_ == FixupRequest::Type::kSplit) {
                changed = ExecuteSplit(req.key_, nullptr);
            } else {
                changed = ExecuteMerge(req.key_, nullptr);
            }
        }
        if (changed) {
            PublishSnapshot();
            ++executed;
            if (executed % 64 == 0) {
                LOG_INFO(fmt::format("SPFresh fixup: {} executed, {} partitions", executed, partition_count_));
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Split / merge
// ═════════════════════════════════════════════════════════════════════════════

void SPFreshIndexInMem::CollectPartitionVectors(PartitionKey key, const SPFreshVectorFetcher *fetcher, std::vector<f32> &out_vecs,
                                                std::vector<u32> &out_row_ids) const {
    out_vecs.clear();
    out_row_ids.clear();

    std::shared_ptr<const TreeSnapshot> snap;
    {
        std::shared_lock lock(snapshot_mtx_);
        snap = snapshot_;
    }
    if (!snap || key >= snap->partition_count_) {
        return;
    }

    const size_t code_bytes = static_cast<size_t>(code_width_) * 8;
    const f32 *centroid = snap->Centroid(key);

    auto append_decoded = [&](u32 row_id, const QuantizedVectorMeta &meta, const u8 *codes) {
        const size_t pos = out_vecs.size();
        out_vecs.resize(pos + dim_);
        f32 *dst = out_vecs.data() + pos;
        if (fetcher != nullptr && fetcher->Fetch(row_id, dst)) {
            // full precision
        } else {
            DecodeRelative(meta, codes, centroid, dst);
        }
        out_row_ids.push_back(row_id);
    };

    // Persisted part (slot table is indexed by slot id, not by partition key).
    const u32 slot_id = (*snap->partitions_)[key].data_slot_;
    if (snap->slots_ && slot_id < snap->slots_->size()) {
        const DataSlot &slot = snap->Slot(slot_id);
        if (slot.count_ > 0 && (slot.flags_ & kSlotFlagTombstone) == 0) {
            const u32 count = slot.count_;
            const u8 *base = snap->RegionBase() + slot.offset_;
            const auto *row_ids = reinterpret_cast<const u32 *>(base);
            const auto *centroid_dists = reinterpret_cast<const f32 *>(base + static_cast<size_t>(count) * 4);
            const auto *qdots = reinterpret_cast<const f32 *>(base + static_cast<size_t>(count) * 8);
            const auto *code_counts = reinterpret_cast<const u32 *>(base + static_cast<size_t>(count) * 12);
            const u8 *codes = base + static_cast<size_t>(count) * 16;
            for (u32 i = 0; i < count; ++i) {
                if (snap->IsDeleted(row_ids[i])) {
                    continue;
                }
                QuantizedVectorMeta meta;
                meta.centroid_distance_ = centroid_dists[i];
                meta.quantized_dot_product_ = qdots[i];
                meta.code_count_ = code_counts[i];
                append_decoded(row_ids[i], meta, codes + static_cast<size_t>(i) * code_bytes);
            }
        }
    }

    // Delta part.
    auto append_delta = [&](const SPFreshDeltaBuffer *delta) {
        if (delta == nullptr) {
            return;
        }
        for (u32 i = 0; i < delta->entry_count_; ++i) {
            if (delta->GetKey(i) != key) {
                continue;
            }
            const u32 row_id = delta->GetRowId(i);
            if (snap->IsDeleted(row_id)) {
                continue;
            }
            append_decoded(row_id, delta->GetMeta(i), delta->GetCodes(i));
        }
    };
    append_delta(delta_a_);
    append_delta(delta_b_);
}


bool SPFreshIndexInMem::ExecuteSplit(PartitionKey key, const SPFreshVectorFetcher *fetcher) {
    // Caller holds tree_mtx_ exclusively.
    if (key >= partition_count_) {
        return false;
    }
    {
        const PartitionMeta &meta = (*partitions_)[key];
        if (!meta.IsReady()) {
            return false;
        }
        if (key < children_.size() && !children_[key].empty()) {
            return false; // internal nodes are rebuilt by OPTIMIZE
        }
        if (meta.TotalCount() <= max_partition_size_ * 3u / 4u) {
            return false;
        }
    }

    std::vector<f32> vecs;
    std::vector<u32> rows;
    CollectPartitionVectors(key, fetcher, vecs, rows);
    if (rows.size() < 4) {
        return false;
    }

    std::vector<f32> centroid_a, centroid_b;
    std::vector<u8> labels;
    BalancedKMeans2(vecs.data(), static_cast<u32>(rows.size()), centroid_a, centroid_b, labels);

    // Reserve capacity *before* taking references: AllocPartition may reallocate
    // partitions_/centroids_ and invalidate outstanding references.
    EnsureCapacity(partition_count_ + 2);

    (*partitions_)[key].state_ = PartitionState::kSplitting;

    const PartitionKey parent = (*partitions_)[key].parent_;
    const PartitionKey key_a = AllocPartition(parent, kLeafLevel, centroid_a.data(), 0, 0);
    const PartitionKey key_b = AllocPartition(parent, kLeafLevel, centroid_b.data(), 0, 0);
    auto &meta = (*partitions_)[key];

    // Candidate targets for reassignment: the two new partitions plus all siblings.
    std::vector<PartitionKey> targets{key_a, key_b};
    if (meta.parent_ != kInvalidPartitionKey && meta.parent_ < children_.size()) {
        for (PartitionKey sibling : children_[meta.parent_]) {
            if (sibling != key && sibling < partition_count_ && (*partitions_)[sibling].IsReady()) {
                targets.push_back(sibling);
            }
        }
    }

    {
        std::lock_guard lock(delta_mtx_);
        const u32 idx = active_delta_idx_.load(std::memory_order_acquire);
        SPFreshDeltaBuffer *active = (idx == 0) ? delta_a_ : delta_b_;
        std::vector<u8> codes(static_cast<size_t>(code_width_) * 8);

        for (size_t i = 0; i < rows.size(); ++i) {
            const f32 *v = vecs.data() + i * dim_;
            PartitionKey best = labels[i] == 0 ? key_a : key_b;
            f64 best_dist = std::numeric_limits<f64>::max();
            for (PartitionKey cand : targets) {
                const f32 *centroid = centroids_->data() + static_cast<size_t>(cand) * dim_;
                f64 sq = 0;
                for (u32 d = 0; d < dim_; ++d) {
                    const f64 diff = static_cast<f64>(v[d]) - centroid[d];
                    sq += diff * diff;
                }
                if (sq < best_dist) {
                    best_dist = sq;
                    best = cand;
                }
            }
            QuantizedVectorMeta qmeta;
            const f32 *centroid = centroids_->data() + static_cast<size_t>(best) * dim_;
            EncodeRelative(v, centroid, qmeta, codes.data());
            active->Append(best, rows[i], qmeta, codes.data());
            delta_bytes_.fetch_add(active->Stride(), std::memory_order_release);
            {
                std::lock_guard counter_lock(counter_mtx_);
                ++(*partitions_)[best].delta_count_;
            }
        }
    }

    if (meta.parent_ == kInvalidPartitionKey) {
        // Splitting the root: promote one level.
        const u32 new_level = height_ + 1;
        std::vector<f32> root_centroid(dim_, 0.0f);
        for (u32 d = 0; d < dim_; ++d) {
            root_centroid[d] = 0.5f * (centroid_a[d] + centroid_b[d]);
        }
        const PartitionKey new_root = AllocPartition(kInvalidPartitionKey, new_level, root_centroid.data(), 0, 0);
        AppendChild(new_root, key_a);
        AppendChild(new_root, key_b);
        root_key_ = new_root;
        height_ = new_level;
    } else {
        ReplaceChild(meta.parent_, key, {key_a, key_b});
    }

    meta.state_ = PartitionState::kDeleting;
    meta.base_count_ = 0;
    meta.delta_count_ = 0;
    ++meta.version_;
    if (meta.data_slot_ < slot_table_->size()) {
        (*slot_table_)[meta.data_slot_].flags_ |= kSlotFlagTombstone;
    }
    return true;
}

std::vector<PartitionKey> SPFreshIndexInMem::SiblingLeaves(PartitionKey key) const {
    std::vector<PartitionKey> result;
    if (key >= partition_count_) {
        return result;
    }
    const PartitionKey parent = (*partitions_)[key].parent_;
    if (parent == kInvalidPartitionKey || parent >= children_.size()) {
        return result;
    }
    for (PartitionKey sibling : children_[parent]) {
        if (sibling != key && sibling < partition_count_ && (*partitions_)[sibling].IsReady()) {
            result.push_back(sibling);
        }
    }
    return result;
}

bool SPFreshIndexInMem::ExecuteMerge(PartitionKey key, const SPFreshVectorFetcher *fetcher) {
    // Caller holds tree_mtx_ exclusively.
    if (key >= partition_count_) {
        return false;
    }
    auto &meta = (*partitions_)[key];
    if (!meta.IsReady()) {
        return false;
    }
    if (!children_.empty() && key < children_.size() && !children_[key].empty()) {
        return false;
    }
    if (meta.TotalCount() >= min_partition_size_ * 3u / 2u) {
        return false;
    }

    std::vector<PartitionKey> siblings = SiblingLeaves(key);
    if (siblings.empty()) {
        return false;
    }
    // Nearest sibling with enough room.
    std::sort(siblings.begin(), siblings.end(), [&](PartitionKey l, PartitionKey r) {
        const f32 *cl = centroids_->data() + static_cast<size_t>(l) * dim_;
        const f32 *cr = centroids_->data() + static_cast<size_t>(r) * dim_;
        const f32 *self = centroids_->data() + static_cast<size_t>(key) * dim_;
        f64 dl = 0, dr = 0;
        for (u32 d = 0; d < dim_; ++d) {
            const f64 a = static_cast<f64>(self[d]) - cl[d];
            const f64 b = static_cast<f64>(self[d]) - cr[d];
            dl += a * a;
            dr += b * b;
        }
        return dl < dr;
    });

    PartitionKey target = kInvalidPartitionKey;
    for (PartitionKey sibling : siblings) {
        if ((*partitions_)[sibling].TotalCount() + meta.TotalCount() <= max_partition_size_) {
            target = sibling;
            break;
        }
    }
    if (target == kInvalidPartitionKey) {
        return false;
    }

    std::vector<f32> vecs;
    std::vector<u32> rows;
    CollectPartitionVectors(key, fetcher, vecs, rows);

    {
        std::lock_guard lock(delta_mtx_);
        const u32 idx = active_delta_idx_.load(std::memory_order_acquire);
        SPFreshDeltaBuffer *active = (idx == 0) ? delta_a_ : delta_b_;
        for (size_t i = 0; i < rows.size(); ++i) {
            MigrateVectorToPartition(vecs.data() + i * dim_, rows[i], target, *active);
            delta_bytes_.fetch_add(active->Stride(), std::memory_order_release);
        }
    }
    {
        std::lock_guard counter_lock(counter_mtx_);
        (*partitions_)[target].delta_count_ += static_cast<u32>(rows.size());
    }

    RemoveChild(meta.parent_, key);
    meta.state_ = PartitionState::kDeleting;
    meta.base_count_ = 0;
    meta.delta_count_ = 0;
    ++meta.version_;
    if (meta.data_slot_ < slot_table_->size()) {
        (*slot_table_)[meta.data_slot_].flags_ |= kSlotFlagTombstone;
    }
    return true;
}

void SPFreshIndexInMem::RunPendingFixupsLocked(const SPFreshVectorFetcher *fetcher) {
    std::deque<FixupRequest> pending;
    {
        std::lock_guard lock(fixup_mtx_);
        pending.swap(fixup_queue_);
    }
    for (const auto &req : pending) {
        if (req.key_ >= partition_count_) {
            continue;
        }
        if (req.type_ == FixupRequest::Type::kSplit) {
            ExecuteSplit(req.key_, fetcher);
        } else {
            ExecuteMerge(req.key_, fetcher);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Compact
// ═════════════════════════════════════════════════════════════════════════════

void SPFreshIndexInMem::TransitionToOwned() {
    if (storage_mode_ != StorageMode::kMmap || mmap_base_ == nullptr) {
        return;
    }
    std::unique_lock tree_lock(tree_mtx_);

    // Centroids currently live in the mapping: copy them into DRAM before the
    // index becomes mutable again.
    if (mmap_centroid_base_ != nullptr) {
        auto owned_centroids = std::make_shared<std::vector<f32>>();
        owned_centroids->resize(static_cast<size_t>(partition_capacity_) * dim_, 0.0f);
        for (u32 key = 0; key < partition_count_; ++key) {
            const auto *src = reinterpret_cast<const f32 *>(mmap_centroid_base_ + static_cast<size_t>(key) * mmap_centroid_stride_);
            std::memcpy(owned_centroids->data() + static_cast<size_t>(key) * dim_, src, static_cast<size_t>(dim_) * sizeof(f32));
        }
        centroids_ = std::move(owned_centroids);
        mmap_centroid_base_ = nullptr;
        mmap_centroid_stride_ = 0;
    }

    const u32 stride = 16u + code_width_ * 8u;
    auto new_slots = std::make_shared<std::vector<DataSlot>>(*slot_table_);

    size_t total_bytes = 0;
    for (const DataSlot &slot : *new_slots) {
        if (slot.count_ > 0 && (slot.flags_ & kSlotFlagTombstone) == 0) {
            total_bytes += static_cast<size_t>(slot.count_) * stride;
        }
    }
    auto new_region = std::make_shared<std::vector<char>>();
    new_region->reserve(total_bytes);
    for (DataSlot &slot : *new_slots) {
        const u64 old_offset = slot.offset_;
        if (slot.count_ == 0 || (slot.flags_ & kSlotFlagTombstone) != 0) {
            slot.offset_ = new_region->size();
            continue;
        }
        const size_t bytes = static_cast<size_t>(slot.count_) * stride;
        slot.offset_ = new_region->size();
        new_region->insert(new_region->end(), mmap_data_region_ + old_offset, mmap_data_region_ + old_offset + bytes);
    }

    slot_table_ = std::move(new_slots);
    data_region_ = std::move(new_region);
    storage_mode_ = StorageMode::kOwned;
    mmap_base_ = nullptr;
    mmap_data_region_ = nullptr;

    tree_lock.unlock();
    PublishSnapshot();
}

void SPFreshIndexInMem::Compact(const SPFreshVectorFetcher *fetcher) {
    std::unique_lock compact_lock(compact_mtx_, std::defer_lock);
    if (!compact_lock.try_lock()) {
        return; // another compact is running
    }
    if (storage_mode_ == StorageMode::kMmap) {
        compact_lock.unlock();
        TransitionToOwned();
        compact_lock.lock();
    }

    {
        std::unique_lock tree_lock(tree_mtx_);
        RunPendingFixupsLocked(fetcher);
    }

    // Freeze the active delta and switch writers to the other buffer.
    SPFreshDeltaBuffer *frozen = nullptr;
    {
        std::lock_guard lock(delta_mtx_);
        const u32 idx = active_delta_idx_.load(std::memory_order_acquire);
        frozen = (idx == 0) ? delta_a_ : delta_b_;
        active_delta_idx_.store(1 - idx, std::memory_order_release);
        delta_bytes_.store(0, std::memory_order_release);
    }
    u64 deleted_snapshot = 0;
    {
        std::lock_guard lock(deleted_mtx_);
        deleted_snapshot = deleted_count_;
    }
    if ((frozen == nullptr || frozen->entry_count_ == 0) && deleted_snapshot == 0) {
        std::lock_guard lock(delta_mtx_);
        if (frozen != nullptr) {
            frozen->Clear();
        }
        return;
    }

    std::unique_lock tree_lock(tree_mtx_);

    // Group frozen delta by partition.
    std::unordered_map<PartitionKey, std::vector<u32>> delta_by_partition;
    for (u32 i = 0; i < frozen->entry_count_; ++i) {
        delta_by_partition[frozen->GetKey(i)].push_back(i);
    }

    const u32 stride = 16u + code_width_ * 8u;
    const u32 slot_capacity = std::max(next_slot_, partition_capacity_);
    auto new_slots = std::make_shared<std::vector<DataSlot>>(slot_capacity);
    auto new_region = std::make_shared<std::vector<char>>();

    size_t est_bytes = 0;
    for (u32 key = 0; key < partition_count_; ++key) {
        est_bytes += static_cast<size_t>((*partitions_)[key].base_count_ + (*partitions_)[key].delta_count_) * stride;
    }
    new_region->reserve(est_bytes);

    u32 new_slot_count = 0;
    u64 live_vectors = 0;

    for (u32 key = 0; key < partition_count_; ++key) {
        auto &meta = (*partitions_)[key];
        const bool is_internal = key < children_.size() && !children_[key].empty();
        if (!meta.IsLive() || is_internal) {
            continue; // internal nodes and tombstones carry no payload
        }

        // Gather rows: persisted (minus deleted) + delta entries.
        std::vector<u32> rows;
        std::vector<QuantizedVectorMeta> metas;
        std::vector<const u8 *> code_ptrs;

        const size_t code_bytes = static_cast<size_t>(code_width_) * 8;
        if (meta.data_slot_ < slot_table_->size()) {
            const DataSlot &old_slot = (*slot_table_)[meta.data_slot_];
            if (old_slot.count_ > 0 && (old_slot.flags_ & kSlotFlagTombstone) == 0) {
                const u32 count = old_slot.count_;
                const u8 *base = reinterpret_cast<const u8 *>(data_region_->data()) + old_slot.offset_;
                const auto *row_ids = reinterpret_cast<const u32 *>(base);
                const auto *centroid_dists = reinterpret_cast<const f32 *>(base + static_cast<size_t>(count) * 4);
                const auto *qdots = reinterpret_cast<const f32 *>(base + static_cast<size_t>(count) * 8);
                const auto *code_counts = reinterpret_cast<const u32 *>(base + static_cast<size_t>(count) * 12);
                const u8 *codes = base + static_cast<size_t>(count) * 16;
                for (u32 i = 0; i < count; ++i) {
                    const u32 row_id = row_ids[i];
                    bool deleted = false;
                    {
                        std::lock_guard lock(deleted_mtx_);
                        const u32 word = row_id >> 6;
                        if (word < deleted_bits_->size()) {
                            deleted = ((*deleted_bits_)[word] >> (row_id & 63u)) & 1ull;
                        }
                    }
                    if (fetcher != nullptr && fetcher->IsDeleted(row_id)) {
                        deleted = true;
                    }
                    if (deleted) {
                        continue;
                    }
                    QuantizedVectorMeta meta_q;
                    meta_q.centroid_distance_ = centroid_dists[i];
                    meta_q.quantized_dot_product_ = qdots[i];
                    meta_q.code_count_ = code_counts[i];
                    rows.push_back(row_id);
                    metas.push_back(meta_q);
                    code_ptrs.push_back(codes + static_cast<size_t>(i) * code_bytes);
                }
            }
        }

        auto it = delta_by_partition.find(key);
        if (it != delta_by_partition.end()) {
            for (u32 di : it->second) {
                const u32 row_id = frozen->GetRowId(di);
                bool deleted = false;
                {
                    std::lock_guard lock(deleted_mtx_);
                    const u32 word = row_id >> 6;
                    if (word < deleted_bits_->size()) {
                        deleted = ((*deleted_bits_)[word] >> (row_id & 63u)) & 1ull;
                    }
                }
                if (fetcher != nullptr && fetcher->IsDeleted(row_id)) {
                    deleted = true;
                }
                if (deleted) {
                    continue;
                }
                rows.push_back(row_id);
                metas.push_back(frozen->GetMeta(di));
                code_ptrs.push_back(frozen->GetCodes(di));
            }
        }

        if (rows.empty()) {
            meta.base_count_ = 0;
            meta.delta_count_ = 0;
            continue;
        }

        const u32 count = static_cast<u32>(rows.size());
        const u32 slot_id = new_slot_count++;
        new_slots->resize(std::max<size_t>(new_slots->size(), slot_id + 1));
        (*new_slots)[slot_id].offset_ = new_region->size();
        (*new_slots)[slot_id].count_ = count;
        (*new_slots)[slot_id].flags_ = 0;
        new_region->resize(new_region->size() + static_cast<size_t>(count) * stride);

        char *base = new_region->data() + (*new_slots)[slot_id].offset_;
        auto *out_rows = reinterpret_cast<u32 *>(base);
        auto *out_cd = reinterpret_cast<f32 *>(base + static_cast<size_t>(count) * 4);
        auto *out_qd = reinterpret_cast<f32 *>(base + static_cast<size_t>(count) * 8);
        auto *out_cc = reinterpret_cast<u32 *>(base + static_cast<size_t>(count) * 12);
        u8 *out_codes = reinterpret_cast<u8 *>(base + static_cast<size_t>(count) * 16);

        for (u32 i = 0; i < count; ++i) {
            out_rows[i] = rows[i];
            out_cd[i] = metas[i].centroid_distance_;
            out_qd[i] = metas[i].quantized_dot_product_;
            out_cc[i] = metas[i].code_count_;
            std::memcpy(out_codes + static_cast<size_t>(i) * code_bytes, code_ptrs[i], code_bytes);
        }

        meta.data_slot_ = slot_id;
        meta.base_count_ = count;
        meta.delta_count_ = 0;
        ++meta.version_;
        live_vectors += count;
    }

    new_slots->resize(new_slot_count);
    slot_table_ = std::move(new_slots);
    data_region_ = std::move(new_region);
    next_slot_ = new_slot_count;
    num_vectors_ = static_cast<u32>(live_vectors);

    {
        std::lock_guard lock(deleted_mtx_);
        deleted_count_ = 0;
        deleted_bits_ = std::make_shared<std::vector<u64>>((live_vectors + 63) / 64 + 1, 0ull);
    }

    frozen->Clear();
    tree_lock.unlock();
    PublishSnapshot();

    LOG_INFO(fmt::format("SPFresh Compact: {} vectors, {} partitions, {} slots", num_vectors_, partition_count_, next_slot_));
}

// ═════════════════════════════════════════════════════════════════════════════
// Persistence
// ═════════════════════════════════════════════════════════════════════════════

size_t SPFreshIndexInMem::Section1Size() const {
    size_t size = sizeof(u32); // partition_count
    size += static_cast<size_t>(partition_count_) * (sizeof(PartitionMetaDisk) + static_cast<size_t>(dim_) * sizeof(f32));
    size += static_cast<size_t>(partition_count_ + 1) * sizeof(u32);
    size_t child_count = 0;
    for (u32 i = 0; i < partition_count_ && i < children_.size(); ++i) {
        child_count += children_[i].size();
    }
    size += child_count * sizeof(u32);
    return size;
}

void SPFreshIndexInMem::SerializeSection1(std::vector<char> &out) const {
    out.clear();
    out.reserve(Section1Size());
    auto append = [&out](const void *data, size_t size) {
        const auto *p = static_cast<const char *>(data);
        out.insert(out.end(), p, p + size);
    };

    append(&partition_count_, sizeof(u32));

    for (u32 key = 0; key < partition_count_; ++key) {
        const PartitionMeta &meta = (*partitions_)[key];
        PartitionMetaDisk disk;
        disk.parent_ = meta.parent_;
        disk.level_ = meta.level_;
        disk.state_ = static_cast<u32>(meta.state_);
        disk.data_slot_ = meta.data_slot_;
        disk.base_count_ = meta.base_count_;
        disk.pad_ = 0;
        disk.version_ = meta.version_;
        append(&disk, sizeof(disk));
        append(centroids_->data() + static_cast<size_t>(key) * dim_, static_cast<size_t>(dim_) * sizeof(f32));
    }

    // Children CSR.
    u32 offset = 0;
    for (u32 key = 0; key < partition_count_; ++key) {
        append(&offset, sizeof(u32));
        if (key < children_.size()) {
            offset += static_cast<u32>(children_[key].size());
        }
    }
    append(&offset, sizeof(u32));
    for (u32 key = 0; key < partition_count_ && key < children_.size(); ++key) {
        if (!children_[key].empty()) {
            append(children_[key].data(), children_[key].size() * sizeof(u32));
        }
    }
}

void SPFreshIndexInMem::DeserializeSection1(const u8 *ptr, const SPFreshFileHeader &header, std::vector<PartitionMeta> &out_partitions,
                                            std::vector<f32> &out_centroids, std::vector<u32> &out_child_offsets,
                                            std::vector<u32> &out_child_ids) {
    auto read = [&ptr](void *dst, size_t size) {
        std::memcpy(dst, ptr, size);
        ptr += size;
    };

    u32 count = 0;
    read(&count, sizeof(u32));
    out_partitions.assign(count, PartitionMeta{});
    out_centroids.assign(static_cast<size_t>(count) * dim_, 0.0f);

    for (u32 key = 0; key < count; ++key) {
        PartitionMetaDisk disk;
        read(&disk, sizeof(disk));
        PartitionMeta meta;
        meta.parent_ = disk.parent_;
        meta.level_ = disk.level_;
        meta.state_ = static_cast<PartitionState>(disk.state_);
        meta.data_slot_ = disk.data_slot_;
        meta.base_count_ = disk.base_count_;
        meta.delta_count_ = 0;
        meta.version_ = disk.version_;
        out_partitions[key] = meta;
        read(out_centroids.data() + static_cast<size_t>(key) * dim_, static_cast<size_t>(dim_) * sizeof(f32));
    }

    out_child_offsets.assign(count + 1, 0);
    for (u32 i = 0; i <= count; ++i) {
        read(&out_child_offsets[i], sizeof(u32));
    }
    const u32 child_count = out_child_offsets.empty() ? 0 : out_child_offsets.back();
    out_child_ids.assign(child_count, 0);
    if (child_count > 0) {
        read(out_child_ids.data(), child_count * sizeof(u32));
    }
    (void)header;
}

void SPFreshIndexInMem::Save(LocalFileHandle &fh) const {
    std::shared_lock tree_lock(tree_mtx_);

    SPFreshFileHeader header;
    header.dim_ = dim_;
    header.pad_dim_ = pad_dim_;
    header.partition_count_ = partition_count_;
    header.partition_capacity_ = partition_capacity_;
    header.num_vectors_ = num_vectors_;
    header.code_width_ = code_width_;
    header.root_key_ = root_key_;
    header.height_ = height_;
    header.rot_seed_ = rot_seed_;
    header.deleted_count_ = deleted_count_;

    std::vector<char> section1;
    SerializeSection1(section1);

    const u64 section1_offset = sizeof(SPFreshFileHeader);
    const u64 slot_table_offset = section1_offset + section1.size();
    const u64 slot_count = next_slot_;
    const u64 data_region_offset = slot_table_offset + slot_count * sizeof(DataSlot);

    header.section1_offset_ = section1_offset;
    header.section1_size_ = section1.size();
    header.slot_table_offset_ = slot_table_offset;
    header.slot_table_capacity_ = slot_count;
    header.data_region_offset_ = data_region_offset;
    header.data_region_size_ = data_region_->size();

    fh.Append(&header, sizeof(header));
    fh.Append(section1.data(), section1.size());
    if (slot_count > 0) {
        fh.Append(slot_table_->data(), slot_count * sizeof(DataSlot));
    }
    if (!data_region_->empty()) {
        fh.Append(data_region_->data(), data_region_->size());
    }
}

void SPFreshIndexInMem::Load(LocalFileHandle &fh, size_t /*file_size*/) {
    SPFreshFileHeader header;
    auto [read_bytes, status] = fh.Read(&header, sizeof(header));
    if (!status.ok() || read_bytes != sizeof(header) || !header.Validate()) {
        UnrecoverableError("SPFresh: invalid index file header");
        return;
    }

    dim_ = header.dim_;
    pad_dim_ = header.pad_dim_;
    code_width_ = header.code_width_ == 0 ? QuantCodeWidth(dim_) : header.code_width_;
    rot_seed_ = header.rot_seed_;
    height_ = header.height_;
    root_key_ = header.root_key_;
    deleted_count_ = header.deleted_count_;
    num_vectors_ = header.num_vectors_;
    GenerateRotationParams(rot_seed_);

    std::vector<char> section1(header.section1_size_);
    fh.Seek(header.section1_offset_);
    if (header.section1_size_ > 0) {
        fh.Read(section1.data(), header.section1_size_);
    }

    std::vector<PartitionMeta> partitions;
    std::vector<f32> centroids;
    std::vector<u32> child_offsets;
    std::vector<u32> child_ids;
    DeserializeSection1(reinterpret_cast<const u8 *>(section1.data()), header, partitions, centroids, child_offsets, child_ids);

    partition_count_ = static_cast<u32>(partitions.size());
    partition_capacity_ = std::max(header.partition_capacity_, partition_count_ + 64);
    partitions_ = std::make_shared<std::vector<PartitionMeta>>(std::move(partitions));
    partitions_->resize(partition_capacity_);
    centroids_ = std::make_shared<std::vector<f32>>(std::move(centroids));
    centroids_->resize(static_cast<size_t>(partition_capacity_) * dim_, 0.0f);

    children_.assign(partition_capacity_, {});
    for (u32 key = 0; key < partition_count_; ++key) {
        for (u32 i = child_offsets[key]; i < child_offsets[key + 1]; ++i) {
            children_[key].push_back(child_ids[i]);
        }
    }
    next_partition_key_ = partition_count_;

    // Slot table + data region (owned mode).
    std::vector<DataSlot> slots(header.slot_table_capacity_);
    fh.Seek(header.slot_table_offset_);
    if (header.slot_table_capacity_ > 0) {
        fh.Read(slots.data(), header.slot_table_capacity_ * sizeof(DataSlot));
    }
    slot_table_ = std::make_shared<std::vector<DataSlot>>(std::move(slots));
    next_slot_ = static_cast<u32>(header.slot_table_capacity_);

    auto region = std::make_shared<std::vector<char>>(header.data_region_size_);
    fh.Seek(header.data_region_offset_);
    if (header.data_region_size_ > 0) {
        fh.Read(region->data(), header.data_region_size_);
    }
    data_region_ = std::move(region);

    storage_mode_ = StorageMode::kOwned;
    mmap_base_ = nullptr;
    mmap_data_region_ = nullptr;

    {
        std::lock_guard lock(deleted_mtx_);
        deleted_bits_ = std::make_shared<std::vector<u64>>((static_cast<size_t>(num_vectors_) + 63) / 64 + 1, 0ull);
    }

    delta_a_->SetEntrySize(QuantVecStride(dim_));
    delta_b_->SetEntrySize(QuantVecStride(dim_));

    PublishSnapshot();
    LOG_INFO(fmt::format("SPFresh Load: {} vectors, {} partitions, height {}", num_vectors_, partition_count_, height_));
}

void SPFreshIndexInMem::LoadFromMmap(const u8 *base, size_t size) {
    if (base == nullptr || size < sizeof(SPFreshFileHeader)) {
        UnrecoverableError("SPFresh: mmap region too small");
        return;
    }
    const auto &header = *reinterpret_cast<const SPFreshFileHeader *>(base);
    if (!header.Validate()) {
        UnrecoverableError("SPFresh: invalid mmap header");
        return;
    }

    dim_ = header.dim_;
    pad_dim_ = header.pad_dim_;
    code_width_ = header.code_width_ == 0 ? QuantCodeWidth(dim_) : header.code_width_;
    rot_seed_ = header.rot_seed_;
    height_ = header.height_;
    root_key_ = header.root_key_;
    num_vectors_ = header.num_vectors_;
    deleted_count_ = header.deleted_count_;
    GenerateRotationParams(rot_seed_);

    // Parse the partition table without materialising centroids: they stay in
    // the mapping and are read on demand (OS page cache), which is what keeps
    // DRAM usage proportional to the number of partitions, not vectors.
    std::vector<PartitionMeta> partitions;
    std::vector<u32> child_offsets;
    {
        const u8 *p = base + header.section1_offset_;
        u32 count = 0;
        std::memcpy(&count, p, sizeof(u32));
        p += sizeof(u32);
        partitions.resize(count);
        const size_t centroid_bytes = static_cast<size_t>(dim_) * sizeof(f32);
        for (u32 i = 0; i < count; ++i) {
            PartitionMetaDisk disk;
            std::memcpy(&disk, p, sizeof(disk));
            p += sizeof(disk) + centroid_bytes;
            PartitionMeta meta;
            meta.parent_ = disk.parent_;
            meta.level_ = disk.level_;
            meta.state_ = static_cast<PartitionState>(disk.state_);
            meta.data_slot_ = disk.data_slot_;
            meta.base_count_ = disk.base_count_;
            meta.delta_count_ = 0;
            meta.version_ = disk.version_;
            partitions[i] = meta;
        }
        child_offsets.resize(count + 1);
        std::memcpy(child_offsets.data(), p, static_cast<size_t>(count + 1) * sizeof(u32));
        p += static_cast<size_t>(count + 1) * sizeof(u32);
        const u32 child_count = child_offsets.empty() ? 0 : child_offsets.back();
        std::vector<u32> child_ids(child_count);
        if (child_count > 0) {
            std::memcpy(child_ids.data(), p, static_cast<size_t>(child_count) * sizeof(u32));
        }
        children_.assign(std::max<size_t>(count, 1), {});
        for (u32 key = 0; key < count; ++key) {
            for (u32 i = child_offsets[key]; i < child_offsets[key + 1]; ++i) {
                children_[key].push_back(child_ids[i]);
            }
        }
    }

    partition_count_ = static_cast<u32>(partitions.size());
    partition_capacity_ = std::max(header.partition_capacity_, partition_count_ + 64);
    partitions_ = std::make_shared<std::vector<PartitionMeta>>(std::move(partitions));
    partitions_->resize(partition_capacity_);
    children_.resize(partition_capacity_);
    centroids_ = std::make_shared<std::vector<f32>>();
    mmap_centroid_base_ = base + header.section1_offset_ + sizeof(u32) + sizeof(PartitionMetaDisk);
    mmap_centroid_stride_ = sizeof(PartitionMetaDisk) + static_cast<u64>(dim_) * sizeof(f32);
    next_partition_key_ = partition_count_;

    storage_mode_ = StorageMode::kMmap;
    mmap_base_ = base;
    mmap_header_ = header;
    const auto *mapped_slots = reinterpret_cast<const DataSlot *>(base + header.slot_table_offset_);
    mmap_slot_capacity_ = static_cast<u32>(header.slot_table_capacity_);
    mmap_data_region_ = base + header.data_region_offset_;
    next_slot_ = mmap_slot_capacity_;

    // Slot descriptors are small (~16B each) and must stay mutable (tombstones),
    // so copy them out of the read-only mapping. Vector payloads stay mapped.
    if (mmap_slot_capacity_ > 0) {
        slot_table_ = std::make_shared<std::vector<DataSlot>>(mapped_slots, mapped_slots + mmap_slot_capacity_);
    } else {
        slot_table_ = std::make_shared<std::vector<DataSlot>>();
    }
    data_region_ = std::make_shared<std::vector<char>>();

    {
        std::lock_guard lock(deleted_mtx_);
        deleted_bits_ = std::make_shared<std::vector<u64>>((static_cast<size_t>(num_vectors_) + 63) / 64 + 1, 0ull);
    }

    delta_a_->SetEntrySize(QuantVecStride(dim_));
    delta_b_->SetEntrySize(QuantVecStride(dim_));

    PublishSnapshot();
    LOG_INFO(fmt::format("SPFresh LoadFromMmap: {} vectors, {} partitions, {} KB", num_vectors_, partition_count_, size / 1024));
}

// ═════════════════════════════════════════════════════════════════════════════
// Dump
// ═════════════════════════════════════════════════════════════════════════════

void SPFreshIndexInMem::TransferTo(SPFreshIndexInMem *target) {
    if (target == nullptr) {
        return;
    }
    StopFixupWorker();
    std::unique_lock tree_lock(tree_mtx_);

    target->begin_row_id_ = begin_row_id_;
    target->dim_ = dim_;
    target->pad_dim_ = pad_dim_;
    target->code_width_ = code_width_;
    target->rot_seed_ = rot_seed_;
    target->rotation_flip_ = rotation_flip_;
    target->max_partition_size_ = max_partition_size_;
    target->min_partition_size_ = min_partition_size_;
    target->base_beam_size_ = base_beam_size_;
    target->kmeans_max_iter_ = kmeans_max_iter_;
    target->max_delta_bytes_ = max_delta_bytes_;
    target->rerank_multiplier_ = rerank_multiplier_;

    target->partitions_ = std::make_shared<std::vector<PartitionMeta>>(*partitions_);
    if (mmap_centroid_base_ != nullptr) {
        auto owned_centroids = std::make_shared<std::vector<f32>>();
        owned_centroids->resize(static_cast<size_t>(partition_capacity_) * dim_, 0.0f);
        for (u32 key = 0; key < partition_count_; ++key) {
            const auto *src = reinterpret_cast<const f32 *>(mmap_centroid_base_ + static_cast<size_t>(key) * mmap_centroid_stride_);
            std::memcpy(owned_centroids->data() + static_cast<size_t>(key) * dim_, src, static_cast<size_t>(dim_) * sizeof(f32));
        }
        target->centroids_ = std::move(owned_centroids);
    } else {
        target->centroids_ = std::make_shared<std::vector<f32>>(*centroids_);
    }
    target->partition_count_ = partition_count_;
    target->partition_capacity_ = partition_capacity_;
    target->root_key_ = root_key_;
    target->height_ = height_;
    target->next_partition_key_ = next_partition_key_;
    target->children_ = children_;

    target->slot_table_ = std::make_shared<std::vector<DataSlot>>(*slot_table_);
    if (data_region_ != nullptr) {
        target->data_region_ = std::make_shared<std::vector<char>>(*data_region_);
    }
    target->next_slot_ = next_slot_;
    target->num_vectors_ = num_vectors_;
    target->deleted_count_ = deleted_count_;
    target->storage_mode_ = StorageMode::kOwned;
    target->mmap_base_ = nullptr;
    target->mmap_data_region_ = nullptr;

    {
        std::lock_guard lock(deleted_mtx_);
        target->deleted_bits_ = std::make_shared<std::vector<u64>>(*deleted_bits_);
    }

    if (target->delta_a_ == nullptr) {
        target->delta_a_ = new SPFreshDeltaBuffer(QuantVecStride(dim_));
        target->delta_b_ = new SPFreshDeltaBuffer(QuantVecStride(dim_));
    }
    target->delta_a_->Clear();
    target->delta_b_->Clear();
    target->active_delta_idx_.store(0, std::memory_order_release);

    tree_lock.unlock();
    target->PublishSnapshot();
}

void SPFreshIndexInMem::Dump(BufferObj *buffer_obj, size_t * /*dump_size_ptr*/) {
    BufferHandle handle = buffer_obj->Load();
    auto *worker = handle.GetFileWorkerMut();
    auto *target = static_cast<SPFreshIndexInMem *>(worker->GetData());
    TransferTo(target);
    chunk_handle_ = std::move(handle);
}

// ═════════════════════════════════════════════════════════════════════════════
// BaseMemIndex interface
// ═════════════════════════════════════════════════════════════════════════════

MemIndexTracerInfo SPFreshIndexInMem::GetInfo() const {
    size_t total = mem_used_;
    if (storage_mode_ == StorageMode::kOwned) {
        total += static_cast<size_t>(partition_capacity_) * dim_ * sizeof(f32);
    }
    total += static_cast<size_t>(partition_capacity_) * sizeof(PartitionMeta);
    total += storage_mode_ == StorageMode::kOwned ? (data_region_ ? data_region_->size() : 0) : 0;
    total += (delta_a_ ? delta_a_->data_.size() : 0) + (delta_b_ ? delta_b_->data_.size() : 0);

    return MemIndexTracerInfo(std::make_shared<std::string>(index_name_), std::make_shared<std::string>(table_name_),
                              std::make_shared<std::string>(db_name_), total, GetRowCount());
}

const ChunkIndexMetaInfo SPFreshIndexInMem::GetChunkIndexMetaInfo() const {
    return ChunkIndexMetaInfo{"spfresh_chunk", begin_row_id_, GetRowCount(), 0, mem_used_};
}

} // namespace infinity

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

// SPFresh (C-SPANN style) index definitions.
//
// Design references:
//   * Microsoft SPFresh / SPANN: nearest-partition assignment, incremental maintenance
//   * CockroachDB C-SPANN: hierarchical K-means tree, per-partition immutable centroid,
//     RaBitQ quantization relative to partition centroid, balanced K-means split,
//     background fixups (split / merge)
//
// Key invariants:
//   1. A partition's centroid is computed once (at creation) and NEVER changes.
//      All quantized vectors in that partition are encoded relative to it.
//   2. Every vector belongs to exactly ONE leaf partition (no replicas).
//   3. Level 1 == leaf level.  Level grows towards the root.
//   4. Partition data lives in a contiguous "slot" inside the data region, so a
//      single range read fetches the whole partition (mmap friendly).

export module infinity_core:spfresh_defs;

import std.compat;

import :infinity_type;

namespace infinity {

// ─────────────────────────────────────────────────────────────────────────────
// File format / magic
// ─────────────────────────────────────────────────────────────────────────────

export constexpr u32 CSPANN_MAGIC = 0x43535041; // "CSPA"
export constexpr u32 CSPANN_VERSION = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Partition identity
// ─────────────────────────────────────────────────────────────────────────────

export using PartitionKey = u32;
export using PartitionLevel = u32;

export constexpr PartitionKey kInvalidPartitionKey = 0xFFFFFFFFu;
export constexpr PartitionKey kRootPartitionKey = 0;
export constexpr PartitionLevel kLeafLevel = 1;

// Partition lifecycle.  In a single-node engine we do not need the full
// CockroachDB 15-step state machine, but the state is still persisted so that a
// crashed/aborted fixup can be detected and replayed.
export enum class PartitionState : u32 {
    kReady = 0,
    kSplitting = 1, // fixup in progress: partition frozen for writes
    kMerging = 2,
    kDeleting = 3, // tombstone: data has been migrated away
};

export constexpr u32 kSlotFlagTombstone = 1u << 0;
export constexpr u32 kSlotFlagOverflow = 1u << 1;

// ─────────────────────────────────────────────────────────────────────────────
// Quantized vector layout (RaBitQ relative to partition centroid)
//
//   for each vector:
//     [QuantizedVectorMeta]  (12 bytes)
//     [u64 code[code_width]] (code_width = ceil(dim / 64))
//
// The whole partition is stored column-wise inside its slot to maximise the
// efficiency of a single sequential read:
//
//   slot := row_ids[count]            (u32)
//           centroid_distances[count] (f32)
//           quantized_dot_products[count] (f32)
//           code_counts[count]        (u32)
//           codes[count][code_width]  (u64)
// ─────────────────────────────────────────────────────────────────────────────

export struct QuantizedVectorMeta {
    f32 centroid_distance_{0.0f};      // ‖o - c‖
    f32 quantized_dot_product_{0.0f};  // 1 / ⟨ō, o⟩ (reciprocal avoids division)
    u32 code_count_{0};                // popcount(codes), used by estimator term2
};
static_assert(sizeof(QuantizedVectorMeta) == 12, "QuantizedVectorMeta must be 12 bytes");

export inline u32 QuantCodeWidth(u32 dim) { return (dim + 63u) / 64u; }
export inline u32 QuantVecStride(u32 dim) { return static_cast<u32>(sizeof(QuantizedVectorMeta)) + QuantCodeWidth(dim) * 8u; }

// ─────────────────────────────────────────────────────────────────────────────
// On-disk POD structures
// ─────────────────────────────────────────────────────────────────────────────

#pragma pack(1)

// One slot == one partition payload inside the data region.
export struct DataSlot {
    u64 offset_{0}; // offset inside the data region
    u32 count_{0};  // vector count
    u32 flags_{0};  // bit 0: tombstone, bit 1: overflow
};

export struct SPFreshFileHeader {
    u32 magic_{CSPANN_MAGIC};
    u32 version_{CSPANN_VERSION};
    u32 dim_{0};
    u32 pad_dim_{0};
    u32 partition_count_{0};
    u32 partition_capacity_{0};
    u32 num_vectors_{0};
    u32 code_width_{0};
    u32 root_key_{kRootPartitionKey};
    u32 height_{1};
    u64 rot_seed_{0};
    u64 section1_offset_{0};
    u64 section1_size_{0};
    u64 slot_table_offset_{0};
    u64 slot_table_capacity_{0};
    u64 data_region_offset_{0};
    u64 data_region_size_{0};
    u64 deleted_count_{0};
    u8 reserved_[256 - 10 * 4 - 8 * 8];

    bool Validate() const { return magic_ == CSPANN_MAGIC && version_ == CSPANN_VERSION; }
};

// Serialized form of PartitionMeta (centroid follows immediately, dim floats).
export struct PartitionMetaDisk {
    u32 parent_{kInvalidPartitionKey};
    u32 level_{kLeafLevel};
    u32 state_{0};
    u32 data_slot_{0};
    u32 base_count_{0};
    u32 pad_{0};
    u64 version_{0};
};

#pragma pack()

static_assert(sizeof(DataSlot) == 16, "DataSlot must be 16 bytes");
static_assert(sizeof(SPFreshFileHeader) == 256, "SPFreshFileHeader must be 256 bytes");

// ─────────────────────────────────────────────────────────────────────────────
// In-memory partition metadata
// ─────────────────────────────────────────────────────────────────────────────

export struct PartitionMeta {
    PartitionKey parent_{kInvalidPartitionKey};
    u32 level_{kLeafLevel};
    PartitionState state_{PartitionState::kReady};
    u32 data_slot_{0};
    u32 base_count_{0};  // vectors persisted in the slot
    u32 delta_count_{0}; // vectors only present in the delta buffer
    u64 version_{0};     // bumped on every structural change

    u32 TotalCount() const { return base_count_ + delta_count_; }
    bool IsReady() const { return state_ == PartitionState::kReady; }
    bool IsLive() const { return state_ != PartitionState::kDeleting; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Delta buffer
//
// Entry layout: [PartitionKey u32][row_id u32][QuantizedVectorMeta 12B][codes]
// A single flat buffer keeps insertion lock-free-ish (per-buffer mutex) and
// allows the searcher to probe it without materialising per-partition lists.
// ─────────────────────────────────────────────────────────────────────────────

export struct SPFreshDeltaBuffer {
    std::vector<char> data_;
    u32 entry_count_{0};
    u32 entry_size_{0}; // QuantVecStride(dim)

    explicit SPFreshDeltaBuffer(u32 entry_size = 0) : entry_size_(entry_size) {}

    void SetEntrySize(u32 size) { entry_size_ = size; }

    size_t Stride() const { return sizeof(u32) * 2 + entry_size_; }

    void Append(PartitionKey key, u32 row_id, const QuantizedVectorMeta &meta, const u8 *codes) {
        const size_t stride = Stride();
        const size_t pos = data_.size();
        data_.resize(pos + stride);
        char *p = data_.data() + pos;
        std::memcpy(p, &key, sizeof(u32));
        std::memcpy(p + sizeof(u32), &row_id, sizeof(u32));
        std::memcpy(p + sizeof(u32) * 2, &meta, sizeof(QuantizedVectorMeta));
        std::memcpy(p + sizeof(u32) * 2 + sizeof(QuantizedVectorMeta), codes, entry_size_ - sizeof(QuantizedVectorMeta));
        ++entry_count_;
    }

    PartitionKey GetKey(size_t i) const {
        PartitionKey k;
        std::memcpy(&k, data_.data() + i * Stride(), sizeof(u32));
        return k;
    }
    u32 GetRowId(size_t i) const {
        u32 r;
        std::memcpy(&r, data_.data() + i * Stride() + sizeof(u32), sizeof(u32));
        return r;
    }
    QuantizedVectorMeta GetMeta(size_t i) const {
        QuantizedVectorMeta m;
        std::memcpy(&m, data_.data() + i * Stride() + sizeof(u32) * 2, sizeof(QuantizedVectorMeta));
        return m;
    }
    const u8 *GetCodes(size_t i) const {
        return reinterpret_cast<const u8 *>(data_.data() + i * Stride() + sizeof(u32) * 2 + sizeof(QuantizedVectorMeta));
    }

    void Clear() {
        data_.clear();
        entry_count_ = 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Background fixup requests
// ─────────────────────────────────────────────────────────────────────────────

export struct FixupRequest {
    enum class Type : u32 { kSplit = 0, kMerge = 1 };
    Type type_{Type::kSplit};
    PartitionKey key_{kInvalidPartitionKey};
    u64 version_{0}; // optimistic concurrency: drop if partition changed
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-level search statistics (for adaptive beam sizing)
// ─────────────────────────────────────────────────────────────────────────────

export struct LevelStats {
    f32 mean_{0.0f};     // mean of the "relative distance gap" samples
    f32 variance_{1.0f}; // variance of the same samples
    u32 samples_{0};
};

// ─────────────────────────────────────────────────────────────────────────────
// Tunables (defaults mirror CockroachDB C-SPANN, adapted for 8M rows/segment)
// ─────────────────────────────────────────────────────────────────────────────

export constexpr u32 kDefaultMaxPartitionSize = 512;
export constexpr u32 kDefaultMinPartitionSize = 16;
export constexpr u32 kDefaultBaseBeamSize = 8;
export constexpr u32 kDefaultKMeansMaxIter = 16;
export constexpr u32 kDefaultAllowImbalancePct = 33;
export constexpr u32 kDefaultRerankMultiplier = 50;
export constexpr u32 kDefaultFixupQueueCapacity = 256;

} // namespace infinity

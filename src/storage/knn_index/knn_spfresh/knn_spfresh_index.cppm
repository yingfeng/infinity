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

export module infinity_core:spfresh_index;

import :spfresh_defs;
import :base_memindex;
import :memindex_tracer;
import :chunk_index_meta;
import :index_base;
import :index_spfresh;
import :local_file_handle;
import :buffer_handle;
import :logger;
import :croaring;

import std.compat;

import internal_types;
import column_def;
import row_id;

namespace infinity {

// Provides full-precision vectors for maintenance operations that must
// re-encode data relative to a *new* centroid (split / merge / compact GC).
//
// The index itself only stores quantized codes; re-encoding therefore either
// needs the original vector (production, backed by the column store) or falls
// back to decoding the quantized code (self-contained degradation path).
export class SPFreshVectorFetcher {
public:
    virtual ~SPFreshVectorFetcher() = default;

    // Returns false when the row is not available (deleted / not found).
    virtual bool Fetch(u32 row_id, f32 *out_vec) const = 0;

    // Returns true when the row is known to be deleted.
    virtual bool IsDeleted(u32 row_id) const { return false; }

    void SetDim(u32 dim) { dim_ = dim; }
    u32 dim() const { return dim_; }

private:
    u32 dim_{0};
};

export class SPFreshIndexInMem : public BaseMemIndex {
public:
    SPFreshIndexInMem();
    SPFreshIndexInMem(RowID begin_row_id, const IndexSPFresh *index_def, u32 embedding_dim, u32 max_vectors, const std::string &base_path = "");
    ~SPFreshIndexInMem() override;

    MemIndexTracerInfo GetInfo() const override;
    const ChunkIndexMetaInfo GetChunkIndexMetaInfo() const override;
    RowID GetBeginRowID() const override { return begin_row_id_; }

    // ── Storage mode ──
    enum class StorageMode { kOwned, kMmap };
    StorageMode GetStorageMode() const { return storage_mode_; }

    // ── Build (bulk) ──
    void Build(const f32 *vectors, u32 count);

    // ── Incremental maintenance ──
    void InsertDelta(const f32 *vec, u32 row_id);
    void MarkDeleted(SegmentOffset offset);

    // Merge delta into base, garbage-collect deleted vectors and run pending fixups.
    // `fetcher` may be null: in that case re-encoding uses decoded approximations.
    void Compact(const SPFreshVectorFetcher *fetcher = nullptr);

    // Scan up to `budget` partitions (round-robin, resumable across calls) and
    // enqueue a kMerge fixup for every undersized leaf that has a sibling to take
    // its data. Returns the number of merges enqueued. Used by the background
    // fixup worker when its queue runs dry, and by tests.
    u32 EnqueueMergeSweep(u32 budget = 0);

    // ── Search ──
    // approx_dist : RaBitQ estimate (squared L2)
    // error_bound : theoretical error bound of `approx_dist`
    //
    // `prune_threshold` is an optional live threshold: any candidate whose
    // optimistic distance `approx_dist - error_bound` is not better than the
    // current value is skipped without invoking the callback. The caller owns the
    // storage and may lower it from inside the callback as its result heap fills.
    using SearchCallback = std::function<void(SegmentOffset, f32 approx_dist, f32 error_bound)>;
    void Search(const f32 *query,
                u32 dim,
                const SearchCallback &callback,
                u32 max_candidates = 0,
                u32 beam_size = 0,
                const f32 *prune_threshold = nullptr) const;

    // ── Persistence ──
    void Save(LocalFileHandle &fh) const;
    void Load(LocalFileHandle &fh, size_t file_size);
    void LoadFromMmap(const u8 *base, size_t size);
    void Dump(BufferObj *buffer_obj, size_t *dump_size_ptr = nullptr);
    void TransferTo(SPFreshIndexInMem *target);

    // ── Accessors ──
    u32 GetRowCount() const;
    u32 GetBaseRowCount() const;
    u32 GetDeltaCount() const;
    u32 GetDeletedCount() const;
    u32 GetNumCentroids() const { return partition_count_; }
    u32 GetTreeHeight() const { return height_; }
    u32 dim() const { return dim_; }
    u32 GetPartitionCount() const { return partition_count_; }
    // Ready, non-internal partitions that actually carry vectors. `GetPartitionCount()`
    // also counts tombstones (split/merge sources), so this is what describes the
    // current clustering quality.
    u32 GetLiveLeafCount() const;

    static u32 QuantizedStride(u32 dim) { return QuantVecStride(dim); }

private:
    // ─────────────────────────────────────────────────────────────────────────
    // Snapshot (RCU)
    // ─────────────────────────────────────────────────────────────────────────
    struct TreeSnapshot {
        StorageMode mode_{StorageMode::kOwned};

        u32 partition_count_{0};
        u32 root_key_{kRootPartitionKey};
        u32 height_{kLeafLevel};
        u32 num_vectors_{0};
        u32 dim_{0};
        u32 code_width_{0};

        // Partition metadata & centroids are pre-allocated and appended in place,
        // so a snapshot only needs to pin the buffers plus its logical count.
        std::shared_ptr<const std::vector<PartitionMeta>> partitions_;
        std::shared_ptr<const std::vector<f32>> centroids_;
        std::shared_ptr<const std::vector<u32>> child_offsets_;
        std::shared_ptr<const std::vector<u32>> child_ids_;
        std::shared_ptr<const std::vector<DataSlot>> slots_;
        std::shared_ptr<const std::vector<char>> data_region_; // kOwned
        const u8 *mmap_data_region_{nullptr};                  // kMmap
        const u8 *mmap_base_{nullptr};
        // In mmap mode centroids are read straight out of the mapped file:
        // only the (small) partition table stays in DRAM.
        const u8 *mmap_centroid_base_{nullptr};
        u64 mmap_centroid_stride_{0};

        const f32 *Centroid(PartitionKey key) const {
            if (mmap_centroid_base_ != nullptr) {
                return reinterpret_cast<const f32 *>(mmap_centroid_base_ + static_cast<size_t>(key) * mmap_centroid_stride_);
            }
            return centroids_->data() + static_cast<size_t>(key) * dim_;
        }
        const DataSlot &Slot(PartitionKey key) const { return (*slots_)[key]; }
        const u8 *RegionBase() const { return mode_ == StorageMode::kOwned ? reinterpret_cast<const u8 *>(data_region_->data()) : mmap_data_region_; }
        u32 ChildBegin(PartitionKey key) const { return (*child_offsets_)[key]; }
        u32 ChildEnd(PartitionKey key) const { return (*child_offsets_)[key + 1]; }
    };

    void PublishSnapshot();
    void EnsureCapacity(u32 required_partitions);

    // The deleted set is not part of the structural snapshot: it changes far more
    // often than the tree and is fetched once per search / maintenance pass.
    std::shared_ptr<const roaring::Roaring> CurrentDeletedSet() const;
    static bool IsDeletedIn(const std::shared_ptr<const roaring::Roaring> &set, u32 row_id);

    // ─────────────────────────────────────────────────────────────────────────
    // Rotation / quantization
    // ─────────────────────────────────────────────────────────────────────────
    void GenerateRotationParams(u64 seed);
    void ApplyRotation(f32 *vec) const;

    struct QueryQuantized {
        std::vector<u64> planes_[4];
        u32 plane_popcount_[4]{0, 0, 0, 0};
        u32 quantized_sum_{0}; // sum of the 4-bit quantized values
        f32 min_val_{0.0f};
        f32 delta_{0.0f};
        f32 sqrt_dim_inv_{0.0f};
        f32 sqrt_dim_{0.0f};
        f32 centroid_distance_{0.0f};
        std::vector<f32> unbias_;
    };

    void QuantizeQuery(QueryQuantized &qq, const f32 *query, const f32 *centroid) const;
    void EncodeRelative(const f32 *vec, const f32 *centroid, QuantizedVectorMeta &meta, u8 *codes) const;
    void DecodeRelative(const QuantizedVectorMeta &meta, const u8 *codes, const f32 *centroid, f32 *out_vec) const;
    f32 EstimateDistance(const QuantizedVectorMeta &meta, const u8 *codes, const QueryQuantized &qq) const;
    f32 ErrorBound(const QuantizedVectorMeta &meta, const QueryQuantized &qq) const;

    // ─────────────────────────────────────────────────────────────────────────
    // K-means
    // ─────────────────────────────────────────────────────────────────────────
    u32 KMeans(const f32 *vectors, u32 count, u32 k, u32 max_iter, std::vector<f32> &out_centroids, std::vector<u32> &out_labels) const;
    void BalancedKMeans2(const f32 *vectors, u32 count, std::vector<f32> &out_a, std::vector<f32> &out_b, std::vector<u8> &out_label) const;

    // ─────────────────────────────────────────────────────────────────────────
    // Tree construction
    // ─────────────────────────────────────────────────────────────────────────
    PartitionKey BuildSubtree(const f32 *vectors, const std::vector<u32> &indices, PartitionKey parent);
    PartitionKey AllocPartition(PartitionKey parent, u32 level, const f32 *centroid, u32 data_slot, u32 base_count);
    void AppendChild(PartitionKey parent, PartitionKey child);
    void ReplaceChild(PartitionKey parent, PartitionKey old_child, const std::vector<PartitionKey> &new_children);
    void RemoveChild(PartitionKey parent, PartitionKey child);

    u32 AllocSlot(u32 count, u32 flags = 0);
    void WriteSlotVectors(u32 slot_id, const f32 *vectors, const std::vector<u32> &indices, const f32 *centroid);

    // ─────────────────────────────────────────────────────────────────────────
    // Search
    // ─────────────────────────────────────────────────────────────────────────
    PartitionKey RouteToLeaf(const f32 *vec) const;
    // `beam_size` == 0 selects the adaptive beam (see AdaptBeamSize).
    void SearchTree(const TreeSnapshot &snap, const f32 *query, u32 beam_size, std::vector<PartitionKey> &out_leaves) const;
    void ScanLeafPartition(const TreeSnapshot &snap,
                           PartitionKey key,
                           const QueryQuantized &qq,
                           const SearchCallback &callback,
                           const f32 *prune_threshold,
                           const std::shared_ptr<const roaring::Roaring> &deleted) const;

    // Adaptive beam sizing helpers.
    u32 AdaptBeamSize(u32 level, u32 base, u32 available, f32 gap) const;

    // ─────────────────────────────────────────────────────────────────────────
    // Fixups (split / merge)
    // ─────────────────────────────────────────────────────────────────────────
    // `version` is the partition version observed by the caller while it held the
    // tree lock; the worker drops the request if the partition has moved on since.
    // Returns false when the request was already pending or the queue is full.
    bool EnqueueFixup(FixupRequest::Type type, PartitionKey key, u64 version);
    void EnsureFixupWorker();
    void StopFixupWorker();
    void FixupWorkerLoop();
    void RunPendingFixupsLocked(const SPFreshVectorFetcher *fetcher);
    bool ExecuteSplit(PartitionKey key, const SPFreshVectorFetcher *fetcher);
    bool ExecuteMerge(PartitionKey key, const SPFreshVectorFetcher *fetcher);
    std::vector<PartitionKey> SiblingLeaves(PartitionKey key) const;

    void
    CollectPartitionVectors(PartitionKey key, const SPFreshVectorFetcher *fetcher, std::vector<f32> &out_vecs, std::vector<u32> &out_row_ids) const;
    void MigrateVectorToPartition(const f32 *vec, u32 row_id, PartitionKey dst, SPFreshDeltaBuffer &delta);

    // ─────────────────────────────────────────────────────────────────────────
    // IO helpers
    // ─────────────────────────────────────────────────────────────────────────
    void TransitionToOwned();

    size_t Section1Size() const;
    void SerializeSection1(std::vector<char> &out) const;
    void DeserializeSection1(const u8 *ptr,
                             const SPFreshFileHeader &header,
                             std::vector<PartitionMeta> &out_partitions,
                             std::vector<f32> &out_centroids,
                             std::vector<u32> &out_child_offsets,
                             std::vector<u32> &out_child_ids);

    // ─────────────────────────────────────────────────────────────────────────
    // Member data
    // ─────────────────────────────────────────────────────────────────────────
    RowID begin_row_id_;
    StorageMode storage_mode_{StorageMode::kOwned};

    u32 dim_{0};
    u32 pad_dim_{1};
    u32 code_width_{1};
    u64 rot_seed_{0};
    std::vector<bool> rotation_flip_;

    // Tunables
    u32 max_partition_size_{kDefaultMaxPartitionSize};
    u32 min_partition_size_{kDefaultMinPartitionSize};
    u32 base_beam_size_{kDefaultBaseBeamSize};
    u32 kmeans_max_iter_{kDefaultKMeansMaxIter};
    u64 max_delta_bytes_{512ULL * 1024 * 1024};
    u32 rerank_multiplier_{kDefaultRerankMultiplier};

    // ── Tree (pre-allocated, appended in place) ──
    std::shared_ptr<std::vector<PartitionMeta>> partitions_;
    std::shared_ptr<std::vector<f32>> centroids_;
    u32 partition_count_{0};
    u32 partition_capacity_{0};
    PartitionKey root_key_{kRootPartitionKey};
    u32 height_{kLeafLevel};
    u32 next_partition_key_{kRootPartitionKey};
    std::vector<std::vector<PartitionKey>> children_;

    // ── COW structures ──
    std::shared_ptr<std::vector<DataSlot>> slot_table_;
    std::shared_ptr<std::vector<char>> data_region_;
    u32 next_slot_{0};

    // ── mmap view ──
    const u8 *mmap_base_{nullptr};
    const u8 *mmap_data_region_{nullptr};
    const u8 *mmap_centroid_base_{nullptr};
    u64 mmap_centroid_stride_{0};
    u32 mmap_slot_capacity_{0};
    SPFreshFileHeader mmap_header_{};

    // ── Counters ──
    u32 num_vectors_{0};
    u64 deleted_count_{0};

    // ── Deleted set ──
    // Roaring keeps the set sparse: row ids are segment offsets, so a handful of
    // deletes in a large segment costs a handful of containers instead of a dense
    // bitmap of `num_vectors_` bits. The published copy is immutable (copy-on-write
    // on every delete) so readers never observe a container being mutated.
    std::shared_ptr<const roaring::Roaring> deleted_bits_;
    mutable std::shared_mutex deleted_mtx_;

    // ── Delta double buffering ──
    SPFreshDeltaBuffer *delta_a_{nullptr};
    SPFreshDeltaBuffer *delta_b_{nullptr};
    std::atomic<u32> active_delta_idx_{0};
    std::mutex delta_mtx_;
    std::atomic<u64> delta_bytes_{0};

    // ── Fixup queue / worker ──
    std::deque<FixupRequest> fixup_queue_;
    std::mutex fixup_mtx_;
    std::condition_variable fixup_cv_;
    std::thread fixup_thread_;
    std::atomic<bool> fixup_stop_{false};
    std::atomic<bool> fixup_running_{false};

    // ── Merge sweep cursor (round-robin over partitions) ──
    std::atomic<u32> merge_sweep_cursor_{0};

    // ── Adaptive beam statistics (per tree level) ──
    bool adaptive_beam_{true};
    mutable std::mutex level_stats_mtx_;
    mutable std::array<LevelStats, kAdaptiveBeamMaxLevels> level_stats_{};

    // ── Snapshot (RCU) ──
    std::shared_ptr<const TreeSnapshot> snapshot_;
    mutable std::shared_mutex snapshot_mtx_;

    // ── Concurrency ──
    mutable std::shared_mutex tree_mtx_;
    mutable std::mutex counter_mtx_;
    mutable std::shared_mutex compact_mtx_;

    // ── Dump guard ──
    BufferHandle chunk_handle_;
    size_t mem_used_{0};
};

} // namespace infinity

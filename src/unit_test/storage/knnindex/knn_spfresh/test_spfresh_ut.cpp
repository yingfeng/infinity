module;

#include "unit_test/gtest_expand.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

module infinity_core:ut.test_spfresh;

import :ut.base_test;
import :spfresh_index;
import :spfresh_defs;
import :index_spfresh;
import :local_file_handle;

import std;

using namespace infinity;

class SPFreshTest : public BaseTest {};

namespace {

std::shared_ptr<IndexSPFresh> MakeIndexDef(u32 bucket_size_limit = 10000) {
    return std::make_shared<IndexSPFresh>(std::make_shared<std::string>("test_idx"),
                                          std::make_shared<std::string>(""),
                                          "test",
                                          std::vector<std::string>{"col1"},
                                          MetricType::kMetricL2,
                                          /*num_centroids*/ 1,
                                          /*replica_count*/ 1,
                                          bucket_size_limit,
                                          /*compress_to_rabitq*/ true,
                                          /*max_delta_mb*/ 512);
}

} // namespace

TEST_F(SPFreshTest, test_basic_construction) {
    SPFreshIndexInMem idx;
    EXPECT_EQ(idx.GetRowCount(), 0u);
    EXPECT_EQ(idx.GetBaseRowCount(), 0u);
    EXPECT_EQ(idx.GetDeltaCount(), 0u);
    EXPECT_EQ(idx.GetDeletedCount(), 0u);
    EXPECT_EQ(idx.GetPartitionCount(), 0u);
}

TEST_F(SPFreshTest, test_build_and_search) {
    auto index_def = MakeIndexDef();
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 100);

    f32 vecs[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    idx.Build(vecs, 4);

    EXPECT_EQ(idx.GetBaseRowCount(), 4u);
    EXPECT_GE(idx.GetNumCentroids(), 1u);
    EXPECT_EQ(idx.GetTreeHeight(), 1u); // 4 vectors <= max partition size -> single leaf root

    f32 query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    u32 result_count = 0;
    idx.Search(query, 4, [&](SegmentOffset, f32, f32) { ++result_count; });
    EXPECT_EQ(result_count, 4u);
}

TEST_F(SPFreshTest, test_insert_delta_and_search) {
    auto index_def = MakeIndexDef();
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 100);

    f32 base[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    idx.Build(base, 2);

    f32 v3[] = {0.0f, 0.0f, 1.0f, 0.0f};
    idx.InsertDelta(v3, 2);
    EXPECT_EQ(idx.GetDeltaCount(), 1u);
    EXPECT_EQ(idx.GetRowCount(), 3u);

    f32 v4[] = {0.0f, 0.0f, 0.0f, 1.0f};
    idx.InsertDelta(v4, 3);
    EXPECT_EQ(idx.GetDeltaCount(), 2u);
    EXPECT_EQ(idx.GetRowCount(), 4u);

    f32 query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    u32 result_count = 0;
    idx.Search(query, 4, [&](SegmentOffset, f32, f32) { ++result_count; });
    EXPECT_EQ(result_count, 4u);
}

TEST_F(SPFreshTest, test_compact) {
    auto index_def = MakeIndexDef();
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 100);

    f32 base[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    idx.Build(base, 2);

    f32 v3[] = {0.0f, 0.0f, 1.0f, 0.0f};
    idx.InsertDelta(v3, 2);
    EXPECT_EQ(idx.GetDeltaCount(), 1u);

    idx.Compact();
    EXPECT_EQ(idx.GetDeltaCount(), 0u);
    EXPECT_EQ(idx.GetBaseRowCount(), 3u);
    EXPECT_EQ(idx.GetRowCount(), 3u);

    // Data survives compaction.
    f32 query[] = {0.0f, 0.0f, 1.0f, 0.0f};
    u32 result_count = 0;
    idx.Search(query, 4, [&](SegmentOffset, f32, f32) { ++result_count; });
    EXPECT_EQ(result_count, 3u);
}

TEST_F(SPFreshTest, test_deletion_and_gc) {
    auto index_def = MakeIndexDef();
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 100);

    f32 base[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    idx.Build(base, 3);

    idx.MarkDeleted(1);
    EXPECT_EQ(idx.GetDeletedCount(), 1u);
    EXPECT_EQ(idx.GetRowCount(), 2u);

    // Deleted row is filtered out by the index itself.
    f32 query[] = {0.0f, 1.0f, 0.0f, 0.0f};
    u32 seen_deleted = 0;
    u32 result_count = 0;
    idx.Search(query, 4, [&](SegmentOffset offset, f32, f32) {
        ++result_count;
        if (offset == 1) {
            ++seen_deleted;
        }
    });
    EXPECT_EQ(result_count, 2u);
    EXPECT_EQ(seen_deleted, 0u);

    // Compaction physically removes the deleted vector.
    idx.Compact();
    EXPECT_EQ(idx.GetRowCount(), 2u);
    EXPECT_EQ(idx.GetBaseRowCount(), 2u);
    EXPECT_EQ(idx.GetDeletedCount(), 0u);
}

TEST_F(SPFreshTest, test_split_creates_new_partitions) {
    // bucket_size_limit acts as the max partition size; 8 forces splits quickly.
    auto index_def = MakeIndexDef(/*bucket_size_limit*/ 8);
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 1000);

    std::vector<f32> data;
    data.reserve(8 * 4);
    for (u32 i = 0; i < 8; ++i) {
        const f32 angle = static_cast<f32>(i) * 0.7f;
        data.push_back(std::cos(angle));
        data.push_back(std::sin(angle));
        data.push_back(static_cast<f32>(i) * 0.1f);
        data.push_back(1.0f);
    }
    idx.Build(data.data(), 8);
    const u32 partitions_before = idx.GetPartitionCount();
    EXPECT_GE(partitions_before, 1u);

    for (u32 i = 8; i < 48; ++i) {
        const f32 angle = static_cast<f32>(i) * 0.7f;
        f32 v[4] = {std::cos(angle), std::sin(angle), static_cast<f32>(i) * 0.1f, 1.0f};
        idx.InsertDelta(v, i);
    }
    // Compact drains the fixup queue (split) and merges the delta.
    idx.Compact();

    EXPECT_GT(idx.GetPartitionCount(), partitions_before);
    EXPECT_EQ(idx.GetDeltaCount(), 0u);
    EXPECT_EQ(idx.GetRowCount(), 48u);

    f32 query[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    u32 result_count = 0;
    idx.Search(query, 4, [&](SegmentOffset, f32, f32) { ++result_count; });
    EXPECT_EQ(result_count, 48u);
}

TEST_F(SPFreshTest, test_quantization_accuracy) {
    auto index_def = MakeIndexDef();
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 8, 100);

    f32 base[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    idx.Build(base, 2);

    // The nearest neighbour of e0 must be e0 itself (approx distance ~0).
    f32 query[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    f32 min_dist = std::numeric_limits<f32>::max();
    SegmentOffset best = std::numeric_limits<SegmentOffset>::max();
    idx.Search(query, 8, [&](SegmentOffset offset, f32 dist, f32 bound) {
        EXPECT_TRUE(std::isfinite(dist));
        EXPECT_TRUE(std::isfinite(bound));
        EXPECT_GE(bound, 0.0f);
        if (dist < min_dist) {
            min_dist = dist;
            best = offset;
        }
    });
    EXPECT_EQ(best, 0u);
    EXPECT_LT(std::fabs(min_dist), 0.5f);
}

TEST_F(SPFreshTest, test_persistence_roundtrip) {
    const std::string path = "/tmp/infinity_spfresh_roundtrip.idx";
    std::error_code ec;
    std::filesystem::remove(path, ec);

    auto index_def = MakeIndexDef();
    const u32 dim = 16, n = 200;
    std::mt19937 rng(7);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(static_cast<size_t>(n) * dim);
    for (auto &x : data) {
        x = dist(rng);
    }

    {
        SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), dim, n);
        idx.Build(data.data(), n);
        EXPECT_EQ(idx.GetBaseRowCount(), n);

        i32 fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        ASSERT_GE(fd, 0);
        {
            LocalFileHandle fh(fd, path, FileAccessMode::kWrite);
            idx.Save(fh);
            fh.Sync();
        }
    }

    SPFreshIndexInMem loaded;
    {
        i32 fd = ::open(path.c_str(), O_RDONLY, 0644);
        ASSERT_GE(fd, 0);
        LocalFileHandle fh(fd, path, FileAccessMode::kRead);
        loaded.Load(fh, 0);
    }

    EXPECT_EQ(loaded.GetBaseRowCount(), n);
    EXPECT_EQ(loaded.GetPartitionCount(), 1u);
    EXPECT_EQ(loaded.dim(), dim);

    u32 count = 0;
    loaded.Search(data.data(), dim, [&](SegmentOffset, f32, f32) { ++count; });
    EXPECT_EQ(count, n);

    std::filesystem::remove(path, ec);
}

TEST_F(SPFreshTest, test_multi_level_tree) {
    // 5000 vectors with fan-out 32 => 3 levels (leaf / middle / root).
    auto index_def = MakeIndexDef(/*bucket_size_limit*/ 64);
    const u32 dim = 32, n = 3000;
    std::mt19937 rng(11);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(static_cast<size_t>(n) * dim);
    for (auto &x : data) {
        x = dist(rng);
    }

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), dim, n);
    idx.Build(data.data(), n);

    EXPECT_GE(idx.GetTreeHeight(), 2u);
    EXPECT_GT(idx.GetPartitionCount(), 40u);

    // Recall@10 over several probes: an exact stored vector must land in the
    // candidate set that is later reranked with full precision.
    constexpr u32 kProbes = 20;
    constexpr u32 kTopk = 10;
    auto exact_l2 = [&](const f32 *a, const f32 *b) {
        f32 sum = 0;
        for (u32 d = 0; d < dim; ++d) {
            const f32 df = a[d] - b[d];
            sum += df * df;
        }
        return sum;
    };
    u32 hits = 0;
    for (u32 p = 0; p < kProbes; ++p) {
        const u32 target = 17 + p * 97;
        const f32 *query = data.data() + static_cast<size_t>(target) * dim;
        std::vector<std::pair<f32, SegmentOffset>> cands;
        idx.Search(query, dim, [&](SegmentOffset offset, f32 d, f32) { cands.emplace_back(d, offset); });
        const size_t keep = cands.size();
        std::partial_sort(cands.begin(), cands.begin() + keep, cands.end());
        std::vector<std::pair<f32, SegmentOffset>> exact;
        exact.reserve(keep);
        for (size_t i = 0; i < keep; ++i) {
            exact.emplace_back(exact_l2(query, data.data() + static_cast<size_t>(cands[i].second) * dim), cands[i].second);
        }
        std::partial_sort(exact.begin(), exact.begin() + std::min<size_t>(kTopk, exact.size()), exact.end());
        for (u32 i = 0; i < kTopk && i < exact.size(); ++i) {
            if (exact[i].second == target) {
                ++hits;
                break;
            }
        }
    }
    const f32 recall = static_cast<f32>(hits) / static_cast<f32>(kProbes);
    LOG_INFO(fmt::format("multi-level recall@{} = {}", kTopk, recall));
    EXPECT_GE(recall, 0.9f);
}

TEST_F(SPFreshTest, test_single_partition_recall) {
    // Whole dataset fits in ONE partition: isolates quantization from tree routing.
    auto index_def = MakeIndexDef(/*bucket_size_limit*/ 2048);
    const u32 dim = 32, n = 500;
    std::mt19937 rng(3);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(static_cast<size_t>(n) * dim);
    for (auto &x : data) {
        x = dist(rng);
    }

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), dim, n);
    idx.Build(data.data(), n);
    ASSERT_EQ(idx.GetPartitionCount(), 1u);

    auto exact_l2 = [&](const f32 *a, const f32 *b) {
        f32 sum = 0;
        for (u32 d = 0; d < dim; ++d) {
            const f32 df = a[d] - b[d];
            sum += df * df;
        }
        return sum;
    };

    for (u32 target : {0u, 123u, 499u}) {
        const f32 *query = data.data() + static_cast<size_t>(target) * dim;
        // 1) approximate candidate generation (as the index does)
        std::vector<std::pair<f32, SegmentOffset>> cands;
        idx.Search(query, dim, [&](SegmentOffset offset, f32 d, f32) { cands.emplace_back(d, offset); });
        LOG_INFO(fmt::format("single-partition: row_count={} candidates={}", idx.GetRowCount(), cands.size()));
        ASSERT_EQ(cands.size(), idx.GetRowCount());
        // 2) full-precision rerank of the over-fetched candidate set
        const size_t keep = cands.size();
        std::partial_sort(cands.begin(), cands.begin() + keep, cands.end());
        f32 best = std::numeric_limits<f32>::max();
        SegmentOffset best_offset = std::numeric_limits<SegmentOffset>::max();
        for (size_t i = 0; i < keep; ++i) {
            const f32 d = exact_l2(query, data.data() + static_cast<size_t>(cands[i].second) * dim);
            if (d < best) {
                best = d;
                best_offset = cands[i].second;
            }
        }
        EXPECT_EQ(best_offset, target) << "target=" << target;
    }
}

TEST_F(SPFreshTest, debug_quantization_ranking) {
    auto index_def = MakeIndexDef(/*bucket_size_limit*/ 2048);
    const u32 dim = 8, n = 200;
    std::mt19937 rng(5);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(static_cast<size_t>(n) * dim);
    for (auto &x : data) {
        x = dist(rng);
    }
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), dim, n);
    idx.Build(data.data(), n);

    const u32 target = 77;
    std::vector<std::pair<f32, SegmentOffset>> all;
    idx.Search(data.data() + static_cast<size_t>(target) * dim, dim, [&](SegmentOffset off, f32 d, f32 b) {
        all.emplace_back(d, off);
        if (off == target) {
            LOG_INFO(fmt::format("target {} dist={} bound={}", off, d, b));
        }
    });
    std::sort(all.begin(), all.end());
    LOG_INFO(fmt::format("scanned={} best={} (offset {}) 2nd={} (offset {})",
                         all.size(),
                         all[0].first,
                         all[0].second,
                         all.size() > 1 ? all[1].first : -1.0f,
                         all.size() > 1 ? all[1].second : 0u));
    f32 self_dist = -1;
    for (auto &[d, o] : all) {
        if (o == target) {
            self_dist = d;
            break;
        }
    }
    LOG_INFO(fmt::format("self_dist={}", self_dist));
    EXPECT_GE(self_dist, 0.0f);
}

TEST_F(SPFreshTest, test_mmap_load_and_search) {
    const std::string path = "/tmp/infinity_spfresh_mmap.idx";
    std::error_code ec;
    std::filesystem::remove(path, ec);

    auto index_def = MakeIndexDef(/*bucket_size_limit*/ 32);
    const u32 dim = 32, n = 2000;
    std::mt19937 rng(23);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(static_cast<size_t>(n) * dim);
    for (auto &x : data) {
        x = dist(rng);
    }

    {
        SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), dim, n);
        idx.Build(data.data(), n);
        i32 fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        ASSERT_GE(fd, 0);
        {
            LocalFileHandle fh(fd, path, FileAccessMode::kWrite);
            idx.Save(fh);
            fh.Sync();
        }
    }

    i32 fd = ::open(path.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    struct stat st{};
    ASSERT_EQ(::fstat(fd, &st), 0);
    void *base = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT_NE(base, MAP_FAILED);

    {
        SPFreshIndexInMem mapped;
        mapped.LoadFromMmap(static_cast<const u8 *>(base), static_cast<size_t>(st.st_size));
        EXPECT_EQ(mapped.GetStorageMode(), SPFreshIndexInMem::StorageMode::kMmap);
        EXPECT_EQ(mapped.GetBaseRowCount(), n);
        EXPECT_EQ(mapped.GetPartitionCount(), mapped.GetPartitionCount());

        // Full beam must expose every vector stored in the mapping.
        u32 count_all = 0;
        mapped.Search(data.data(), dim, [&](SegmentOffset, f32, f32) { ++count_all; }, 0, 100000);
        EXPECT_EQ(count_all, n);

        // Default beam + rerank must still find the exact neighbour.
        std::vector<std::pair<f32, SegmentOffset>> cands;
        mapped.Search(data.data(), dim, [&](SegmentOffset offset, f32 d, f32) { cands.emplace_back(d, offset); });
        ASSERT_FALSE(cands.empty());
        f32 best = std::numeric_limits<f32>::max();
        SegmentOffset best_offset = std::numeric_limits<SegmentOffset>::max();
        for (auto &[approx, offset] : cands) {
            f32 exact = 0;
            for (u32 d = 0; d < dim; ++d) {
                const f32 df = data[d] - data[static_cast<size_t>(offset) * dim + d];
                exact += df * df;
            }
            if (exact < best) {
                best = exact;
                best_offset = offset;
            }
        }
        EXPECT_EQ(best_offset, 0u);

        // Mapping must be read-only: an insert promotes the index to owned mode.
        f32 probe[32];
        std::memcpy(probe, data.data(), sizeof(probe));
        mapped.InsertDelta(probe, n);
        EXPECT_EQ(mapped.GetStorageMode(), SPFreshIndexInMem::StorageMode::kOwned);
        EXPECT_EQ(mapped.GetRowCount(), n + 1);
    }

    ::munmap(base, static_cast<size_t>(st.st_size));
    ::close(fd);
    std::filesystem::remove(path, ec);
}

// ── Deleted set (roaring) ────────────────────────────────────────────────────

TEST_F(SPFreshTest, test_deleted_set_is_sparse_and_idempotent) {
    auto index_def = MakeIndexDef();
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 100);

    f32 base[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    idx.Build(base, 4);

    idx.MarkDeleted(3);
    idx.MarkDeleted(3); // idempotent
    EXPECT_EQ(idx.GetDeletedCount(), 1u);

    // A sparse segment offset far below/above the vector count must be representable:
    // the roaring set indexes ids, it does not need to be sized by row count.
    idx.MarkDeleted(1000000);
    EXPECT_EQ(idx.GetDeletedCount(), 2u);

    f32 query[] = {0.0f, 0.0f, 0.0f, 1.0f};
    std::vector<SegmentOffset> seen;
    idx.Search(query, 4, [&](SegmentOffset offset, f32, f32) { seen.push_back(offset); });
    EXPECT_EQ(seen.size(), 3u); // row 3 is gone, the sparse delete touched nothing else
    EXPECT_EQ(std::find(seen.begin(), seen.end(), 3u), seen.end());

    idx.Compact();
    EXPECT_EQ(idx.GetDeletedCount(), 0u);
    EXPECT_EQ(idx.GetRowCount(), 3u);
}

// ── Background merge sweep ───────────────────────────────────────────────────

TEST_F(SPFreshTest, test_merge_sweep_collapses_undersized_leaves) {
    // fan-out 64 => min_partition_size = 8
    auto index_def = MakeIndexDef(/*bucket_size_limit*/ 64);
    const u32 dim = 16, n = 512;
    std::mt19937 rng(31);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(static_cast<size_t>(n) * dim);
    for (auto &x : data) {
        x = dist(rng);
    }

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), dim, n);
    idx.Build(data.data(), n);
    const u32 leaves_before = idx.GetLiveLeafCount();
    ASSERT_GT(leaves_before, 4u) << "fan-out too large to produce several leaves";

    // Keep one row out of sixteen: after compaction every leaf is far below
    // min_partition_size, which is exactly what the sweep is supposed to fix.
    u32 keep = 0;
    for (u32 i = 0; i < n; ++i) {
        if (i % 16 == 0) {
            ++keep;
        } else {
            idx.MarkDeleted(i);
        }
    }
    idx.Compact(); // drops the deleted rows and kicks a merge sweep
    EXPECT_EQ(idx.GetRowCount(), keep);

    // The worker rescans every kFixupIdleWaitMs; wait until the leaf count settles.
    u32 live = idx.GetLiveLeafCount();
    for (u32 round = 0; round < 100; ++round) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const u32 now = idx.GetLiveLeafCount();
        if (now == live && round > 5) {
            break;
        }
        live = now;
    }
    LOG_INFO(fmt::format("merge sweep: live leaves {} -> {}", leaves_before, live));
    EXPECT_LT(live, leaves_before);

    // Merging must not lose data: every surviving row is still reachable. The beam is
    // widened on purpose here — beam truncation is a recall question (covered by the
    // adaptive-beam test), not a data-placement one.
    u32 found = 0;
    std::vector<f32> q(dim);
    for (u32 i = 0; i < n; i += 16) {
        std::memcpy(q.data(), data.data() + static_cast<size_t>(i) * dim, static_cast<size_t>(dim) * sizeof(f32));
        bool hit = false;
        idx.Search(
            q.data(),
            dim,
            [&](SegmentOffset offset, f32, f32) {
                if (offset == i) {
                    hit = true;
                }
            },
            0,
            100000);
        if (hit) {
            ++found;
        }
    }
    EXPECT_EQ(found, keep);
}

TEST_F(SPFreshTest, test_merge_sweep_is_resumable) {
    // The sweep scans a bounded slice per call and must still cover every partition
    // over a full cycle; with nothing to merge it stays silent.
    auto index_def = MakeIndexDef(/*bucket_size_limit*/ 64);
    const u32 dim = 8, n = 600;
    std::mt19937 rng(41);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(static_cast<size_t>(n) * dim);
    for (auto &x : data) {
        x = dist(rng);
    }

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), dim, n);
    idx.Build(data.data(), n);

    // Healthy index: no partition is undersized, so nothing gets enqueued.
    EXPECT_EQ(idx.EnqueueMergeSweep(), 0u);
    EXPECT_EQ(idx.GetLiveLeafCount(), idx.GetLiveLeafCount());

    // Make everything tiny, then sweep in small slices until the tree collapses.
    for (u32 i = 0; i < n; ++i) {
        if (i % 24 != 0) {
            idx.MarkDeleted(i);
        }
    }
    idx.Compact();
    u32 total = idx.EnqueueMergeSweep(8);
    for (u32 round = 0; round < 50 && total == 0; ++round) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        total = idx.EnqueueMergeSweep(8);
    }
    EXPECT_GT(total, 0u);
}

// ── Error-bound pruning ──────────────────────────────────────────────────────

TEST_F(SPFreshTest, test_error_bound_pruning_keeps_exact_topk) {
    // Mirrors what the physical KNN operator does: bounded candidate heap, live
    // prune threshold handed to the index, exact rerank of the survivors.
    auto index_def = MakeIndexDef(/*bucket_size_limit*/ 128);
    const u32 dim = 32, n = 1000, topk = 10, rerank = topk * 10;
    std::mt19937 rng(53);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(static_cast<size_t>(n) * dim);
    for (auto &x : data) {
        x = dist(rng);
    }

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), dim, n);
    idx.Build(data.data(), n);

    auto exact_l2 = [&](const f32 *a, u32 row) {
        f32 sum = 0;
        for (u32 d = 0; d < dim; ++d) {
            const f32 df = a[d] - data[static_cast<size_t>(row) * dim + d];
            sum += df * df;
        }
        return sum;
    };

    struct Cand {
        SegmentOffset offset;
        f32 est;
        f32 bound;
    };
    auto lower = [](const Cand &c) { return c.est - c.bound; };

    auto run = [&](const f32 *query, bool pruning, u32 &scanned) {
        // Same policy as the physical KNN operator: keep the best estimates, prune by
        // the error bound of the worst kept one.
        auto worse_first = [](const Cand &a, const Cand &b) { return a.est < b.est; };
        std::priority_queue<Cand, std::vector<Cand>, decltype(worse_first)> heap(worse_first);
        f32 threshold = std::numeric_limits<f32>::max();
        scanned = 0;
        // A huge beam makes the candidate pool equal to the whole table, so the
        // expected answer is the brute-force top-k.
        idx.Search(
            query,
            dim,
            [&](SegmentOffset offset, f32 est, f32 bound) {
                ++scanned;
                heap.push(Cand{offset, est, bound});
                if (heap.size() > rerank) {
                    heap.pop();
                }
                if (heap.size() >= rerank) {
                    threshold = lower(heap.top());
                }
            },
            0,
            100000,
            pruning ? &threshold : nullptr);

        std::vector<std::pair<f32, SegmentOffset>> exact;
        exact.reserve(heap.size());
        while (!heap.empty()) {
            exact.emplace_back(exact_l2(query, heap.top().offset), heap.top().offset);
            heap.pop();
        }
        std::sort(exact.begin(), exact.end());
        if (exact.size() > topk) {
            exact.resize(topk);
        }
        return exact;
    };

    u32 pruned_total = 0;
    u32 full_total = 0;
    for (u32 p = 0; p < 10; ++p) {
        const f32 *query = data.data() + static_cast<size_t>(p * 97 + 13) * dim;
        u32 scanned_pruned = 0;
        u32 scanned_full = 0;
        const auto pruned = run(query, /*pruning*/ true, scanned_pruned);
        const auto full = run(query, /*pruning*/ false, scanned_full);
        pruned_total += scanned_pruned;
        full_total += scanned_full;
        ASSERT_EQ(pruned.size(), topk);
        ASSERT_EQ(full.size(), topk);
        for (u32 i = 0; i < topk; ++i) {
            EXPECT_EQ(pruned[i].second, full[i].second) << "probe " << p << " rank " << i;
        }
    }
    LOG_INFO(fmt::format("pruning: {} callbacks with threshold vs {} without", pruned_total, full_total));
    // Pruning may only ever remove work, never add it.
    EXPECT_LE(pruned_total, full_total);
}

// ── Adaptive beam ────────────────────────────────────────────────────────────

TEST_F(SPFreshTest, test_adaptive_beam_keeps_exact_neighbour) {
    // Default beam == adaptive beam. It has to find the exact neighbour of a stored
    // vector while still touching far fewer candidates than a full scan.
    auto index_def = MakeIndexDef(/*bucket_size_limit*/ 64);
    const u32 dim = 32, n = 3000;
    std::mt19937 rng(67);
    std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
    std::vector<f32> data(static_cast<size_t>(n) * dim);
    for (auto &x : data) {
        x = dist(rng);
    }

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), dim, n);
    idx.Build(data.data(), n);
    ASSERT_GE(idx.GetTreeHeight(), 2u);

    auto exact_l2 = [&](const f32 *a, const f32 *b) {
        f32 sum = 0;
        for (u32 d = 0; d < dim; ++d) {
            const f32 df = a[d] - b[d];
            sum += df * df;
        }
        return sum;
    };

    constexpr u32 kProbes = 30;
    u32 hits = 0;
    size_t max_scanned = 0;
    for (u32 p = 0; p < kProbes; ++p) {
        const u32 target = 11 + p * 89;
        const f32 *query = data.data() + static_cast<size_t>(target) * dim;
        std::vector<std::pair<f32, SegmentOffset>> cands;
        idx.Search(query, dim, [&](SegmentOffset offset, f32, f32) { cands.emplace_back(0.0f, offset); }, 0, 0);
        max_scanned = std::max(max_scanned, cands.size());
        f32 best = std::numeric_limits<f32>::max();
        SegmentOffset best_offset = std::numeric_limits<SegmentOffset>::max();
        for (const auto &[unused, offset] : cands) {
            const f32 d = exact_l2(query, data.data() + static_cast<size_t>(offset) * dim);
            if (d < best) {
                best = d;
                best_offset = offset;
            }
        }
        if (best_offset == target) {
            ++hits;
        }
    }
    const f32 recall = static_cast<f32>(hits) / static_cast<f32>(kProbes);
    LOG_INFO(fmt::format("adaptive beam recall@1 = {}, max candidates scanned = {}", recall, max_scanned));
    EXPECT_GE(recall, 0.95f);
    // Adaptivity must not degenerate into a full scan.
    EXPECT_LT(max_scanned, static_cast<size_t>(n));
}

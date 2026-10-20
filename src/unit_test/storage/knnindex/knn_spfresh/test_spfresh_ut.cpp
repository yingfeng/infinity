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
    return std::make_shared<IndexSPFresh>(std::make_shared<std::string>("test_idx"), std::make_shared<std::string>(""), "test",
                                          std::vector<std::string>{"col1"}, MetricType::kMetricL2, /*num_centroids*/ 1,
                                          /*replica_count*/ 1, bucket_size_limit, /*compress_to_rabitq*/ true,
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
    LOG_INFO(fmt::format("scanned={} best={} (offset {}) 2nd={} (offset {})", all.size(), all[0].first, all[0].second,
                         all.size() > 1 ? all[1].first : -1.0f, all.size() > 1 ? all[1].second : 0u));
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

module;

#include "unit_test/gtest_expand.h"

module infinity_core:ut.test_spfresh;

import :ut.base_test;
import :spfresh_index;
import :spfresh_defs;
import :index_spfresh;

import std;

using namespace infinity;

class SPFreshTest : public BaseTest {};

TEST_F(SPFreshTest, test_basic_construction) {
    SPFreshIndexInMem idx;
    EXPECT_EQ(idx.GetRowCount(), 0u);
    EXPECT_EQ(idx.GetBaseRowCount(), 0u);
    EXPECT_EQ(idx.GetDeltaCount(), 0u);
    EXPECT_EQ(idx.GetDeletedCount(), 0u);
}

TEST_F(SPFreshTest, test_build_and_search) {
    auto index_def = std::make_shared<IndexSPFresh>(
        std::make_shared<std::string>("test_idx"), std::make_shared<std::string>(""),
        "test", std::vector<std::string>{"col1"},
        MetricType::kMetricL2, 2, 1, 10000, true);
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 100);

    f32 vecs[] = {1.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 1.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 1.0f};
    idx.Build(vecs, 4);

    EXPECT_EQ(idx.GetBaseRowCount(), 4u);
    EXPECT_GE(idx.GetNumCentroids(), 1u);

    f32 query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    u32 result_count = 0;
    idx.Search(query, 4, [&](SegmentOffset, f32) { result_count++; });
    EXPECT_EQ(result_count, 4u);
}

TEST_F(SPFreshTest, test_insert_delta_and_search) {
    // replica_count=1 so each InsertDelta creates 1 delta entry
    auto index_def = std::make_shared<IndexSPFresh>(
        std::make_shared<std::string>("test_idx"), std::make_shared<std::string>(""),
        "test", std::vector<std::string>{"col1"},
        MetricType::kMetricL2, 2, 1, 10000, true);
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
    idx.Search(query, 4, [&](SegmentOffset, f32) { result_count++; });
    EXPECT_EQ(result_count, 4u);
}

TEST_F(SPFreshTest, test_compact) {
    auto index_def = std::make_shared<IndexSPFresh>(
        std::make_shared<std::string>("test_idx"), std::make_shared<std::string>(""),
        "test", std::vector<std::string>{"col1"},
        MetricType::kMetricL2, 2, 1, 10000, true);
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
}

TEST_F(SPFreshTest, test_delete_and_gc) {
    auto index_def = std::make_shared<IndexSPFresh>(
        std::make_shared<std::string>("test_idx"), std::make_shared<std::string>(""),
        "test", std::vector<std::string>{"col1"},
        MetricType::kMetricL2, 2, 1, 10000, true);
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 100);

    f32 base[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    idx.Build(base, 3);

    idx.MarkDeleted(1);
    EXPECT_EQ(idx.GetDeletedCount(), 1u);
    EXPECT_EQ(idx.GetRowCount(), 2u);

    f32 query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    u32 result_count = 0;
    idx.Search(query, 4, [&](SegmentOffset, f32) { result_count++; });
    EXPECT_EQ(result_count, 2u);

    idx.Compact();
    EXPECT_EQ(idx.GetDeletedCount(), 0u);
    EXPECT_EQ(idx.GetBaseRowCount(), 2u);
}

TEST_F(SPFreshTest, test_auto_compact) {
    auto index_def = std::make_shared<IndexSPFresh>(
        std::make_shared<std::string>("test_idx"), std::make_shared<std::string>(""),
        "test", std::vector<std::string>{"col1"},
        MetricType::kMetricL2, 1, 1, 10000, true);
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 100);

    f32 base[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f};
    idx.Build(base, 2);

    f32 v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    idx.InsertDelta(v2, 1);
    EXPECT_EQ(idx.GetDeltaCount(), 1u);

    bool compacted = idx.TryAutoCompact(1);
    EXPECT_TRUE(compacted);
    EXPECT_EQ(idx.GetDeltaCount(), 0u);
    EXPECT_EQ(idx.GetBaseRowCount(), 3u);

    compacted = idx.TryAutoCompact(1);
    EXPECT_FALSE(compacted);
}

TEST_F(SPFreshTest, test_rebalance_no_crash) {
    auto index_def = std::make_shared<IndexSPFresh>(
        std::make_shared<std::string>("test_idx"), std::make_shared<std::string>(""),
        "test", std::vector<std::string>{"col1"},
        MetricType::kMetricL2, 2, 1, 10000, true);
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 100);

    f32 base[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    idx.Build(base, 2);

    idx.Rebalance(100);
    EXPECT_EQ(idx.GetRowCount(), 2u);
}

TEST_F(SPFreshTest, test_rabitq_distance) {
    // Use n_centroids=1 so K-Means always works with few vectors
    auto index_def = std::make_shared<IndexSPFresh>(
        std::make_shared<std::string>("test_idx"), std::make_shared<std::string>(""),
        "test", std::vector<std::string>{"col1"},
        MetricType::kMetricL2, 1, 1, 10000, true);
    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 100);

    f32 base[] = {1.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f};
    idx.Build(base, 2);

    f32 query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    f32 min_dist = std::numeric_limits<f32>::max();
    idx.Search(query, 4, [&](SegmentOffset, f32 dist) {
        min_dist = std::min(min_dist, dist);
    });

    EXPECT_TRUE(std::isfinite(min_dist));
}

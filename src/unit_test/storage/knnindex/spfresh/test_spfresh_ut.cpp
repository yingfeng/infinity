module;

#include "unit_test/gtest_expand.h"

module infinity_core:ut.test_spfresh;

import :ut.base_test;
import :spfresh_index;
import :spfresh_defs;
import :index_spfresh;
import :local_file_handle;

import std;

using namespace infinity;

class SPFreshTest : public BaseTest {};

TEST_F(SPFreshTest, test_basic_construction) {
    // Test: create empty index, verify initial state
    SPFreshIndexInMem idx;
    EXPECT_EQ(idx.GetRowCount(), 0u);
    EXPECT_EQ(idx.GetBaseRowCount(), 0u);
    EXPECT_EQ(idx.GetDeltaCount(), 0u);
}

TEST_F(SPFreshTest, test_insert_delta) {
    // Test: insert vectors into delta
    auto index_name = std::make_shared<std::string>("test_idx");
    auto index_comment = std::make_shared<std::string>("");
    auto index_def = std::make_shared<IndexSPFresh>(index_name, index_comment, "test", std::vector<std::string>{"col1"},
                                                     MetricType::kMetricL2, 10, 8, 10000, true);

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 1000);

    // Insert first vector
    f32 v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    idx.InsertDelta(v1, 0);
    EXPECT_EQ(idx.GetDeltaCount(), 1u);
    EXPECT_EQ(idx.GetBaseRowCount(), 0u);
    EXPECT_EQ(idx.GetRowCount(), 1u);

    // Insert second vector
    f32 v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    idx.InsertDelta(v2, 1);
    EXPECT_EQ(idx.GetDeltaCount(), 2u);
    EXPECT_EQ(idx.GetRowCount(), 2u);
}

TEST_F(SPFreshTest, test_compact) {
    // Test: compact delta into base
    auto index_name = std::make_shared<std::string>("test_idx");
    auto index_comment = std::make_shared<std::string>("");
    auto index_def = std::make_shared<IndexSPFresh>(index_name, index_comment, "test", std::vector<std::string>{"col1"},
                                                     MetricType::kMetricL2, 10, 8, 10000, true);

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 1000);

    f32 v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    f32 v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    idx.InsertDelta(v1, 0);
    idx.InsertDelta(v2, 1);
    EXPECT_EQ(idx.GetDeltaCount(), 2u);

    idx.Compact();
    EXPECT_EQ(idx.GetDeltaCount(), 0u);
    EXPECT_EQ(idx.GetBaseRowCount(), 2u);
    EXPECT_EQ(idx.GetRowCount(), 2u);
}

TEST_F(SPFreshTest, test_search) {
    // Test: search returns approximate distances (not exact, but verifies no crash and returns row count)
    auto index_name = std::make_shared<std::string>("test_idx");
    auto index_comment = std::make_shared<std::string>("");
    auto index_def = std::make_shared<IndexSPFresh>(index_name, index_comment, "test", std::vector<std::string>{"col1"},
                                                     MetricType::kMetricL2, 10, 8, 10000, true);

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 1000);

    // Insert 3 vectors
    f32 v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    f32 v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    f32 v3[] = {0.0f, 0.0f, 1.0f, 0.0f};
    idx.InsertDelta(v1, 0);
    idx.InsertDelta(v2, 1);
    idx.InsertDelta(v3, 2);

    // Compact so we have base vectors too
    idx.Compact();

    // Insert delta after compact
    f32 v4[] = {0.0f, 0.0f, 0.0f, 1.0f};
    idx.InsertDelta(v4, 3);

    // Search
    f32 query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    u32 result_count = 0;
    idx.Search(query, 4, [&](SegmentOffset, f32) {
        result_count++;
    });

    // Should scan all 4 vectors (3 base + 1 delta)
    EXPECT_EQ(result_count, 4u);
}

TEST_F(SPFreshTest, test_delta_and_compact_cycle) {
    // Test: multiple insert/compact cycles
    auto index_name = std::make_shared<std::string>("test_idx");
    auto index_comment = std::make_shared<std::string>("");
    auto index_def = std::make_shared<IndexSPFresh>(index_name, index_comment, "test", std::vector<std::string>{"col1"},
                                                     MetricType::kMetricL2, 10, 8, 10000, true);

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 1000);

    // Cycle 1: insert and compact
    f32 v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    f32 v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    idx.InsertDelta(v1, 0);
    idx.InsertDelta(v2, 1);
    idx.Compact();
    EXPECT_EQ(idx.GetRowCount(), 2u);
    EXPECT_EQ(idx.GetBaseRowCount(), 2u);
    EXPECT_EQ(idx.GetDeltaCount(), 0u);

    // Cycle 2: insert more deltas
    f32 v3[] = {0.0f, 0.0f, 1.0f, 0.0f};
    idx.InsertDelta(v3, 2);
    EXPECT_EQ(idx.GetRowCount(), 3u);
    EXPECT_EQ(idx.GetDeltaCount(), 1u);

    // Search should find all 3
    f32 query[] = {0.0f, 0.0f, 1.0f, 0.0f};
    u32 result_count = 0;
    idx.Search(query, 4, [&](SegmentOffset, f32) { result_count++; });
    EXPECT_EQ(result_count, 3u);

    // Compact again
    idx.Compact();
    EXPECT_EQ(idx.GetRowCount(), 3u);
    EXPECT_EQ(idx.GetBaseRowCount(), 3u);
    EXPECT_EQ(idx.GetDeltaCount(), 0u);
}

TEST_F(SPFreshTest, test_try_auto_compact) {
    // Test: TryAutoCompact works correctly
    auto index_name = std::make_shared<std::string>("test_idx");
    auto index_comment = std::make_shared<std::string>("");
    auto index_def = std::make_shared<IndexSPFresh>(index_name, index_comment, "test", std::vector<std::string>{"col1"},
                                                     MetricType::kMetricL2, 10, 8, 10000, true);

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 1000);

    f32 v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    idx.InsertDelta(v1, 0);
    EXPECT_EQ(idx.GetDeltaCount(), 1u);
    EXPECT_EQ(idx.GetBaseRowCount(), 0u);

    // Manual TryAutoCompact with low threshold
    bool compacted = idx.TryAutoCompact(1);
    EXPECT_TRUE(compacted);
    EXPECT_EQ(idx.GetDeltaCount(), 0u);
    EXPECT_EQ(idx.GetBaseRowCount(), 1u);

    // No delta, no compact
    compacted = idx.TryAutoCompact(1);
    EXPECT_FALSE(compacted);
}

TEST_F(SPFreshTest, test_rebalance) {
    // Test: rebalance without crash
    auto index_name = std::make_shared<std::string>("test_idx");
    auto index_comment = std::make_shared<std::string>("");
    auto index_def = std::make_shared<IndexSPFresh>(index_name, index_comment, "test", std::vector<std::string>{"col1"},
                                                     MetricType::kMetricL2, 10, 8, 10000, true);

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 1000);

    f32 v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    f32 v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    idx.InsertDelta(v1, 0);
    idx.InsertDelta(v2, 1);
    idx.Compact();

    idx.Rebalance(100);
    EXPECT_EQ(idx.GetRowCount(), 2u);
}

TEST_F(SPFreshTest, test_split_bucket) {
    // Test: split bucket
    auto index_name = std::make_shared<std::string>("test_idx");
    auto index_comment = std::make_shared<std::string>("");
    auto index_def = std::make_shared<IndexSPFresh>(index_name, index_comment, "test", std::vector<std::string>{"col1"},
                                                     MetricType::kMetricL2, 10, 8, 10000, true);

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 1000);
    auto ids = idx.SplitBucket(0);
    // Simplified split returns empty for now
    EXPECT_TRUE(ids.empty());
}

TEST_F(SPFreshTest, test_mark_delete) {
    // Test: mark vector as deleted, verify it's excluded from search and GC'd on compact
    auto index_name = std::make_shared<std::string>("test_idx");
    auto index_comment = std::make_shared<std::string>("");
    auto index_def = std::make_shared<IndexSPFresh>(index_name, index_comment, "test", std::vector<std::string>{"col1"},
                                                     MetricType::kMetricL2, 10, 8, 10000, true);

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 1000);

    f32 v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    f32 v2[] = {0.0f, 1.0f, 0.0f, 0.0f};
    f32 v3[] = {0.0f, 0.0f, 1.0f, 0.0f};
    idx.InsertDelta(v1, 0);
    idx.InsertDelta(v2, 1);
    idx.InsertDelta(v3, 2);
    idx.Compact();
    EXPECT_EQ(idx.GetRowCount(), 3u);

    // Mark v2 as deleted
    idx.MarkDeleted(1);
    EXPECT_EQ(idx.GetDeletedCount(), 1u);
    EXPECT_EQ(idx.GetRowCount(), 2u); // 3 - 1 deleted

    // Search should only return 2 results (v1 and v3)
    u32 result_count = 0;
    idx.Search(v1, 4, [&](SegmentOffset, f32) { result_count++; });
    EXPECT_EQ(result_count, 2u);

    // Compact should GC the deleted vector
    idx.Compact();
    EXPECT_EQ(idx.GetDeletedCount(), 0u);
    EXPECT_EQ(idx.GetRowCount(), 2u);
}

TEST_F(SPFreshTest, test_enconde_distance) {
    // Test: RaBitQ encode approximates L2 distance (without rotation, simplified)
    // The simplified RaBitQ (sign-only, no rotation) gives low precision
    // but should still return finite values
    auto index_name = std::make_shared<std::string>("test_idx");
    auto index_comment = std::make_shared<std::string>("");
    auto index_def = std::make_shared<IndexSPFresh>(index_name, index_comment, "test", std::vector<std::string>{"col1"},
                                                     MetricType::kMetricL2, 10, 8, 10000, true);

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), 4, 1000);

    f32 v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    idx.InsertDelta(v1, 0);
    idx.Compact();

    // Query same vector - simplified RaBitQ gives approximate distance
    f32 query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    f32 min_dist = std::numeric_limits<f32>::max();
    idx.Search(query, 4, [&](SegmentOffset, f32 dist) {
        min_dist = std::min(min_dist, dist);
    });

    // Distance should be finite
    EXPECT_TRUE(std::isfinite(min_dist));
}

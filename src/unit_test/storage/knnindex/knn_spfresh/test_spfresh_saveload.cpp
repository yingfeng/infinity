import :spfresh_index;
import :spfresh_defs;
import :index_spfresh;
import :local_file_handle;
import :logger;
import std;

using namespace infinity;

TEST_F(SPFreshTest, debug_save_load_search) {
    std::string tmp_path = "/tmp/spfresh_test.idx";

    auto index_def = std::make_shared<IndexSPFresh>(
        std::make_shared<std::string>("test_idx"), std::make_shared<std::string>(""),
        "test", std::vector<std::string>{"col1"},
        MetricType::kMetricL2, 5, 1, 10000, true, 512);

    u32 dim = 128, n = 100;
    std::vector<f32> data(n * dim);
    std::mt19937 rng(42);
    for (u32 i = 0; i < n * dim; ++i) data[i] = (f32)rng() / (f32)rng.max() * 2.0f - 1.0f;
    for (u32 i = 0; i < n; ++i) {
        f32 *v = data.data() + i * dim;
        f64 sum = 0;
        for (u32 d = 0; d < dim; ++d) sum += static_cast<f64>(v[d]) * v[d];
        if (sum > 1e-30) { f64 inv = 1.0 / std::sqrt(sum); for (u32 d = 0; d < dim; ++d) v[d] *= inv; }
    }

    SPFreshIndexInMem idx(RowID(0, 0), index_def.get(), dim, n);
    idx.Build(data.data(), n);
    EXPECT_EQ(idx.GetBaseRowCount(), n);

    // Save
    {
        auto fh = LocalFileHandle::MakeShared(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        idx.Save(*fh);
        fh->Close();
    }

    // Load
    SPFreshIndexInMem loaded;
    {
        auto fh = LocalFileHandle::MakeShared(tmp_path, O_RDONLY, 0644);
        loaded.Load(*fh, 0);
        fh->Close();
    }
    EXPECT_EQ(loaded.GetBaseRowCount(), n);
    EXPECT_EQ(loaded.GetNumCentroids(), idx.GetNumCentroids());

    // Search on loaded
    f32 *query = data.data();
    u32 cnt = 0;
    loaded.Search(query, dim, [&](SegmentOffset, f32) { cnt++; });
    LOG_INFO(fmt::format("Loaded search returned {} candidates (expected {})", cnt, n));
    EXPECT_EQ(cnt, n) << "Loaded index must return all vectors";

    // Search on original (after TransferTo simulation: save/load doesn't affect original)
    u32 cnt2 = 0;
    idx.Search(query, dim, [&](SegmentOffset, f32) { cnt2++; });
    LOG_INFO(fmt::format("Original search returned {} candidates", cnt2));
    EXPECT_EQ(cnt2, n) << "Original index must also return all vectors";

    std::filesystem::remove(tmp_path);
}

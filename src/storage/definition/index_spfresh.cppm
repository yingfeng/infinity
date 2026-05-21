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

export module infinity_core:index_spfresh;

import :index_base;
import :base_table_ref;

import third_party;

import create_index_info;
import statement_common;

namespace infinity {

export class IndexSPFresh final : public IndexBase {
public:
    static std::shared_ptr<IndexBase> Make(std::shared_ptr<std::string> index_name,
                                           std::shared_ptr<std::string> index_comment,
                                           const std::string &file_name,
                                           std::vector<std::string> column_names,
                                           const std::vector<InitParameter *> &index_param_list);

    IndexSPFresh(std::shared_ptr<std::string> index_name,
                 std::shared_ptr<std::string> index_comment,
                 const std::string &file_name,
                 std::vector<std::string> column_names,
                 MetricType metric_type,
                 u32 num_centroids,
                 u32 replica_count,
                 u32 bucket_size_limit,
                 bool compress_to_rabitq,
                 u32 max_delta_mb)
        : IndexBase(IndexType::kSPFresh, index_name, index_comment, file_name, std::move(column_names)), metric_type_(metric_type),
          num_centroids_(num_centroids), replica_count_(replica_count), bucket_size_limit_(bucket_size_limit),
          compress_to_rabitq_(compress_to_rabitq), max_delta_mb_(max_delta_mb) {}

    ~IndexSPFresh() final = default;

    bool operator==(const IndexSPFresh &other) const;
    bool operator!=(const IndexSPFresh &other) const;

public:
    i32 GetSizeInBytes() const override;
    void WriteAdv(char *&ptr) const override;
    std::string ToString() const override;
    std::string BuildOtherParamsString() const override;
    nlohmann::json Serialize() const override;

public:
    static void ValidateColumnDataType(const std::shared_ptr<BaseTableRef> &base_table_ref,
                                       const std::string &column_name,
                                       const std::vector<InitParameter *> &index_param_list);

public:
    const MetricType metric_type_{MetricType::kInvalid};
    const u32 num_centroids_{1000};
    const u32 replica_count_{8};
    const u32 bucket_size_limit_{10000};
    const bool compress_to_rabitq_{true};
    const u32 max_delta_mb_{512};
};

} // namespace infinity

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

module infinity_core:index_spfresh.impl;

import :index_spfresh;
import :status;
import :index_base;
import :infinity_exception;
import :default_values;

import std;
import third_party;

import data_type;
import embedding_info;
import internal_types;
import logical_type;
import statement_common;
import serialize;

namespace infinity {

std::shared_ptr<IndexBase> IndexSPFresh::Make(std::shared_ptr<std::string> index_name,
                                              std::shared_ptr<std::string> index_comment,
                                              const std::string &file_name,
                                              std::vector<std::string> column_names,
                                              const std::vector<InitParameter *> &index_param_list) {
    MetricType metric_type = MetricType::kInvalid;
    u32 num_centroids = 1000;
    u32 replica_count = 8;
    u32 bucket_size_limit = 10000;
    bool compress_to_rabitq = true;

    for (const auto *param : index_param_list) {
        if (param->param_name_ == "metric") {
            metric_type = StringToMetricType(param->param_value_);
        } else if (param->param_name_ == "num_centroids") {
            num_centroids = std::stoul(param->param_value_);
        } else if (param->param_name_ == "replica_count") {
            replica_count = std::stoul(param->param_value_);
        } else if (param->param_name_ == "bucket_size_limit") {
            bucket_size_limit = std::stoul(param->param_value_);
        } else if (param->param_name_ == "compress_to_rabitq") {
            compress_to_rabitq = (param->param_value_ == "true" || param->param_value_ == "1");
        } else {
            Status status = Status::InvalidIndexParam(param->param_name_);
            RecoverableError(status);
        }
    }

    if (metric_type == MetricType::kInvalid) {
        Status status = Status::InvalidIndexParam("Metric type");
        RecoverableError(status);
    }

    return std::make_shared<IndexSPFresh>(index_name,
                                          index_comment,
                                          file_name,
                                          std::move(column_names),
                                          metric_type,
                                          num_centroids,
                                          replica_count,
                                          bucket_size_limit,
                                          compress_to_rabitq);
}

bool IndexSPFresh::operator==(const IndexSPFresh &other) const {
    return IndexBase::operator==(other) && metric_type_ == other.metric_type_ && num_centroids_ == other.num_centroids_ &&
           replica_count_ == other.replica_count_ && bucket_size_limit_ == other.bucket_size_limit_ &&
           compress_to_rabitq_ == other.compress_to_rabitq_;
}

bool IndexSPFresh::operator!=(const IndexSPFresh &other) const { return !(*this == other); }

i32 IndexSPFresh::GetSizeInBytes() const {
    size_t size = IndexBase::GetSizeInBytes();
    size += sizeof(metric_type_);
    size += sizeof(num_centroids_);
    size += sizeof(replica_count_);
    size += sizeof(bucket_size_limit_);
    size += sizeof(compress_to_rabitq_);
    return i32(size);
}

void IndexSPFresh::WriteAdv(char *&ptr) const {
    IndexBase::WriteAdv(ptr);
    WriteBufAdv(ptr, metric_type_);
    WriteBufAdv(ptr, num_centroids_);
    WriteBufAdv(ptr, replica_count_);
    WriteBufAdv(ptr, bucket_size_limit_);
    WriteBufAdv(ptr, compress_to_rabitq_);
}

std::string IndexSPFresh::ToString() const {
    std::stringstream ss;
    ss << IndexBase::ToString() << ", " << MetricTypeToString(metric_type_)
       << ", num_centroids=" << num_centroids_ << ", replica_count=" << replica_count_
       << ", bucket_size_limit=" << bucket_size_limit_ << ", rabitq=" << (compress_to_rabitq_ ? "true" : "false");
    return ss.str();
}

std::string IndexSPFresh::BuildOtherParamsString() const {
    std::stringstream ss;
    ss << "metric = " << MetricTypeToString(metric_type_)
       << ", num_centroids = " << num_centroids_
       << ", replica_count = " << replica_count_
       << ", bucket_size_limit = " << bucket_size_limit_
       << ", compress_to_rabitq = " << (compress_to_rabitq_ ? "true" : "false");
    return ss.str();
}

nlohmann::json IndexSPFresh::Serialize() const {
    nlohmann::json res = IndexBase::Serialize();
    res["metric_type"] = MetricTypeToString(metric_type_);
    res["num_centroids"] = num_centroids_;
    res["replica_count"] = replica_count_;
    res["bucket_size_limit"] = bucket_size_limit_;
    res["compress_to_rabitq"] = compress_to_rabitq_;
    return res;
}

void IndexSPFresh::ValidateColumnDataType(const std::shared_ptr<BaseTableRef> &base_table_ref,
                                          const std::string &column_name,
                                          const std::vector<InitParameter *> &index_param_list) {
    const auto &column_names_vector = *(base_table_ref->column_names_);
    const auto &column_types_vector = *(base_table_ref->column_types_);
    const size_t column_id = std::find(column_names_vector.begin(), column_names_vector.end(), column_name) - column_names_vector.begin();
    if (column_id == column_names_vector.size()) {
        RecoverableError(Status::ColumnNotExist(column_name));
    }
    const DataType *data_type_ptr = column_types_vector[column_id].get();
    switch (data_type_ptr->type()) {
        case LogicalType::kEmbedding:
        case LogicalType::kMultiVector: {
            break;
        }
        default: {
            RecoverableError(Status::InvalidIndexDefinition(
                fmt::format("Attempt to create SPFresh index on column: {}, data type: {}.", column_name, data_type_ptr->ToString())));
        }
    }
    const auto embedding_info = dynamic_cast<const EmbeddingInfo *>(data_type_ptr->type_info().get());
    const EmbeddingDataType embedding_data_type = embedding_info->Type();

    // Check if RaBitQ compression is requested
    bool use_rabitq = true;
    for (const auto *param : index_param_list) {
        if (param->param_name_ == "compress_to_rabitq") {
            use_rabitq = (param->param_value_ == "true" || param->param_value_ == "1");
        }
    }
    if (use_rabitq) {
        if (embedding_data_type != EmbeddingDataType::kElemFloat) {
            RecoverableError(Status::InvalidIndexDefinition(
                fmt::format("SPFresh with RaBitQ compression only supports float element type.")));
        }
    }
}

} // namespace infinity

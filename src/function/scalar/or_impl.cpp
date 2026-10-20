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

module infinity_core:or_func.impl;

import :or_func;
import :new_catalog;
import :infinity_exception;
import :scalar_function;
import :scalar_function_set;
import :column_vector;
import :data_block;
import :null_value;
import :status;

import std;

import logical_type;
import internal_types;
import data_type;

namespace infinity {

// Three-valued OR logic:
//   NULL OR true   → true   (not null)
//   true  OR NULL  → true   (not null)
//   NULL OR false  → NULL
//   false OR NULL  → NULL
//   NULL OR NULL   → NULL
static void OrFunctionNullAware(const DataBlock &input, std::shared_ptr<ColumnVector> &output) {
    if (input.column_count() != 2) {
        UnrecoverableError("OR function requires exactly 2 arguments.");
    }
    size_t row_count = input.row_count();
    const auto &left = input.column_vectors_[0];
    const auto &right = input.column_vectors_[1];

    if (output.get() == left.get() || output.get() == right.get()) {
        UnrecoverableError("OR function: output should not be same as input.");
    }

    auto is_null = [](const auto &cv, size_t i) -> bool {
        if (cv->vector_type() == ColumnVectorType::kConstant) {
            return !cv->nulls_ptr_->IsTrue(0);
        }
        return !cv->nulls_ptr_->IsTrue(i);
    };

    auto get_bool = [](const auto &cv, size_t i) -> BooleanT {
        size_t idx = (cv->vector_type() == ColumnVectorType::kConstant) ? 0 : i;
        return cv->buffer_->GetCompactBit(idx);
    };

    auto &result_null = output->nulls_ptr_;
    result_null->SetAllTrue();
    for (size_t i = 0; i < row_count; ++i) {
        bool ln = is_null(left, i);
        bool rn = is_null(right, i);
        if (!ln && !rn) {
            output->buffer_->SetCompactBit(i, get_bool(left, i) || get_bool(right, i));
        } else if (ln && !rn && get_bool(right, i)) {
            output->buffer_->SetCompactBit(i, true);
        } else if (!ln && rn && get_bool(left, i)) {
            output->buffer_->SetCompactBit(i, true);
        } else {
            result_null->SetFalse(i);
            output->buffer_->SetCompactBit(i, NullValue<BooleanT>());
        }
    }
    output->Finalize(row_count);
}

static void GenerateOrFunction(std::shared_ptr<ScalarFunctionSet> &function_set_ptr) {
    std::string func_name = "OR";
    ScalarFunction or_function(func_name,
                               {DataType(LogicalType::kBoolean), DataType(LogicalType::kBoolean)},
                               {DataType(LogicalType::kBoolean)},
                               OrFunctionNullAware);
    function_set_ptr->AddFunction(or_function);
}

void RegisterOrFunction(NewCatalog *catalog_ptr) {
    std::string func_name = "OR";

    std::shared_ptr<ScalarFunctionSet> function_set_ptr = std::make_shared<ScalarFunctionSet>(func_name);

    GenerateOrFunction(function_set_ptr);

    NewCatalog::AddFunctionSet(catalog_ptr, function_set_ptr);
}

} // namespace infinity
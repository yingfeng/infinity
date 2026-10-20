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

module;

#include "unit_test/gtest_expand.h"

module infinity_core:ut.or_functions;

import :ut.base_test;
import :infinity_exception;
import third_party;
import :logger;
import :infinity_context;
import :new_catalog;
import :scalar_function;
import :scalar_function_set;
import :function_set;
import :function;
import :column_expression;
import :value;
import :default_values;
import :data_block;
import :base_expression;
import :column_vector;
import :or_func;
import :config;
import :status;
import :kv_store;

import global_resource_usage;
import logical_type;
import internal_types;
import data_type;
import :null_value;

using namespace infinity;
class OrFunctionsTest : public BaseTest {};

TEST_F(OrFunctionsTest, or_func) {
    using namespace infinity;

    std::unique_ptr<Config> config_ptr = std::make_unique<Config>();
    Status status = config_ptr->Init(nullptr, nullptr);
    EXPECT_TRUE(status.ok());
    std::unique_ptr<KVStore> kv_store_ptr = std::make_unique<KVStore>();
    status = kv_store_ptr->Init(config_ptr->CatalogDir());
    EXPECT_TRUE(status.ok());
    std::unique_ptr<NewCatalog> catalog_ptr = std::make_unique<NewCatalog>(kv_store_ptr.get());

    RegisterOrFunction(catalog_ptr.get());

    std::string op = "or";
    std::shared_ptr<FunctionSet> function_set = NewCatalog::GetFunctionSetByName(catalog_ptr.get(), op);
    EXPECT_EQ(function_set->type_, FunctionType::kScalar);
    std::shared_ptr<ScalarFunctionSet> scalar_function_set = std::static_pointer_cast<ScalarFunctionSet>(function_set);

    {
        std::vector<std::shared_ptr<BaseExpression>> inputs;

        std::shared_ptr<DataType> data_type = std::make_shared<DataType>(LogicalType::kBoolean);
        std::shared_ptr<DataType> result_type = std::make_shared<DataType>(LogicalType::kBoolean);
        std::shared_ptr<ColumnExpression> col1_expr_ptr = std::make_shared<ColumnExpression>(*data_type, "t1", 1, "c1", 0, 0);
        std::shared_ptr<ColumnExpression> col2_expr_ptr = std::make_shared<ColumnExpression>(*data_type, "t1", 1, "c2", 1, 0);

        inputs.emplace_back(col1_expr_ptr);
        inputs.emplace_back(col2_expr_ptr);

        ScalarFunction func = scalar_function_set->GetMostMatchFunction(inputs);
        EXPECT_STREQ("OR(Boolean, Boolean)->Boolean", func.ToString().c_str());

        std::vector<std::shared_ptr<DataType>> column_types;
        column_types.emplace_back(data_type);
        column_types.emplace_back(data_type);

        size_t row_count = DEFAULT_VECTOR_SIZE;

        DataBlock data_block;
        data_block.Init(column_types);

        for (size_t i = 0; i < row_count; ++i) {
            if (i % 2 == 0) {
                data_block.AppendValue(0, Value::MakeBool(true));
                data_block.AppendValue(1, Value::MakeBool(false));
            } else {
                data_block.AppendValue(0, Value::MakeBool(false));
                data_block.AppendValue(1, Value::MakeBool(true));
            }
        }
        data_block.Finalize();

        for (size_t i = 0; i < row_count; ++i) {
            Value v1 = data_block.GetValue(0, i);
            Value v2 = data_block.GetValue(1, i);
            EXPECT_EQ(v1.type_.type(), LogicalType::kBoolean);
            EXPECT_EQ(v2.type_.type(), LogicalType::kBoolean);
            if (i % 2 == 0) {
                EXPECT_EQ(v1.value_.boolean, true);
                EXPECT_EQ(v2.value_.boolean, false);
            } else {
                EXPECT_EQ(v1.value_.boolean, false);
                EXPECT_EQ(v2.value_.boolean, true);
            }
        }

        std::shared_ptr<ColumnVector> result = std::make_shared<ColumnVector>(result_type);
        result->Initialize();
        func.function_(data_block, result);

        for (size_t i = 0; i < row_count; ++i) {
            Value v = result->GetValueByIndex(i);
            EXPECT_EQ(v.type_.type(), LogicalType::kBoolean);
            EXPECT_EQ(v.value_.boolean, true);
        }
    }
}

TEST_F(OrFunctionsTest, or_func_nullable_flat) {
    using namespace infinity;

    std::unique_ptr<Config> config_ptr = std::make_unique<Config>();
    Status status = config_ptr->Init(nullptr, nullptr);
    EXPECT_TRUE(status.ok());
    std::unique_ptr<KVStore> kv_store_ptr = std::make_unique<KVStore>();
    status = kv_store_ptr->Init(config_ptr->CatalogDir());
    EXPECT_TRUE(status.ok());
    std::unique_ptr<NewCatalog> catalog_ptr = std::make_unique<NewCatalog>(kv_store_ptr.get());

    RegisterOrFunction(catalog_ptr.get());

    std::string op = "or";
    std::shared_ptr<FunctionSet> function_set = NewCatalog::GetFunctionSetByName(catalog_ptr.get(), op);
    EXPECT_EQ(function_set->type_, FunctionType::kScalar);
    std::shared_ptr<ScalarFunctionSet> scalar_function_set = std::static_pointer_cast<ScalarFunctionSet>(function_set);

    std::vector<std::shared_ptr<BaseExpression>> inputs;
    std::shared_ptr<DataType> data_type = std::make_shared<DataType>(LogicalType::kBoolean);
    std::shared_ptr<DataType> result_type = std::make_shared<DataType>(LogicalType::kBoolean);
    std::shared_ptr<ColumnExpression> col1_expr_ptr = std::make_shared<ColumnExpression>(*data_type, "t1", 1, "c1", 0, 0);
    std::shared_ptr<ColumnExpression> col2_expr_ptr = std::make_shared<ColumnExpression>(*data_type, "t1", 1, "c2", 1, 0);
    inputs.emplace_back(col1_expr_ptr);
    inputs.emplace_back(col2_expr_ptr);

    ScalarFunction func = scalar_function_set->GetMostMatchFunction(inputs);
    EXPECT_STREQ("OR(Boolean, Boolean)->Boolean", func.ToString().c_str());

    // Both flat vectors with mixed NULL patterns
    // Row:  0      1      2      3      4      5      6      7
    // left:  true,  false, true,  false, NULL,  NULL,  true,  false
    // right: false, true,  false, NULL,  true,  false, NULL,  NULL
    // OR:    true,  true,  true,  NULL,  true,  NULL,  true,  NULL
    {
        const size_t row_count = 8;
        const bool left_vals[] = {true, false, true, false, true, true, true, false};
        const bool left_nulls[] = {false, false, false, false, true, true, false, false};
        const bool right_vals[] = {false, true, false, false, true, false, false, false};
        const bool right_nulls[] = {false, false, false, true, false, false, true, true};
        const bool expected_vals[] = {true, true, true, false, true, false, true, false};
        const bool expected_nulls[] = {false, false, false, true, false, true, false, true};

        std::vector<std::shared_ptr<DataType>> column_types;
        column_types.emplace_back(data_type);
        column_types.emplace_back(data_type);

        DataBlock data_block;
        data_block.Init(column_types);

        for (size_t i = 0; i < row_count; ++i) {
            data_block.AppendValue(0, Value::MakeBool(left_vals[i]));
            data_block.AppendValue(1, Value::MakeBool(right_vals[i]));
        }
        data_block.Finalize();

        for (size_t i = 0; i < row_count; ++i) {
            if (left_nulls[i]) {
                data_block.column_vectors_[0]->nulls_ptr_->SetFalse(i);
            }
            if (right_nulls[i]) {
                data_block.column_vectors_[1]->nulls_ptr_->SetFalse(i);
            }
        }

        std::shared_ptr<ColumnVector> result = std::make_shared<ColumnVector>(result_type);
        result->Initialize();
        func.function_(data_block, result);

        for (size_t i = 0; i < row_count; ++i) {
            SCOPED_TRACE(fmt::format("Row {}", i));
            bool is_not_null = result->nulls_ptr_->IsTrue(i);
            EXPECT_EQ(is_not_null, !expected_nulls[i]);
            if (!expected_nulls[i]) {
                Value v = result->GetValueByIndex(i);
                EXPECT_EQ(v.type_.type(), LogicalType::kBoolean);
                EXPECT_EQ(v.value_.boolean, expected_vals[i]);
            }
        }
    }
}

TEST_F(OrFunctionsTest, or_func_nullable_constant) {
    using namespace infinity;

    std::unique_ptr<Config> config_ptr = std::make_unique<Config>();
    Status status = config_ptr->Init(nullptr, nullptr);
    EXPECT_TRUE(status.ok());
    std::unique_ptr<KVStore> kv_store_ptr = std::make_unique<KVStore>();
    status = kv_store_ptr->Init(config_ptr->CatalogDir());
    EXPECT_TRUE(status.ok());
    std::unique_ptr<NewCatalog> catalog_ptr = std::make_unique<NewCatalog>(kv_store_ptr.get());

    RegisterOrFunction(catalog_ptr.get());

    std::string op = "or";
    std::shared_ptr<FunctionSet> function_set = NewCatalog::GetFunctionSetByName(catalog_ptr.get(), op);
    EXPECT_EQ(function_set->type_, FunctionType::kScalar);
    std::shared_ptr<ScalarFunctionSet> scalar_function_set = std::static_pointer_cast<ScalarFunctionSet>(function_set);

    std::vector<std::shared_ptr<BaseExpression>> inputs;
    std::shared_ptr<DataType> data_type = std::make_shared<DataType>(LogicalType::kBoolean);
    std::shared_ptr<DataType> result_type = std::make_shared<DataType>(LogicalType::kBoolean);
    std::shared_ptr<ColumnExpression> col1_expr_ptr = std::make_shared<ColumnExpression>(*data_type, "t1", 1, "c1", 0, 0);
    std::shared_ptr<ColumnExpression> col2_expr_ptr = std::make_shared<ColumnExpression>(*data_type, "t1", 1, "c2", 1, 0);
    inputs.emplace_back(col1_expr_ptr);
    inputs.emplace_back(col2_expr_ptr);

    ScalarFunction func = scalar_function_set->GetMostMatchFunction(inputs);
    EXPECT_STREQ("OR(Boolean, Boolean)->Boolean", func.ToString().c_str());

    // Helper: create a constant boolean ColumnVector
    auto make_const_bool = [&](bool val, bool is_null) -> std::shared_ptr<ColumnVector> {
        auto cv = ColumnVector::Make(data_type);
        cv->Initialize(ColumnVectorType::kConstant, 1);
        cv->buffer_->SetCompactBit(0, val);
        if (is_null) {
            cv->nulls_ptr_->SetFalse(0);
        }
        return cv;
    };

    // Helper: create a flat boolean ColumnVector with mixed values and nulls
    auto make_flat_bool = [&](const std::vector<bool> &vals, const std::vector<bool> &nulls) -> std::shared_ptr<ColumnVector> {
        auto cv = ColumnVector::Make(data_type);
        cv->Initialize(ColumnVectorType::kCompactBit, vals.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            cv->AppendValue(Value::MakeBool(vals[i]));
            if (nulls[i]) {
                cv->nulls_ptr_->SetFalse(i);
            }
        }
        return cv;
    };

    // Test 1: Left constant NULL, right flat [true, false, NULL]
    // NULL OR true = true, NULL OR false = NULL, NULL OR NULL = NULL
    {
        SCOPED_TRACE("const NULL OR flat");
        auto left = make_const_bool(true, true); // value doesn't matter, is null
        auto right = make_flat_bool({true, false, false}, {false, false, true});
        DataBlock data_block;
        data_block.Init({left, right});

        std::shared_ptr<ColumnVector> result = std::make_shared<ColumnVector>(result_type);
        result->Initialize();
        func.function_(data_block, result);

        // Row 0: NULL OR true = true
        EXPECT_TRUE(result->nulls_ptr_->IsTrue(0));
        EXPECT_EQ(result->GetValueByIndex(0).value_.boolean, true);
        // Row 1: NULL OR false = NULL
        EXPECT_FALSE(result->nulls_ptr_->IsTrue(1));
        // Row 2: NULL OR NULL = NULL
        EXPECT_FALSE(result->nulls_ptr_->IsTrue(2));
    }

    // Test 2: Left constant true, right flat [true, false, NULL]
    // true OR true = true, true OR false = true, true OR NULL = true
    {
        SCOPED_TRACE("const true OR flat");
        auto left = make_const_bool(true, false);
        auto right = make_flat_bool({true, false, false}, {false, false, true});
        DataBlock data_block;
        data_block.Init({left, right});

        std::shared_ptr<ColumnVector> result = std::make_shared<ColumnVector>(result_type);
        result->Initialize();
        func.function_(data_block, result);

        for (size_t i = 0; i < 3; ++i) {
            EXPECT_TRUE(result->nulls_ptr_->IsTrue(i));
            Value v = result->GetValueByIndex(i);
            EXPECT_EQ(v.type_.type(), LogicalType::kBoolean);
            EXPECT_EQ(v.value_.boolean, true);
        }
    }

    // Test 3: Left constant false, right flat [true, false, NULL]
    // false OR true = true, false OR false = false, false OR NULL = NULL
    {
        SCOPED_TRACE("const false OR flat");
        auto left = make_const_bool(false, false);
        auto right = make_flat_bool({true, false, false}, {false, false, true});
        DataBlock data_block;
        data_block.Init({left, right});

        std::shared_ptr<ColumnVector> result = std::make_shared<ColumnVector>(result_type);
        result->Initialize();
        func.function_(data_block, result);

        // Row 0: false OR true = true
        EXPECT_TRUE(result->nulls_ptr_->IsTrue(0));
        EXPECT_EQ(result->GetValueByIndex(0).value_.boolean, true);
        // Row 1: false OR false = false
        EXPECT_TRUE(result->nulls_ptr_->IsTrue(1));
        EXPECT_EQ(result->GetValueByIndex(1).value_.boolean, false);
        // Row 2: false OR NULL = NULL
        EXPECT_FALSE(result->nulls_ptr_->IsTrue(2));
    }

    // Test 4: Left flat [true, false, NULL], right constant NULL
    // true OR NULL = true, false OR NULL = NULL, NULL OR NULL = NULL
    {
        SCOPED_TRACE("flat OR const NULL");
        auto left = make_flat_bool({true, false, false}, {false, false, true});
        auto right = make_const_bool(true, true); // value doesn't matter, is null
        DataBlock data_block;
        data_block.Init({left, right});

        std::shared_ptr<ColumnVector> result = std::make_shared<ColumnVector>(result_type);
        result->Initialize();
        func.function_(data_block, result);

        // Row 0: true OR NULL = true
        EXPECT_TRUE(result->nulls_ptr_->IsTrue(0));
        EXPECT_EQ(result->GetValueByIndex(0).value_.boolean, true);
        // Row 1: false OR NULL = NULL
        EXPECT_FALSE(result->nulls_ptr_->IsTrue(1));
        // Row 2: NULL OR NULL = NULL
        EXPECT_FALSE(result->nulls_ptr_->IsTrue(2));
    }

    // Test 5: Left flat [true, false, NULL], right constant true
    // true OR true = true, false OR true = true, NULL OR true = true
    {
        SCOPED_TRACE("flat OR const true");
        auto left = make_flat_bool({true, false, false}, {false, false, true});
        auto right = make_const_bool(true, false);
        DataBlock data_block;
        data_block.Init({left, right});

        std::shared_ptr<ColumnVector> result = std::make_shared<ColumnVector>(result_type);
        result->Initialize();
        func.function_(data_block, result);

        for (size_t i = 0; i < 3; ++i) {
            EXPECT_TRUE(result->nulls_ptr_->IsTrue(i));
            Value v = result->GetValueByIndex(i);
            EXPECT_EQ(v.type_.type(), LogicalType::kBoolean);
            EXPECT_EQ(v.value_.boolean, true);
        }
    }

    // Test 6: Left flat [true, false, NULL], right constant false
    // true OR false = true, false OR false = false, NULL OR false = NULL
    {
        SCOPED_TRACE("flat OR const false");
        auto left = make_flat_bool({true, false, false}, {false, false, true});
        auto right = make_const_bool(false, false);
        DataBlock data_block;
        data_block.Init({left, right});

        std::shared_ptr<ColumnVector> result = std::make_shared<ColumnVector>(result_type);
        result->Initialize();
        func.function_(data_block, result);

        // Row 0: true OR false = true
        EXPECT_TRUE(result->nulls_ptr_->IsTrue(0));
        EXPECT_EQ(result->GetValueByIndex(0).value_.boolean, true);
        // Row 1: false OR false = false
        EXPECT_TRUE(result->nulls_ptr_->IsTrue(1));
        EXPECT_EQ(result->GetValueByIndex(1).value_.boolean, false);
        // Row 2: NULL OR false = NULL
        EXPECT_FALSE(result->nulls_ptr_->IsTrue(2));
    }
}

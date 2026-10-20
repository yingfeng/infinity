module infinity_core:is_null.impl;

import :is_null;
import :new_catalog;
import :status;
import :infinity_exception;
import :scalar_function;
import :scalar_function_set;

import std.compat;

import logical_type;
import internal_types;
import data_type;
import :column_vector;
import :data_block;
import :vector_buffer;

namespace infinity {

// IS NULL / IS NOT NULL are special: they must *detect* null rather than
// propagate it. A normal scalar function treats a null input row as a null
// output (and may skip Run for it), which is the opposite of what we want.
// So we implement a custom executor that reads the input column's null bitmap
// directly and writes a boolean result that is itself never null.
struct IsNullFunction {
    static void IsNullExecute(const DataBlock &input, std::shared_ptr<ColumnVector> &output) {
        const ColumnVector &input_cv = *input.column_vectors_[0];
        bool is_constant = (input_cv.vector_type() == ColumnVectorType::kConstant);
        bool base_is_null = !input_cv.nulls_ptr_->IsTrue(0);
        size_t row_count = input.row_count();
        for (size_t idx = 0; idx < row_count; ++idx) {
            bool row_is_null = is_constant ? base_is_null : !input_cv.nulls_ptr_->IsTrue(idx);
            output->buffer_->SetCompactBit(idx, row_is_null);
        }
        output->Finalize(row_count);
    }

    static void IsNotNullExecute(const DataBlock &input, std::shared_ptr<ColumnVector> &output) {
        const ColumnVector &input_cv = *input.column_vectors_[0];
        bool is_constant = (input_cv.vector_type() == ColumnVectorType::kConstant);
        bool base_is_null = !input_cv.nulls_ptr_->IsTrue(0);
        size_t row_count = input.row_count();
        for (size_t idx = 0; idx < row_count; ++idx) {
            bool row_is_null = is_constant ? base_is_null : !input_cv.nulls_ptr_->IsTrue(idx);
            output->buffer_->SetCompactBit(idx, !row_is_null);
        }
        output->Finalize(row_count);
    }
};

void RegisterIsNullFunction(NewCatalog *catalog_ptr) {
    // Register for the common, non-parametric logical types. Parametric types
    // (embedding / tensor / sparse / array / tuple ...) are intentionally
    // omitted: a single overload cannot cover every dimension/element type, and
    // IS NULL on those is an uncommon case.
    const std::vector<LogicalType> arg_types = {
        LogicalType::kBoolean, LogicalType::kTinyInt, LogicalType::kSmallInt, LogicalType::kInteger,   LogicalType::kBigInt,   LogicalType::kHugeInt,
        LogicalType::kFloat,   LogicalType::kDouble,  LogicalType::kFloat16,  LogicalType::kBFloat16,  LogicalType::kVarchar,  LogicalType::kJson,
        LogicalType::kDate,    LogicalType::kTime,    LogicalType::kDateTime, LogicalType::kTimestamp, LogicalType::kInterval, LogicalType::kPoint,
        LogicalType::kLine,    LogicalType::kLineSeg, LogicalType::kBox,      LogicalType::kCircle,    LogicalType::kUuid,     LogicalType::kRowID,
    };

    auto make_set = [&](const std::string &name, void (*exec)(const DataBlock &, std::shared_ptr<ColumnVector> &)) {
        auto set_ptr = std::make_shared<ScalarFunctionSet>(name);
        for (LogicalType arg_type : arg_types) {
            ScalarFunction func(name, {DataType(arg_type)}, DataType(LogicalType::kBoolean), exec);
            set_ptr->AddFunction(func);
        }
        NewCatalog::AddFunctionSet(catalog_ptr, set_ptr);
    };

    make_set("is_null", &IsNullFunction::IsNullExecute);
    make_set("is_not_null", &IsNullFunction::IsNotNullExecute);
}

} // namespace infinity

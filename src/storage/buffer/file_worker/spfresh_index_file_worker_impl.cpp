// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.

module infinity_core:spfresh_index_file_worker.impl;

import :spfresh_index_file_worker;
import :spfresh_index;
import :local_file_handle;
import :infinity_exception;

import std;

namespace infinity {

SPFreshIndexFileWorker::~SPFreshIndexFileWorker() {
    if (data_ != nullptr) {
        FreeInMemory();
        data_ = nullptr;
    }
}

void SPFreshIndexFileWorker::AllocateInMemory() {
    if (data_ != nullptr) {
        UnrecoverableError("AllocateInMemory: Already allocated.");
    }
    data_ = new SPFreshIndexInMem();
}

void SPFreshIndexFileWorker::FreeInMemory() {
    if (data_ != nullptr) {
        auto *index = static_cast<SPFreshIndexInMem *>(data_);
        delete index;
        data_ = nullptr;
    }
}

bool SPFreshIndexFileWorker::WriteToFileImpl(bool to_spill, bool &prepare_success, const FileWorkerSaveCtx &ctx) {
    if (data_ == nullptr) {
        UnrecoverableError("WriteToFileImpl: Data is not allocated.");
    }
    auto *index = static_cast<SPFreshIndexInMem *>(data_);
    index->Save(*file_handle_);
    prepare_success = true;
    return true;
}

void SPFreshIndexFileWorker::ReadFromFileImpl(size_t file_size, bool from_spill) {
    if (data_ == nullptr) {
        data_ = new SPFreshIndexInMem();
    }
    auto *index = static_cast<SPFreshIndexInMem *>(data_);
    index->Load(*file_handle_, file_size);
}

} // namespace infinity

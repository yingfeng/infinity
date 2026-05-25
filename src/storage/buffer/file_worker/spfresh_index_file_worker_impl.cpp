// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.

module infinity_core:spfresh_index_file_worker.impl;

import :spfresh_index_file_worker;
import :spfresh_index;
import :spfresh_defs;
import :local_file_handle;
import :infinity_exception;
import :logger;

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
        UnrecoverableError("SPFreshIndexFileWorker::AllocateInMemory: Already allocated.");
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
        UnrecoverableError("SPFreshIndexFileWorker::WriteToFileImpl: Data is not allocated.");
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

bool SPFreshIndexFileWorker::ReadFromMmapImpl(const void *ptr, size_t size) {
    if (mmap_data_ != nullptr) {
        UnrecoverableError("SPFreshIndexFileWorker::ReadFromMmapImpl: Mmap data is already allocated.");
    }
    if (data_ == nullptr) {
        data_ = new SPFreshIndexInMem();
    }
    auto *index = static_cast<SPFreshIndexInMem *>(data_);
    index->LoadFromMmap(static_cast<const u8 *>(ptr), size);
    mmap_data_ = reinterpret_cast<u8 *>(data_);
    return true;
}

void SPFreshIndexFileWorker::FreeFromMmapImpl() {
    if (mmap_data_ == nullptr) {
        UnrecoverableError("SPFreshIndexFileWorker::FreeFromMmapImpl: Mmap data is not allocated.");
    }
    auto *index = reinterpret_cast<SPFreshIndexInMem *>(mmap_data_);
    delete index;
    mmap_data_ = nullptr;
    data_ = nullptr;
}

size_t SPFreshIndexFileWorker::GetMemoryCost() const {
    if (data_ == nullptr && mmap_data_ == nullptr)
        return 0;
    auto *index = data_ != nullptr ? static_cast<SPFreshIndexInMem *>(data_)
                                   : reinterpret_cast<SPFreshIndexInMem *>(mmap_data_);
    // In mmap mode, memory cost is minimal (just centroid data in DRAM)
    // In owned mode, memory cost includes all bucket data
    size_t cost = index->GetNumCentroids() * index->dim() * sizeof(f32); // centroids
    cost += index->GetNumCentroids() * sizeof(SPFreshBucketMeta); // metadatas
    cost += index->GetNumCentroids() * sizeof(SPFreshRunningMean); // running means (rough)
    // delta buffers
    cost += index->GetDeltaCount() * (sizeof(u32) * 2 + SPFreshIndexInMem::RabitQVecSize(index->dim()));
    return cost;
}

} // namespace infinity

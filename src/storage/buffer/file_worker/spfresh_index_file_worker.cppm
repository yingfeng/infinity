// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.

module;

export module infinity_core:spfresh_index_file_worker;

import :index_file_worker;
import :file_worker_type;

import std.compat;

import column_def;

namespace infinity {

class SPFreshIndexInMem;

export class SPFreshIndexFileWorker final : public IndexFileWorker {
public:
    SPFreshIndexFileWorker(std::shared_ptr<std::string> data_dir,
                           std::shared_ptr<std::string> temp_dir,
                           std::shared_ptr<std::string> file_dir,
                           std::shared_ptr<std::string> file_name,
                           std::shared_ptr<IndexBase> index_base,
                           std::shared_ptr<ColumnDef> column_def,
                           PersistenceManager *persistence_manager)
        : IndexFileWorker(std::move(data_dir), std::move(temp_dir), std::move(file_dir), std::move(file_name), std::move(index_base),
                          std::move(column_def), persistence_manager) {}

    ~SPFreshIndexFileWorker() override;
    void AllocateInMemory() override;
    void FreeInMemory() override;
    FileWorkerType Type() const override { return FileWorkerType::kSPFreshIndexFile; }
    size_t GetMemoryCost() const override;

protected:
    bool WriteToFileImpl(bool to_spill, bool &prepare_success, const FileWorkerSaveCtx &ctx) override;
    void ReadFromFileImpl(size_t file_size, bool from_spill) override;
    bool ReadFromMmapImpl(const void *ptr, size_t size) override;
    void FreeFromMmapImpl() override;
};

} // namespace infinity

// Copyright(C) 2023 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

module infinity_core:spfresh_index.impl;

import :spfresh_index;
import :spfresh_defs;
import :index_spfresh;
import :kmeans_partition;
import :infinity_exception;
import :local_file_handle;
import :logger;

import std;

import internal_types;

namespace infinity {

// ---------- Construction / Destruction ----------

SPFreshIndexInMem::SPFreshIndexInMem(RowID begin_row_id, const IndexSPFresh *index_def, u32 embedding_dim, u32 max_vectors)
    : begin_row_id_(begin_row_id), rabitq_meta_(nullptr), rabitq_data_(nullptr),
      num_vectors_(0), max_vectors_(max_vectors), dim_(embedding_dim),
      delta_compacted_seq_(0), num_centroids_(index_def ? index_def->num_centroids_ : 1000) {

    size_t code_size = sizeof(RabitQVec) + dim_ / 8;
    size_t data_size = static_cast<size_t>(max_vectors) * code_size;
    rabitq_data_ = new char[data_size];
    std::memset(rabitq_data_, 0, data_size);
    mem_used_ += data_size;

    centroids_.resize(static_cast<size_t>(num_centroids_) * dim_);
    bucket_metas_.resize(num_centroids_);
    bucket_assignments_.resize(max_vectors_ * index_def->replica_count_, u32(-1));
}

SPFreshIndexInMem::~SPFreshIndexInMem() {
    delete[] reinterpret_cast<char *>(rabitq_meta_);
    delete[] reinterpret_cast<char *>(rabitq_data_);
}

// ---------- RaBitQ Encode ----------

size_t SPFreshIndexInMem::Encode(f32 *code_buf, const f32 *vec, u32 dim, f32 *out_norm) {
    auto *code_ptr = reinterpret_cast<RabitQVec *>(code_buf);
    std::memset(code_buf, 0, sizeof(RabitQVec) + dim / 8);

    f32 raw_norm = 0;
    for (u32 d = 0; d < dim; ++d) raw_norm += vec[d] * vec[d];
    code_ptr->raw_norm_ = raw_norm;
    code_ptr->norm_ = std::sqrt(raw_norm);
    code_ptr->sum_ = 0;
    code_ptr->error_ = 0.8f;

    for (u32 d = 0; d < dim; ++d) {
        if (vec[d] >= 0) {
            code_ptr->compress_[d / 8] |= (1 << (d % 8));
            code_ptr->sum_ += 1.0f;
        }
    }
    if (out_norm) *out_norm = code_ptr->norm_;
    return sizeof(RabitQVec) + dim / 8;
}

// ---------- Bulk Build Insert ----------

void SPFreshIndexInMem::InsertBlockData(const f32 *vectors, u32 count) {
    for (u32 i = 0; i < count && num_vectors_ < max_vectors_; ++i) {
        InsertVector(vectors + i * dim_, num_vectors_);
    }
}

void SPFreshIndexInMem::InsertVector(const f32 *vec, u32 vector_id) {
    if (vector_id >= max_vectors_) return;

    // Find nearest centroid
    f32 min_dist = std::numeric_limits<f32>::max();
    u32 best_centroid = 0;
    for (u32 c = 0; c < num_centroids_; ++c) {
        f32 dist = 0;
        for (u32 d = 0; d < dim_; ++d) {
            f32 diff = vec[d] - centroids_[static_cast<size_t>(c) * dim_ + d];
            dist += diff * diff;
        }
        if (dist < min_dist) { min_dist = dist; best_centroid = c; }
    }

    bucket_assignments_[vector_id] = best_centroid;
    ++bucket_metas_[best_centroid].base_count_;

    size_t code_size = sizeof(RabitQVec) + dim_ / 8;
    auto *code_ptr = reinterpret_cast<RabitQVec *>(static_cast<char *>(rabitq_data_) + static_cast<size_t>(vector_id) * code_size);
    Encode(reinterpret_cast<f32 *>(code_ptr), vec, dim_, nullptr);
    ++num_vectors_;
}

// ---------- Incremental Insert (Delta) ----------

void SPFreshIndexInMem::InsertDelta(const f32 *vec, u32 row_id) {
    // Find nearest centroid (simplified RNGSelection: pick top-1 nearest)
    f32 min_dist = std::numeric_limits<f32>::max();
    u32 best_centroid = 0;
    for (u32 c = 0; c < num_centroids_; ++c) {
        f32 dist = 0;
        for (u32 d = 0; d < dim_; ++d) {
            f32 diff = vec[d] - centroids_[static_cast<size_t>(c) * dim_ + d];
            dist += diff * diff;
        }
        if (dist < min_dist) { min_dist = dist; best_centroid = c; }
    }

    // Encode to RaBitQ and append to delta
    size_t code_size = sizeof(RabitQVec) + dim_ / 8;
    std::vector<char> entry(code_size);
    Encode(reinterpret_cast<f32 *>(entry.data()), vec, dim_, nullptr);

    delta_entries_.push_back(std::move(entry));
    bucket_metas_[best_centroid].delta_count_++;

    // Auto-compact when delta grows large
    if (delta_entries_.size() > 4096) {
        Compact();
    }
}

// ---------- Search (base + delta, with delete filter) ----------

void SPFreshIndexInMem::Search(const f32 *query,
                                u32 dim,
                                const SearchCallback &callback) const {
    if (dim != dim_) return;
    f32 inv_sqrt_d = 1.0f / std::sqrt(static_cast<f32>(dim_));

    f32 query_norm = 0;
    for (u32 d = 0; d < dim_; ++d) query_norm += query[d] * query[d];
    query_norm = std::sqrt(query_norm);
    if (query_norm < 1e-8f) query_norm = 1.0f;

    size_t code_size = sizeof(RabitQVec) + dim_ / 8;

    // Scan base array (skip deleted)
    for (u32 i = 0; i < num_vectors_; ++i) {
        if (deleted_set_.find(i) != deleted_set_.end()) continue;
        const auto *code = reinterpret_cast<const RabitQVec *>(
            static_cast<const char *>(rabitq_data_) + static_cast<size_t>(i) * code_size);
        f32 dist = RabitQDistance(code, query, dim_, inv_sqrt_d, query_norm);
        callback(i, dist);
    }

    // Scan delta entries (delta indices start after base)
    // For simplicity, delta entries don't have per-entry deletion tracking
    for (size_t di = 0; di < delta_entries_.size(); ++di) {
        u32 row_id = num_vectors_ + static_cast<u32>(di);
        if (deleted_set_.find(row_id) != deleted_set_.end()) continue;
        const auto *code = reinterpret_cast<const RabitQVec *>(delta_entries_[di].data());
        f32 dist = RabitQDistance(code, query, dim_, inv_sqrt_d, query_norm);
        callback(row_id, dist);
    }
}

// ---------- Delete ----------

void SPFreshIndexInMem::MarkDeleted(u32 row_id) {
    deleted_set_.insert(row_id);
    LOG_TRACE(fmt::format("SPFresh MarkDeleted: row_id={}, total_deleted={}", row_id, deleted_set_.size()));
}

f32 SPFreshIndexInMem::RabitQDistance(const RabitQVec *code, const f32 *query, u32 dim, f32 inv_sqrt_d, f32 query_norm) {
    f32 ip_estimate = 0;
    for (u32 d = 0; d < dim; ++d) {
        bool bit = (code->compress_[d / 8] >> (d % 8)) & 1;
        ip_estimate += bit ? query[d] : -query[d];
    }

    f32 cos_approx = inv_sqrt_d * 2.0f * ip_estimate / (code->norm_ * query_norm);
    cos_approx = std::clamp(cos_approx, -1.0f, 1.0f);
    return code->norm_ * code->norm_ + query_norm * query_norm
           - 2 * code->norm_ * query_norm * cos_approx;
}

// ---------- Compaction ----------

void SPFreshIndexInMem::Compact() {
    size_t code_size = sizeof(RabitQVec) + dim_ / 8;

    // Remove deleted vectors from base: rebuild base without deleted entries
    if (!deleted_set_.empty() && num_vectors_ > 0) {
        std::vector<u32> live_indices;
        live_indices.reserve(num_vectors_);
        for (u32 i = 0; i < num_vectors_; ++i) {
            if (deleted_set_.find(i) == deleted_set_.end()) {
                live_indices.push_back(i);
            }
        }

        u32 live_count = static_cast<u32>(live_indices.size());
        if (live_count < num_vectors_) {
            auto *new_data = new char[static_cast<size_t>(live_count) * code_size];
            for (u32 j = 0; j < live_count; ++j) {
                std::memcpy(new_data + static_cast<size_t>(j) * code_size,
                            static_cast<const char *>(rabitq_data_) + static_cast<size_t>(live_indices[j]) * code_size,
                            code_size);
            }
            delete[] static_cast<char *>(rabitq_data_);
            rabitq_data_ = new_data;
            num_vectors_ = live_count;
            max_vectors_ = live_count;

            // Remap deleted_set_ - after compaction, row IDs are re-indexed
            // For simplicity, clear and let callers re-insert if needed
            deleted_set_.clear();
        }
    }

    // Now merge delta into base
    if (delta_entries_.empty()) return;

    u32 new_count = num_vectors_ + static_cast<u32>(delta_entries_.size());
    auto *new_data = new char[static_cast<size_t>(new_count) * code_size];

    if (num_vectors_ > 0) {
        std::memcpy(new_data, rabitq_data_, static_cast<size_t>(num_vectors_) * code_size);
    }
    for (size_t i = 0; i < delta_entries_.size(); ++i) {
        std::memcpy(new_data + (static_cast<size_t>(num_vectors_) + i) * code_size,
                    delta_entries_[i].data(), code_size);
    }

    delete[] static_cast<char *>(rabitq_data_);
    rabitq_data_ = new_data;
    num_vectors_ = new_count;
    max_vectors_ = new_count;
    delta_entries_.clear();
    delta_compacted_seq_++;

    LOG_TRACE(fmt::format("SPFresh Compact: merged, total={}, deleted={}", num_vectors_, deleted_set_.size()));
}

// ---------- Persistence ----------

void SPFreshIndexInMem::Save(LocalFileHandle &file_handle) const {
    u32 magic = 0x50504652; // "SPFR"
    u32 version = 2;
    file_handle.Append(&magic, sizeof(magic));
    file_handle.Append(&version, sizeof(version));
    file_handle.Append(&num_vectors_, sizeof(num_vectors_));
    file_handle.Append(&dim_, sizeof(dim_));

    size_t code_size = sizeof(RabitQVec) + dim_ / 8;
    u64 bucket_bytes = static_cast<u64>(num_vectors_) * code_size;
    file_handle.Append(&bucket_bytes, sizeof(bucket_bytes));
    file_handle.Append(static_cast<const char *>(rabitq_data_), bucket_bytes);

    file_handle.Append(&num_centroids_, sizeof(num_centroids_));
    for (auto &meta : bucket_metas_) {
        file_handle.Append(&meta, sizeof(meta));
    }
    file_handle.Append(const_cast<f32 *>(centroids_.data()), static_cast<size_t>(num_centroids_) * dim_ * sizeof(f32));

    // Write delta entries count (for WAL replay)
    u32 delta_count = static_cast<u32>(delta_entries_.size());
    file_handle.Append(&delta_count, sizeof(delta_count));
}

void SPFreshIndexInMem::Load(LocalFileHandle &file_handle, size_t file_size) {
    u32 magic = 0;
    file_handle.Read(&magic, sizeof(magic));
    if (magic != 0x50504652) {
        UnrecoverableError("SPFreshIndexInMem: Invalid magic number");
        return;
    }
    u32 version = 0;
    file_handle.Read(&version, sizeof(version));

    file_handle.Read(&num_vectors_, sizeof(num_vectors_));
    file_handle.Read(&dim_, sizeof(dim_));

    u64 bucket_bytes = 0;
    file_handle.Read(&bucket_bytes, sizeof(bucket_bytes));

    size_t code_size = sizeof(RabitQVec) + dim_ / 8;
    size_t expected = static_cast<size_t>(num_vectors_) * code_size;
    if (bucket_bytes != expected) {
        UnrecoverableError("SPFreshIndexInMem: bucket size mismatch");
        return;
    }

    if (max_vectors_ < num_vectors_) {
        delete[] static_cast<char *>(rabitq_data_);
        max_vectors_ = num_vectors_;
        rabitq_data_ = new char[static_cast<size_t>(max_vectors_) * code_size];
    }
    file_handle.Read(static_cast<char *>(rabitq_data_), bucket_bytes);

    if (version >= 1) {
        file_handle.Read(&num_centroids_, sizeof(num_centroids_));
        bucket_metas_.resize(num_centroids_);
        for (auto &meta : bucket_metas_) file_handle.Read(&meta, sizeof(meta));
        centroids_.resize(static_cast<size_t>(num_centroids_) * dim_);
        file_handle.Read(centroids_.data(), static_cast<size_t>(num_centroids_) * dim_ * sizeof(f32));
        bucket_assignments_.resize(num_vectors_);
    }
    // Delta entries are replayed from WAL, not persisted in file
}

void SPFreshIndexInMem::Dump(BufferObj *buffer_obj, size_t *p_dump_size) {}

// ---------- BaseMemIndex interface ----------

MemIndexTracerInfo SPFreshIndexInMem::GetInfo() const {
    return MemIndexTracerInfo(std::make_shared<std::string>(index_name_),
                              std::make_shared<std::string>(table_name_),
                              std::make_shared<std::string>(db_name_),
                              mem_used_ + delta_entries_.size() * (sizeof(RabitQVec) + dim_ / 8),
                              GetRowCount());
}

const ChunkIndexMetaInfo SPFreshIndexInMem::GetChunkIndexMetaInfo() const {
    return ChunkIndexMetaInfo{"spfresh_chunk", begin_row_id_, GetRowCount(), 0, mem_used_};
}

// ---------- Phase 3: Rebalancer ----------

bool SPFreshIndexInMem::TryAutoCompact(u32 delta_threshold) {
    if (delta_entries_.size() >= delta_threshold) {
        Compact();
        return true;
    }
    return false;
}

void SPFreshIndexInMem::Rebalance(u32 bucket_size_limit) {
    // Step 1: Compact delta first
    if (!delta_entries_.empty()) {
        Compact();
    }

    // Step 2: Scan buckets and split overloaded ones
    if (num_centroids_ == 0 || centroids_.empty()) return;

    // Build vector-to-centroid mapping from base data
    std::vector<u32> bucket_counts(num_centroids_, 0);
    for (u32 i = 0; i < num_vectors_; ++i) {
        u32 centroid = bucket_assignments_.empty() ? 0 : bucket_assignments_[i];
        if (centroid < num_centroids_) {
            bucket_counts[centroid]++;
        }
    }

    // Split overloaded buckets
    bool any_split = false;
    for (u32 c = 0; c < num_centroids_; ++c) {
        if (bucket_counts[c] > bucket_size_limit) {
            auto new_ids = SplitBucket(c);
            if (!new_ids.empty()) {
                any_split = true;
            }
        }
    }

    if (any_split) {
        LOG_TRACE(fmt::format("SPFresh Rebalance: split completed, centroids now {}", num_centroids_));
    }
}

std::vector<u32> SPFreshIndexInMem::SplitBucket(u32 bucket_id) {
    // Simplified split: just mark as needing rebalance
    // Full k-means(k=2) split would re-encode all vectors in the bucket
    // For now, we log and return empty (no actual split)
    LOG_TRACE(fmt::format("SPFresh SplitBucket: bucket {} exceeds limit, split request noted", bucket_id));
    return {};
}

} // namespace infinity

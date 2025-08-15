// Copyright(C) 2025 InfiniFlow, Inc. All rights reserved.
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

#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>

module infinity_core:kentro.impl;

import :kentro;
import :stl;
import :status;
import :infinity_exception;
import :third_party;

namespace infinity {

KMeans::KMeans(SizeT n_clusters) 
    : n_clusters_(n_clusters), iters_(25), euclidean_(false), balanced_(false),
      max_balance_diff_(16), verbose_(false), trained_(false), use_medoids_(false) {
    if (n_clusters == 0) {
        String error_message = "Number of clusters must be positive";
        UnrecoverableError(error_message);
    }
    cluster_sizes_.resize(n_clusters_, 0.0f);
}

KMeans& KMeans::WithIterations(SizeT iters) {
    if (iters == 0) {
        String error_message = "Number of iterations must be positive";
        UnrecoverableError(error_message);
    }
    iters_ = iters;
    return *this;
}

KMeans& KMeans::WithEuclidean(bool euclidean) {
    euclidean_ = euclidean;
    return *this;
}

KMeans& KMeans::WithBalanced(bool balanced) {
    balanced_ = balanced;
    return *this;
}

KMeans& KMeans::WithMaxBalanceDiff(SizeT max_balance_diff) {
    if (max_balance_diff == 0) {
        String error_message = "Max balance difference must be positive";
        UnrecoverableError(error_message);
    }
    max_balance_diff_ = max_balance_diff;
    return *this;
}

KMeans& KMeans::WithVerbose(bool verbose) {
    verbose_ = verbose;
    return *this;
}

KMeans& KMeans::WithUseMedoids(bool use_medoids) {
    use_medoids_ = use_medoids;
    return *this;
}

Vector<Vector<SizeT>> KMeans::Train(const Vector<Vector<f32>>& data, SizeT num_threads) {
    SizeT n = data.size();
    if (n == 0) {
        String error_message = "Data cannot be empty";
        UnrecoverableError(error_message);
    }

    if (trained_) {
        String error_message = "Clustering has already been trained";
        UnrecoverableError(error_message);
    }

    if (n < n_clusters_) {
        String error_message = fmt::format("Number of points ({}) must be at least as large as number of clusters ({})", n, n_clusters_);
        UnrecoverableError(error_message);
    }

    // Initialize with proper memory reservation
    assignments_.resize(n);
    cluster_sizes_.assign(n_clusters_, 0.0f);

    // Compute data norms for Euclidean distance
    Vector<f32> data_norms;
    if (euclidean_) {
        data_norms.reserve(n);  // Pre-allocate memory
        data_norms = ComputeDataNorms(data);
    }

    // Initialize centroids by sampling random points
    centroids_ = SampleRows(data);
    PostprocessCentroids();

    // Main K-Means iterations
    for (SizeT iter = 0; iter < iters_; ++iter) {
        AssignClusters(data, euclidean_ ? &data_norms : nullptr);
        if (use_medoids_) {
            UpdateMedoids(data);
        } else {
            UpdateCentroids(data);
        }
        SplitClusters(data);
        PostprocessCentroids();

        if (verbose_) {
            f32 cost = ComputeCost(data);
            std::cout << fmt::format("Iteration {}/{} | Cost: {:.6f}", iter + 1, iters_, cost) << std::endl;
        }
    }

    // Final assignment
    AssignClusters(data, euclidean_ ? &data_norms : nullptr);

    // Balance clusters if requested
    if (balanced_) {
        if (verbose_) {
            std::cout << "Balancing clusters..." << std::endl;
        }
        BalanceClusters(data, euclidean_ ? &data_norms : nullptr);
    }

    // Convert assignments to cluster vectors with pre-allocation
    Vector<Vector<SizeT>> result(n_clusters_);

    // First pass: count points per cluster to reserve memory
    Vector<SizeT> cluster_counts(n_clusters_, 0);
    for (SizeT assignment : assignments_) {
        cluster_counts[assignment]++;
    }

    // Reserve memory for each cluster
    for (SizeT i = 0; i < n_clusters_; ++i) {
        result[i].reserve(cluster_counts[i]);
    }

    // Second pass: assign points to clusters
    for (SizeT point_idx = 0; point_idx < assignments_.size(); ++point_idx) {
        result[assignments_[point_idx]].push_back(point_idx);
    }

    trained_ = true;
    return result;
}

Vector<Vector<SizeT>> KMeans::Assign(const Vector<Vector<f32>>& data, SizeT k) const {
    if (!trained_) {
        String error_message = "Clustering has not been trained yet";
        UnrecoverableError(error_message);
    }

    if (k == 0) {
        String error_message = "k must be positive";
        UnrecoverableError(error_message);
    }

    SizeT n = data.size();
    if (n == 0) {
        return Vector<Vector<SizeT>>(n_clusters_);
    }

    SizeT m = data[0].size();
    if (centroids_.empty() || m != centroids_[0].size()) {
        String error_message = fmt::format("Dimension mismatch: expected {}, got {}",
                                         centroids_.empty() ? 0 : centroids_[0].size(), m);
        UnrecoverableError(error_message);
    }

    // Pre-compute centroid norms once
    Vector<f32> centroid_norms;
    if (euclidean_) {
        centroid_norms.reserve(n_clusters_);
        centroid_norms = ComputeDataNorms(centroids_);
    }

    Vector<Vector<SizeT>> result(n_clusters_);

    // Pre-allocate temporary vectors to avoid repeated allocations
    Vector<f32> dots(n_clusters_);
    Vector<f32> distances(n_clusters_);
    Vector<SizeT> indices(n_clusters_);
    std::iota(indices.begin(), indices.end(), 0);

    for (SizeT i = 0; i < n; ++i) {
        const auto& point = data[i];

        // Compute dot products with all centroids (vectorized operation)
        for (SizeT j = 0; j < n_clusters_; ++j) {
            dots[j] = DotProduct(centroids_[j], point);
        }

        // Compute distances
        if (euclidean_) {
            f32 point_norm = DotProduct(point, point);
            for (SizeT j = 0; j < n_clusters_; ++j) {
                distances[j] = centroid_norms[j] - 2.0f * dots[j] + point_norm;
            }
        } else {
            for (SizeT j = 0; j < n_clusters_; ++j) {
                distances[j] = -dots[j];
            }
        }

        // Find k nearest clusters using partial sort for better performance
        if (k < n_clusters_) {
            std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                            [&distances](SizeT a, SizeT b) {
                                return distances[a] < distances[b];
                            });
        } else {
            std::sort(indices.begin(), indices.end(), [&distances](SizeT a, SizeT b) {
                return distances[a] < distances[b];
            });
        }

        for (SizeT idx = 0; idx < std::min(k, n_clusters_); ++idx) {
            result[indices[idx]].push_back(i);
        }
    }

    return result;
}

// Helper functions
Vector<f32> KMeans::ComputeDataNorms(const Vector<Vector<f32>>& data) const {
    Vector<f32> norms(data.size());
    for (SizeT i = 0; i < data.size(); ++i) {
        norms[i] = DotProduct(data[i], data[i]);
    }
    return norms;
}

f32 KMeans::DotProduct(const Vector<f32>& a, const Vector<f32>& b) const {
    f32 result = 0.0f;
    SizeT size = std::min(a.size(), b.size());

    // Unroll loop for better performance (process 4 elements at a time)
    SizeT unroll_size = size - (size % 4);
    SizeT i = 0;

    for (; i < unroll_size; i += 4) {
        result += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
    }

    // Handle remaining elements
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

f32 KMeans::EuclideanDistance(const Vector<f32>& a, const Vector<f32>& b) const {
    f32 sum = 0.0f;
    SizeT size = std::min(a.size(), b.size());
    for (SizeT i = 0; i < size; ++i) {
        f32 diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

f32 KMeans::CosineDistance(const Vector<f32>& a, const Vector<f32>& b) const {
    return 1.0f - DotProduct(a, b);
}

void KMeans::NormalizeVector(Vector<f32>& vec) const {
    f32 norm = std::sqrt(DotProduct(vec, vec));
    if (norm > 0.0f) {
        for (f32& val : vec) {
            val /= norm;
        }
    }
}

Vector<Vector<f32>> KMeans::SampleRows(const Vector<Vector<f32>>& data) {
    std::random_device rd;
    std::mt19937 rng(rd());
    SizeT n = data.size();
    SizeT m = data[0].size();

    // Use partial shuffle for better performance when n_clusters << n
    Vector<SizeT> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Only shuffle the first n_clusters elements
    for (SizeT i = 0; i < n_clusters_; ++i) {
        std::uniform_int_distribution<SizeT> dist(i, n - 1);
        SizeT j = dist(rng);
        std::swap(indices[i], indices[j]);
    }

    // Pre-allocate centroids with proper dimensions
    Vector<Vector<f32>> centroids(n_clusters_);
    for (SizeT i = 0; i < n_clusters_; ++i) {
        centroids[i].reserve(m);
    }

    if (use_medoids_) {
        // Initialize medoid indices
        medoid_indices_.clear();
        medoid_indices_.reserve(n_clusters_);
        for (SizeT i = 0; i < n_clusters_; ++i) {
            medoid_indices_.push_back(indices[i]);
        }

        // Create centroids from medoid points
        for (SizeT i = 0; i < n_clusters_; ++i) {
            centroids[i] = data[medoid_indices_[i]];
        }
    } else {
        // Standard centroid initialization
        for (SizeT i = 0; i < n_clusters_; ++i) {
            centroids[i] = data[indices[i]];
        }
    }

    return centroids;
}

void KMeans::AssignClusters(const Vector<Vector<f32>>& data, const Vector<f32>* data_norms) {
    SizeT n = data.size();

    // Pre-allocate temporary vectors to avoid repeated allocations
    static thread_local Vector<f32> dots;
    static thread_local Vector<f32> distances_or_similarities;

    dots.resize(n_clusters_);
    distances_or_similarities.resize(n_clusters_);

    if (euclidean_) {
        // Pre-compute centroid norms once
        Vector<f32> centroid_norms;
        centroid_norms.reserve(n_clusters_);
        centroid_norms = ComputeDataNorms(centroids_);

        for (SizeT i = 0; i < n; ++i) {
            const auto& point = data[i];

            // Compute dot products with all centroids (vectorized)
            for (SizeT j = 0; j < n_clusters_; ++j) {
                dots[j] = DotProduct(centroids_[j], point);
            }

            // Compute distances using pre-computed norms
            f32 point_norm = (*data_norms)[i];
            for (SizeT j = 0; j < n_clusters_; ++j) {
                distances_or_similarities[j] = centroid_norms[j] - 2.0f * dots[j] + point_norm;
            }

            // Find minimum distance cluster
            auto min_it = std::min_element(distances_or_similarities.begin(), distances_or_similarities.end());
            assignments_[i] = std::distance(distances_or_similarities.begin(), min_it);
        }
    } else {
        for (SizeT i = 0; i < n; ++i) {
            const auto& point = data[i];

            // Compute similarities (dot products) with all centroids
            for (SizeT j = 0; j < n_clusters_; ++j) {
                distances_or_similarities[j] = DotProduct(centroids_[j], point);
            }

            // Find maximum similarity cluster
            auto max_it = std::max_element(distances_or_similarities.begin(), distances_or_similarities.end());
            assignments_[i] = std::distance(distances_or_similarities.begin(), max_it);
        }
    }

    // Update cluster sizes efficiently
    std::fill(cluster_sizes_.begin(), cluster_sizes_.end(), 0.0f);
    for (SizeT assignment : assignments_) {
        cluster_sizes_[assignment] += 1.0f;
    }
}

void KMeans::UpdateCentroids(const Vector<Vector<f32>>& data) {
    SizeT m = data[0].size();

    // Initialize centroids to zero more efficiently
    for (SizeT i = 0; i < n_clusters_; ++i) {
        centroids_[i].assign(m, 0.0f);
    }

    // Accumulate points for each cluster
    for (SizeT i = 0; i < data.size(); ++i) {
        SizeT cluster = assignments_[i];
        const auto& point = data[i];
        auto& centroid = centroids_[cluster];

        // Vectorized addition
        for (SizeT j = 0; j < m; ++j) {
            centroid[j] += point[j];
        }
    }

    // Normalize by cluster sizes
    for (SizeT j = 0; j < n_clusters_; ++j) {
        if (cluster_sizes_[j] > 0.0f) {
            f32 inv_size = 1.0f / cluster_sizes_[j];  // Multiply instead of divide
            auto& centroid = centroids_[j];
            for (f32& val : centroid) {
                val *= inv_size;
            }
        }
    }
}

void KMeans::UpdateMedoids(const Vector<Vector<f32>>& data) {
    // Pre-allocate cluster_points vector to avoid repeated allocations
    static thread_local Vector<SizeT> cluster_points;

    // For each cluster, find the point that minimizes total distance to all points in cluster
    for (SizeT cluster_id = 0; cluster_id < n_clusters_; ++cluster_id) {
        if (cluster_sizes_[cluster_id] == 0.0f) {
            continue;
        }

        // Get all points assigned to this cluster
        cluster_points.clear();
        cluster_points.reserve(static_cast<SizeT>(cluster_sizes_[cluster_id]));

        for (SizeT point_idx = 0; point_idx < assignments_.size(); ++point_idx) {
            if (assignments_[point_idx] == cluster_id) {
                cluster_points.push_back(point_idx);
            }
        }

        if (cluster_points.empty()) {
            continue;
        }

        // Find the point that minimizes total distance to all other points in the cluster
        SizeT best_medoid = cluster_points[0];
        f32 best_cost = std::numeric_limits<f32>::infinity();

        for (SizeT candidate_idx : cluster_points) {
            const auto& candidate_point = data[candidate_idx];
            f32 total_cost = 0.0f;

            // Optimized distance computation
            for (SizeT other_idx : cluster_points) {
                if (candidate_idx != other_idx) {
                    const auto& other_point = data[other_idx];
                    f32 distance;
                    if (euclidean_) {
                        distance = EuclideanDistance(candidate_point, other_point);
                    } else {
                        distance = CosineDistance(candidate_point, other_point);
                    }
                    total_cost += distance;
                }
            }

            if (total_cost < best_cost) {
                best_cost = total_cost;
                best_medoid = candidate_idx;
            }
        }

        // Update medoid index and centroid
        medoid_indices_[cluster_id] = best_medoid;
        centroids_[cluster_id] = data[best_medoid];
    }
}

void KMeans::PostprocessCentroids() {
    if (!euclidean_) {
        // Normalize centroids for spherical k-means
        for (auto& centroid : centroids_) {
            NormalizeVector(centroid);
        }
    }
}

void KMeans::SplitClusters(const Vector<Vector<f32>>& data) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<f32> uniform(0.0f, 1.0f);

    for (SizeT i = 0; i < n_clusters_; ++i) {
        if (cluster_sizes_[i] == 0.0f) {
            // Find cluster to split
            SizeT j = 0;
            while (true) {
                f32 p = (cluster_sizes_[j] - 1.0f) / (static_cast<f32>(data.size()) - static_cast<f32>(n_clusters_));
                f32 r = uniform(rng);
                if (r < p) {
                    break;
                }
                j = (j + 1) % n_clusters_;
            }

            if (use_medoids_) {
                // For medoids, find a random point from cluster j to be the new medoid
                Vector<SizeT> cluster_j_points;
                for (SizeT point_idx = 0; point_idx < assignments_.size(); ++point_idx) {
                    if (assignments_[point_idx] == j) {
                        cluster_j_points.push_back(point_idx);
                    }
                }

                if (!cluster_j_points.empty()) {
                    std::uniform_int_distribution<SizeT> point_dist(0, cluster_j_points.size() - 1);
                    SizeT random_point_idx = cluster_j_points[point_dist(rng)];
                    medoid_indices_[i] = random_point_idx;
                    centroids_[i] = data[random_point_idx];
                }
            } else {
                // Split cluster j (original centroid logic)
                centroids_[i] = centroids_[j];

                // Apply small symmetric perturbation
                for (SizeT k = 0; k < centroids_[i].size(); ++k) {
                    if (k % 2 == 0) {
                        centroids_[i][k] *= 1.0f + EPS;
                        centroids_[j][k] *= 1.0f - EPS;
                    } else {
                        centroids_[i][k] *= 1.0f - EPS;
                        centroids_[j][k] *= 1.0f + EPS;
                    }
                }
            }

            // Split cluster sizes evenly
            cluster_sizes_[i] = cluster_sizes_[j] / 2.0f;
            cluster_sizes_[j] -= cluster_sizes_[i];
        }
    }
}

f32 KMeans::ComputeCost(const Vector<Vector<f32>>& data) const {
    f32 total_cost = 0.0f;

    if (euclidean_) {
        for (SizeT i = 0; i < data.size(); ++i) {
            const auto& point = data[i];
            const auto& centroid = centroids_[assignments_[i]];
            total_cost += EuclideanDistance(point, centroid);
        }
    } else {
        for (SizeT i = 0; i < data.size(); ++i) {
            const auto& point = data[i];
            const auto& centroid = centroids_[assignments_[i]];
            total_cost += DotProduct(point, centroid);
        }
    }

    return total_cost / static_cast<f32>(data.size());
}

void KMeans::BalanceClusters(const Vector<Vector<f32>>& data, const Vector<f32>* data_norms) {
    SizeT m = data[0].size();
    Vector<Vector<f32>> unnormalized_centroids(n_clusters_, Vector<f32>(m, 0.0f));

    // Compute unnormalized centroids
    for (SizeT i = 0; i < data.size(); ++i) {
        SizeT cluster = assignments_[i];
        for (SizeT j = 0; j < m; ++j) {
            unnormalized_centroids[cluster][j] += data[i][j];
        }
    }

    f32 n_min = *std::min_element(cluster_sizes_.begin(), cluster_sizes_.end());
    f32 n_max = *std::max_element(cluster_sizes_.begin(), cluster_sizes_.end());

    SizeT iters = 0;
    f32 p_now = 0.0f;
    f32 p_next = std::numeric_limits<f32>::infinity();

    while (n_max - n_min > 0.5f + static_cast<f32>(max_balance_diff_)) {
        for (SizeT i = 0; i < data.size(); ++i) {
            SizeT old_cluster = assignments_[i];
            f32 n_old = cluster_sizes_[old_cluster];
            const auto& point = data[i];

            // Remove point from old cluster
            for (SizeT j = 0; j < m; ++j) {
                unnormalized_centroids[old_cluster][j] -= point[j];
            }

            // Update old centroid
            if (n_old > 0.0f) {
                if (use_medoids_) {
                    // For medoids, find the best medoid from remaining points in old cluster
                    UpdateMedoidForCluster(data, old_cluster);
                } else {
                    // For standard k-means, compute averaged centroid
                    for (SizeT j = 0; j < m; ++j) {
                        centroids_[old_cluster][j] = unnormalized_centroids[old_cluster][j] / (n_old - 1.0f);
                    }
                    if (!euclidean_) {
                        NormalizeVector(centroids_[old_cluster]);
                    }
                }
            }

            cluster_sizes_[old_cluster] = cluster_sizes_[old_cluster] - 1.0f + PARTLY_REMAINING_FACTOR;

            // Compute distances and costs
            Vector<f32> distances(n_clusters_);
            if (euclidean_) {
                Vector<f32> dots(n_clusters_);
                for (SizeT j = 0; j < n_clusters_; ++j) {
                    dots[j] = DotProduct(centroids_[j], point);
                }
                Vector<f32> centroid_norms = ComputeDataNorms(centroids_);
                f32 point_norm = (*data_norms)[i];
                for (SizeT j = 0; j < n_clusters_; ++j) {
                    distances[j] = centroid_norms[j] - 2.0f * dots[j] + point_norm;
                }
            } else {
                for (SizeT j = 0; j < n_clusters_; ++j) {
                    distances[j] = -DotProduct(centroids_[j], point);
                }
            }

            Vector<f32> costs(n_clusters_);
            for (SizeT j = 0; j < n_clusters_; ++j) {
                costs[j] = distances[j] + cluster_sizes_[j] * p_now;
            }

            auto min_it = std::min_element(costs.begin(), costs.end());
            SizeT min_cluster = std::distance(costs.begin(), min_it);

            // Update penalties
            Vector<f32> penalties_1(n_clusters_);
            Vector<f32> penalties_2(n_clusters_);
            for (SizeT p = 0; p < n_clusters_; ++p) {
                penalties_1[p] = distances[p] - distances[old_cluster];
                penalties_2[p] = cluster_sizes_[old_cluster] - cluster_sizes_[p];
            }

            f32 min_p_value = std::numeric_limits<f32>::infinity();
            for (SizeT p = 0; p < n_clusters_; ++p) {
                if (cluster_sizes_[old_cluster] > cluster_sizes_[p] && penalties_2[p] != 0.0f) {
                    f32 penalty = penalties_1[p] / penalties_2[p];
                    if (penalty < min_p_value) {
                        min_p_value = penalty;
                    }
                }
            }

            if (p_now < min_p_value && min_p_value < p_next) {
                p_next = min_p_value;
            }

            // Assign to new cluster
            cluster_sizes_[min_cluster] += 1.0f;
            for (SizeT j = 0; j < m; ++j) {
                unnormalized_centroids[min_cluster][j] += point[j];
            }

            // Update new centroid
            if (use_medoids_) {
                // For medoids, find the best medoid from points in new cluster
                UpdateMedoidForCluster(data, min_cluster);
            } else {
                // For standard k-means, compute averaged centroid
                for (SizeT j = 0; j < m; ++j) {
                    centroids_[min_cluster][j] = unnormalized_centroids[min_cluster][j] / cluster_sizes_[min_cluster];
                }
                if (!euclidean_) {
                    NormalizeVector(centroids_[min_cluster]);
                }
            }

            cluster_sizes_[old_cluster] -= PARTLY_REMAINING_FACTOR;
            assignments_[i] = min_cluster;
        }

        n_min = *std::min_element(cluster_sizes_.begin(), cluster_sizes_.end());
        n_max = *std::max_element(cluster_sizes_.begin(), cluster_sizes_.end());
        p_now = PENALTY_FACTOR * p_next;
        p_next = std::numeric_limits<f32>::infinity();

        iters++;

        if (verbose_) {
            f32 cost = ComputeCost(data);
            std::cout << fmt::format("Balance iteration {} | Cost: {:.6f} | Max diff: {:.2f}",
                                   iters, cost, n_max - n_min) << std::endl;
        }
    }

    // After balancing, update medoid indices if using medoids
    if (use_medoids_) {
        UpdateMedoidsAfterBalancing(data);
    }
}

void KMeans::UpdateMedoidsAfterBalancing(const Vector<Vector<f32>>& data) {
    // For each cluster, find the data point that best represents the current centroid
    for (SizeT cluster_id = 0; cluster_id < n_clusters_; ++cluster_id) {
        if (cluster_sizes_[cluster_id] == 0.0f) {
            continue;
        }

        // Get all points assigned to this cluster
        Vector<SizeT> cluster_points;
        for (SizeT point_idx = 0; point_idx < assignments_.size(); ++point_idx) {
            if (assignments_[point_idx] == cluster_id) {
                cluster_points.push_back(point_idx);
            }
        }

        if (cluster_points.empty()) {
            continue;
        }

        // Find the point closest to the current centroid
        const auto& current_centroid = centroids_[cluster_id];
        SizeT best_medoid = cluster_points[0];
        f32 best_distance = std::numeric_limits<f32>::infinity();

        for (SizeT candidate_idx : cluster_points) {
            const auto& candidate_point = data[candidate_idx];
            f32 distance = euclidean_ ? EuclideanDistance(candidate_point, current_centroid)
                                      : CosineDistance(candidate_point, current_centroid);

            if (distance < best_distance) {
                best_distance = distance;
                best_medoid = candidate_idx;
            }
        }

        // Update medoid index and centroid to match the selected medoid
        medoid_indices_[cluster_id] = best_medoid;
        centroids_[cluster_id] = data[best_medoid];
    }
}

void KMeans::UpdateMedoidForCluster(const Vector<Vector<f32>>& data, SizeT cluster_id) {
    // Get all points assigned to this cluster
    Vector<SizeT> cluster_points;
    for (SizeT point_idx = 0; point_idx < assignments_.size(); ++point_idx) {
        if (assignments_[point_idx] == cluster_id) {
            cluster_points.push_back(point_idx);
        }
    }

    if (cluster_points.empty()) {
        return;
    }

    // Find the point closest to the current centroid
    const auto& current_centroid = centroids_[cluster_id];
    SizeT best_medoid = cluster_points[0];
    f32 best_distance = std::numeric_limits<f32>::infinity();

    for (SizeT candidate_idx : cluster_points) {
        const auto& candidate_point = data[candidate_idx];
        f32 distance = euclidean_ ? EuclideanDistance(candidate_point, current_centroid)
                                  : CosineDistance(candidate_point, current_centroid);

        if (distance < best_distance) {
            best_distance = distance;
            best_medoid = candidate_idx;
        }
    }

    // Update medoid index and centroid to match the selected medoid
    medoid_indices_[cluster_id] = best_medoid;
    centroids_[cluster_id] = data[best_medoid];
}

// Move-optimized versions for better performance
Vector<Vector<SizeT>> KMeans::Train(Vector<Vector<f32>>&& data, SizeT num_threads) {
    // For move version, we can potentially optimize by avoiding some copies
    // For now, delegate to the const reference version
    return Train(data, num_threads);
}

Vector<Vector<SizeT>> KMeans::Assign(Vector<Vector<f32>>&& data, SizeT k) const {
    // For move version, we can potentially optimize by avoiding some copies
    // For now, delegate to the const reference version
    return Assign(data, k);
}

} // namespace infinity

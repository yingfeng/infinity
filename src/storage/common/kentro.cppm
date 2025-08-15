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

export module infinity_core:kentro;

import :stl;
import :status;
import :infinity_exception;

namespace infinity {

/// Constants for the balanced K-Means algorithm
export constexpr f32 EPS = 1.0f / 1024.0f;
export constexpr f32 PARTLY_REMAINING_FACTOR = 0.15f;
export constexpr f32 PENALTY_FACTOR = 2.5f;

/// Errors that can occur during K-Means clustering
export enum class KMeansErrorType {
    InvalidParameter,
    AlreadyTrained,
    NotTrained,
    InsufficientPoints,
    DimensionMismatch,
};

export class KMeansError {
public:
    KMeansErrorType type_;
    String message_;

    KMeansError(KMeansErrorType type, String message) : type_(type), message_(std::move(message)) {}

    const String& what() const { return message_; }
};

/// High-performance K-Means clustering implementation
///
/// This implementation provides both standard and balanced K-Means clustering
/// with support for Euclidean and cosine similarity metrics, as well as
/// K-medoids clustering using the PAM (Partition Around Medoids) algorithm.
export class KMeans {
public:
    /// Create a new K-Means instance
    ///
    /// # Arguments
    ///
    /// * `n_clusters` - The number of clusters (k)
    ///
    /// # Panics
    ///
    /// Panics if `n_clusters` is 0
    explicit KMeans(SizeT n_clusters);

    /// Set the number of iterations (default: 25)
    KMeans& WithIterations(SizeT iters);

    /// Use Euclidean distance instead of cosine similarity (default: false)
    KMeans& WithEuclidean(bool euclidean);

    /// Enable balanced K-Means clustering (default: false)
    KMeans& WithBalanced(bool balanced);

    /// Set maximum balance difference for balanced clustering (default: 16)
    KMeans& WithMaxBalanceDiff(SizeT max_balance_diff);

    /// Enable verbose output (default: false)
    KMeans& WithVerbose(bool verbose);

    /// Enable K-medoids clustering (default: false)
    KMeans& WithUseMedoids(bool use_medoids);

    /// Perform K-Means clustering on the provided data
    ///
    /// # Arguments
    ///
    /// * `data` - The data matrix (n_points × n_dimensions)
    /// * `num_threads` - Number of threads to use (0 for automatic)
    ///
    /// # Returns
    ///
    /// A vector of vectors where each inner vector contains the indices of points
    /// assigned to the corresponding cluster
    Vector<Vector<SizeT>> Train(const Vector<Vector<f32>>& data, SizeT num_threads = 0);

    /// Assign data points to their k nearest clusters
    ///
    /// # Arguments
    ///
    /// * `data` - The data matrix (n_points × n_dimensions)
    /// * `k` - Number of nearest clusters to assign each point to
    ///
    /// # Returns
    ///
    /// A vector of vectors where each inner vector contains the indices of points
    /// assigned to the corresponding cluster
    Vector<Vector<SizeT>> Assign(const Vector<Vector<f32>>& data, SizeT k) const;

    /// Move-optimized version of Train for better performance
    Vector<Vector<SizeT>> Train(Vector<Vector<f32>>&& data, SizeT num_threads = 0);

    /// Move-optimized version of Assign for better performance
    Vector<Vector<SizeT>> Assign(Vector<Vector<f32>>&& data, SizeT k) const;

    /// Get the number of clusters
    SizeT NClusters() const { return n_clusters_; }

    /// Get the number of iterations
    SizeT Iterations() const { return iters_; }

    /// Check if using Euclidean distance
    bool IsEuclidean() const { return euclidean_; }

    /// Check if using balanced clustering
    bool IsBalanced() const { return balanced_; }

    /// Check if using medoids clustering
    bool IsUseMedoids() const { return use_medoids_; }

    /// Get the cluster centroids
    ///
    /// Returns empty vector if the model hasn't been trained yet
    const Vector<Vector<f32>>& Centroids() const { return centroids_; }

    /// Get the medoid indices (only available when using medoids)
    ///
    /// Returns the indices of the medoid points in the original dataset
    const Vector<SizeT>& MedoidIndices() const { return medoid_indices_; }

    /// Check if the model has been trained
    bool IsTrained() const { return trained_; }

private:
    // Configuration
    SizeT n_clusters_;
    SizeT iters_;
    bool euclidean_;
    bool balanced_;
    SizeT max_balance_diff_;
    bool verbose_;
    bool trained_;
    bool use_medoids_;

    // Internal state
    Vector<Vector<f32>> centroids_;
    Vector<SizeT> medoid_indices_; // Indices of medoid points in the original data
    Vector<SizeT> assignments_;
    Vector<f32> cluster_sizes_;

    // Private implementation methods
    Vector<Vector<f32>> SampleRows(const Vector<Vector<f32>>& data);
    void AssignClusters(const Vector<Vector<f32>>& data, const Vector<f32>* data_norms = nullptr);
    void UpdateCentroids(const Vector<Vector<f32>>& data);
    void UpdateMedoids(const Vector<Vector<f32>>& data);
    void PostprocessCentroids();
    void SplitClusters(const Vector<Vector<f32>>& data);
    f32 ComputeCost(const Vector<Vector<f32>>& data) const;
    void BalanceClusters(const Vector<Vector<f32>>& data, const Vector<f32>* data_norms = nullptr);
    void UpdateMedoidsAfterBalancing(const Vector<Vector<f32>>& data);
    void UpdateMedoidForCluster(const Vector<Vector<f32>>& data, SizeT cluster_id);

    // Helper functions
    Vector<f32> ComputeDataNorms(const Vector<Vector<f32>>& data) const;
    f32 DotProduct(const Vector<f32>& a, const Vector<f32>& b) const;
    f32 EuclideanDistance(const Vector<f32>& a, const Vector<f32>& b) const;
    f32 CosineDistance(const Vector<f32>& a, const Vector<f32>& b) const;
    void NormalizeVector(Vector<f32>& vec) const;
};

} // namespace infinity

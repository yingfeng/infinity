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

#ifdef CI
#include "gtest/gtest.h"
import infinity_core;
import base_test;
#else
module;

#include "gtest/gtest.h"

module infinity_core:ut.kentro;

import :ut.base_test;
import :kentro;
import :stl;
#endif

using namespace infinity;

class KentroTest : public BaseTest {};

TEST_F(KentroTest, BasicKMeans) {
    // Create sample data: 6 points in 2D space forming 3 clusters
    Vector<Vector<f32>> data = {
        {1.0f, 1.0f},   // Cluster 1
        {1.1f, 1.1f},   // Cluster 1
        {5.0f, 5.0f},   // Cluster 2
        {5.1f, 5.1f},   // Cluster 2
        {9.0f, 9.0f},   // Cluster 3
        {9.1f, 9.1f}    // Cluster 3
    };

    KMeans kmeans(3);
    auto clusters = kmeans.Train(data);

    EXPECT_EQ(clusters.size(), 3);
    EXPECT_TRUE(kmeans.IsTrained());
    
    // Check that all points are assigned
    SizeT total_points = 0;
    for (const auto& cluster : clusters) {
        total_points += cluster.size();
    }
    EXPECT_EQ(total_points, 6);
}

TEST_F(KentroTest, EuclideanKMeans) {
    Vector<Vector<f32>> data = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f}
    };

    KMeans kmeans(2);
    kmeans.WithEuclidean(true);
    auto clusters = kmeans.Train(data);

    EXPECT_EQ(clusters.size(), 2);
    EXPECT_TRUE(kmeans.IsEuclidean());
}

TEST_F(KentroTest, BalancedKMeans) {
    // Create data with uneven natural clusters
    Vector<Vector<f32>> data = {
        {1.0f, 1.0f}, {1.1f, 1.1f}, {1.2f, 1.2f},  // 3 points near (1,1)
        {5.0f, 5.0f}, {5.1f, 5.1f}, {5.2f, 5.2f},  // 3 points near (5,5)
        {9.0f, 9.0f}, {9.1f, 9.1f}, {9.2f, 9.2f},  // 3 points near (9,9)
        {13.0f, 13.0f}  // 1 point far away
    };

    KMeans kmeans(3);
    kmeans.WithBalanced(true).WithMaxBalanceDiff(2);
    auto clusters = kmeans.Train(data);

    EXPECT_EQ(clusters.size(), 3);
    EXPECT_TRUE(kmeans.IsBalanced());

    // Check that clusters are reasonably balanced
    Vector<SizeT> sizes;
    for (const auto& cluster : clusters) {
        sizes.push_back(cluster.size());
    }
    
    auto [min_size, max_size] = std::minmax_element(sizes.begin(), sizes.end());
    EXPECT_LE(*max_size - *min_size, 2);  // Should be reasonably balanced
}

TEST_F(KentroTest, MedoidsKMeans) {
    Vector<Vector<f32>> data = {
        {1.0f, 1.0f}, {1.1f, 1.1f},
        {5.0f, 5.0f}, {5.1f, 5.1f},
        {9.0f, 9.0f}, {9.1f, 9.1f}
    };

    KMeans kmeans(3);
    kmeans.WithUseMedoids(true).WithEuclidean(true);
    auto clusters = kmeans.Train(data);

    EXPECT_EQ(clusters.size(), 3);
    EXPECT_TRUE(kmeans.IsTrained());
    EXPECT_TRUE(kmeans.IsUseMedoids());

    // Check that medoid indices are valid
    const auto& medoid_indices = kmeans.MedoidIndices();
    EXPECT_EQ(medoid_indices.size(), 3);
    for (SizeT medoid_idx : medoid_indices) {
        EXPECT_LT(medoid_idx, data.size());
    }
}

TEST_F(KentroTest, AssignMethod) {
    Vector<Vector<f32>> train_data = {
        {1.0f, 1.0f}, {1.1f, 1.1f},
        {5.0f, 5.0f}, {5.1f, 5.1f},
        {9.0f, 9.0f}, {9.1f, 9.1f}
    };

    Vector<Vector<f32>> test_data = {
        {1.05f, 1.05f},  // Should be close to cluster 1
        {9.05f, 9.05f}   // Should be close to cluster 3
    };

    KMeans kmeans(3);
    kmeans.Train(train_data);

    auto assignments = kmeans.Assign(test_data, 1);
    EXPECT_EQ(assignments.size(), 3);
    
    // Check that test points are assigned
    SizeT total_assigned = 0;
    for (const auto& cluster : assignments) {
        total_assigned += cluster.size();
    }
    EXPECT_EQ(total_assigned, 2);
}

TEST_F(KentroTest, GettersAndSetters) {
    KMeans kmeans(5);
    
    EXPECT_EQ(kmeans.NClusters(), 5);
    EXPECT_EQ(kmeans.Iterations(), 25);  // Default value
    EXPECT_FALSE(kmeans.IsEuclidean());  // Default value
    EXPECT_FALSE(kmeans.IsBalanced());   // Default value
    EXPECT_FALSE(kmeans.IsUseMedoids()); // Default value
    EXPECT_FALSE(kmeans.IsTrained());    // Not trained yet

    kmeans.WithIterations(50)
          .WithEuclidean(true)
          .WithBalanced(true)
          .WithUseMedoids(true);

    EXPECT_EQ(kmeans.Iterations(), 50);
    EXPECT_TRUE(kmeans.IsEuclidean());
    EXPECT_TRUE(kmeans.IsBalanced());
    EXPECT_TRUE(kmeans.IsUseMedoids());
}

TEST_F(KentroTest, ErrorHandling) {
    // Test insufficient points
    Vector<Vector<f32>> small_data = {
        {1.0f, 1.0f},
        {2.0f, 2.0f}
    };

    KMeans kmeans(3);  // More clusters than points
    
    // This should throw an error
    EXPECT_THROW(kmeans.Train(small_data), UnrecoverableException);
}

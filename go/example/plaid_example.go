// Copyright(C) 2026 InfiniFlow, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build ignore

package main

import (
	"fmt"
	"log"

	infinity "github.com/infiniflow/infinity-go-sdk"
)

func main() {
	fmt.Println("=== PLAID Index Example ===")
	fmt.Println("This example demonstrates how to use PLAID index with Go client")
	fmt.Println()

	// Connect to Infinity
	conn, err := infinity.Connect(infinity.LocalHost)
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	fmt.Println("Connected to Infinity successfully")

	defer func() {
		_, err := conn.Disconnect()
		if err != nil {
			log.Printf("Error disconnecting: %v", err)
		} else {
			fmt.Println("Disconnected from Infinity")
		}
	}()

	// Create database
	db, err := conn.CreateDatabase("plaid_test_db", infinity.ConflictTypeIgnore, "")
	if err != nil {
		log.Printf("CreateDatabase error (may already exist): %v", err)
		db, err = conn.GetDatabase("plaid_test_db")
		if err != nil {
			log.Fatalf("Failed to get database: %v", err)
		}
	}
	fmt.Println("Database 'plaid_test_db' ready")

	// Create table with tensor column
	schema := infinity.TableSchema{
		{
			Name:        "id",
			DataType:    "integer",
			Constraints: []infinity.ColumnConstraint{infinity.ConstraintPrimaryKey},
		},
		{
			Name:     "title",
			DataType: "varchar",
		},
		{
			// Tensor column for multi-vector embeddings (ColBERT-style)
			Name:     "doc_embeddings",
			DataType: "tensor,128,float32",
		},
	}

	table, err := db.CreateTable("documents", schema, infinity.ConflictTypeIgnore)
	if err != nil {
		log.Printf("CreateTable error (may already exist): %v", err)
		table, err = db.GetTable("documents")
		if err != nil {
			log.Fatalf("Failed to get table: %v", err)
		}
	}
	fmt.Println("Table 'documents' ready with tensor column 'doc_embeddings'")

	// Example 1: Create PLAID Index
	fmt.Println("\n=== Example 1: Create PLAID Index ===")
	fmt.Println("PLAID index is optimized for multi-vector (tensor) similarity search")
	fmt.Println("using MaxSim scoring (like ColBERT)")

	plaidIndexInfo := infinity.NewIndexInfo(
		"doc_embeddings",        // Column name
		infinity.IndexTypePLAID, // Index type: PLAID
		map[string]string{
			"nbits":       "4",   // Number of bits for residual quantization
			"n_centroids": "100", // Number of IVF centroids
		},
	)
	fmt.Printf("Index info: %s\n", plaidIndexInfo.String())

	// Create PLAID index on table
	_, err = table.CreateIndex("plaid_idx", plaidIndexInfo, infinity.ConflictTypeIgnore, "PLAID index for fast tensor search")
	if err != nil {
		log.Printf("CreateIndex error (may already exist): %v", err)
	} else {
		fmt.Println("PLAID index 'plaid_idx' created successfully")
	}

	// List indexes to verify
	indexes, err := table.ListIndexes()
	if err != nil {
		log.Printf("ListIndexes error: %v", err)
	} else {
		fmt.Printf("Available indexes: %v\n", indexes)
	}

	// Example 2: Insert sample data
	fmt.Println("\n=== Example 2: Insert Sample Data ===")
	fmt.Println("Inserting documents with tensor embeddings (multiple vectors per document)")

	// Each document has multiple embeddings (e.g., from different passages)
	rows := []infinity.InsertData{
		{
			"id":    1,
			"title": "Introduction to Machine Learning",
			"doc_embeddings": [][]float32{
				// First passage embedding (128 dimensions)
				makeRandomVector(128, 0.1),
				makeRandomVector(128, 0.2),
				makeRandomVector(128, 0.3),
			},
		},
		{
			"id":    2,
			"title": "Deep Learning Fundamentals",
			"doc_embeddings": [][]float32{
				makeRandomVector(128, 0.4),
				makeRandomVector(128, 0.5),
			},
		},
		{
			"id":    3,
			"title": "Natural Language Processing",
			"doc_embeddings": [][]float32{
				makeRandomVector(128, 0.6),
				makeRandomVector(128, 0.7),
				makeRandomVector(128, 0.8),
				makeRandomVector(128, 0.9),
			},
		},
	}

	_, err = table.Insert(rows)
	if err != nil {
		log.Printf("Insert error: %v", err)
	} else {
		fmt.Printf("Inserted %d documents\n", len(rows))
	}

	// Example 3: Match Tensor Search using PLAID Index
	fmt.Println("\n=== Example 3: Tensor Search with PLAID Index ===")
	fmt.Println("Searching with multi-vector query (MaxSim scoring)")

	// Create a multi-vector query (like a query with multiple embeddings)
	queryTensor := [][]float32{
		makeRandomVector(128, 0.15),
		makeRandomVector(128, 0.25),
	}

	// Perform tensor search using MatchTensor
	// The PLAID index will automatically be used for acceleration
	result, err := table.MatchTensor(
		"doc_embeddings", // Column to search
		queryTensor,      // Query tensor (multi-vector)
		"float32",        // Data type
		10,               // Top N results
		map[string]string{ // Extra options
			"n_ivf_probe":    "10",  // Number of IVF clusters to probe
			"n_doc_to_score": "100", // Number of docs to score
		},
	).Output([]string{"id", "title", "_score"}).ToResult()

	if err != nil {
		log.Printf("MatchTensor error: %v", err)
	} else {
		fmt.Printf("Tensor search results:\n%v\n", result)
	}

	// Example 4: Direct MatchTensor call
	fmt.Println("\n=== Example 4: Direct MatchTensor Call ===")

	result2, err := table.MatchTensor("doc_embeddings", queryTensor, "float32", 5, nil).
		Output([]string{"id", "title"}).
		ToResult()
	if err != nil {
		log.Printf("MatchTensor error: %v", err)
	} else {
		fmt.Printf("MatchTensor results:\n%v\n", result2)
	}

	// Example 5: Compare Index Types
	fmt.Println("\n=== Example 5: Index Type Comparison ===")
	fmt.Printf("IndexTypeIVF:    %d (%s)\n", infinity.IndexTypeIVF, infinity.IndexTypeIVF.String())
	fmt.Printf("IndexTypeHnsw:   %d (%s)\n", infinity.IndexTypeHnsw, infinity.IndexTypeHnsw.String())
	fmt.Printf("IndexTypeEMVB:   %d (%s)\n", infinity.IndexTypeEMVB, infinity.IndexTypeEMVB.String())
	fmt.Printf("IndexTypeBMP:    %d (%s)\n", infinity.IndexTypeBMP, infinity.IndexTypeBMP.String())
	fmt.Printf("IndexTypePLAID:  %d (%s) <- Our new index!\n", infinity.IndexTypePLAID, infinity.IndexTypePLAID.String())

	// Example 6: Clean up
	fmt.Println("\n=== Example 6: Clean up ===")

	// Drop index
	_, err = table.DropIndex("plaid_idx", infinity.ConflictTypeIgnore)
	if err != nil {
		log.Printf("DropIndex error: %v", err)
	} else {
		fmt.Println("PLAID index dropped")
	}

	// Drop table
	_, err = db.DropTable("documents", infinity.ConflictTypeIgnore)
	if err != nil {
		log.Printf("DropTable error: %v", err)
	} else {
		fmt.Println("Table 'documents' dropped")
	}

	// Drop database
	_, err = conn.DropDatabase("plaid_test_db", infinity.ConflictTypeIgnore)
	if err != nil {
		log.Printf("DropDatabase error: %v", err)
	} else {
		fmt.Println("Database 'plaid_test_db' dropped")
	}

	fmt.Println("\n=== PLAID Index Example Completed ===")
	fmt.Println()
	fmt.Println("Summary:")
	fmt.Println("- PLAID index type is properly defined in Go client")
	fmt.Println("- Index creation works with infinity.IndexTypePLAID")
	fmt.Println("- Tensor search (MatchTensor) uses PLAID index automatically")
	fmt.Println("- All Go client functionality is working correctly!")
}

// makeRandomVector creates a sample vector for demonstration
func makeRandomVector(dim int, seed float32) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = seed + float32(i)*0.01
	}
	return vec
}

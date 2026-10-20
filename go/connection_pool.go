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

package infinity

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ConnectionPoolConfig contains configuration for the connection pool
type ConnectionPoolConfig struct {
	URI                 URI
	InitialSize         int           // Initial number of idle connections to create.
	MaxOpen             int           // Maximum total open connections.
	MaxIdle             int           // Maximum number of idle connections kept in the pool.
	MaxIdleTime         time.Duration // Maximum time a connection can be idle.
	HealthCheckInterval time.Duration // Interval for health checks (ignored).
}

// DefaultConnectionPoolConfig returns a default configuration
func DefaultConnectionPoolConfig(uri URI) ConnectionPoolConfig {
	return ConnectionPoolConfig{
		URI:                 uri,
		InitialSize:         0,
		MaxOpen:             10,
		MaxIdle:             2,
		MaxIdleTime:         30 * time.Minute,
		HealthCheckInterval: 5 * time.Minute,
	}
}

// PooledConnection wraps a connection with pool metadata
type PooledConnection struct {
	conn       *InfinityConnection
	createdAt  time.Time
	lastUsedAt time.Time
	inUse      bool
}

// ConnectionPool manages a pool of Infinity connections
type ConnectionPool struct {
	uri          URI
	config       ConnectionPoolConfig
	factory      func(URI) (*InfinityConnection, error)
	mu           sync.Mutex
	cond         *sync.Cond
	closed       bool
	available    []*PooledConnection
	createdCount int
	openingCount int
	connToPooled map[*InfinityConnection]*PooledConnection
}

// NewConnectionPool creates a new connection pool
func NewConnectionPool(config ConnectionPoolConfig, factory func(URI) (*InfinityConnection, error)) (*ConnectionPool, error) {
	if config.InitialSize < 0 {
		config.InitialSize = 0
	}
	if config.MaxIdleTime <= 0 {
		config.MaxIdleTime = 30 * time.Minute
	}
	if config.MaxOpen <= 0 {
		if config.InitialSize > 0 {
			config.MaxOpen = config.InitialSize
		} else {
			config.MaxOpen = 10
		}
	}
	if config.MaxIdle < 0 {
		config.MaxIdle = 0
	}
	if config.MaxIdle > config.MaxOpen {
		config.MaxIdle = config.MaxOpen
	}
	if config.InitialSize > config.MaxIdle {
		config.InitialSize = config.MaxIdle
	}
	if factory == nil {
		factory = Connect
	}

	pool := &ConnectionPool{
		uri:          config.URI,
		config:       config,
		factory:      factory,
		available:    make([]*PooledConnection, 0, config.MaxIdle),
		connToPooled: make(map[*InfinityConnection]*PooledConnection),
	}
	pool.cond = sync.NewCond(&pool.mu)

	// Create initial connections
	pool.mu.Lock()
	for i := 0; i < config.InitialSize; i++ {
		pooledConn, err := pool.createConnection()
		if err != nil {
			for conn := range pool.connToPooled {
				conn.Disconnect()
			}
			pool.connToPooled = make(map[*InfinityConnection]*PooledConnection)
			pool.available = nil
			pool.createdCount = 0
			pool.mu.Unlock()
			return nil, fmt.Errorf("failed to create initial connection %d: %w", i, err)
		}
		pool.registerConnection(pooledConn)
		pool.available = append(pool.available, pooledConn)
	}
	pool.mu.Unlock()

	return pool, nil
}

// createConnection creates a new pooled connection without registering it
func (p *ConnectionPool) createConnection() (*PooledConnection, error) {
	conn, err := p.factory(p.uri)
	if err != nil {
		return nil, err
	}

	now := time.Now()
	return &PooledConnection{
		conn:       conn,
		createdAt:  now,
		lastUsedAt: now,
	}, nil
}

// registerConnection adds a newly created connection to pool accounting.
// Caller must hold p.mu lock.
func (p *ConnectionPool) registerConnection(pooledConn *PooledConnection) {
	p.connToPooled[pooledConn.conn] = pooledConn
	p.createdCount++
}

// reserveOpenSlot claims capacity for a connection that will be opened outside the lock.
// Caller must hold p.mu lock.
func (p *ConnectionPool) reserveOpenSlot() bool {
	if p.createdCount+p.openingCount >= p.config.MaxOpen {
		return false
	}
	p.openingCount++
	return true
}

// Get gets a connection from the pool.
// It blocks until a connection is available or the pool is closed.
func (p *ConnectionPool) Get() (*InfinityConnection, error) {
	return p.GetContext(context.Background())
}

// GetContext gets a connection from the pool with context
func (p *ConnectionPool) GetContext(ctx context.Context) (*InfinityConnection, error) {
	// Check if context is already done
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			p.mu.Lock()
			p.cond.Broadcast()
			p.mu.Unlock()
		case <-done:
		}
	}()

	for {
		if p.closed {
			return nil, NewInfinityException(int(ErrorCodeClientClose), "Connection pool is closed")
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		valid := p.available[:0]
		for _, pooledConn := range p.available {
			if !pooledConn.conn.IsConnected() {
				p.removeConnection(pooledConn)
				continue
			}
			if time.Since(pooledConn.lastUsedAt) > p.config.MaxIdleTime {
				p.removeConnection(pooledConn)
				continue
			}
			valid = append(valid, pooledConn)
		}
		p.available = valid

		if len(p.available) > 0 {
			pooledConn := p.available[len(p.available)-1]
			p.available = p.available[:len(p.available)-1]
			pooledConn.lastUsedAt = time.Now()
			pooledConn.inUse = true
			return pooledConn.conn, nil
		}

		if p.reserveOpenSlot() {
			p.mu.Unlock()
			pooledConn, err := p.createConnection()
			p.mu.Lock()
			p.openingCount--
			if err != nil {
				p.cond.Signal()
				return nil, err
			}
			if p.closed {
				pooledConn.conn.Disconnect()
				p.cond.Signal()
				return nil, NewInfinityException(int(ErrorCodeClientClose), "Connection pool is closed")
			}
			if err := ctx.Err(); err != nil {
				pooledConn.conn.Disconnect()
				p.cond.Signal()
				return nil, err
			}
			p.registerConnection(pooledConn)
			pooledConn.inUse = true
			return pooledConn.conn, nil
		}

		p.cond.Wait()
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
}

// Put returns a connection to the pool
func (p *ConnectionPool) Put(conn *InfinityConnection) error {
	if conn == nil {
		return NewInfinityException(int(ErrorCodeInvalidParameterValue), "Cannot return nil connection to pool")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		conn.Disconnect()
		return NewInfinityException(int(ErrorCodeClientClose), "Connection pool is closed")
	}

	pooledConn, ok := p.connToPooled[conn]
	if !ok {
		// Connection not from this pool, close it
		conn.Disconnect()
		return NewInfinityException(int(ErrorCodeInvalidParameterValue), "Connection not from this pool")
	}

	// Check if connection is still valid
	if !conn.IsConnected() {
		p.removeConnection(pooledConn)
		return NewInfinityException(int(ErrorCodeClientClose), "Connection is dead, removed from pool")
	}

	if !pooledConn.inUse {
		return NewInfinityException(int(ErrorCodeInvalidParameterValue), "Connection is already in pool (double release)")
	}

	// Release the connection back to the pool; if the idle list is at
	// MaxIdle capacity, close the excess connection instead of keeping it.
	pooledConn.lastUsedAt = time.Now()
	pooledConn.inUse = false
	if len(p.available) < p.config.MaxIdle {
		p.available = append(p.available, pooledConn)
		p.cond.Signal()
		return nil
	}

	p.removeConnection(pooledConn)
	return nil
}

// removeConnection closes a pooled connection and removes it from tracking
func (p *ConnectionPool) removeConnection(pooledConn *PooledConnection) {
	if pooledConn.conn == nil {
		return
	}

	conn := pooledConn.conn
	pooledConn.conn = nil
	pooledConn.inUse = false
	delete(p.connToPooled, conn)
	conn.Disconnect()
	p.createdCount--
	p.cond.Signal()
}

// isClosed checks if the pool is closed
func (p *ConnectionPool) isClosed() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.closed
}

// Close closes all connections in the pool
func (p *ConnectionPool) Close() error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return NewInfinityException(int(ErrorCodeClientClose), "Connection pool already closed")
	}
	p.closed = true

	// Close all connections
	for conn := range p.connToPooled {
		conn.Disconnect()
	}
	// Clear maps and slices
	p.connToPooled = make(map[*InfinityConnection]*PooledConnection)
	p.available = nil
	p.createdCount = 0

	p.cond.Broadcast()
	p.mu.Unlock()
	return nil
}

// Stats returns statistics about the pool
type PoolStats struct {
	TotalConnections     int
	AvailableConnections int
	InUseConnections     int
	Closed               bool
}

// Stats returns current pool statistics
func (p *ConnectionPool) Stats() PoolStats {
	p.mu.Lock()
	defer p.mu.Unlock()

	total := p.createdCount
	available := len(p.available)
	inUse := total - available

	return PoolStats{
		TotalConnections:     total,
		AvailableConnections: available,
		InUseConnections:     inUse,
		Closed:               p.closed,
	}
}

// Size returns the current number of connections in the pool
func (p *ConnectionPool) Size() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.createdCount
}

// Available returns the number of available connections in the pool
func (p *ConnectionPool) Available() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.available)
}

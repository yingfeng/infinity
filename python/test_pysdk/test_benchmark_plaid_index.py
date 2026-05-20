"""
Unit tests for benchmark_plaid_index.py setup_table method.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add example directory to path to import benchmark module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'example'))

from benchmark_plaid_index import PlaidBenchmark


class TestPlaidBenchmarkSetupTable:
    """Tests for PlaidBenchmark.setup_table method."""

    def test_setup_table_opens_existing_table(self, capsys):
        """Test that setup_table successfully opens an existing table."""
        # Arrange
        benchmark = PlaidBenchmark()
        benchmark.table_name = "test_table"
        
        # Mock db_instance and table instance
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.get_table.return_value = mock_table
        benchmark.db_instance = mock_db
        
        # Act
        benchmark.setup_table(drop_existing=True)
        
        # Assert
        mock_db.get_table.assert_called_once_with("test_table")
        assert benchmark.table_instance == mock_table
        
        # Check output messages
        captured = capsys.readouterr()
        assert "Opening existing table 'test_table'" in captured.out
        assert "Table opened" in captured.out

    def test_setup_table_raises_exception_when_table_not_found(self):
        """Test that setup_table raises exception when table doesn't exist."""
        # Arrange
        benchmark = PlaidBenchmark()
        benchmark.table_name = "nonexistent_table"
        
        # Mock db_instance to raise exception
        mock_db = MagicMock()
        mock_db.get_table.side_effect = Exception("Table not found")
        benchmark.db_instance = mock_db
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            benchmark.setup_table(drop_existing=True)
        
        assert "Table not found" in str(exc_info.value)

    def test_setup_table_raises_exception_when_db_instance_none(self):
        """Test that setup_table raises exception when db_instance is None."""
        # Arrange
        benchmark = PlaidBenchmark()
        benchmark.db_instance = None
        benchmark.table_name = "test_table"
        
        # Act & Assert
        with pytest.raises(AttributeError):
            benchmark.setup_table(drop_existing=True)

    def test_setup_table_uses_correct_table_name(self):
        """Test that setup_table uses the configured table name."""
        # Arrange
        benchmark = PlaidBenchmark()
        benchmark.table_name = "custom_benchmark_table"
        
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.get_table.return_value = mock_table
        benchmark.db_instance = mock_db
        
        # Act
        benchmark.setup_table(drop_existing=True)
        
        # Assert
        mock_db.get_table.assert_called_once_with("custom_benchmark_table")

    def test_setup_table_drop_existing_parameter_ignored(self, capsys):
        """Test that drop_existing parameter is accepted but ignored (table is opened, not dropped)."""
        # Arrange
        benchmark = PlaidBenchmark()
        benchmark.table_name = "test_table"
        
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.get_table.return_value = mock_table
        benchmark.db_instance = mock_db
        
        # Act - both True and False should behave the same
        benchmark.setup_table(drop_existing=True)
        benchmark.table_instance = None  # Reset
        
        benchmark.setup_table(drop_existing=False)
        
        # Assert - get_table called twice (no drop_table called)
        assert mock_db.get_table.call_count == 2
        assert benchmark.table_instance == mock_table

    def test_setup_table_with_different_table_names(self):
        """Test setup_table with various table names."""
        table_names = ["simple_table", "table_with_underscore", "TableWithCamelCase", "table123"]
        
        for table_name in table_names:
            # Arrange
            benchmark = PlaidBenchmark()
            benchmark.table_name = table_name
            
            mock_table = MagicMock()
            mock_db = MagicMock()
            mock_db.get_table.return_value = mock_table
            benchmark.db_instance = mock_db
            
            # Act
            benchmark.setup_table(drop_existing=True)
            
            # Assert
            mock_db.get_table.assert_called_once_with(table_name)
            assert benchmark.table_instance == mock_table


class TestPlaidBenchmarkIntegration:
    """Integration tests that require a running Infinity server."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for integration tests."""
        # Skip these tests if no server is available
        pytest.skip("Integration tests require running Infinity server")

    def test_setup_table_integration(self):
        """Integration test for setup_table with real server."""
        # This test requires a running Infinity server
        benchmark = PlaidBenchmark()
        benchmark.connect()
        benchmark.setup_table(drop_existing=True)
        assert benchmark.table_instance is not None
        benchmark.cleanup()

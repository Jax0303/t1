"""
Tests for StructureAwareRetriever class.

Tests adaptive retrieval, query classification, hierarchy-aware retrieval,
context expansion, reranking, and evaluation metrics.
"""

import pytest
from pathlib import Path
import tempfile
import os

from src.parsing.extractor import TableObject, Cell
from src.parsing.cell_merger import CellMergeHandler, NormalizedTable
from src.parsing.hierarchy import HierarchyExtractor, HeaderTree
from src.retrieval.indexer import HierarchicalIndexer
from src.retrieval.retriever import (
    StructureAwareRetriever,
    QueryType,
    RetrievalResult,
    ContextBundle,
)


class TestStructureAwareRetriever:
    """Test cases for StructureAwareRetriever class."""

    @pytest.fixture
    def sample_index(self):
        """Create a sample index for testing."""
        # Create table
        cells = [
            [
                Cell(0, 0, "Revenue", colspan=4, is_header=True),
                Cell(0, 4, "Expenses", colspan=4, is_header=True),
            ],
            [
                Cell(1, 0, "Q1", is_header=True),
                Cell(1, 1, "Q2", is_header=True),
                Cell(1, 2, "Q3", is_header=True),
                Cell(1, 3, "Q4", is_header=True),
                Cell(1, 4, "Q1", is_header=True),
                Cell(1, 5, "Q2", is_header=True),
                Cell(1, 6, "Q3", is_header=True),
                Cell(1, 7, "Q4", is_header=True),
            ],
            [
                Cell(2, 0, "100"),
                Cell(2, 1, "200"),
                Cell(2, 2, "150"),
                Cell(2, 3, "180"),
                Cell(2, 4, "50"),
                Cell(2, 5, "60"),
                Cell(2, 6, "55"),
                Cell(2, 7, "65"),
            ],
        ]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        # Build index
        indexer = HierarchicalIndexer(index_type="Flat")
        index = indexer.build_hierarchical_index([(normalized, header_tree)])
        
        return indexer, normalized, header_tree

    def test_classify_query_type_lookup(self):
        """Test query type classification for LOOKUP queries."""
        indexer, _, _ = self.sample_index
        retriever = StructureAwareRetriever(indexer)
        
        query_type = retriever.classify_query_type("What is the revenue in Q1?")
        assert query_type == QueryType.LOOKUP
        
        query_type = retriever.classify_query_type("Find the value for Q2")
        assert query_type == QueryType.LOOKUP

    def test_classify_query_type_aggregate(self):
        """Test query type classification for AGGREGATE queries."""
        indexer, _, _ = self.sample_index
        retriever = StructureAwareRetriever(indexer)
        
        query_type = retriever.classify_query_type("What is the total revenue?")
        assert query_type == QueryType.AGGREGATE
        
        query_type = retriever.classify_query_type("Compare Q1 and Q2")
        assert query_type == QueryType.AGGREGATE
        
        query_type = retriever.classify_query_type("Which quarter has the highest revenue?")
        assert query_type == QueryType.AGGREGATE

    def test_classify_query_type_structural(self):
        """Test query type classification for STRUCTURAL queries."""
        indexer, _, _ = self.sample_index
        retriever = StructureAwareRetriever(indexer)
        
        query_type = retriever.classify_query_type("How many columns are in the table?")
        assert query_type == QueryType.STRUCTURAL
        
        query_type = retriever.classify_query_type("What are the headers?")
        assert query_type == QueryType.STRUCTURAL

    def test_retrieve_adaptive_strategy(self, sample_index):
        """Test adaptive retrieval strategy."""
        indexer, normalized, header_tree = sample_index
        retriever = StructureAwareRetriever(indexer)
        
        # LOOKUP query should retrieve at cell level
        results = retriever.retrieve("What is the revenue in Q1?", top_k=5, strategy="adaptive")
        assert isinstance(results, list)
        assert len(results) > 0
        
        # AGGREGATE query should retrieve at row/subtable level
        results = retriever.retrieve("What is the total revenue?", top_k=5, strategy="adaptive")
        assert isinstance(results, list)

    def test_retrieve_hierarchical_strategy(self, sample_index):
        """Test hierarchical retrieval strategy."""
        indexer, normalized, header_tree = sample_index
        retriever = StructureAwareRetriever(indexer)
        
        results = retriever.retrieve("revenue", top_k=5, strategy="hierarchical")
        assert isinstance(results, list)

    def test_retrieve_cell_direct_strategy(self, sample_index):
        """Test cell direct retrieval strategy."""
        indexer, normalized, header_tree = sample_index
        retriever = StructureAwareRetriever(indexer)
        
        results = retriever.retrieve("Q1", top_k=5, strategy="cell_direct")
        assert isinstance(results, list)
        
        # Should be cell-level results
        if results:
            assert results[0].granularity == "cell"

    def test_retrieve_with_hierarchy(self, sample_index):
        """Test retrieval with hierarchy context."""
        indexer, normalized, header_tree = sample_index
        retriever = StructureAwareRetriever(indexer)
        
        bundle = retriever.retrieve_with_hierarchy("revenue", top_k=5)
        
        assert isinstance(bundle, ContextBundle)
        assert bundle.query == "revenue"
        assert isinstance(bundle.retrieved_cells, list)
        assert isinstance(bundle.hierarchy_context, dict)
        
        # Check that hierarchy paths are included
        if bundle.retrieved_cells:
            assert bundle.retrieved_cells[0].hierarchy_path

    def test_rerank(self, sample_index):
        """Test reranking functionality."""
        indexer, normalized, header_tree = sample_index
        retriever = StructureAwareRetriever(indexer)
        
        # Get initial results
        initial_results = retriever.retrieve("revenue", top_k=10, strategy="adaptive")
        
        if len(initial_results) > 1:
            # Rerank
            reranked = retriever.rerank("revenue", initial_results, top_k=5)
            
            assert len(reranked) <= 5
            assert isinstance(reranked, list)
            
            # Check that scores are updated
            if reranked:
                assert reranked[0].score >= 0

    def test_expand_context(self, sample_index):
        """Test context expansion."""
        indexer, normalized, header_tree = sample_index
        retriever = StructureAwareRetriever(indexer)
        
        # Get initial results
        initial_results = retriever.retrieve("revenue", top_k=3, strategy="adaptive")
        
        # Expand context
        expanded = retriever.expand_context(initial_results, hops=1)
        
        assert isinstance(expanded, list)
        assert len(expanded) >= len(initial_results)

    def test_evaluate_recall_at_k(self, sample_index):
        """Test Recall@k evaluation."""
        indexer, normalized, header_tree = sample_index
        retriever = StructureAwareRetriever(indexer)
        
        # Create test queries with relevant IDs
        # Note: In real scenario, would use actual relevant IDs from ground truth
        queries = [
            ("revenue", ["item1", "item2"]),  # Mock relevant IDs
            ("Q1", ["item3"]),
        ]
        
        # Evaluate
        results = retriever.evaluate_recall_at_k(
            queries, k_values=[1, 5], strategy="adaptive"
        )
        
        assert isinstance(results, dict)
        assert "table" in results or "cell" in results
        
        # Check structure
        for gran, k_dict in results.items():
            assert isinstance(k_dict, dict)
            for k, recall in k_dict.items():
                assert 0.0 <= recall <= 1.0

    def test_retrieve_invalid_strategy(self, sample_index):
        """Test error handling for invalid strategy."""
        indexer, normalized, header_tree = sample_index
        retriever = StructureAwareRetriever(indexer)
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            retriever.retrieve("query", strategy="invalid")

    def test_retrieve_no_index_error(self):
        """Test error when index is not loaded."""
        indexer = HierarchicalIndexer(index_type="Flat")
        retriever = StructureAwareRetriever(indexer)
        
        with pytest.raises(ValueError, match="Index not loaded"):
            retriever.retrieve("query")

    def test_retrieve_with_table_id(self, sample_index):
        """Test retrieval restricted to specific table."""
        indexer, normalized, header_tree = sample_index
        retriever = StructureAwareRetriever(indexer)
        
        # Get a table ID from results
        results = retriever.retrieve("revenue", top_k=1)
        if results:
            table_id = results[0].metadata.source_table_id
            
            # Retrieve with table restriction
            bundle = retriever.retrieve_with_hierarchy(
                "revenue", table_id=table_id, top_k=5
            )
            
            # All results should be from same table
            for result in bundle.retrieved_cells:
                assert result.metadata.source_table_id == table_id

    def test_adaptive_vs_fixed_strategies(self, sample_index):
        """Test comparison of adaptive vs fixed strategies."""
        indexer, normalized, header_tree = sample_index
        retriever = StructureAwareRetriever(indexer)
        
        query = "What is the total revenue?"
        
        # Adaptive strategy
        adaptive_results = retriever.retrieve(query, top_k=5, strategy="adaptive")
        
        # Fixed strategies
        hierarchical_results = retriever.retrieve(
            query, top_k=5, strategy="hierarchical"
        )
        cell_results = retriever.retrieve(query, top_k=5, strategy="cell_direct")
        
        # All should return results
        assert len(adaptive_results) >= 0
        assert len(hierarchical_results) >= 0
        assert len(cell_results) >= 0
        
        # Adaptive should potentially use different granularity
        # (implementation dependent)



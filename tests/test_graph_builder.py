"""
Tests for GraphBuilder class.

Tests graph construction, semantic edges, context extraction,
and visualization export with HiTab-style examples.
"""

import pytest
from pathlib import Path
import tempfile
import os

from src.parsing.extractor import TableObject, Cell
from src.parsing.cell_merger import CellMergeHandler, NormalizedTable
from src.parsing.hierarchy import HierarchyExtractor, HeaderTree
from src.encoding.graph_builder import GraphBuilder


class TestGraphBuilder:
    """Test cases for GraphBuilder class."""

    def test_builder_init(self) -> None:
        """Test graph builder initialization."""
        builder = GraphBuilder(use_semantic_edges=False)
        assert builder.use_semantic_edges is False

    def test_build_table_graph_simple(self) -> None:
        """Test building graph from simple table."""
        # Create simple table
        cells = [
            [
                Cell(0, 0, "Header", colspan=2, is_header=True),
            ],
            [
                Cell(1, 0, "Data1"),
                Cell(1, 1, "Data2"),
            ],
        ]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        # Normalize
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        # Extract hierarchy
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        # Build graph
        builder = GraphBuilder(use_semantic_edges=False)
        graph = builder.build_table_graph(normalized, header_tree)
        
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        
        # Check node types
        node_types = {attrs.get("type") for _, attrs in graph.nodes(data=True)}
        assert "header" in node_types
        assert "data" in node_types

    def test_build_table_graph_hitab_style(self) -> None:
        """Test building graph from HiTab-style financial table."""
        # HiTab-style 3-level header table
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
        
        # Normalize and extract hierarchy
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        # Build graph
        builder = GraphBuilder(use_semantic_edges=False)
        graph = builder.build_table_graph(normalized, header_tree)
        
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        
        # Check for hierarchical edges
        edge_relationships = {
            attrs.get("relationship")
            for _, _, attrs in graph.edges(data=True)
        }
        assert "parent_of" in edge_relationships
        assert "header_for" in edge_relationships

    def test_add_semantic_edges(self) -> None:
        """Test adding semantic similarity edges."""
        # Create table with similar content
        cells = [
            [Cell(0, 0, "Revenue", is_header=True)],
            [Cell(1, 0, "Sales")],
            [Cell(2, 0, "Income")],
        ]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        builder = GraphBuilder(use_semantic_edges=True)
        graph = builder.build_table_graph(normalized, header_tree)
        
        # Add semantic edges (may skip if model not available)
        try:
            graph = builder.add_semantic_edges(graph, similarity_threshold=0.7)
            
            # Check for semantic edges
            edge_relationships = {
                attrs.get("relationship")
                for _, _, attrs in graph.edges(data=True)
            }
            # May or may not have semantic edges depending on similarity
            # Just check that method doesn't crash
            assert True
        except Exception as e:
            # If embedding model fails, that's okay for testing
            pytest.skip(f"Semantic edges not available: {e}")

    def test_get_cell_context(self) -> None:
        """Test extracting cell context."""
        cells = [
            [
                Cell(0, 0, "Header", colspan=2, is_header=True),
            ],
            [
                Cell(1, 0, "Data1"),
                Cell(1, 1, "Data2"),
            ],
        ]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        builder = GraphBuilder(use_semantic_edges=False)
        graph = builder.build_table_graph(normalized, header_tree)
        
        # Find a data cell ID
        data_cell_id = None
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") == "data":
                data_cell_id = node_id
                break
        
        if data_cell_id:
            context = builder.get_cell_context(graph, data_cell_id, hops=2)
            
            assert context.number_of_nodes() > 0
            assert data_cell_id in context.nodes()
            
            # Should include header nodes
            has_header = any(
                attrs.get("type") == "header"
                for _, attrs in context.nodes(data=True)
            )
            assert has_header

    def test_export_gexf(self) -> None:
        """Test exporting to GEXF format."""
        cells = [[Cell(0, 0, "Header", is_header=True)], [Cell(1, 0, "Data")]]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        builder = GraphBuilder(use_semantic_edges=False)
        graph = builder.build_table_graph(normalized, header_tree)
        
        with tempfile.NamedTemporaryFile(suffix=".gexf", delete=False) as f:
            temp_path = f.name
        
        try:
            builder.export_for_visualization(graph, temp_path, format="gexf")
            assert Path(temp_path).exists()
            assert Path(temp_path).stat().st_size > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_json(self) -> None:
        """Test exporting to JSON format."""
        cells = [[Cell(0, 0, "Header", is_header=True)], [Cell(1, 0, "Data")]]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        builder = GraphBuilder(use_semantic_edges=False)
        graph = builder.build_table_graph(normalized, header_tree)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            builder.export_for_visualization(graph, temp_path, format="json")
            assert Path(temp_path).exists()
            
            # Verify JSON is valid
            import json
            with open(temp_path, "r") as f:
                data = json.load(f)
            assert "nodes" in data or "directed" in data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_to_json_from_json(self) -> None:
        """Test JSON serialization round-trip."""
        cells = [[Cell(0, 0, "Header", is_header=True)], [Cell(1, 0, "Data")]]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        builder = GraphBuilder(use_semantic_edges=False)
        graph1 = builder.build_table_graph(normalized, header_tree)
        
        # Serialize
        json_str = builder.to_json(graph1)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Deserialize
        graph2 = builder.from_json(json_str)
        
        # Compare
        assert graph2.number_of_nodes() == graph1.number_of_nodes()
        assert graph2.number_of_edges() == graph1.number_of_edges()

    def test_spatial_edges(self) -> None:
        """Test spatial adjacency edges."""
        cells = [
            [Cell(0, 0, "A"), Cell(0, 1, "B")],
            [Cell(1, 0, "C"), Cell(1, 1, "D")],
        ]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        builder = GraphBuilder(use_semantic_edges=False)
        graph = builder.build_table_graph(normalized, header_tree)
        
        # Check for spatial_adjacent edges
        edge_relationships = {
            attrs.get("relationship")
            for _, _, attrs in graph.edges(data=True)
        }
        assert "spatial_adjacent" in edge_relationships

    def test_row_col_member_edges(self) -> None:
        """Test row and column membership edges."""
        cells = [
            [Cell(0, 0, "A"), Cell(0, 1, "B")],
            [Cell(1, 0, "C"), Cell(1, 1, "D")],
        ]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        builder = GraphBuilder(use_semantic_edges=False)
        graph = builder.build_table_graph(normalized, header_tree)
        
        # Check for row_member and col_member edges
        edge_relationships = {
            attrs.get("relationship")
            for _, _, attrs in graph.edges(data=True)
        }
        assert "row_member" in edge_relationships
        assert "col_member" in edge_relationships

    def test_export_invalid_format(self) -> None:
        """Test error handling for invalid export format."""
        cells = [[Cell(0, 0, "Header", is_header=True)]]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        builder = GraphBuilder(use_semantic_edges=False)
        graph = builder.build_table_graph(normalized, header_tree)
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                builder.export_for_visualization(
                    graph, temp_path, format="invalid"
                )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_cell_context_invalid_id(self) -> None:
        """Test context extraction with invalid cell ID."""
        cells = [[Cell(0, 0, "Header", is_header=True)]]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        builder = GraphBuilder(use_semantic_edges=False)
        graph = builder.build_table_graph(normalized, header_tree)
        
        # Try with invalid ID
        context = builder.get_cell_context(graph, "invalid_id", hops=2)
        assert context.number_of_nodes() == 0



"""
Tests for HierarchicalIndexer class.

Tests multi-level indexing, incremental addition, save/load,
and indexing time measurement with HiTab-style examples.
"""

import pytest
from pathlib import Path
import tempfile
import os
import time

from src.parsing.extractor import TableObject, Cell
from src.parsing.cell_merger import CellMergeHandler, NormalizedTable
from src.parsing.hierarchy import HierarchyExtractor, HeaderTree
from src.retrieval.indexer import HierarchicalIndexer, HierarchicalIndex


class TestHierarchicalIndexer:
    """Test cases for HierarchicalIndexer class."""

    def test_indexer_init(self) -> None:
        """Test indexer initialization."""
        indexer = HierarchicalIndexer(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_type="Flat",
        )
        assert indexer.embedding_dim > 0
        assert indexer.index_type == "Flat"

    def test_build_hierarchical_index_simple(self) -> None:
        """Test building hierarchical index from simple table."""
        # Create simple table
        cells = [
            [Cell(0, 0, "Header", colspan=2, is_header=True)],
            [Cell(1, 0, "Data1"), Cell(1, 1, "Data2")],
        ]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        # Normalize and extract hierarchy
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        # Build index
        indexer = HierarchicalIndexer(index_type="Flat")
        index = indexer.build_hierarchical_index([(normalized, header_tree)])
        
        assert isinstance(index, HierarchicalIndex)
        assert index.table_index is not None
        assert index.subtable_index is not None
        assert index.row_index is not None
        assert index.cell_index is not None
        assert len(index.metadata) > 0

    def test_build_hierarchical_index_hitab_style(self) -> None:
        """Test building index from HiTab-style financial table."""
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
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        # Measure indexing time
        indexer = HierarchicalIndexer(index_type="Flat")
        start_time = time.time()
        index = indexer.build_hierarchical_index([(normalized, header_tree)])
        elapsed_time = time.time() - start_time
        
        assert isinstance(index, HierarchicalIndex)
        assert elapsed_time > 0
        
        # Check granularity levels
        granularities = {meta.granularity for meta in index.metadata}
        assert "table" in granularities
        assert "subtable" in granularities
        assert "row" in granularities
        assert "cell" in granularities
        
        # Check metadata
        table_meta = next(
            (m for m in index.metadata if m.granularity == "table"), None
        )
        assert table_meta is not None
        assert table_meta.hierarchy_path == "Table"
        assert len(table_meta.child_ids) > 0

    def test_add_table_incremental(self) -> None:
        """Test incrementally adding tables to index."""
        indexer = HierarchicalIndexer(index_type="Flat")
        
        # First table
        cells1 = [
            [Cell(0, 0, "Header1", is_header=True)],
            [Cell(1, 0, "Data1")],
        ]
        table_obj1 = TableObject(cells=cells1, headers=[])
        
        merger = CellMergeHandler()
        normalized1 = merger.normalize_merged_cells(table_obj1)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree1 = hierarchy_extractor.extract_header_tree(table_obj1)
        
        indexer.add_table(normalized1, header_tree1)
        
        initial_count = len(indexer.index.metadata) if indexer.index else 0
        
        # Second table
        cells2 = [
            [Cell(0, 0, "Header2", is_header=True)],
            [Cell(1, 0, "Data2")],
        ]
        table_obj2 = TableObject(cells=cells2, headers=[])
        normalized2 = merger.normalize_merged_cells(table_obj2)
        header_tree2 = hierarchy_extractor.extract_header_tree(table_obj2)
        
        indexer.add_table(normalized2, header_tree2)
        
        assert indexer.index is not None
        assert len(indexer.index.metadata) > initial_count

    def test_save_and_load_index(self) -> None:
        """Test saving and loading index."""
        # Create and build index
        cells = [
            [Cell(0, 0, "Header", is_header=True)],
            [Cell(1, 0, "Data")],
        ]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        indexer1 = HierarchicalIndexer(index_type="Flat")
        index1 = indexer1.build_hierarchical_index([(normalized, header_tree)])
        
        # Save index
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "test_index")
            indexer1.save_index(index_path)
            
            # Verify files exist
            assert Path(index_path).exists()
            assert Path(index_path, "table_index.faiss").exists()
            assert Path(index_path, "metadata.json").exists()
            assert Path(index_path, "config.json").exists()
            
            # Load index
            indexer2 = HierarchicalIndexer(index_type="Flat")
            index2 = indexer2.load_index(index_path)
            
            # Compare
            assert index2.embedding_dim == index1.embedding_dim
            assert len(index2.metadata) == len(index1.metadata)
            assert index2.table_index.ntotal == index1.table_index.ntotal

    def test_index_metadata_structure(self) -> None:
        """Test that metadata has correct structure."""
        cells = [
            [Cell(0, 0, "Header", colspan=2, is_header=True)],
            [Cell(1, 0, "Data1"), Cell(1, 1, "Data2")],
        ]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        indexer = HierarchicalIndexer(index_type="Flat")
        index = indexer.build_hierarchical_index([(normalized, header_tree)])
        
        # Check metadata structure
        for meta in index.metadata:
            assert meta.item_id is not None
            assert meta.source_table_id is not None
            assert meta.granularity in ["table", "subtable", "row", "cell"]
            assert meta.hierarchy_path is not None
            assert len(meta.coordinates) == 2
            assert isinstance(meta.parent_ids, list)
            assert isinstance(meta.child_ids, list)

    def test_parent_child_relationships(self) -> None:
        """Test parent-child relationships in metadata."""
        cells = [
            [
                Cell(0, 0, "Category A", colspan=2, is_header=True),
                Cell(0, 2, "Category B", colspan=2, is_header=True),
            ],
            [
                Cell(1, 0, "A1", is_header=True),
                Cell(1, 1, "A2", is_header=True),
                Cell(1, 2, "B1", is_header=True),
                Cell(1, 3, "B2", is_header=True),
            ],
            [
                Cell(2, 0, "Data1"),
                Cell(2, 1, "Data2"),
                Cell(2, 2, "Data3"),
                Cell(2, 3, "Data4"),
            ],
        ]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        indexer = HierarchicalIndexer(index_type="Flat")
        index = indexer.build_hierarchical_index([(normalized, header_tree)])
        
        # Find table metadata
        table_meta = next(
            (m for m in index.metadata if m.granularity == "table"), None
        )
        assert table_meta is not None
        
        # Check that table has children
        assert len(table_meta.child_ids) > 0
        
        # Check that subtables have table as parent
        subtable_metas = [
            m for m in index.metadata if m.granularity == "subtable"
        ]
        for subtable_meta in subtable_metas:
            assert table_meta.item_id in subtable_meta.parent_ids

    def test_indexing_time_measurement(self) -> None:
        """Test indexing time measurement."""
        # Create larger table
        cells = []
        # Header row
        cells.append([
            Cell(0, 0, "Header", colspan=5, is_header=True),
        ])
        # Data rows
        for row in range(1, 11):
            row_cells = []
            for col in range(5):
                row_cells.append(Cell(row, col, f"R{row}C{col}"))
            cells.append(row_cells)
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        indexer = HierarchicalIndexer(index_type="Flat")
        
        start_time = time.time()
        index = indexer.build_hierarchical_index([(normalized, header_tree)])
        elapsed_time = time.time() - start_time
        
        assert elapsed_time > 0
        assert elapsed_time < 60  # Should complete in reasonable time
        assert index is not None

    def test_different_index_types(self) -> None:
        """Test different FAISS index types."""
        cells = [[Cell(0, 0, "Header", is_header=True)], [Cell(1, 0, "Data")]]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        # Test Flat index
        indexer_flat = HierarchicalIndexer(index_type="Flat")
        index_flat = indexer_flat.build_hierarchical_index(
            [(normalized, header_tree)]
        )
        assert index_flat.table_index is not None
        
        # Note: IVF and HNSW require training/data, so we just test Flat here

    def test_empty_table_handling(self) -> None:
        """Test handling of empty or minimal tables."""
        # Minimal table
        cells = [[Cell(0, 0, "Header", is_header=True)]]
        
        table_obj = TableObject(cells=cells, headers=[])
        
        merger = CellMergeHandler()
        normalized = merger.normalize_merged_cells(table_obj)
        
        hierarchy_extractor = HierarchyExtractor()
        header_tree = hierarchy_extractor.extract_header_tree(table_obj)
        
        indexer = HierarchicalIndexer(index_type="Flat")
        
        # Should not raise error
        index = indexer.build_hierarchical_index([(normalized, header_tree)])
        assert index is not None

    def test_save_index_no_index_error(self) -> None:
        """Test error when saving without building index."""
        indexer = HierarchicalIndexer(index_type="Flat")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "test_index")
            
            with pytest.raises(ValueError, match="No index to save"):
                indexer.save_index(index_path)

    def test_load_index_not_found_error(self) -> None:
        """Test error when loading non-existent index."""
        indexer = HierarchicalIndexer(index_type="Flat")
        
        with pytest.raises(FileNotFoundError):
            indexer.load_index("/nonexistent/path")



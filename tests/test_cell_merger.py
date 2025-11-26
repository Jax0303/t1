"""
Tests for CellMergeHandler class.

Covers simple 2x2 merged cells, complex financial tables with multiple
merge regions, and edge cases like entire row/column merges.
"""

import pytest
from typing import List

from src.parsing.extractor import TableObject, Cell
from src.parsing.cell_merger import (
    CellMergeHandler,
    NormalizedTable,
    NormalizedCell,
    CoordinateMap,
    MergedCellInfo,
)


class TestCellMergeHandler:
    """Test cases for CellMergeHandler class."""

    def test_handler_init(self) -> None:
        """Test handler initialization."""
        handler = CellMergeHandler()
        assert handler is not None

    def test_simple_2x2_merged_cells(self) -> None:
        """Test normalization of simple 2x2 merged cells."""
        handler = CellMergeHandler()
        
        # Create table with 2x2 merged cell at (0,0)
        cells = [
            [
                Cell(0, 0, "Merged", rowspan=2, colspan=2),
                Cell(0, 2, "C"),
            ],
            [
                Cell(1, 2, "D"),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        
        assert isinstance(normalized, NormalizedTable)
        assert normalized.num_rows == 2
        assert normalized.num_cols == 3
        
        # Check root cell
        root_cell = normalized.get_cell(0, 0)
        assert root_cell is not None
        assert root_cell.is_merged is True
        assert root_cell.merge_root == (0, 0)
        assert root_cell.merge_span == (2, 2)
        assert root_cell.content == "Merged"
        
        # Check merged cell references
        ref_cell = normalized.get_cell(0, 1)
        assert ref_cell is not None
        assert ref_cell.is_merged is True
        assert ref_cell.merge_root == (0, 0)
        assert ref_cell.content == "Merged"  # Same content as root
        
        ref_cell2 = normalized.get_cell(1, 0)
        assert ref_cell2 is not None
        assert ref_cell2.is_merged is True
        assert ref_cell2.merge_root == (0, 0)

    def test_complex_financial_table_merges(self) -> None:
        """Test normalization of complex financial table with multiple merge regions."""
        handler = CellMergeHandler()
        
        # Financial table with multiple merged regions
        # Row 0: [Revenue (span=4), Expenses (span=4)]
        # Row 1: [Q1, Q2, Q3, Q4, Q1, Q2, Q3, Q4]
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
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        
        assert isinstance(normalized, NormalizedTable)
        assert len(normalized.merged_regions) == 2  # Revenue and Expenses
        
        # Check Revenue merge
        revenue_cell = normalized.get_cell(0, 0)
        assert revenue_cell is not None
        assert revenue_cell.is_merged is True
        assert revenue_cell.merge_span == (1, 4)
        
        # Check that cells (0,1), (0,2), (0,3) reference (0,0)
        for col in [1, 2, 3]:
            ref_cell = normalized.get_cell(0, col)
            assert ref_cell is not None
            assert ref_cell.is_merged is True
            assert ref_cell.merge_root == (0, 0)

    def test_entire_row_merge(self) -> None:
        """Test normalization of entire row merge."""
        handler = CellMergeHandler()
        
        # Table with entire row merged
        cells = [
            [
                Cell(0, 0, "Full Row", colspan=3),
            ],
            [
                Cell(1, 0, "A"),
                Cell(1, 1, "B"),
                Cell(1, 2, "C"),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        
        assert isinstance(normalized, NormalizedTable)
        
        # Check merged row
        root_cell = normalized.get_cell(0, 0)
        assert root_cell is not None
        assert root_cell.is_merged is True
        assert root_cell.merge_span == (1, 3)
        
        # Check references
        for col in [1, 2]:
            ref_cell = normalized.get_cell(0, col)
            assert ref_cell is not None
            assert ref_cell.is_merged is True
            assert ref_cell.merge_root == (0, 0)

    def test_entire_column_merge(self) -> None:
        """Test normalization of entire column merge."""
        handler = CellMergeHandler()
        
        # Table with entire column merged
        cells = [
            [
                Cell(0, 0, "Full Col", rowspan=3),
                Cell(0, 1, "B"),
                Cell(0, 2, "C"),
            ],
            [
                Cell(1, 1, "D"),
                Cell(1, 2, "E"),
            ],
            [
                Cell(2, 1, "F"),
                Cell(2, 2, "G"),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        
        assert isinstance(normalized, NormalizedTable)
        
        # Check merged column root
        root_cell = normalized.get_cell(0, 0)
        assert root_cell is not None
        assert root_cell.is_merged is True
        assert root_cell.merge_span == (3, 1)
        
        # Check references in column 0
        for row in [1, 2]:
            ref_cell = normalized.get_cell(row, 0)
            assert ref_cell is not None
            assert ref_cell.is_merged is True
            assert ref_cell.merge_root == (0, 0)

    def test_create_coordinate_system(self) -> None:
        """Test creation of coordinate system."""
        handler = CellMergeHandler()
        
        cells = [
            [Cell(0, 0, "A"), Cell(0, 1, "B")],
            [Cell(1, 0, "C"), Cell(1, 1, "D")],
        ]
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        coord_map = handler.create_coordinate_system(normalized)
        
        assert isinstance(coord_map, CoordinateMap)
        
        # Check Excel mappings
        assert coord_map.get_excel(0, 0) == "A1"
        assert coord_map.get_excel(0, 1) == "B1"
        assert coord_map.get_excel(1, 0) == "A2"
        assert coord_map.get_excel(1, 1) == "B2"
        
        # Check tuple mappings
        assert coord_map.get_tuple("A1") == (0, 0)
        assert coord_map.get_tuple("B1") == (0, 1)
        assert coord_map.get_tuple("A2") == (1, 0)
        assert coord_map.get_tuple("B2") == (1, 1)

    def test_resolve_cell_reference_tuple(self) -> None:
        """Test resolving cell reference using tuple coordinates."""
        handler = CellMergeHandler()
        
        cells = [
            [
                Cell(0, 0, "Merged", rowspan=2, colspan=2),
                Cell(0, 2, "C"),
            ],
            [
                Cell(1, 2, "D"),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        coord_map = handler.create_coordinate_system(normalized)
        
        # Resolve root cell
        cell = handler.resolve_cell_reference((0, 0), coord_map)
        assert cell is not None
        assert cell.content == "Merged"
        assert cell.is_merged is True
        
        # Resolve merged cell reference (should return root)
        cell_ref = handler.resolve_cell_reference((0, 1), coord_map)
        assert cell_ref is not None
        assert cell_ref.content == "Merged"
        assert cell_ref.merge_root == (0, 0)
        
        # Resolve regular cell
        cell_c = handler.resolve_cell_reference((0, 2), coord_map)
        assert cell_c is not None
        assert cell_c.content == "C"
        assert cell_c.is_merged is False

    def test_resolve_cell_reference_excel(self) -> None:
        """Test resolving cell reference using Excel-style coordinates."""
        handler = CellMergeHandler()
        
        cells = [
            [
                Cell(0, 0, "Merged", rowspan=2, colspan=2),
                Cell(0, 2, "C"),
            ],
            [
                Cell(1, 2, "D"),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        coord_map = handler.create_coordinate_system(normalized)
        
        # Resolve using Excel reference
        cell = handler.resolve_cell_reference("A1", coord_map)
        assert cell is not None
        assert cell.content == "Merged"
        
        # Resolve merged cell reference
        cell_ref = handler.resolve_cell_reference("B1", coord_map)
        assert cell_ref is not None
        assert cell_ref.content == "Merged"
        assert cell_ref.merge_root == (0, 0)
        
        # Resolve regular cell
        cell_c = handler.resolve_cell_reference("C1", coord_map)
        assert cell_c is not None
        assert cell_c.content == "C"

    def test_resolve_invalid_reference(self) -> None:
        """Test resolving invalid cell reference."""
        handler = CellMergeHandler()
        
        cells = [[Cell(0, 0, "A")]]
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        coord_map = handler.create_coordinate_system(normalized)
        
        # Invalid Excel reference
        cell = handler.resolve_cell_reference("ZZ999", coord_map)
        assert cell is None
        
        # Invalid tuple (out of bounds)
        cell = handler.resolve_cell_reference((100, 100), coord_map)
        assert cell is None
        
        # Invalid format
        cell = handler.resolve_cell_reference("invalid", coord_map)
        assert cell is None

    def test_visualize_coordinate_grid(self) -> None:
        """Test visualization of coordinate grid."""
        handler = CellMergeHandler()
        
        cells = [
            [
                Cell(0, 0, "Merged", rowspan=2, colspan=2),
                Cell(0, 2, "C"),
            ],
            [
                Cell(1, 2, "D"),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        coord_map = handler.create_coordinate_system(normalized)
        
        visualization = handler.visualize_coordinate_grid(
            normalized, coord_map, show_merges=True
        )
        
        assert isinstance(visualization, str)
        assert "Merged" in visualization
        assert "M" in visualization or "→" in visualization  # Merge indicators

    def test_multiple_overlapping_merges(self) -> None:
        """Test table with multiple overlapping merge regions."""
        handler = CellMergeHandler()
        
        # Complex table with multiple merges
        cells = [
            [
                Cell(0, 0, "Header1", colspan=2, is_header=True),
                Cell(0, 2, "Header2", colspan=2, is_header=True),
            ],
            [
                Cell(1, 0, "Sub1", rowspan=2),
                Cell(1, 1, "Sub2"),
                Cell(1, 2, "Sub3", rowspan=2),
                Cell(1, 3, "Sub4"),
            ],
            [
                Cell(2, 1, "Data1"),
                Cell(2, 3, "Data2"),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        
        assert isinstance(normalized, NormalizedTable)
        assert len(normalized.merged_regions) >= 2
        
        # Check that all merges are properly normalized
        for row in range(normalized.num_rows):
            for col in range(normalized.num_cols):
                cell = normalized.get_cell(row, col)
                assert cell is not None

    def test_empty_table_error(self) -> None:
        """Test error handling for empty table."""
        handler = CellMergeHandler()
        
        table = TableObject(cells=[], headers=[])
        
        with pytest.raises(ValueError, match="Table has no cells"):
            handler.normalize_merged_cells(table)

    def test_excel_coordinate_conversion(self) -> None:
        """Test Excel coordinate conversion utilities."""
        handler = CellMergeHandler()
        
        # Test tuple to Excel
        assert handler._tuple_to_excel(0, 0) == "A1"
        assert handler._tuple_to_excel(0, 1) == "B1"
        assert handler._tuple_to_excel(1, 0) == "A2"
        assert handler._tuple_to_excel(25, 25) == "Z26"
        assert handler._tuple_to_excel(0, 26) == "AA1"
        
        # Test Excel to tuple
        assert handler._excel_to_tuple("A1") == (0, 0)
        assert handler._excel_to_tuple("B1") == (0, 1)
        assert handler._excel_to_tuple("A2") == (1, 0)
        assert handler._excel_to_tuple("Z26") == (25, 25)
        assert handler._excel_to_tuple("AA1") == (0, 26)
        
        # Test invalid Excel reference
        assert handler._excel_to_tuple("invalid") is None
        assert handler._excel_to_tuple("123") is None

    def test_find_merged_region(self) -> None:
        """Test finding merged region covering coordinates."""
        handler = CellMergeHandler()
        
        cells = [
            [
                Cell(0, 0, "Merged", rowspan=2, colspan=2),
                Cell(0, 2, "C"),
            ],
            [
                Cell(1, 2, "D"),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        
        # Find region covering (0, 0)
        region = normalized.find_merged_region(0, 0)
        assert region is not None
        assert region.merge_root == (0, 0)
        
        # Find region covering (0, 1) - should be same region
        region2 = normalized.find_merged_region(0, 1)
        assert region2 is not None
        assert region2.merge_root == (0, 0)
        
        # Find region covering (1, 0) - should be same region
        region3 = normalized.find_merged_region(1, 0)
        assert region3 is not None
        assert region3.merge_root == (0, 0)
        
        # No region covering (0, 2)
        region4 = normalized.find_merged_region(0, 2)
        assert region4 is None

    def test_large_table_coordinates(self) -> None:
        """Test coordinate system with large table."""
        handler = CellMergeHandler()
        
        # Create 10x10 table
        cells = []
        for row in range(10):
            row_cells = []
            for col in range(10):
                row_cells.append(Cell(row, col, f"R{row}C{col}"))
            cells.append(row_cells)
        
        table = TableObject(cells=cells, headers=[])
        normalized = handler.normalize_merged_cells(table)
        coord_map = handler.create_coordinate_system(normalized)
        
        # Check that all coordinates are mapped
        assert len(coord_map.excel_to_tuple) == 100
        assert len(coord_map.tuple_to_excel) == 100
        
        # Check some edge cases
        assert coord_map.get_excel(9, 9) == "J10"
        assert coord_map.get_tuple("J10") == (9, 9)



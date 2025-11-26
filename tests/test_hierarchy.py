"""
Tests for HierarchyExtractor class.

Covers simple 2-level headers, complex 3-level headers with irregular spans,
and mixed horizontal/vertical hierarchies.
"""

import pytest
from typing import List

from src.parsing.extractor import TableObject, Cell
from src.parsing.hierarchy import (
    HierarchyExtractor,
    HeaderTree,
    HeaderNode,
)


class TestHeaderNode:
    """Test cases for HeaderNode dataclass."""

    def test_node_creation(self) -> None:
        """Test basic node creation."""
        node = HeaderNode(
            text="Header", level=1, span=2, col_start=0, col_end=2
        )
        assert node.text == "Header"
        assert node.level == 1
        assert node.span == 2
        assert node.is_leaf() is True

    def test_node_add_child(self) -> None:
        """Test adding child nodes."""
        parent = HeaderNode(text="Parent", level=1, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=2, span=1, col_start=0, col_end=1)
        
        parent.add_child(child)
        
        assert len(parent.children) == 1
        assert child.parent == parent
        assert parent.is_leaf() is False

    def test_get_all_leaves(self) -> None:
        """Test getting all leaf nodes."""
        root = HeaderNode(text="Root", level=0, span=4, col_start=0, col_end=4)
        child1 = HeaderNode(text="Child1", level=1, span=2, col_start=0, col_end=2)
        child2 = HeaderNode(text="Child2", level=1, span=2, col_start=2, col_end=4)
        leaf1 = HeaderNode(text="Leaf1", level=2, span=1, col_start=0, col_end=1)
        leaf2 = HeaderNode(text="Leaf2", level=2, span=1, col_start=1, col_end=2)
        
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(leaf1)
        child1.add_child(leaf2)
        
        leaves = root.get_all_leaves()
        assert len(leaves) == 2
        assert leaf1 in leaves
        assert leaf2 in leaves


class TestHeaderTree:
    """Test cases for HeaderTree dataclass."""

    def test_tree_creation(self) -> None:
        """Test tree creation."""
        root = HeaderNode(text="Root", level=0, span=4, col_start=0, col_end=4)
        tree = HeaderTree(root=root)
        
        assert tree.root == root
        assert tree.max_level == 0
        assert tree.num_columns == 1

    def test_get_all_leaves(self) -> None:
        """Test getting all leaves from tree."""
        root = HeaderNode(text="Root", level=0, span=4, col_start=0, col_end=4)
        child1 = HeaderNode(text="C1", level=1, span=2, col_start=0, col_end=2)
        child2 = HeaderNode(text="C2", level=1, span=2, col_start=2, col_end=4)
        leaf1 = HeaderNode(text="L1", level=2, span=1, col_start=0, col_end=1)
        leaf2 = HeaderNode(text="L2", level=2, span=1, col_start=1, col_end=2)
        leaf3 = HeaderNode(text="L3", level=2, span=1, col_start=2, col_end=3)
        leaf4 = HeaderNode(text="L4", level=2, span=1, col_start=3, col_end=4)
        
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(leaf1)
        child1.add_child(leaf2)
        child2.add_child(leaf3)
        child2.add_child(leaf4)
        
        tree = HeaderTree(root=root)
        
        leaves = tree.get_all_leaves()
        assert len(leaves) == 4
        assert tree.num_columns == 4

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        tree = HeaderTree(root=root)
        
        json_str = tree.to_json()
        assert isinstance(json_str, str)
        assert "Root" in json_str
        assert "max_level" in json_str

    def test_visualize_ascii(self) -> None:
        """Test ASCII visualization."""
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=1, span=1, col_start=0, col_end=1)
        root.add_child(child)
        
        tree = HeaderTree(root=root)
        ascii_tree = tree.visualize_ascii()
        
        assert isinstance(ascii_tree, str)
        assert "Root" in ascii_tree
        assert "Child" in ascii_tree


class TestHierarchyExtractor:
    """Test cases for HierarchyExtractor class."""

    def test_extractor_init(self) -> None:
        """Test extractor initialization."""
        extractor = HierarchyExtractor()
        assert extractor.max_header_rows == 5
        assert extractor.min_header_rows == 1

    def test_simple_2_level_headers(self) -> None:
        """Test extraction from simple 2-level header table."""
        extractor = HierarchyExtractor()
        
        # Create a simple 2-level header table
        # Level 1: [A, A, B, B]
        # Level 2: [A1, A2, B1, B2]
        cells = [
            [
                Cell(0, 0, "A", colspan=2, is_header=True),
                Cell(0, 2, "B", colspan=2, is_header=True),
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
        
        table = TableObject(cells=cells, headers=[])
        tree = extractor.extract_header_tree(table)
        
        assert isinstance(tree, HeaderTree)
        assert tree.max_level >= 2
        leaves = tree.get_all_leaves()
        assert len(leaves) == 4  # A1, A2, B1, B2

    def test_complex_3_level_headers(self) -> None:
        """Test extraction from complex 3-level header table."""
        extractor = HierarchyExtractor()
        
        # Create a 3-level header table with irregular spans
        # Level 1: [X (span=4)]
        # Level 2: [A (span=2), B (span=2)]
        # Level 3: [A1, A2, B1, B2]
        cells = [
            [Cell(0, 0, "X", colspan=4, is_header=True)],
            [
                Cell(1, 0, "A", colspan=2, is_header=True),
                Cell(1, 2, "B", colspan=2, is_header=True),
            ],
            [
                Cell(2, 0, "A1", is_header=True),
                Cell(2, 1, "A2", is_header=True),
                Cell(2, 2, "B1", is_header=True),
                Cell(2, 3, "B2", is_header=True),
            ],
            [
                Cell(3, 0, "D1"),
                Cell(3, 1, "D2"),
                Cell(3, 2, "D3"),
                Cell(3, 3, "D4"),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        tree = extractor.extract_header_tree(table)
        
        assert isinstance(tree, HeaderTree)
        assert tree.max_level >= 3
        leaves = tree.get_all_leaves()
        assert len(leaves) == 4

    def test_irregular_spans_3_level(self) -> None:
        """Test extraction with irregular spans in 3-level headers."""
        extractor = HierarchyExtractor()
        
        # Irregular spans:
        # Level 1: [X (span=3), Y (span=2)]
        # Level 2: [A (span=2), B, C (span=2)]
        # Level 3: [A1, A2, B1, C1, C2]
        cells = [
            [
                Cell(0, 0, "X", colspan=3, is_header=True),
                Cell(0, 3, "Y", colspan=2, is_header=True),
            ],
            [
                Cell(1, 0, "A", colspan=2, is_header=True),
                Cell(1, 2, "B", is_header=True),
                Cell(1, 3, "C", colspan=2, is_header=True),
            ],
            [
                Cell(2, 0, "A1", is_header=True),
                Cell(2, 1, "A2", is_header=True),
                Cell(2, 2, "B1", is_header=True),
                Cell(2, 3, "C1", is_header=True),
                Cell(2, 4, "C2", is_header=True),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        tree = extractor.extract_header_tree(table)
        
        assert isinstance(tree, HeaderTree)
        leaves = tree.get_all_leaves()
        assert len(leaves) == 5  # A1, A2, B1, C1, C2

    def test_mixed_horizontal_vertical_hierarchy(self) -> None:
        """Test extraction with mixed horizontal and vertical hierarchies."""
        extractor = HierarchyExtractor()
        
        # Mixed hierarchy with rowspan:
        # Row 0: [X (rowspan=2, colspan=1), A (colspan=2), B (colspan=2)]
        # Row 1: [A1, A2, B1, B2]
        cells = [
            [
                Cell(0, 0, "X", rowspan=2, colspan=1, is_header=True),
                Cell(0, 1, "A", colspan=2, is_header=True),
                Cell(0, 3, "B", colspan=2, is_header=True),
            ],
            [
                Cell(1, 1, "A1", is_header=True),
                Cell(1, 2, "A2", is_header=True),
                Cell(1, 3, "B1", is_header=True),
                Cell(1, 4, "B2", is_header=True),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        tree = extractor.extract_header_tree(table)
        
        assert isinstance(tree, HeaderTree)
        # Should handle rowspan correctly
        leaves = tree.get_all_leaves()
        assert len(leaves) >= 4

    def test_hitab_style_table(self) -> None:
        """Test HiTab-style table with financial data headers."""
        extractor = HierarchyExtractor()
        
        # Financial table style:
        # Level 1: [Revenue, Expenses, Net Income]
        # Level 2: [Q1, Q2, Q3, Q4, Q1, Q2, Q3, Q4, Total]
        # (Simplified - actual would have more complex nesting)
        cells = [
            [
                Cell(0, 0, "Revenue", colspan=4, is_header=True),
                Cell(0, 4, "Expenses", colspan=4, is_header=True),
                Cell(0, 8, "Net Income", colspan=1, is_header=True),
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
                Cell(1, 8, "Total", is_header=True),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        tree = extractor.extract_header_tree(table)
        
        assert isinstance(tree, HeaderTree)
        assert tree.max_level >= 2
        leaves = tree.get_all_leaves()
        assert len(leaves) == 9  # 9 columns

    def test_fintabnet_style_table(self) -> None:
        """Test FinTabNet-style table with 3-level headers."""
        extractor = HierarchyExtractor()
        
        # FinTabNet style:
        # Level 1: [Assets, Liabilities]
        # Level 2: [Current, Non-Current, Current, Non-Current]
        # Level 3: [Cash, Inventory, ..., Accounts Payable, ...]
        cells = [
            [
                Cell(0, 0, "Assets", colspan=3, is_header=True),
                Cell(0, 3, "Liabilities", colspan=3, is_header=True),
            ],
            [
                Cell(1, 0, "Current", colspan=2, is_header=True),
                Cell(1, 2, "Non-Current", is_header=True),
                Cell(1, 3, "Current", colspan=2, is_header=True),
                Cell(1, 5, "Non-Current", is_header=True),
            ],
            [
                Cell(2, 0, "Cash", is_header=True),
                Cell(2, 1, "Inventory", is_header=True),
                Cell(2, 2, "Property", is_header=True),
                Cell(2, 3, "A/P", is_header=True),
                Cell(2, 4, "Accruals", is_header=True),
                Cell(2, 5, "Long-term Debt", is_header=True),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        tree = extractor.extract_header_tree(table)
        
        assert isinstance(tree, HeaderTree)
        assert tree.max_level >= 3
        leaves = tree.get_all_leaves()
        assert len(leaves) == 6

    def test_empty_table(self) -> None:
        """Test error handling for empty table."""
        extractor = HierarchyExtractor()
        
        table = TableObject(cells=[], headers=[])
        
        with pytest.raises(ValueError, match="Table has no cells"):
            extractor.extract_header_tree(table)

    def test_single_row_table(self) -> None:
        """Test extraction from single row table."""
        extractor = HierarchyExtractor()
        
        cells = [[Cell(0, 0, "Col1"), Cell(0, 1, "Col2")]]
        table = TableObject(cells=cells, headers=[])
        
        tree = extractor.extract_header_tree(table)
        
        assert isinstance(tree, HeaderTree)
        # Should create flat structure
        assert tree.max_level >= 1

    def test_table_with_no_headers(self) -> None:
        """Test extraction from table with no header markers."""
        extractor = HierarchyExtractor()
        
        # Table with all numeric/data cells
        cells = [
            [Cell(0, 0, "1"), Cell(0, 1, "2")],
            [Cell(1, 0, "3"), Cell(1, 1, "4")],
        ]
        
        table = TableObject(cells=cells, headers=[])
        tree = extractor.extract_header_tree(table)
        
        # Should still create a structure (using first row as header)
        assert isinstance(tree, HeaderTree)

    def test_nested_merged_cells(self) -> None:
        """Test extraction with deeply nested merged cells."""
        extractor = HierarchyExtractor()
        
        # Complex nesting:
        # Level 1: [All (span=6)]
        # Level 2: [Group1 (span=3), Group2 (span=3)]
        # Level 3: [A (span=2), B, C (span=2), D]
        # Level 4: [A1, A2, B1, C1, C2, D1]
        cells = [
            [Cell(0, 0, "All", colspan=6, is_header=True)],
            [
                Cell(1, 0, "Group1", colspan=3, is_header=True),
                Cell(1, 3, "Group2", colspan=3, is_header=True),
            ],
            [
                Cell(2, 0, "A", colspan=2, is_header=True),
                Cell(2, 2, "B", is_header=True),
                Cell(2, 3, "C", colspan=2, is_header=True),
                Cell(2, 5, "D", is_header=True),
            ],
            [
                Cell(3, 0, "A1", is_header=True),
                Cell(3, 1, "A2", is_header=True),
                Cell(3, 2, "B1", is_header=True),
                Cell(3, 3, "C1", is_header=True),
                Cell(3, 4, "C2", is_header=True),
                Cell(3, 5, "D1", is_header=True),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        tree = extractor.extract_header_tree(table)
        
        assert isinstance(tree, HeaderTree)
        assert tree.max_level >= 4
        leaves = tree.get_all_leaves()
        assert len(leaves) == 6

    def test_tree_visualization(self) -> None:
        """Test tree visualization output."""
        extractor = HierarchyExtractor()
        
        cells = [
            [Cell(0, 0, "A", colspan=2, is_header=True)],
            [
                Cell(1, 0, "A1", is_header=True),
                Cell(1, 1, "A2", is_header=True),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        tree = extractor.extract_header_tree(table)
        
        ascii_tree = tree.visualize_ascii()
        assert isinstance(ascii_tree, str)
        assert len(ascii_tree) > 0

    def test_tree_json_serialization(self) -> None:
        """Test JSON serialization of tree."""
        extractor = HierarchyExtractor()
        
        cells = [
            [Cell(0, 0, "Header", colspan=2, is_header=True)],
            [
                Cell(1, 0, "Col1", is_header=True),
                Cell(1, 1, "Col2", is_header=True),
            ],
        ]
        
        table = TableObject(cells=cells, headers=[])
        tree = extractor.extract_header_tree(table)
        
        json_str = tree.to_json()
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        import json as json_lib
        parsed = json_lib.loads(json_str)
        assert "root" in parsed
        assert "max_level" in parsed



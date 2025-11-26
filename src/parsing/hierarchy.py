"""
Hierarchical structure detection for tables.

Implements header hierarchy extraction based on colspan/rowspan relationships
and spatial overlap analysis. Supports HiTab/FinTabNet style multi-level headers.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
import json
import logging

from src.parsing.extractor import TableObject, Cell

logger = logging.getLogger(__name__)


@dataclass
class HeaderNode:
    """
    Represents a node in the header hierarchy tree.
    
    Attributes:
        text: Header text content
        level: Hierarchy level (0 = root/table, 1 = first header row, etc.)
        span: Column span (colspan) - number of columns this header covers
        row_span: Row span (rowspan) - number of rows this header spans
        col_start: Starting column index (0-based)
        col_end: Ending column index (exclusive)
        row_start: Starting row index (0-based)
        row_end: Ending row index (exclusive)
        children: List of child HeaderNode objects
        parent: Parent HeaderNode (None for root)
        cell_ref: Reference to original Cell object
    """
    text: str
    level: int
    span: int  # colspan
    row_span: int = 1  # rowspan
    col_start: int = 0
    col_end: int = 0
    row_start: int = 0
    row_end: int = 0
    children: List["HeaderNode"] = field(default_factory=list)
    parent: Optional["HeaderNode"] = None
    cell_ref: Optional[Cell] = None

    def __post_init__(self) -> None:
        """Validate node attributes."""
        if self.span < 1:
            raise ValueError(f"span must be >= 1, got {self.span}")
        if self.row_span < 1:
            raise ValueError(f"row_span must be >= 1, got {self.row_span}")
        if self.col_end <= self.col_start:
            self.col_end = self.col_start + self.span

    def add_child(self, child: "HeaderNode") -> None:
        """
        Add a child node and set parent reference.
        
        Args:
            child: Child HeaderNode to add
        """
        if child not in self.children:
            self.children.append(child)
            child.parent = self

    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf (has no children).
        
        Returns:
            True if leaf node, False otherwise
        """
        return len(self.children) == 0

    def get_all_leaves(self) -> List["HeaderNode"]:
        """
        Get all leaf nodes in the subtree rooted at this node.
        
        Returns:
            List of leaf HeaderNode objects
        """
        leaves: List[HeaderNode] = []
        
        if self.is_leaf():
            leaves.append(self)
        else:
            for child in self.children:
                leaves.extend(child.get_all_leaves())
        
        return leaves

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary (for JSON serialization).
        
        Returns:
            Dictionary representation
        """
        return {
            "text": self.text,
            "level": self.level,
            "span": self.span,
            "row_span": self.row_span,
            "col_start": self.col_start,
            "col_end": self.col_end,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "children": [child.to_dict() for child in self.children],
        }

    def get_path(self) -> List[str]:
        """
        Get the path from root to this node.
        
        Returns:
            List of text values from root to this node
        """
        path = [self.text]
        current = self.parent
        while current is not None:
            path.insert(0, current.text)
            current = current.parent
        return path


@dataclass
class HeaderTree:
    """
    Represents a hierarchical header structure for a table.
    
    Attributes:
        root: Root HeaderNode (represents the table)
        max_level: Maximum hierarchy level in the tree
        num_columns: Total number of data columns
    """
    root: HeaderNode
    max_level: int = 0
    num_columns: int = 0

    def __post_init__(self) -> None:
        """Initialize and validate tree structure."""
        if self.root is None:
            raise ValueError("Root node cannot be None")
        self.max_level = self._calculate_max_level(self.root)
        self.num_columns = self._count_leaf_nodes()

    def _calculate_max_level(self, node: HeaderNode) -> int:
        """
        Calculate maximum level in the tree.
        
        Args:
            node: Starting node
            
        Returns:
            Maximum level depth
        """
        if node.is_leaf():
            return node.level
        
        max_child_level = max(
            (self._calculate_max_level(child) for child in node.children),
            default=node.level
        )
        return max(node.level, max_child_level)

    def _count_leaf_nodes(self) -> int:
        """
        Count total number of leaf nodes (data columns).
        
        Returns:
            Number of leaf nodes
        """
        return len(self.root.get_all_leaves())

    def get_all_leaves(self) -> List[HeaderNode]:
        """
        Get all leaf nodes (actual data column headers).
        
        Returns:
            List of leaf HeaderNode objects
        """
        return self.root.get_all_leaves()

    def get_nodes_at_level(self, level: int) -> List[HeaderNode]:
        """
        Get all nodes at a specific level.
        
        Args:
            level: Target level
            
        Returns:
            List of HeaderNode objects at the specified level
        """
        nodes: List[HeaderNode] = []
        self._collect_nodes_at_level(self.root, level, nodes)
        return nodes

    def _collect_nodes_at_level(
        self, node: HeaderNode, target_level: int, result: List[HeaderNode]
    ) -> None:
        """
        Recursively collect nodes at target level.
        
        Args:
            node: Current node
            target_level: Target level to collect
            result: List to append results to
        """
        if node.level == target_level:
            result.append(node)
        
        for child in node.children:
            self._collect_nodes_at_level(child, target_level, result)

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize tree to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        tree_dict = {
            "root": self.root.to_dict(),
            "max_level": self.max_level,
            "num_columns": self.num_columns,
        }
        return json.dumps(tree_dict, indent=indent, ensure_ascii=False)

    def visualize_ascii(self, prefix: str = "", is_last: bool = True) -> str:
        """
        Visualize tree as ASCII art (for debugging).
        
        Args:
            prefix: Prefix string for indentation
            is_last: Whether this is the last child
            
        Returns:
            ASCII tree representation
        """
        lines: List[str] = []
        self._visualize_node(self.root, "", True, lines)
        return "\n".join(lines)

    def _visualize_node(
        self,
        node: HeaderNode,
        prefix: str,
        is_last: bool,
        lines: List[str],
    ) -> None:
        """
        Recursively visualize a node and its children.
        
        Args:
            node: Node to visualize
            prefix: Current prefix string
            is_last: Whether this is the last child
            lines: List to append lines to
        """
        # Current node line
        connector = "└── " if is_last else "├── "
        node_info = f"{node.text} (L{node.level}, span={node.span})"
        lines.append(prefix + connector + node_info)
        
        # Update prefix for children
        extension = "    " if is_last else "│   "
        new_prefix = prefix + extension
        
        # Visualize children
        for i, child in enumerate(node.children):
            is_last_child = i == len(node.children) - 1
            self._visualize_node(child, new_prefix, is_last_child, lines)


class HierarchyExtractor:
    """
    Extract hierarchical header structures from tables.
    
    Implements algorithms based on HiTab/FinTabNet approaches:
    - Uses colspan/rowspan to determine parent-child relationships
    - Builds parent-child links based on spatial overlap
    - Handles both horizontal (column) and vertical (row) hierarchies
    """

    def __init__(
        self,
        max_header_rows: int = 5,
        min_header_rows: int = 1,
        use_spatial_overlap: bool = True,
    ) -> None:
        """
        Initialize hierarchy extractor.
        
        Args:
            max_header_rows: Maximum number of rows to consider as headers
            min_header_rows: Minimum number of header rows
            use_spatial_overlap: Use bounding box intersection for relationships
        """
        self.max_header_rows = max_header_rows
        self.min_header_rows = min_header_rows
        self.use_spatial_overlap = use_spatial_overlap
        logger.info("HierarchyExtractor initialized")

    def extract_header_tree(self, table: TableObject) -> HeaderTree:
        """
        Extract hierarchical header structure from a table.
        
        Identifies header rows and builds a tree where:
        - Root node represents the table
        - Each level represents a header row
        - Leaf nodes represent data columns
        
        Args:
            table: TableObject to extract hierarchy from
            
        Returns:
            HeaderTree instance with extracted hierarchy
            
        Raises:
            ValueError: If table is empty or invalid
        """
        if not table.cells or len(table.cells) == 0:
            raise ValueError("Table has no cells")
        
        logger.info(f"Extracting header tree from table with {table.num_rows()} rows")
        
        # Identify header rows
        header_rows = self._identify_header_rows(table)
        
        if len(header_rows) == 0:
            logger.warning("No header rows identified, creating flat structure")
            header_rows = [0]  # Use first row as header
        
        # Build root node
        root = HeaderNode(
            text="TABLE_ROOT",
            level=0,
            span=table.num_cols(),
            col_start=0,
            col_end=table.num_cols(),
            row_start=0,
            row_end=table.num_rows(),
        )
        
        # Build hierarchy level by level
        if header_rows:
            self._build_hierarchy_levels(table, root, header_rows)
        else:
            # Fallback: create flat structure from first row
            self._build_flat_hierarchy(table, root)
        
        # Create HeaderTree
        header_tree = HeaderTree(root=root)
        
        logger.info(
            f"Extracted header tree: {header_tree.max_level} levels, "
            f"{header_tree.num_columns} columns"
        )
        
        return header_tree

    def _identify_header_rows(self, table: TableObject) -> List[int]:
        """
        Identify which rows are header rows.
        
        Typically first N rows with merged cells or header markers.
        
        Args:
            table: TableObject to analyze
            
        Returns:
            List of row indices (0-based) that are headers
        """
        header_rows: List[int] = []
        max_rows_to_check = min(self.max_header_rows, table.num_rows())
        
        for row_idx in range(max_rows_to_check):
            if row_idx >= len(table.cells):
                break
            
            row = table.cells[row_idx]
            
            # Check if row contains header cells
            is_header_row = False
            
            # Method 1: Check if cells are marked as headers
            header_cell_count = sum(1 for cell in row if cell.is_header)
            if header_cell_count > len(row) * 0.5:  # More than 50% are headers
                is_header_row = True
            
            # Method 2: Check for merged cells (colspan > 1 or rowspan > 1)
            if not is_header_row:
                merged_cell_count = sum(
                    1 for cell in row if cell.colspan > 1 or cell.rowspan > 1
                )
                if merged_cell_count > 0:
                    is_header_row = True
            
            # Method 3: Check if row contains mostly text (not numeric data)
            if not is_header_row and len(row) > 0:
                text_cell_count = sum(
                    1 for cell in row
                    if cell.content and not self._is_numeric(cell.content)
                )
                if text_cell_count > len(row) * 0.7:  # More than 70% text
                    is_header_row = True
            
            if is_header_row:
                header_rows.append(row_idx)
            else:
                # Stop if we hit a data row
                if len(header_rows) >= self.min_header_rows:
                    break
        
        return header_rows

    def _is_numeric(self, text: str) -> bool:
        """
        Check if text is numeric.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be numeric
        """
        if not text:
            return False
        
        text = text.strip().replace(",", "").replace("%", "")
        
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _build_hierarchy_levels(
        self,
        table: TableObject,
        root: HeaderNode,
        header_rows: List[int],
    ) -> None:
        """
        Build hierarchy levels from header rows.
        
        Args:
            table: TableObject to process
            root: Root HeaderNode
            header_rows: List of header row indices
        """
        # Create a grid to track which columns are covered by which nodes
        num_cols = table.num_cols()
        level_nodes: List[List[Optional[HeaderNode]]] = [
            [None] * num_cols for _ in range(len(header_rows) + 1)
        ]
        
        # Process each header row level
        for level_idx, row_idx in enumerate(header_rows):
            level = level_idx + 1
            row = table.cells[row_idx]
            
            # Process cells in this row
            col_idx = 0
            for cell in row:
                # Skip to next available column
                while col_idx < num_cols and level_nodes[level_idx][col_idx] is not None:
                    col_idx += 1
                
                if col_idx >= num_cols:
                    break
                
                # Calculate column span
                col_span = cell.colspan
                col_end = min(col_idx + col_span, num_cols)
                
                # Create header node
                node = HeaderNode(
                    text=cell.content.strip() or f"Header_{level}_{col_idx}",
                    level=level,
                    span=col_span,
                    row_span=cell.rowspan,
                    col_start=col_idx,
                    col_end=col_end,
                    row_start=row_idx,
                    row_end=row_idx + cell.rowspan,
                    cell_ref=cell,
                )
                
                # Find parent node (from previous level)
                parent = self._find_parent_node(
                    node, level_nodes[level_idx - 1] if level_idx > 0 else None, root
                )
                
                if parent:
                    parent.add_child(node)
                else:
                    # If no parent found, attach to root
                    root.add_child(node)
                
                # Mark columns as covered
                for c in range(col_idx, col_end):
                    if c < num_cols:
                        level_nodes[level_idx][c] = node
                
                col_idx = col_end
        
        # Create leaf nodes for data columns
        self._create_leaf_nodes(table, level_nodes, len(header_rows), root)

    def _find_parent_node(
        self,
        node: HeaderNode,
        parent_level_nodes: Optional[List[Optional[HeaderNode]]],
        root: HeaderNode,
    ) -> Optional[HeaderNode]:
        """
        Find parent node for a given node based on spatial overlap.
        
        Args:
            node: Child node to find parent for
            parent_level_nodes: Nodes from parent level (None for root level)
            root: Root node as fallback
            
        Returns:
            Parent HeaderNode or None
        """
        if parent_level_nodes is None:
            return root
        
        # Find parent by spatial overlap
        # Parent should cover the same or more columns
        best_parent: Optional[HeaderNode] = None
        best_overlap = 0
        
        for parent_node in parent_level_nodes:
            if parent_node is None:
                continue
            
            # Check if parent's column range overlaps with node's range
            overlap_start = max(parent_node.col_start, node.col_start)
            overlap_end = min(parent_node.col_end, node.col_end)
            overlap = max(0, overlap_end - overlap_start)
            
            # Parent should contain the node's columns
            if (
                parent_node.col_start <= node.col_start
                and parent_node.col_end >= node.col_end
                and overlap > best_overlap
            ):
                best_parent = parent_node
                best_overlap = overlap
        
        return best_parent

    def _create_leaf_nodes(
        self,
        table: TableObject,
        level_nodes: List[List[Optional[HeaderNode]]],
        last_header_level: int,
        root: HeaderNode,
    ) -> None:
        """
        Create leaf nodes for data columns.
        
        Args:
            table: TableObject
            level_nodes: Grid of nodes by level and column
            last_header_level: Index of last header level
            root: Root node
        """
        num_cols = table.num_cols()
        last_level = level_nodes[last_header_level] if last_header_level < len(level_nodes) else None
        
        if last_level is None:
            return
        
        # Find parent for each column
        for col_idx in range(num_cols):
            parent = last_level[col_idx]
            
            if parent is None:
                # No parent found, create leaf attached to root
                leaf = HeaderNode(
                    text=f"Column_{col_idx}",
                    level=last_header_level + 1,
                    span=1,
                    col_start=col_idx,
                    col_end=col_idx + 1,
                    row_start=table.num_rows(),
                    row_end=table.num_rows(),
                )
                root.add_child(leaf)
            else:
                # Check if parent already has children (shouldn't for leaf level)
                # If not, this column is a leaf
                if parent.is_leaf():
                    # Parent becomes intermediate, create leaf
                    leaf = HeaderNode(
                        text=parent.text + f"_col{col_idx}",
                        level=last_header_level + 1,
                        span=1,
                        col_start=col_idx,
                        col_end=col_idx + 1,
                        row_start=table.num_rows(),
                        row_end=table.num_rows(),
                    )
                    parent.add_child(leaf)

    def _build_flat_hierarchy(
        self, table: TableObject, root: HeaderNode
    ) -> None:
        """
        Build a flat hierarchy (single level) from first row.
        
        Args:
            table: TableObject
            root: Root node
        """
        if not table.cells or len(table.cells) == 0:
            return
        
        first_row = table.cells[0]
        
        for col_idx, cell in enumerate(first_row):
            node = HeaderNode(
                text=cell.content.strip() or f"Column_{col_idx}",
                level=1,
                span=cell.colspan,
                row_span=cell.rowspan,
                col_start=col_idx,
                col_end=col_idx + cell.colspan,
                row_start=0,
                row_end=cell.rowspan,
                cell_ref=cell,
            )
            root.add_child(node)

    def _calculate_spatial_overlap(
        self, node1: HeaderNode, node2: HeaderNode
    ) -> float:
        """
        Calculate spatial overlap between two nodes.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        # Column overlap
        col_overlap_start = max(node1.col_start, node2.col_start)
        col_overlap_end = min(node1.col_end, node2.col_end)
        col_overlap = max(0, col_overlap_end - col_overlap_start)
        
        # Row overlap
        row_overlap_start = max(node1.row_start, node2.row_start)
        row_overlap_end = min(node1.row_end, node2.row_end)
        row_overlap = max(0, row_overlap_end - row_overlap_start)
        
        # Calculate intersection over union
        node1_area = (node1.col_end - node1.col_start) * (
            node1.row_end - node1.row_start
        )
        node2_area = (node2.col_end - node2.col_start) * (
            node2.row_end - node2.row_start
        )
        intersection = col_overlap * row_overlap
        union = node1_area + node2_area - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union


# Alias for backward compatibility
HierarchyDetector = HierarchyExtractor

"""
Cell merging and normalization utilities.

Handles merged cells by creating explicit coordinate mappings and
supporting both Excel-style and zero-indexed coordinate systems.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
import re

from src.parsing.extractor import TableObject, Cell

logger = logging.getLogger(__name__)


@dataclass
class MergedCellInfo:
    """
    Information about a merged cell.
    
    Attributes:
        merge_root: Root coordinate (row, col) of the merged cell
        merge_span: Span dimensions (rowspan, colspan)
        content: Cell content
        is_merged: Whether this is a merged cell
    """
    merge_root: Tuple[int, int]
    merge_span: Tuple[int, int]  # (rowspan, colspan)
    content: str
    is_merged: bool = False

    def covers(self, row: int, col: int) -> bool:
        """
        Check if this merged cell covers the given coordinate.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if coordinate is covered by this merged cell
        """
        root_row, root_col = self.merge_root
        rowspan, colspan = self.merge_span
        
        return (
            root_row <= row < root_row + rowspan
            and root_col <= col < root_col + colspan
        )


@dataclass
class NormalizedCell:
    """
    Normalized cell with explicit coordinate mapping.
    
    Attributes:
        row: Row index (0-based)
        col: Column index (0-based)
        content: Cell content
        is_merged: Whether this cell is part of a merged region
        merge_root: Root coordinate if merged, None otherwise
        merge_span: Span dimensions if merged, None otherwise
        logical_address: Logical address (Excel-style or tuple)
    """
    row: int
    col: int
    content: str
    is_merged: bool = False
    merge_root: Optional[Tuple[int, int]] = None
    merge_span: Optional[Tuple[int, int]] = None
    logical_address: Optional[Union[str, Tuple[int, int]]] = None

    def get_physical_coords(self) -> Tuple[int, int]:
        """
        Get physical coordinates (may be merged root).
        
        Returns:
            Tuple of (row, col) coordinates
        """
        if self.is_merged and self.merge_root:
            return self.merge_root
        return (self.row, self.col)


@dataclass
class NormalizedTable:
    """
    Normalized table with explicit coordinate mappings for merged cells.
    
    Attributes:
        cells: 2D grid of NormalizedCell objects [row][col]
        num_rows: Number of rows
        num_cols: Number of columns
        merged_regions: List of MergedCellInfo objects
        original_table: Reference to original TableObject
    """
    cells: List[List[NormalizedCell]] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    merged_regions: List[MergedCellInfo] = field(default_factory=list)
    original_table: Optional[TableObject] = None

    def get_cell(self, row: int, col: int) -> Optional[NormalizedCell]:
        """
        Get normalized cell at specified coordinates.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            NormalizedCell or None if out of bounds
        """
        if 0 <= row < self.num_rows and 0 <= col < self.num_cols:
            return self.cells[row][col]
        return None

    def find_merged_region(self, row: int, col: int) -> Optional[MergedCellInfo]:
        """
        Find merged region covering the given coordinates.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            MergedCellInfo if found, None otherwise
        """
        for region in self.merged_regions:
            if region.covers(row, col):
                return region
        return None


@dataclass
class CoordinateMap:
    """
    Bidirectional mapping between coordinates and cell references.
    
    Attributes:
        excel_to_tuple: Mapping from Excel-style (A1) to tuple (0,0)
        tuple_to_excel: Mapping from tuple (0,0) to Excel-style (A1)
        tuple_to_cell: Mapping from tuple to NormalizedCell
        excel_to_cell: Mapping from Excel-style to NormalizedCell
    """
    excel_to_tuple: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    tuple_to_excel: Dict[Tuple[int, int], str] = field(default_factory=dict)
    tuple_to_cell: Dict[Tuple[int, int], NormalizedCell] = field(default_factory=dict)
    excel_to_cell: Dict[str, NormalizedCell] = field(default_factory=dict)

    def get_tuple(self, excel_ref: str) -> Optional[Tuple[int, int]]:
        """
        Convert Excel reference to tuple.
        
        Args:
            excel_ref: Excel-style reference (e.g., "A1", "B2")
            
        Returns:
            Tuple (row, col) or None if invalid
        """
        return self.excel_to_tuple.get(excel_ref.upper())

    def get_excel(self, row: int, col: int) -> Optional[str]:
        """
        Convert tuple to Excel reference.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Excel-style reference or None if not found
        """
        return self.tuple_to_excel.get((row, col))

    def get_cell_by_tuple(self, row: int, col: int) -> Optional[NormalizedCell]:
        """
        Get cell by tuple coordinates.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            NormalizedCell or None
        """
        return self.tuple_to_cell.get((row, col))

    def get_cell_by_excel(self, excel_ref: str) -> Optional[NormalizedCell]:
        """
        Get cell by Excel reference.
        
        Args:
            excel_ref: Excel-style reference
            
        Returns:
            NormalizedCell or None
        """
        return self.excel_to_cell.get(excel_ref.upper())


class CellMergeHandler:
    """
    Handle merged cells by creating explicit coordinate mappings.
    
    Converts merged cells to normalized form where each physical cell
    maps to logical addresses, supporting both Excel-style and tuple coordinates.
    """

    def __init__(self) -> None:
        """Initialize cell merge handler."""
        logger.info("CellMergeHandler initialized")

    def normalize_merged_cells(self, table: TableObject) -> NormalizedTable:
        """
        Convert merged cells to explicit coordinates.
        
        Each physical cell maps to one or more logical addresses.
        Example: A merged cell spanning rows 2-4, cols 1-2 creates mappings:
        - (2,1) -> Cell(content, is_merged=True, merge_root=(2,1), merge_span=(3,2))
        - (2,2), (3,1), (3,2), (4,1), (4,2) -> references to (2,1)
        
        Args:
            table: TableObject to normalize
            
        Returns:
            NormalizedTable with explicit coordinate mappings
        """
        logger.info("Normalizing merged cells")
        
        if not table.cells or len(table.cells) == 0:
            raise ValueError("Table has no cells")
        
        num_rows = table.num_rows()
        num_cols = table.num_cols()
        
        # Create a grid to track which cells are covered by merged cells
        cell_grid: List[List[Optional[Cell]]] = [
            [None] * num_cols for _ in range(num_rows)
        ]
        
        # Track merged regions
        merged_regions: List[MergedCellInfo] = []
        
        # First pass: identify merged cells and mark covered areas
        for row_idx, row in enumerate(table.cells):
            for cell in row:
                if cell.rowspan > 1 or cell.colspan > 1:
                    # This is a merged cell
                    merge_info = MergedCellInfo(
                        merge_root=(cell.row, cell.col),
                        merge_span=(cell.rowspan, cell.colspan),
                        content=cell.content,
                        is_merged=True,
                    )
                    merged_regions.append(merge_info)
                    
                    # Mark all covered cells
                    for r in range(cell.row, cell.row + cell.rowspan):
                        for c in range(cell.col, cell.col + cell.colspan):
                            if r < num_rows and c < num_cols:
                                cell_grid[r][c] = cell
        
        # Second pass: create normalized cells
        normalized_cells: List[List[NormalizedCell]] = []
        
        for row_idx in range(num_rows):
            normalized_row: List[NormalizedCell] = []
            
            for col_idx in range(num_cols):
                original_cell = cell_grid[row_idx][col_idx]
                
                if original_cell is None:
                    # Empty cell (shouldn't happen in well-formed tables)
                    normalized_cell = NormalizedCell(
                        row=row_idx,
                        col=col_idx,
                        content="",
                        is_merged=False,
                    )
                elif original_cell.row == row_idx and original_cell.col == col_idx:
                    # This is the root cell of a merged region or a regular cell
                    is_merged = original_cell.rowspan > 1 or original_cell.colspan > 1
                    
                    normalized_cell = NormalizedCell(
                        row=row_idx,
                        col=col_idx,
                        content=original_cell.content,
                        is_merged=is_merged,
                        merge_root=(original_cell.row, original_cell.col) if is_merged else None,
                        merge_span=(original_cell.rowspan, original_cell.colspan) if is_merged else None,
                    )
                else:
                    # This cell is covered by a merged cell
                    # Reference the root cell
                    root_row = original_cell.row
                    root_col = original_cell.col
                    
                    normalized_cell = NormalizedCell(
                        row=row_idx,
                        col=col_idx,
                        content=original_cell.content,  # Same content as root
                        is_merged=True,
                        merge_root=(root_row, root_col),
                        merge_span=(original_cell.rowspan, original_cell.colspan),
                    )
                
                normalized_row.append(normalized_cell)
            
            normalized_cells.append(normalized_row)
        
        # Create NormalizedTable
        normalized_table = NormalizedTable(
            cells=normalized_cells,
            num_rows=num_rows,
            num_cols=num_cols,
            merged_regions=merged_regions,
            original_table=table,
        )
        
        logger.info(
            f"Normalized table: {num_rows}x{num_cols}, "
            f"{len(merged_regions)} merged regions"
        )
        
        return normalized_table

    def create_coordinate_system(
        self, table: NormalizedTable
    ) -> CoordinateMap:
        """
        Assign unique IDs to each logical cell.
        
        Supports both:
        - Excel-style (A1, B2, etc.)
        - Zero-indexed tuples ((0,0), (1,2), etc.)
        
        Stores bidirectional mapping.
        
        Args:
            table: NormalizedTable to create coordinate system for
            
        Returns:
            CoordinateMap with bidirectional mappings
        """
        logger.info("Creating coordinate system")
        
        coord_map = CoordinateMap()
        
        for row_idx in range(table.num_rows):
            for col_idx in range(table.num_cols):
                cell = table.get_cell(row_idx, col_idx)
                if cell is None:
                    continue
                
                # Create Excel-style reference
                excel_ref = self._tuple_to_excel(row_idx, col_idx)
                
                # Store mappings
                coord_map.excel_to_tuple[excel_ref] = (row_idx, col_idx)
                coord_map.tuple_to_excel[(row_idx, col_idx)] = excel_ref
                coord_map.tuple_to_cell[(row_idx, col_idx)] = cell
                coord_map.excel_to_cell[excel_ref] = cell
                
                # Update cell's logical address
                cell.logical_address = excel_ref
        
        logger.info(
            f"Created coordinate system: {len(coord_map.excel_to_tuple)} cells"
        )
        
        return coord_map

    def resolve_cell_reference(
        self,
        coord: Union[str, Tuple[int, int]],
        coord_map: CoordinateMap,
    ) -> Optional[NormalizedCell]:
        """
        Given any coordinate, return the actual cell content.
        
        Handles merged cell references correctly by resolving to the root cell.
        
        Args:
            coord: Coordinate as Excel string (e.g., "A1") or tuple (row, col)
            coord_map: CoordinateMap to use for resolution
            
        Returns:
            NormalizedCell or None if coordinate is invalid
        """
        # Determine coordinate type and convert to tuple
        if isinstance(coord, str):
            # Excel-style reference
            tuple_coord = coord_map.get_tuple(coord)
            if tuple_coord is None:
                logger.warning(f"Invalid Excel reference: {coord}")
                return None
            row, col = tuple_coord
        elif isinstance(coord, tuple) and len(coord) == 2:
            row, col = coord
        else:
            logger.warning(f"Invalid coordinate format: {coord}")
            return None
        
        # Get cell by tuple
        cell = coord_map.get_cell_by_tuple(row, col)
        
        if cell is None:
            return None
        
        # If cell is merged, resolve to root cell
        if cell.is_merged and cell.merge_root:
            root_row, root_col = cell.merge_root
            root_cell = coord_map.get_cell_by_tuple(root_row, root_col)
            return root_cell if root_cell else cell
        
        return cell

    def visualize_coordinate_grid(
        self,
        table: NormalizedTable,
        coord_map: Optional[CoordinateMap] = None,
        show_merges: bool = True,
        max_rows: Optional[int] = 20,
        max_cols: Optional[int] = 10,
    ) -> str:
        """
        Display the coordinate grid with merge indicators.
        
        Args:
            table: NormalizedTable to visualize
            coord_map: Optional CoordinateMap for Excel references
            show_merges: Whether to show merge indicators
            max_rows: Maximum rows to display (None for all)
            max_cols: Maximum columns to display (None for all)
            
        Returns:
            String representation of the coordinate grid
        """
        lines: List[str] = []
        
        num_rows = min(table.num_rows, max_rows) if max_rows else table.num_rows
        num_cols = min(table.num_cols, max_cols) if max_cols else table.num_cols
        
        # Header row with column labels
        header_parts = ["Row\\Col"]
        if coord_map:
            for col in range(num_cols):
                excel_ref = coord_map.get_excel(0, col) or f"C{col}"
                header_parts.append(f"{excel_ref:>8}")
        else:
            for col in range(num_cols):
                header_parts.append(f"{col:>8}")
        
        lines.append(" | ".join(header_parts))
        lines.append("-" * len(" | ".join(header_parts)))
        
        # Data rows
        for row_idx in range(num_rows):
            row_parts = [f"{row_idx:>5}"]
            
            for col_idx in range(num_cols):
                cell = table.get_cell(row_idx, col_idx)
                if cell is None:
                    row_parts.append(" " * 8)
                    continue
                
                # Format cell content
                content = cell.content[:6] if len(cell.content) > 6 else cell.content
                content = content.replace("\n", " ").replace("\r", "")
                
                # Add merge indicator
                if show_merges and cell.is_merged:
                    if cell.merge_root == (row_idx, col_idx):
                        # Root cell
                        indicator = "M"
                    else:
                        # Referenced cell
                        indicator = "→"
                    content = f"{indicator}{content}"[:8]
                
                row_parts.append(f"{content:>8}")
            
            lines.append(" | ".join(row_parts))
        
        # Add merge legend if showing merges
        if show_merges:
            lines.append("")
            lines.append("Legend: M = Merge root, → = Merged cell reference")
        
        return "\n".join(lines)

    def _tuple_to_excel(self, row: int, col: int) -> str:
        """
        Convert zero-indexed tuple to Excel-style reference.
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            
        Returns:
            Excel-style reference (e.g., "A1", "B2")
        """
        # Convert column to letters (A, B, ..., Z, AA, AB, ...)
        col_letters = []
        col_num = col
        
        while col_num >= 0:
            col_letters.insert(0, chr(ord("A") + (col_num % 26)))
            col_num = col_num // 26 - 1
            if col_num < 0:
                break
        
        col_str = "".join(col_letters)
        
        # Excel rows are 1-indexed
        row_num = row + 1
        
        return f"{col_str}{row_num}"

    def _excel_to_tuple(self, excel_ref: str) -> Optional[Tuple[int, int]]:
        """
        Convert Excel-style reference to zero-indexed tuple.
        
        Args:
            excel_ref: Excel-style reference (e.g., "A1", "B2")
            
        Returns:
            Tuple (row, col) or None if invalid
        """
        # Match pattern: letters followed by numbers
        match = re.match(r"^([A-Z]+)(\d+)$", excel_ref.upper())
        if not match:
            return None
        
        col_str, row_str = match.groups()
        
        # Convert column letters to number
        col = 0
        for char in col_str:
            col = col * 26 + (ord(char) - ord("A") + 1)
        col -= 1  # Convert to 0-indexed
        
        # Convert row number (Excel is 1-indexed)
        row = int(row_str) - 1
        
        return (row, col)


# Alias for backward compatibility
CellMerger = CellMergeHandler

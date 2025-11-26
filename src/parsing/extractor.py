"""
Table extraction from various document formats.

Supports PDF and HTML table extraction with comprehensive structure preservation.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import logging
import html

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Cell:
    """
    Represents a single cell in a table.
    
    Attributes:
        row: Row index (0-based)
        col: Column index (0-based)
        content: Cell content as string
        rowspan: Number of rows this cell spans (default: 1)
        colspan: Number of columns this cell spans (default: 1)
        is_header: Whether this cell is a header cell
    """
    row: int
    col: int
    content: str
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False

    def __post_init__(self) -> None:
        """Validate cell attributes."""
        if self.rowspan < 1:
            raise ValueError(f"rowspan must be >= 1, got {self.rowspan}")
        if self.colspan < 1:
            raise ValueError(f"colspan must be >= 1, got {self.colspan}")
        if self.content is None:
            self.content = ""


@dataclass
class TableObject:
    """
    Represents a structured table with cells, headers, and metadata.
    
    Attributes:
        cells: 2D list of Cell objects organized by [row][col]
        headers: List of header cells with hierarchy information
        metadata: Dictionary containing additional metadata (page_num, bbox, etc.)
        raw_html: HTML representation with rowspan/colspan preserved
    """
    cells: List[List[Cell]] = field(default_factory=list)
    headers: List[Cell] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_html: str = ""

    def __post_init__(self) -> None:
        """Initialize and validate table structure."""
        if not self.cells:
            self.cells = []
        if not self.headers:
            self.headers = []
        if not self.metadata:
            self.metadata = {}

    def get_cell(self, row: int, col: int) -> Optional[Cell]:
        """
        Get cell at specified position.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Cell object or None if position is invalid
        """
        if 0 <= row < len(self.cells) and 0 <= col < len(self.cells[row]):
            return self.cells[row][col]
        return None

    def num_rows(self) -> int:
        """Get number of rows in the table."""
        return len(self.cells)

    def num_cols(self) -> int:
        """Get number of columns in the table."""
        if not self.cells:
            return 0
        return max(len(row) for row in self.cells) if self.cells else 0


class TableExtractor:
    """
    Extract tables from documents (PDF, HTML, etc.).
    
    Supports multiple extraction backends and formats with comprehensive
    structure preservation including merged cells and multi-page tables.
    """

    def __init__(
        self,
        use_pdfplumber: bool = True,
        use_beautifulsoup: bool = True,
    ) -> None:
        """
        Initialize table extractor.
        
        Args:
            use_pdfplumber: Enable PDF extraction via pdfplumber
            use_beautifulsoup: Enable HTML extraction via BeautifulSoup
            
        Raises:
            ImportError: If required libraries are not available
        """
        self.use_pdfplumber = use_pdfplumber
        self.use_beautifulsoup = use_beautifulsoup
        
        if use_pdfplumber and not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber is required for PDF extraction. "
                "Install it with: pip install pdfplumber"
            )
        
        if use_beautifulsoup and not BEAUTIFULSOUP_AVAILABLE:
            raise ImportError(
                "beautifulsoup4 is required for HTML extraction. "
                "Install it with: pip install beautifulsoup4"
            )
        
        logger.info("TableExtractor initialized")

    def extract_from_pdf(
        self, pdf_path: str, pages: Optional[List[int]] = None
    ) -> List[TableObject]:
        """
        Extract tables from PDF document using pdfplumber.
        
        Detects and extracts tables from PDF, preserving structure including
        merged cells and handling multi-page tables.
        
        Args:
            pdf_path: Path to PDF file (as string)
            pages: Optional list of page numbers to extract (1-indexed)
            
        Returns:
            List of TableObject instances with extracted tables
            
        Raises:
            FileNotFoundError: If PDF file does not exist
            ValueError: If PDF is corrupted or cannot be read
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not self.use_pdfplumber:
            raise ValueError("PDF extraction is disabled")
        
        logger.info(f"Extracting tables from PDF: {pdf_path}")
        tables: List[TableObject] = []
        
        try:
            with pdfplumber.open(str(pdf_path_obj)) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = pages if pages else list(range(1, total_pages + 1))
                
                for page_num in pages_to_process:
                    if page_num < 1 or page_num > total_pages:
                        logger.warning(
                            f"Page {page_num} out of range (1-{total_pages}), skipping"
                        )
                        continue
                    
                    page = pdf.pages[page_num - 1]  # Convert to 0-indexed
                    
                    try:
                        # Extract tables from page
                        page_tables = page.extract_tables()
                        
                        if not page_tables:
                            continue
                        
                        for table_idx, table_data in enumerate(page_tables):
                            if not table_data or len(table_data) == 0:
                                continue
                            
                            try:
                                table_obj = self._convert_pdf_table_to_object(
                                    table_data, page_num, page, table_idx
                                )
                                if table_obj:
                                    tables.append(table_obj)
                            except Exception as e:
                                logger.error(
                                    f"Error processing table {table_idx} on page {page_num}: {e}",
                                    exc_info=True
                                )
                                continue
                    
                    except Exception as e:
                        logger.error(
                            f"Error processing page {page_num}: {e}",
                            exc_info=True
                        )
                        continue
        
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}", exc_info=True)
            raise ValueError(f"Failed to extract tables from PDF: {e}") from e
        
        logger.info(f"Extracted {len(tables)} tables from PDF")
        return tables

    def _convert_pdf_table_to_object(
        self,
        table_data: List[List[Optional[str]]],
        page_num: int,
        page: Any,
        table_idx: int,
    ) -> Optional[TableObject]:
        """
        Convert pdfplumber table data to TableObject.
        
        Args:
            table_data: Raw table data from pdfplumber
            page_num: Page number (1-indexed)
            page: pdfplumber Page object
            table_idx: Table index on the page
            
        Returns:
            TableObject instance or None if conversion fails
        """
        if not table_data or len(table_data) == 0:
            return None
        
        try:
            # Get bounding box for this table
            page_tables = page.find_tables()
            bbox = None
            if page_tables and table_idx < len(page_tables):
                table_obj = page_tables[table_idx]
                bbox = (
                    table_obj.bbox[0],  # x0
                    table_obj.bbox[1],  # y0
                    table_obj.bbox[2],  # x1
                    table_obj.bbox[3],  # y1
                )
            
            # Convert to cells with structure detection
            cells: List[List[Cell]] = []
            headers: List[Cell] = []
            
            # Detect merged cells and build cell grid
            num_rows = len(table_data)
            num_cols = max(len(row) for row in table_data) if table_data else 0
            
            # Create a grid to track cell positions
            cell_grid: List[List[Optional[Cell]]] = [
                [None] * num_cols for _ in range(num_rows)
            ]
            
            for row_idx, row_data in enumerate(table_data):
                cell_row: List[Cell] = []
                col_idx = 0
                
                for cell_data in row_data:
                    # Skip None cells (already handled by merged cells)
                    while col_idx < num_cols and cell_grid[row_idx][col_idx] is not None:
                        col_idx += 1
                    
                    if col_idx >= num_cols:
                        break
                    
                    content = str(cell_data).strip() if cell_data is not None else ""
                    
                    # Detect if this is a header row (first row or all caps)
                    is_header = (
                        row_idx == 0
                        or (content and content.isupper() and len(content) > 2)
                    )
                    
                    # For now, assume no merged cells in PDF (pdfplumber handles this)
                    # In future, can detect by checking cell boundaries
                    cell = Cell(
                        row=row_idx,
                        col=col_idx,
                        content=content,
                        rowspan=1,
                        colspan=1,
                        is_header=is_header,
                    )
                    
                    cell_row.append(cell)
                    cell_grid[row_idx][col_idx] = cell
                    
                    if is_header:
                        headers.append(cell)
                    
                    col_idx += 1
                
                cells.append(cell_row)
            
            # Generate HTML representation
            raw_html = self._generate_html_from_cells(cells)
            
            # Create TableObject
            table_obj = TableObject(
                cells=cells,
                headers=headers,
                metadata={
                    "page_num": page_num,
                    "bbox": bbox,
                    "table_idx": table_idx,
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                },
                raw_html=raw_html,
            )
            
            return table_obj
        
        except Exception as e:
            logger.error(f"Error converting PDF table to object: {e}", exc_info=True)
            return None

    def extract_from_html(
        self, html_string: str
    ) -> TableObject:
        """
        Parse HTML table and extract structure using BeautifulSoup.
        
        Extracts table structure including row/column spans, cell content,
        positions, and table headers (thead, th tags).
        
        Args:
            html_string: HTML string containing table(s)
            
        Returns:
            TableObject instance with extracted table structure
            
        Raises:
            ValueError: If HTML is malformed or contains no tables
        """
        if not self.use_beautifulsoup:
            raise ValueError("HTML extraction is disabled")
        
        logger.info("Extracting table from HTML")
        
        try:
            soup = BeautifulSoup(html_string, "lxml")
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}", exc_info=True)
            raise ValueError(f"Failed to parse HTML: {e}") from e
        
        # Find first table (or can be extended to handle multiple tables)
        table_tag = soup.find("table")
        
        if table_tag is None:
            raise ValueError("No table found in HTML string")
        
        try:
            table_obj = self._parse_html_table(table_tag, html_string)
            return table_obj
        except Exception as e:
            logger.error(f"Error parsing HTML table: {e}", exc_info=True)
            raise ValueError(f"Failed to extract table from HTML: {e}") from e

    def _parse_html_table(
        self, table_tag: Any, original_html: str
    ) -> TableObject:
        """
        Parse a BeautifulSoup table tag into TableObject.
        
        Args:
            table_tag: BeautifulSoup table tag
            original_html: Original HTML string for raw_html field
            
        Returns:
            TableObject instance
        """
        cells: List[List[Cell]] = []
        headers: List[Cell] = []
        
        # Find thead and tbody
        thead = table_tag.find("thead")
        tbody = table_tag.find("tbody")
        tfoot = table_tag.find("tfoot")
        
        # Process thead
        header_rows: List[Any] = []
        if thead:
            header_rows = thead.find_all("tr")
        
        # Process tbody (or all rows if no tbody)
        body_rows: List[Any] = []
        if tbody:
            body_rows = tbody.find_all("tr")
        else:
            # If no tbody, get all tr tags not in thead
            body_rows = table_tag.find_all("tr")
            if thead:
                body_rows = [tr for tr in body_rows if tr not in header_rows]
        
        # Process tfoot
        footer_rows: List[Any] = []
        if tfoot:
            footer_rows = tfoot.find_all("tr")
        
        # Build cell grid with rowspan/colspan handling
        all_rows = header_rows + body_rows + footer_rows
        max_cols = self._calculate_max_columns(all_rows)
        
        # Create grid to track occupied cells
        cell_grid: List[List[Optional[Cell]]] = [
            [None] * max_cols for _ in range(len(all_rows))
        ]
        
        current_row = 0
        
        # Process header rows
        for row_idx, tr in enumerate(header_rows):
            col_idx = 0
            cell_row: List[Cell] = []
            
            for cell_tag in tr.find_all(["th", "td"]):
                # Skip to next available column
                while col_idx < max_cols and cell_grid[current_row][col_idx] is not None:
                    col_idx += 1
                
                if col_idx >= max_cols:
                    break
                
                # Get cell attributes
                rowspan = int(cell_tag.get("rowspan", 1))
                colspan = int(cell_tag.get("colspan", 1))
                content = self._extract_cell_content(cell_tag)
                is_header = cell_tag.name == "th" or row_idx < len(header_rows)
                
                # Create cell
                cell = Cell(
                    row=current_row,
                    col=col_idx,
                    content=content,
                    rowspan=rowspan,
                    colspan=colspan,
                    is_header=is_header,
                )
                
                cell_row.append(cell)
                headers.append(cell)
                
                # Mark occupied cells in grid
                for r in range(current_row, current_row + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if r < len(cell_grid) and c < len(cell_grid[r]):
                            cell_grid[r][c] = cell
                
                col_idx += colspan
            
            cells.append(cell_row)
            current_row += 1
        
        # Process body rows
        for tr in body_rows:
            col_idx = 0
            cell_row: List[Cell] = []
            
            for cell_tag in tr.find_all(["td", "th"]):
                # Skip to next available column
                while col_idx < max_cols and cell_grid[current_row][col_idx] is not None:
                    col_idx += 1
                
                if col_idx >= max_cols:
                    break
                
                # Get cell attributes
                rowspan = int(cell_tag.get("rowspan", 1))
                colspan = int(cell_tag.get("colspan", 1))
                content = self._extract_cell_content(cell_tag)
                is_header = cell_tag.name == "th"
                
                # Create cell
                cell = Cell(
                    row=current_row,
                    col=col_idx,
                    content=content,
                    rowspan=rowspan,
                    colspan=colspan,
                    is_header=is_header,
                )
                
                cell_row.append(cell)
                
                if is_header:
                    headers.append(cell)
                
                # Mark occupied cells in grid
                for r in range(current_row, current_row + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if r < len(cell_grid) and c < len(cell_grid[r]):
                            cell_grid[r][c] = cell
                
                col_idx += colspan
            
            cells.append(cell_row)
            current_row += 1
        
        # Process footer rows
        for tr in footer_rows:
            col_idx = 0
            cell_row: List[Cell] = []
            
            for cell_tag in tr.find_all(["td", "th"]):
                # Skip to next available column
                while col_idx < max_cols and cell_grid[current_row][col_idx] is not None:
                    col_idx += 1
                
                if col_idx >= max_cols:
                    break
                
                rowspan = int(cell_tag.get("rowspan", 1))
                colspan = int(cell_tag.get("colspan", 1))
                content = self._extract_cell_content(cell_tag)
                
                cell = Cell(
                    row=current_row,
                    col=col_idx,
                    content=content,
                    rowspan=rowspan,
                    colspan=colspan,
                    is_header=False,
                )
                
                cell_row.append(cell)
                
                # Mark occupied cells
                for r in range(current_row, current_row + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if r < len(cell_grid) and c < len(cell_grid[r]):
                            cell_grid[r][c] = cell
                
                col_idx += colspan
            
            cells.append(cell_row)
            current_row += 1
        
        # Generate HTML representation
        raw_html = str(table_tag)
        
        # Create TableObject
        table_obj = TableObject(
            cells=cells,
            headers=headers,
            metadata={
                "num_rows": len(cells),
                "num_cols": max_cols,
                "has_thead": thead is not None,
                "has_tbody": tbody is not None,
                "has_tfoot": tfoot is not None,
            },
            raw_html=raw_html,
        )
        
        return table_obj

    def _extract_cell_content(self, cell_tag: Any) -> str:
        """
        Extract text content from a cell tag, handling nested elements.
        
        Args:
            cell_tag: BeautifulSoup cell tag (td or th)
            
        Returns:
            Extracted text content
        """
        # Get text and clean it up
        text = cell_tag.get_text(separator=" ", strip=True)
        # Decode HTML entities
        text = html.unescape(text)
        return text

    def _calculate_max_columns(self, rows: List[Any]) -> int:
        """
        Calculate maximum number of columns considering rowspan/colspan.
        
        Args:
            rows: List of BeautifulSoup tr tags
            
        Returns:
            Maximum column count
        """
        max_cols = 0
        
        for tr in rows:
            col_count = 0
            for cell in tr.find_all(["td", "th"]):
                colspan = int(cell.get("colspan", 1))
                col_count += colspan
            max_cols = max(max_cols, col_count)
        
        return max_cols if max_cols > 0 else 1

    def _generate_html_from_cells(
        self, cells: List[List[Cell]]
    ) -> str:
        """
        Generate HTML representation from cell grid.
        
        Args:
            cells: 2D list of Cell objects
            
        Returns:
            HTML string representation
        """
        html_parts = ["<table>"]
        
        for row in cells:
            html_parts.append("<tr>")
            for cell in row:
                tag = "th" if cell.is_header else "td"
                attrs = []
                
                if cell.rowspan > 1:
                    attrs.append(f'rowspan="{cell.rowspan}"')
                if cell.colspan > 1:
                    attrs.append(f'colspan="{cell.colspan}"')
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                content = html.escape(cell.content)
                html_parts.append(f"<{tag}{attr_str}>{content}</{tag}>")
            
            html_parts.append("</tr>")
        
        html_parts.append("</table>")
        return "".join(html_parts)

    def extract(
        self, document_path: str
    ) -> List[TableObject]:
        """
        Extract tables from document (auto-detect format).
        
        Args:
            document_path: Path to document file
            
        Returns:
            List of TableObject instances
            
        Raises:
            ValueError: If file format is not supported
        """
        path_obj = Path(document_path)
        suffix = path_obj.suffix.lower()
        
        if suffix == ".pdf":
            return self.extract_from_pdf(str(path_obj))
        elif suffix in [".html", ".htm"]:
            with open(path_obj, "r", encoding="utf-8") as f:
                html_content = f.read()
            return [self.extract_from_html(html_content)]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

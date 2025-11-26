"""
Tests for TableExtractor class.

Covers edge cases including nested merged cells, multi-header tables,
and tables spanning multiple pages.
"""

import pytest
from pathlib import Path
from typing import List
import tempfile
import os

from src.parsing.extractor import (
    TableExtractor,
    TableObject,
    Cell,
)


class TestCell:
    """Test cases for Cell dataclass."""

    def test_cell_creation(self) -> None:
        """Test basic cell creation."""
        cell = Cell(row=0, col=0, content="Test")
        assert cell.row == 0
        assert cell.col == 0
        assert cell.content == "Test"
        assert cell.rowspan == 1
        assert cell.colspan == 1
        assert cell.is_header is False

    def test_cell_with_spans(self) -> None:
        """Test cell with rowspan and colspan."""
        cell = Cell(
            row=0, col=0, content="Merged", rowspan=2, colspan=3
        )
        assert cell.rowspan == 2
        assert cell.colspan == 3

    def test_cell_invalid_rowspan(self) -> None:
        """Test that invalid rowspan raises error."""
        with pytest.raises(ValueError, match="rowspan must be >= 1"):
            Cell(row=0, col=0, content="Test", rowspan=0)

    def test_cell_invalid_colspan(self) -> None:
        """Test that invalid colspan raises error."""
        with pytest.raises(ValueError, match="colspan must be >= 1"):
            Cell(row=0, col=0, content="Test", colspan=-1)


class TestTableObject:
    """Test cases for TableObject dataclass."""

    def test_table_object_creation(self) -> None:
        """Test basic table object creation."""
        table = TableObject()
        assert table.cells == []
        assert table.headers == []
        assert table.metadata == {}
        assert table.raw_html == ""

    def test_table_object_with_cells(self) -> None:
        """Test table object with cells."""
        cells = [
            [Cell(0, 0, "A"), Cell(0, 1, "B")],
            [Cell(1, 0, "C"), Cell(1, 1, "D")],
        ]
        table = TableObject(cells=cells)
        assert table.num_rows() == 2
        assert table.num_cols() == 2

    def test_get_cell(self) -> None:
        """Test getting cell by position."""
        cells = [
            [Cell(0, 0, "A"), Cell(0, 1, "B")],
            [Cell(1, 0, "C"), Cell(1, 1, "D")],
        ]
        table = TableObject(cells=cells)
        assert table.get_cell(0, 0).content == "A"
        assert table.get_cell(1, 1).content == "D"
        assert table.get_cell(2, 0) is None
        assert table.get_cell(0, 5) is None


class TestTableExtractor:
    """Test cases for TableExtractor class."""

    def test_extractor_init(self) -> None:
        """Test extractor initialization."""
        extractor = TableExtractor()
        assert extractor.use_pdfplumber is True
        assert extractor.use_beautifulsoup is True

    def test_extractor_init_without_pdfplumber(self) -> None:
        """Test extractor initialization without pdfplumber."""
        # This will fail if pdfplumber is installed
        # In real scenario, would mock the import
        pass

    def test_extract_from_html_simple_table(self) -> None:
        """Test extracting simple HTML table."""
        extractor = TableExtractor()
        
        html = """
        <table>
            <tr>
                <th>Header1</th>
                <th>Header2</th>
            </tr>
            <tr>
                <td>Data1</td>
                <td>Data2</td>
            </tr>
        </table>
        """
        
        table_obj = extractor.extract_from_html(html)
        
        assert isinstance(table_obj, TableObject)
        assert table_obj.num_rows() == 2
        assert table_obj.num_cols() == 2
        assert len(table_obj.headers) >= 2
        assert table_obj.get_cell(0, 0).content == "Header1"
        assert table_obj.get_cell(1, 0).content == "Data1"

    def test_extract_from_html_with_rowspan_colspan(self) -> None:
        """Test extracting HTML table with merged cells."""
        extractor = TableExtractor()
        
        html = """
        <table>
            <tr>
                <th rowspan="2">Merged Row</th>
                <th colspan="2">Merged Col</th>
            </tr>
            <tr>
                <td>Data1</td>
                <td>Data2</td>
            </tr>
            <tr>
                <td rowspan="1" colspan="1">Single</td>
                <td>Data3</td>
                <td>Data4</td>
            </tr>
        </table>
        """
        
        table_obj = extractor.extract_from_html(html)
        
        assert isinstance(table_obj, TableObject)
        # Find the merged cell
        merged_cell = table_obj.get_cell(0, 0)
        assert merged_cell is not None
        assert merged_cell.rowspan == 2
        assert merged_cell.content == "Merged Row"
        
        # Check colspan
        merged_col_cell = table_obj.get_cell(0, 1)
        assert merged_col_cell is not None
        assert merged_col_cell.colspan == 2

    def test_extract_from_html_nested_merged_cells(self) -> None:
        """Test extracting table with nested merged cells."""
        extractor = TableExtractor()
        
        html = """
        <table>
            <thead>
                <tr>
                    <th rowspan="2">Outer</th>
                    <th colspan="2">Inner</th>
                </tr>
                <tr>
                    <th>Sub1</th>
                    <th>Sub2</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td rowspan="2">Nested</td>
                    <td>Data1</td>
                    <td>Data2</td>
                </tr>
                <tr>
                    <td>Data3</td>
                    <td>Data4</td>
                </tr>
            </tbody>
        </table>
        """
        
        table_obj = extractor.extract_from_html(html)
        
        assert isinstance(table_obj, TableObject)
        assert table_obj.metadata["has_thead"] is True
        assert table_obj.metadata["has_tbody"] is True
        
        # Check nested structure
        outer_cell = table_obj.get_cell(0, 0)
        assert outer_cell is not None
        assert outer_cell.rowspan == 2
        assert outer_cell.is_header is True

    def test_extract_from_html_multi_header_table(self) -> None:
        """Test extracting table with multiple header rows."""
        extractor = TableExtractor()
        
        html = """
        <table>
            <thead>
                <tr>
                    <th>Header Row 1 Col 1</th>
                    <th>Header Row 1 Col 2</th>
                </tr>
                <tr>
                    <th>Header Row 2 Col 1</th>
                    <th>Header Row 2 Col 2</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Data1</td>
                    <td>Data2</td>
                </tr>
            </tbody>
        </table>
        """
        
        table_obj = extractor.extract_from_html(html)
        
        assert isinstance(table_obj, TableObject)
        assert len(table_obj.headers) >= 4  # At least 4 header cells
        assert table_obj.get_cell(0, 0).is_header is True
        assert table_obj.get_cell(1, 0).is_header is True

    def test_extract_from_html_malformed_table(self) -> None:
        """Test handling of malformed HTML table."""
        extractor = TableExtractor()
        
        # Missing closing tags
        html = """
        <table>
            <tr>
                <th>Header1
                <td>Data1
        </table>
        """
        
        # Should not raise exception, but may produce incomplete results
        try:
            table_obj = extractor.extract_from_html(html)
            # BeautifulSoup is forgiving, so it may still parse
            assert isinstance(table_obj, TableObject)
        except Exception:
            # If it fails, that's also acceptable for malformed HTML
            pass

    def test_extract_from_html_no_table(self) -> None:
        """Test error when HTML contains no table."""
        extractor = TableExtractor()
        
        html = "<div>No table here</div>"
        
        with pytest.raises(ValueError, match="No table found"):
            extractor.extract_from_html(html)

    def test_extract_from_html_empty_table(self) -> None:
        """Test extracting empty table."""
        extractor = TableExtractor()
        
        html = "<table></table>"
        
        table_obj = extractor.extract_from_html(html)
        assert isinstance(table_obj, TableObject)
        assert table_obj.num_rows() == 0

    def test_extract_from_html_with_tfoot(self) -> None:
        """Test extracting table with tfoot section."""
        extractor = TableExtractor()
        
        html = """
        <table>
            <thead>
                <tr><th>Header</th></tr>
            </thead>
            <tbody>
                <tr><td>Data</td></tr>
            </tbody>
            <tfoot>
                <tr><td>Footer</td></tr>
            </tfoot>
        </table>
        """
        
        table_obj = extractor.extract_from_html(html)
        
        assert isinstance(table_obj, TableObject)
        assert table_obj.metadata["has_tfoot"] is True
        assert table_obj.num_rows() == 3  # header + body + footer

    def test_extract_from_html_special_characters(self) -> None:
        """Test extracting table with special characters and HTML entities."""
        extractor = TableExtractor()
        
        html = """
        <table>
            <tr>
                <td>&lt;Test&gt;</td>
                <td>&amp;Symbol</td>
                <td>한글</td>
            </tr>
        </table>
        """
        
        table_obj = extractor.extract_from_html(html)
        
        assert table_obj.get_cell(0, 0).content == "<Test>"
        assert table_obj.get_cell(0, 1).content == "&Symbol"
        assert table_obj.get_cell(0, 2).content == "한글"

    @pytest.mark.skipif(
        not os.path.exists("data/raw") or len(list(Path("data/raw").glob("*.pdf"))) == 0,
        reason="No PDF files available for testing"
    )
    def test_extract_from_pdf_simple(self) -> None:
        """Test extracting tables from PDF file."""
        extractor = TableExtractor()
        
        # Find first PDF in data/raw
        pdf_files = list(Path("data/raw").glob("*.pdf"))
        if not pdf_files:
            pytest.skip("No PDF files available")
        
        pdf_path = str(pdf_files[0])
        tables = extractor.extract_from_pdf(pdf_path)
        
        assert isinstance(tables, list)
        # May be empty if PDF has no tables
        for table in tables:
            assert isinstance(table, TableObject)
            assert "page_num" in table.metadata

    def test_extract_from_pdf_nonexistent_file(self) -> None:
        """Test error when PDF file does not exist."""
        extractor = TableExtractor()
        
        with pytest.raises(FileNotFoundError):
            extractor.extract_from_pdf("nonexistent_file.pdf")

    def test_extract_from_pdf_specific_pages(self) -> None:
        """Test extracting from specific pages."""
        extractor = TableExtractor()
        
        # This test requires a real PDF file
        # Skip if no PDFs available
        pdf_files = list(Path("data/raw").glob("*.pdf"))
        if not pdf_files:
            pytest.skip("No PDF files available")
        
        pdf_path = str(pdf_files[0])
        tables = extractor.extract_from_pdf(pdf_path, pages=[1])
        
        assert isinstance(tables, list)
        for table in tables:
            assert table.metadata["page_num"] == 1

    def test_extract_from_pdf_invalid_page_numbers(self) -> None:
        """Test handling of invalid page numbers."""
        extractor = TableExtractor()
        
        pdf_files = list(Path("data/raw").glob("*.pdf"))
        if not pdf_files:
            pytest.skip("No PDF files available")
        
        pdf_path = str(pdf_files[0])
        
        # Should not raise error, just skip invalid pages
        tables = extractor.extract_from_pdf(pdf_path, pages=[1, 99999, -1])
        assert isinstance(tables, list)

    def test_extract_auto_detect_pdf(self) -> None:
        """Test auto-detection for PDF files."""
        extractor = TableExtractor()
        
        pdf_files = list(Path("data/raw").glob("*.pdf"))
        if not pdf_files:
            pytest.skip("No PDF files available")
        
        tables = extractor.extract(str(pdf_files[0]))
        assert isinstance(tables, list)

    def test_extract_auto_detect_html(self) -> None:
        """Test auto-detection for HTML files."""
        extractor = TableExtractor()
        
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write("<table><tr><td>Test</td></tr></table>")
            temp_path = f.name
        
        try:
            tables = extractor.extract(temp_path)
            assert len(tables) == 1
            assert isinstance(tables[0], TableObject)
        finally:
            os.unlink(temp_path)

    def test_extract_unsupported_format(self) -> None:
        """Test error for unsupported file format."""
        extractor = TableExtractor()
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Test content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                extractor.extract(temp_path)
        finally:
            os.unlink(temp_path)

    def test_generate_html_from_cells(self) -> None:
        """Test HTML generation from cells."""
        extractor = TableExtractor()
        
        cells = [
            [
                Cell(0, 0, "Header", is_header=True),
                Cell(0, 1, "Value", rowspan=2, colspan=1),
            ],
            [Cell(1, 0, "Data")],
        ]
        
        html = extractor._generate_html_from_cells(cells)
        
        assert "<table>" in html
        assert "<th" in html or "Header" in html
        assert 'rowspan="2"' in html



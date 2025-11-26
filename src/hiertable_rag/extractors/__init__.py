"""
Table extraction modules for various document formats.
"""

from hiertable_rag.extractors.pdf_extractor import PDFTableExtractor

__all__ = ["PDFTableExtractor", "TableExtractor"]

# Alias for backward compatibility
TableExtractor = PDFTableExtractor



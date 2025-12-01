"""
HierTable-RAG: Hierarchical Table-based Retrieval Augmented Generation

A research project for extracting, structuring, and querying hierarchical
table data using RAG techniques.
"""

__version__ = "0.1.0"

from src.hiertable_rag.core.rag import HierTableRAG
# from src.hiertable_rag.extractors import TableExtractor
# from src.hiertable_rag.processors import TableProcessor

__all__ = [
    "HierTableRAG",
    "TableExtractor",
    "TableProcessor",
    "__version__",
]



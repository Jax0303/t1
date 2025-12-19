"""
HierTable-RAG: Hierarchical Table-based Retrieval Augmented Generation
"""

__version__ = "0.1.0"

from src.hiertable_rag.core.rag import HierTableRAG
from src.hiertable_rag.core.indexer import MultiGranularityIndexer, MultiGranularityRetriever
from src.hiertable_rag.core.evaluator import AgenticEvaluator

__all__ = [
    "HierTableRAG",
    "MultiGranularityIndexer",
    "MultiGranularityRetriever",
    "AgenticEvaluator",
    "__version__",
]

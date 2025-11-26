"""
HierTable-RAG: Hierarchical Table-based Retrieval Augmented Generation

Main package for hierarchical table extraction, processing, and RAG.
"""

__version__ = "0.1.0"

# Import main modules
from src.parsing import TableExtractor, HierarchyDetector, CellMerger
from src.encoding import TreeEncoder, GraphBuilder
from src.retrieval import HierarchicalIndexer, HierarchicalRetriever
from src.generation import PromptBuilder, LLMGenerator
# Evaluation metrics may not be fully implemented
try:
    from src.evaluation import (
        RetrievalMetrics,
        GenerationMetrics,
        HierarchyMetrics,
    )
except ImportError:
    # These classes may not exist yet
    RetrievalMetrics = None
    GenerationMetrics = None
    HierarchyMetrics = None

__all__ = [
    "TableExtractor",
    "HierarchyDetector",
    "CellMerger",
    "TreeEncoder",
    "GraphBuilder",
    "HierarchicalIndexer",
    "HierarchicalRetriever",
    "PromptBuilder",
    "LLMGenerator",
    "RetrievalMetrics",
    "GenerationMetrics",
    "HierarchyMetrics",
    "__version__",
]

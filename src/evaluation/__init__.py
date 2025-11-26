"""
Evaluation modules for measuring system performance.
"""

from src.evaluation.metrics import (
    QAMetrics,
    StructureMetrics,
    FaithfulnessMetrics,
    EfficiencyMetrics,
    EvaluationFramework,
    QA,
)

# These classes are not yet implemented in metrics.py
# Placeholder classes for future implementation
class RetrievalMetrics:
    pass

class GenerationMetrics:
    pass

class HierarchyMetrics:
    pass

from src.evaluation.baselines import (
    FlatRAG,
    TableRAG,
    DirectLLM,
    BaselineComparator,
    ExperimentRun,
    TableMarkdownConverter,
)

__all__ = [
    "RetrievalMetrics",
    "GenerationMetrics",
    "HierarchyMetrics",
    "QAMetrics",
    "StructureMetrics",
    "FaithfulnessMetrics",
    "EfficiencyMetrics",
    "EvaluationFramework",
    "QA",
    "FlatRAG",
    "TableRAG",
    "DirectLLM",
    "BaselineComparator",
    "ExperimentRun",
    "TableMarkdownConverter",
]

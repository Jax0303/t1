"""
Evaluation metrics for HierTable-RAG system.

Provides comprehensive evaluation framework including:
- Standard QA metrics (exact match, F1)
- Structure-aware metrics (structure recall, hierarchy preservation)
- Faithfulness metrics (citation accuracy)
- Efficiency metrics (context efficiency)
- Dataset-level evaluation and baseline comparison
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging
import re
import json
import math
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats

from src.generation.prompting import Answer, CellRef
from src.retrieval.retriever import ContextBundle, RetrievalResult
from src.parsing.hierarchy import HeaderTree, HeaderNode

logger = logging.getLogger(__name__)


@dataclass
class QA:
    """
    Question-Answer pair with ground truth information.
    
    Attributes:
        question: Question text
        answer: Ground truth answer text
        gold_cells: List of cell references that should be cited
        query_type: Type of query (e.g., "aggregation", "lookup", "comparison")
        required_hierarchy: Required hierarchy levels for answering
        metadata: Additional metadata
    """
    question: str
    answer: str
    gold_cells: List[CellRef] = field(default_factory=list)
    query_type: str = ""
    required_hierarchy: Optional[HeaderTree] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QAMetrics:
    """
    Standard QA evaluation metrics.
    
    Includes exact match, F1 score with numerical tolerance.
    """

    @staticmethod
    def compute_exact_match(pred: str, gold: str) -> float:
        """
        Compute exact match score.
        
        Normalizes whitespace and case before comparison.
        
        Args:
            pred: Predicted answer
            gold: Ground truth answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Normalize whitespace and case
        pred_normalized = re.sub(r'\s+', ' ', pred.strip().lower())
        gold_normalized = re.sub(r'\s+', ' ', gold.strip().lower())
        
        if pred_normalized == gold_normalized:
            return 1.0
        
        # Try numerical comparison with tolerance
        if QAMetrics._is_numerical_match(pred_normalized, gold_normalized):
            return 1.0
        
        return 0.0

    @staticmethod
    def compute_f1(pred: str, gold: str) -> float:
        """
        Compute F1 score between predicted and gold answers.
        
        Uses token-level matching with numerical tolerance.
        
        Args:
            pred: Predicted answer
            gold: Ground truth answer
            
        Returns:
            F1 score (0.0 to 1.0)
        """
        # Tokenize (split on whitespace and punctuation)
        pred_tokens = QAMetrics._tokenize(pred)
        gold_tokens = QAMetrics._tokenize(gold)
        
        if len(gold_tokens) == 0:
            return 1.0 if len(pred_tokens) == 0 else 0.0
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # Count matches with numerical tolerance
        matches = 0
        gold_matched = set()
        
        for pred_token in pred_tokens:
            for i, gold_token in enumerate(gold_tokens):
                if i in gold_matched:
                    continue
                
                if pred_token.lower() == gold_token.lower():
                    matches += 1
                    gold_matched.add(i)
                    break
                elif QAMetrics._is_numerical_match(pred_token, gold_token):
                    matches += 1
                    gold_matched.add(i)
                    break
        
        precision = matches / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = matches / len(gold_tokens) if len(gold_tokens) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation, keep tokens
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    @staticmethod
    def _is_numerical_match(pred: str, gold: str, tolerance: float = 0.01) -> bool:
        """
        Check if two strings represent numerically equivalent values.
        
        Args:
            pred: Predicted value string
            gold: Ground truth value string
            tolerance: Relative tolerance for comparison (default: 0.01 = 1%)
            
        Returns:
            True if numerically equivalent within tolerance
        """
        try:
            # Try to parse as numbers
            pred_num = float(re.sub(r'[^\d.-]', '', pred))
            gold_num = float(re.sub(r'[^\d.-]', '', gold))
            
            # Check if within tolerance
            if abs(gold_num) < 1e-10:  # Near zero
                return abs(pred_num - gold_num) < tolerance
            else:
                relative_error = abs(pred_num - gold_num) / abs(gold_num)
                return relative_error <= tolerance
        except (ValueError, TypeError):
            # Not numerical, fall back to exact match
            return False


class StructureMetrics:
    """
    Structure-aware evaluation metrics.
    
    Measures how well the system retrieves and uses table structure.
    """

    @staticmethod
    def compute_structure_recall(
        pred_cells: List[CellRef],
        gold_cells: List[CellRef]
    ) -> float:
        """
        Measure if correct cells were retrieved/cited.
        
        Args:
            pred_cells: Predicted/cited cell references
            gold_cells: Ground truth cell references
            
        Returns:
            Recall score (0.0 to 1.0)
        """
        if len(gold_cells) == 0:
            return 1.0 if len(pred_cells) == 0 else 0.0
        
        # Convert to sets of tuples for comparison
        pred_set = set(
            cell.to_tuple(zero_indexed=True) for cell in pred_cells
        )
        gold_set = set(
            cell.to_tuple(zero_indexed=True) for cell in gold_cells
        )
        
        # Calculate recall
        correct = len(pred_set & gold_set)
        recall = correct / len(gold_set) if len(gold_set) > 0 else 0.0
        
        return recall

    @staticmethod
    def compute_hierarchy_preservation(
        retrieved_context: ContextBundle,
        required_hierarchy: Optional[HeaderTree]
    ) -> float:
        """
        Check if necessary hierarchy information was included.
        
        Args:
            retrieved_context: Retrieved context bundle
            required_hierarchy: Required hierarchy tree
            
        Returns:
            Hierarchy preservation score (0.0 to 1.0)
        """
        if required_hierarchy is None:
            # No hierarchy requirement, return 1.0
            return 1.0
        
        if not retrieved_context.hierarchy_context:
            # No hierarchy in retrieved context
            return 0.0
        
        # Extract hierarchy levels from retrieved context
        retrieved_levels = StructureMetrics._extract_hierarchy_levels(
            retrieved_context.hierarchy_context
        )
        
        # Extract required hierarchy levels
        required_levels = StructureMetrics._extract_required_levels(
            required_hierarchy
        )
        
        if len(required_levels) == 0:
            return 1.0
        
        # Check how many required levels are present
        present_levels = retrieved_levels & required_levels
        score = len(present_levels) / len(required_levels)
        
        return score

    @staticmethod
    def _extract_hierarchy_levels(
        hierarchy_context: Dict[str, Any]
    ) -> Set[int]:
        """
        Extract hierarchy levels from context bundle.
        
        Args:
            hierarchy_context: Hierarchy context dictionary
            
        Returns:
            Set of hierarchy levels present
        """
        levels = set()
        
        if "nodes" in hierarchy_context:
            for node in hierarchy_context["nodes"]:
                if "granularity" in node:
                    granularity = node["granularity"]
                    # Map granularity to level
                    level_map = {
                        "table": 0,
                        "subtable": 1,
                        "row": 2,
                        "cell": 3,
                    }
                    if granularity in level_map:
                        levels.add(level_map[granularity])
        
        return levels

    @staticmethod
    def _extract_required_levels(header_tree: HeaderTree) -> Set[int]:
        """
        Extract required hierarchy levels from header tree.
        
        Args:
            header_tree: Header tree structure
            
        Returns:
            Set of required hierarchy levels
        """
        levels = set()
        
        def collect_levels(node: HeaderNode):
            levels.add(node.level)
            for child in node.children:
                collect_levels(child)
        
        collect_levels(header_tree.root)
        
        return levels


class FaithfulnessMetrics:
    """
    Faithfulness evaluation metrics.
    
    Measures citation accuracy and hallucination detection.
    """

    @staticmethod
    def compute_citation_accuracy(
        answer: Answer,
        context: ContextBundle
    ) -> Dict[str, float]:
        """
        Compute citation accuracy metrics.
        
        Args:
            answer: Answer object with citations
            context: Context bundle with retrieved cells
            
        Returns:
            Dictionary with citation metrics:
            - citation_precision: cited cells that are in context
            - citation_recall: context cells that should be cited
            - hallucination_rate: cited cells not in context
        """
        # Build coordinate set from context
        context_coords = set()
        for result in context.retrieved_cells:
            coords = result.metadata.coordinates
            (row_start, row_end), (col_start, col_end) = coords
            
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    context_coords.add((row, col))
        
        # Build coordinate set from citations
        cited_coords = set(
            cell.to_tuple(zero_indexed=True) for cell in answer.cited_cells
        )
        
        # Calculate metrics
        if len(cited_coords) == 0:
            citation_precision = 1.0  # No citations, consider perfect precision
            hallucination_rate = 0.0
        else:
            valid_citations = cited_coords & context_coords
            citation_precision = (
                len(valid_citations) / len(cited_coords)
                if len(cited_coords) > 0
                else 0.0
            )
            hallucination_rate = (
                len(cited_coords - context_coords) / len(cited_coords)
                if len(cited_coords) > 0
                else 0.0
            )
        
        # Citation recall: how many context cells were cited
        if len(context_coords) == 0:
            citation_recall = 1.0 if len(cited_coords) == 0 else 0.0
        else:
            cited_from_context = cited_coords & context_coords
            citation_recall = (
                len(cited_from_context) / len(context_coords)
                if len(context_coords) > 0
                else 0.0
            )
        
        return {
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "hallucination_rate": hallucination_rate,
        }


class EfficiencyMetrics:
    """
    Efficiency evaluation metrics.
    
    Measures how efficiently the system uses retrieved context.
    """

    @staticmethod
    def compute_context_efficiency(
        context: ContextBundle,
        answer: Answer
    ) -> Dict[str, float]:
        """
        Compute context efficiency metrics.
        
        Args:
            context: Retrieved context bundle
            answer: Generated answer
            
        Returns:
            Dictionary with efficiency metrics:
            - useful_cell_ratio: cited cells / total retrieved cells
            - token_efficiency: useful tokens / total context tokens
        """
        total_cells = len(context.retrieved_cells)
        cited_cells = len(answer.cited_cells)
        
        # Useful cell ratio
        useful_cell_ratio = (
            cited_cells / total_cells if total_cells > 0 else 0.0
        )
        
        # Token efficiency
        # Estimate tokens from content length (rough approximation: 1 token ≈ 4 chars)
        total_tokens = sum(
            len(result.content) / 4 for result in context.retrieved_cells
        )
        
        # Count tokens in cited cells
        cited_coords = set(
            cell.to_tuple(zero_indexed=True) for cell in answer.cited_cells
        )
        
        useful_tokens = 0
        for result in context.retrieved_cells:
            coords = result.metadata.coordinates
            (row_start, row_end), (col_start, col_end) = coords
            
            # Check if any cell in this result is cited
            is_cited = any(
                (row, col) in cited_coords
                for row in range(row_start, row_end)
                for col in range(col_start, col_end)
            )
            
            if is_cited:
                useful_tokens += len(result.content) / 4
        
        token_efficiency = (
            useful_tokens / total_tokens if total_tokens > 0 else 0.0
        )
        
        return {
            "useful_cell_ratio": useful_cell_ratio,
            "token_efficiency": token_efficiency,
        }


class EvaluationFramework:
    """
    Comprehensive evaluation framework for HierTable-RAG.
    
    Provides dataset-level evaluation and baseline comparison.
    """

    def __init__(self) -> None:
        """Initialize evaluation framework."""
        logger.info("EvaluationFramework initialized")

    def evaluate_on_dataset(
        self,
        predictions: List[Answer],
        ground_truth: List[QA],
        contexts: Optional[List[ContextBundle]] = None,
    ) -> Dict[str, Any]:
        """
        Run all metrics on full dataset.
        
        Args:
            predictions: List of predicted answers
            ground_truth: List of ground truth QA pairs
            contexts: Optional list of context bundles for each prediction
            
        Returns:
            Dictionary with comprehensive evaluation results:
            - overall_metrics: Aggregate metrics
            - per_query_metrics: Metrics per query
            - error_analysis: Detailed error analysis
            - confusion_matrix: Confusion matrix for query types
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground truth "
                f"({len(ground_truth)}) must have same length"
            )
        
        if contexts is not None and len(contexts) != len(predictions):
            raise ValueError(
                f"Contexts ({len(contexts) if contexts else 0}) must match "
                f"predictions ({len(predictions)}) length"
            )
        
        logger.info(f"Evaluating {len(predictions)} predictions")
        
        # Initialize metrics accumulators
        em_scores = []
        f1_scores = []
        structure_recalls = []
        hierarchy_scores = []
        citation_precisions = []
        citation_recalls = []
        hallucination_rates = []
        useful_cell_ratios = []
        token_efficiencies = []
        
        per_query_metrics = []
        error_analysis = []
        
        # Query type tracking
        query_type_stats = defaultdict(lambda: {
            "count": 0,
            "em": [],
            "f1": [],
            "structure_recall": [],
        })
        
        # Evaluate each prediction
        for i, (pred, gold) in enumerate(zip(predictions, ground_truth)):
            # Standard QA metrics
            em = QAMetrics.compute_exact_match(pred.answer_text, gold.answer)
            f1 = QAMetrics.compute_f1(pred.answer_text, gold.answer)
            
            em_scores.append(em)
            f1_scores.append(f1)
            
            # Structure metrics
            structure_recall = StructureMetrics.compute_structure_recall(
                pred.cited_cells, gold.gold_cells
            )
            structure_recalls.append(structure_recall)
            
            # Hierarchy preservation
            if contexts and contexts[i] is not None:
                hierarchy_score = StructureMetrics.compute_hierarchy_preservation(
                    contexts[i], gold.required_hierarchy
                )
            else:
                hierarchy_score = 0.0
            hierarchy_scores.append(hierarchy_score)
            
            # Faithfulness metrics
            if contexts and contexts[i] is not None:
                citation_metrics = FaithfulnessMetrics.compute_citation_accuracy(
                    pred, contexts[i]
                )
                citation_precision = citation_metrics["citation_precision"]
                citation_recall = citation_metrics["citation_recall"]
                hallucination_rate = citation_metrics["hallucination_rate"]
                citation_precisions.append(citation_precision)
                citation_recalls.append(citation_recall)
                hallucination_rates.append(hallucination_rate)
            else:
                citation_precision = 0.0
                citation_recall = 0.0
                hallucination_rate = 0.0
            
            # Efficiency metrics
            if contexts and contexts[i] is not None:
                efficiency_metrics = EfficiencyMetrics.compute_context_efficiency(
                    contexts[i], pred
                )
                useful_cell_ratio = efficiency_metrics["useful_cell_ratio"]
                token_efficiency = efficiency_metrics["token_efficiency"]
                useful_cell_ratios.append(useful_cell_ratio)
                token_efficiencies.append(token_efficiency)
            else:
                useful_cell_ratio = 0.0
                token_efficiency = 0.0
            
            # Track query type
            query_type = gold.query_type or "unknown"
            query_type_stats[query_type]["count"] += 1
            query_type_stats[query_type]["em"].append(em)
            query_type_stats[query_type]["f1"].append(f1)
            query_type_stats[query_type]["structure_recall"].append(
                structure_recall
            )
            
            # Per-query metrics
            query_metrics = {
                "query_id": i,
                "question": gold.question,
                "exact_match": em,
                "f1": f1,
                "structure_recall": structure_recall,
                "hierarchy_score": hierarchy_score,
                "citation_precision": citation_precision,
                "citation_recall": citation_recall,
                "hallucination_rate": hallucination_rate,
                "useful_cell_ratio": useful_cell_ratio,
                "token_efficiency": token_efficiency,
            }
            per_query_metrics.append(query_metrics)
            
            # Error analysis for failed cases
            if em < 1.0 or f1 < 0.8:
                error_analysis.append({
                    "query_id": i,
                    "question": gold.question,
                    "predicted": pred.answer_text,
                    "gold": gold.answer,
                    "exact_match": em,
                    "f1": f1,
                    "structure_recall": structure_recall,
                    "error_type": self._classify_error(pred, gold),
                })
        
        # Aggregate metrics
        overall_metrics = {
            "exact_match": np.mean(em_scores),
            "f1": np.mean(f1_scores),
            "structure_recall": np.mean(structure_recalls),
            "hierarchy_preservation": np.mean(hierarchy_scores),
            "citation_precision": (
                np.mean(citation_precisions) if citation_precisions else 0.0
            ),
            "citation_recall": (
                np.mean(citation_recalls) if citation_recalls else 0.0
            ),
            "hallucination_rate": (
                np.mean(hallucination_rates) if hallucination_rates else 0.0
            ),
            "useful_cell_ratio": (
                np.mean(useful_cell_ratios) if useful_cell_ratios else 0.0
            ),
            "token_efficiency": (
                np.mean(token_efficiencies) if token_efficiencies else 0.0
            ),
        }
        
        # Query type breakdown
        query_type_breakdown = {}
        for qtype, stats_dict in query_type_stats.items():
            query_type_breakdown[qtype] = {
                "count": stats_dict["count"],
                "exact_match": np.mean(stats_dict["em"]),
                "f1": np.mean(stats_dict["f1"]),
                "structure_recall": np.mean(stats_dict["structure_recall"]),
            }
        
        # Confusion matrix (simplified - for query types)
        confusion_matrix = self._build_confusion_matrix(
            predictions, ground_truth
        )
        
        return {
            "overall_metrics": overall_metrics,
            "per_query_metrics": per_query_metrics,
            "query_type_breakdown": query_type_breakdown,
            "error_analysis": error_analysis,
            "confusion_matrix": confusion_matrix,
        }

    def _classify_error(self, pred: Answer, gold: QA) -> str:
        """
        Classify error type.
        
        Args:
            pred: Predicted answer
            gold: Ground truth
            
        Returns:
            Error type string
        """
        em = QAMetrics.compute_exact_match(pred.answer_text, gold.answer)
        f1 = QAMetrics.compute_f1(pred.answer_text, gold.answer)
        
        if em == 1.0:
            return "none"
        elif f1 < 0.3:
            return "completely_wrong"
        elif f1 < 0.7:
            return "partially_correct"
        else:
            return "minor_error"

    def _build_confusion_matrix(
        self, predictions: List[Answer], ground_truth: List[QA]
    ) -> Dict[str, Any]:
        """
        Build confusion matrix for query types.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth
            
        Returns:
            Confusion matrix dictionary
        """
        # Simplified confusion matrix
        # In practice, this would compare predicted vs actual query types
        matrix = defaultdict(lambda: defaultdict(int))
        
        for pred, gold in zip(predictions, ground_truth):
            gold_type = gold.query_type or "unknown"
            # For now, we don't have predicted query type
            # This is a placeholder structure
            matrix[gold_type][gold_type] += 1
        
        return dict(matrix)

    def compare_baselines(
        self,
        systems: Dict[str, Any],
        dataset: List[QA],
    ) -> pd.DataFrame:
        """
        Compare multiple systems side-by-side.
        
        Args:
            systems: Dictionary mapping system names to system objects
                Each system should have a method: generate_answer(query: str) -> Answer
            dataset: List of QA pairs for evaluation
            
        Returns:
            DataFrame with comparison results including statistical tests
        """
        logger.info(f"Comparing {len(systems)} systems on {len(dataset)} examples")
        
        # Evaluate each system
        system_results = {}
        
        for system_name, system in systems.items():
            logger.info(f"Evaluating system: {system_name}")
            
            predictions = []
            for qa in dataset:
                try:
                    # Generate answer (assuming system has this method)
                    if hasattr(system, "generate_answer"):
                        answer = system.generate_answer(qa.question)
                    elif hasattr(system, "query"):
                        result = system.query(qa.question)
                        # Convert to Answer format
                        answer = Answer(
                            answer_text=result.get("answer", ""),
                            model=system_name,
                        )
                    else:
                        raise ValueError(
                            f"System {system_name} must have "
                            f"generate_answer or query method"
                        )
                    predictions.append(answer)
                except Exception as e:
                    logger.error(
                        f"Error evaluating {system_name} on query: {e}"
                    )
                    # Add empty answer as placeholder
                    predictions.append(
                        Answer(answer_text="", model=system_name)
                    )
            
            # Evaluate predictions
            eval_results = self.evaluate_on_dataset(predictions, dataset)
            system_results[system_name] = {
                "predictions": predictions,
                "metrics": eval_results["overall_metrics"],
            }
        
        # Build comparison DataFrame
        comparison_data = []
        
        for system_name, results in system_results.items():
            metrics = results["metrics"]
            comparison_data.append({
                "system": system_name,
                "exact_match": metrics["exact_match"],
                "f1": metrics["f1"],
                "structure_recall": metrics["structure_recall"],
                "citation_precision": metrics["citation_precision"],
                "citation_recall": metrics["citation_recall"],
                "hallucination_rate": metrics["hallucination_rate"],
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Statistical significance tests
        if len(systems) >= 2:
            system_names = list(systems.keys())
            baseline_name = system_names[0]
            
            significance_results = {}
            
            for metric in ["exact_match", "f1", "structure_recall"]:
                baseline_scores = self._extract_metric_scores(
                    system_results[baseline_name]["predictions"],
                    dataset,
                    metric,
                )
                
                for other_system in system_names[1:]:
                    other_scores = self._extract_metric_scores(
                        system_results[other_system]["predictions"],
                        dataset,
                        metric,
                    )
                    
                    # T-test
                    t_stat, p_value = stats.ttest_rel(baseline_scores, other_scores)
                    
                    # Bootstrap confidence interval
                    ci_lower, ci_upper = self._bootstrap_ci(
                        baseline_scores, other_scores
                    )
                    
                    key = f"{baseline_name}_vs_{other_system}_{metric}"
                    significance_results[key] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
            
            df.attrs["significance_tests"] = significance_results
        
        return df

    def _extract_metric_scores(
        self, predictions: List[Answer], ground_truth: List[QA], metric: str
    ) -> List[float]:
        """
        Extract metric scores for statistical testing.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth
            metric: Metric name
            
        Returns:
            List of scores
        """
        scores = []
        
        for pred, gold in zip(predictions, ground_truth):
            if metric == "exact_match":
                score = QAMetrics.compute_exact_match(
                    pred.answer_text, gold.answer
                )
            elif metric == "f1":
                score = QAMetrics.compute_f1(pred.answer_text, gold.answer)
            elif metric == "structure_recall":
                score = StructureMetrics.compute_structure_recall(
                    pred.cited_cells, gold.gold_cells
                )
            else:
                score = 0.0
            
            scores.append(score)
        
        return scores

    def _bootstrap_ci(
        self, scores1: List[float], scores2: List[float], n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for difference.
        
        Args:
            scores1: First set of scores
            scores2: Second set of scores
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound) for 95% CI
        """
        differences = [s2 - s1 for s1, s2 in zip(scores1, scores2)]
        
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(
                differences, size=len(differences), replace=True
            )
            bootstrap_diffs.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        return ci_lower, ci_upper

    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = "json",
    ) -> None:
        """
        Export evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Output file path
            format: Output format ("json" or "latex")
        """
        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif format == "latex":
            # Convert to LaTeX table format
            latex_table = self._results_to_latex(results)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _results_to_latex(self, results: Dict[str, Any]) -> str:
        """
        Convert results to LaTeX table format.
        
        Args:
            results: Evaluation results
            
        Returns:
            LaTeX table string
        """
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{|l|c|c|c|}")
        lines.append("\\hline")
        lines.append("Metric & Value \\\\")
        lines.append("\\hline")
        
        if "overall_metrics" in results:
            metrics = results["overall_metrics"]
            for metric_name, value in metrics.items():
                metric_display = metric_name.replace("_", " ").title()
                lines.append(f"{metric_display} & {value:.4f} \\\\")
        
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Evaluation Results}")
        lines.append("\\label{tab:eval_results}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)

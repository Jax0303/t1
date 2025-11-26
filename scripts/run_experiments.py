#!/usr/bin/env python3
"""
Experiment runner for HierTable-RAG.

Runs comprehensive experiments with different configurations:
- Different embedding models
- Different LLMs
- Different retrieval strategies
- Different prompt formats

Includes logging, error analysis, and paper figure generation.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import logging
import json
import time
import pickle
from pathlib import Path
from collections import defaultdict
import hashlib

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import pandas as pd
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.generation.prompting import Answer, LLMReasoner, StructuredPromptBuilder
from src.retrieval.retriever import StructureAwareRetriever, ContextBundle
from src.evaluation.metrics import QA, EvaluationFramework
from src.evaluation.baselines import FlatRAG, TableRAG, DirectLLM
from src.parsing.extractor import TableObject

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """
    Experiment configuration.
    
    Attributes:
        experiment_id: Unique experiment identifier
        embedding_model: Embedding model name (e.g., "all-MiniLM-L6-v2", "BGE")
        llm_model: LLM model name (e.g., "gpt-4", "claude-3-opus")
        llm_provider: LLM provider ("openai", "anthropic")
        retrieval_strategy: Retrieval strategy ("adaptive", "hierarchical", "flat")
        prompt_format: Prompt format ("explicit", "natural", "code", "table_markdown")
        top_k: Number of retrieved items
        use_gpu: Whether to use GPU
        checkpoint_dir: Directory for checkpoints
        log_to_wandb: Whether to log to Weights & Biases
        log_to_mlflow: Whether to log to MLflow
        wandb_project: W&B project name
        mlflow_experiment: MLflow experiment name
    """
    experiment_id: str
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-4"
    llm_provider: str = "openai"
    retrieval_strategy: str = "adaptive"
    prompt_format: str = "explicit"
    top_k: int = 5
    use_gpu: bool = False
    checkpoint_dir: str = "experiments/checkpoints"
    log_to_wandb: bool = False
    log_to_mlflow: bool = False
    wandb_project: str = "hiertable-rag"
    mlflow_experiment: str = "hiertable-rag"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_hash(self) -> str:
        """Get hash of configuration for checkpointing."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


@dataclass
class ExperimentResults:
    """
    Experiment results.
    
    Attributes:
        config: Experiment configuration
        predictions: List of Answer objects
        ground_truth: List of QA objects
        contexts: List of ContextBundle objects
        metrics: Overall metrics dictionary
        per_query_metrics: Per-query metrics list
        timing: Timing statistics
        token_usage: Token usage statistics
        errors: List of error cases
    """
    config: ExperimentConfig
    predictions: List[Answer] = field(default_factory=list)
    ground_truth: List[QA] = field(default_factory=list)
    contexts: List[ContextBundle] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    per_query_metrics: List[Dict[str, Any]] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "per_query_metrics": self.per_query_metrics,
            "timing": self.timing,
            "token_usage": self.token_usage,
            "errors": self.errors,
        }
    
    def save(self, output_path: Path) -> None:
        """Save results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_path = output_path.with_suffix(".json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save full results (with predictions) as pickle
        full_path = output_path.with_suffix(".pkl")
        with open(full_path, "wb") as f:
            pickle.dump(self, f)
        
        logger.info(f"Saved results to {summary_path} and {full_path}")
    
    @classmethod
    def load(cls, checkpoint_path: Path) -> "ExperimentResults":
        """Load results from checkpoint."""
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)


class ExperimentRunner:
    """
    Main experiment runner.
    """
    
    def __init__(self, output_dir: str = "experiments") -> None:
        """
        Initialize experiment runner.
        
        Args:
            output_dir: Output directory for experiments
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_framework = EvaluationFramework()
        
        logger.info(f"ExperimentRunner initialized (output_dir: {output_dir})")
    
    def run_full_pipeline(
        self,
        config: ExperimentConfig,
        dataset_path: str,
        resume: bool = True,
    ) -> ExperimentResults:
        """
        Run full experiment pipeline.
        
        Args:
            config: Experiment configuration
            dataset_path: Path to dataset JSON file
            resume: Whether to resume from checkpoint if available
            
        Returns:
            ExperimentResults object
        """
        logger.info(f"Running experiment: {config.experiment_id}")
        
        # Check for checkpoint
        checkpoint_path = self._get_checkpoint_path(config)
        if resume and checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            try:
                return ExperimentResults.load(checkpoint_path)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
        
        # Initialize logging
        run_id = self._init_logging(config)
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = self._load_dataset(dataset_path)
        
        # Initialize systems
        logger.info("Initializing systems...")
        systems = self._initialize_systems(config)
        
        # Run experiments
        logger.info("Running experiments...")
        results = self._run_experiments(config, dataset, systems, run_id)
        
        # Save checkpoint
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        results.save(checkpoint_path)
        
        # Finalize logging
        self._finalize_logging(run_id, results)
        
        logger.info(f"Experiment complete: {config.experiment_id}")
        return results
    
    def _load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load dataset from JSON file."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _initialize_systems(
        self, config: ExperimentConfig
    ) -> Dict[str, Any]:
        """Initialize all systems (HierTable-RAG + baselines)."""
        systems = {}
        
        # Check GPU availability
        use_gpu = config.use_gpu
        if use_gpu and TORCH_AVAILABLE:
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("GPU requested but not available, using CPU")
        else:
            use_gpu = False
        
        # Initialize LLM reasoner
        llm_reasoner = LLMReasoner(
            provider=config.llm_provider,
            default_model=config.llm_model,
        )
        
        # HierTable-RAG (placeholder - would need actual implementation)
        # For now, we'll use a wrapper that matches the interface
        systems["hiertable_rag"] = self._create_hiertable_rag_system(
            config, llm_reasoner
        )
        
        # Baselines
        systems["flat_rag"] = FlatRAG(llm_reasoner=llm_reasoner)
        systems["table_rag"] = TableRAG(llm_reasoner=llm_reasoner)
        systems["direct_llm"] = DirectLLM(llm_reasoner=llm_reasoner)
        
        return systems
    
    def _create_hiertable_rag_system(
        self, config: ExperimentConfig, llm_reasoner: LLMReasoner
    ) -> Any:
        """Create HierTable-RAG system."""
        # This would create the actual HierTable-RAG system
        # For now, return a wrapper that matches the interface
        
        class HierTableRAGWrapper:
            def __init__(self, config, llm_reasoner):
                self.config = config
                self.llm_reasoner = llm_reasoner
                self.prompt_builder = StructuredPromptBuilder()
                # Would initialize retriever here
                
            def answer_question(
                self, query: str, tables: List[TableObject]
            ) -> Tuple[Answer, Dict[str, Any]]:
                """Answer question using HierTable-RAG."""
                # Placeholder implementation
                # Would use StructureAwareRetriever and prompt builder
                prompt = f"Answer: {query}"
                answer = self.llm_reasoner.generate_answer(prompt)
                return answer, {}
        
        return HierTableRAGWrapper(config, llm_reasoner)
    
    def _run_experiments(
        self,
        config: ExperimentConfig,
        dataset: Dict[str, Any],
        systems: Dict[str, Any],
        run_id: Optional[str],
    ) -> ExperimentResults:
        """Run experiments on all systems."""
        results = ExperimentResults(config=config)
        
        # Extract QA pairs and tables from dataset
        qa_pairs = dataset.get("qa_pairs", [])
        tables_dict = dataset.get("tables", {})
        
        # Convert QA pairs to QA objects
        from src.evaluation.metrics import QA
        from src.generation.prompting import CellRef
        
        ground_truth = []
        for qa_data in qa_pairs:
            cell_refs = [
                CellRef(
                    row=cell["row"],
                    col=cell["col"],
                    format=cell.get("format", "tuple"),
                    original_text=cell.get("original_text", ""),
                )
                for cell in qa_data.get("required_cells", [])
            ]
            
            qa = QA(
                question=qa_data["question"],
                answer=qa_data["answer"],
                gold_cells=cell_refs,
                query_type=qa_data.get("question_type", ""),
                metadata=qa_data.get("metadata", {}),
            )
            ground_truth.append(qa)
        
        results.ground_truth = ground_truth
        
        # Run each system
        all_predictions = []
        all_contexts = []
        all_timings = []
        all_token_usage = []
        
        iterator = (
            tqdm(enumerate(qa_pairs), desc="Processing queries", total=len(qa_pairs))
            if TQDM_AVAILABLE
            else enumerate(qa_pairs)
        )
        
        for i, qa_data in iterator:
            query = qa_data["question"]
            table_id = qa_data.get("table_id", "")
            
            # Get tables for this query
            tables = self._get_tables_for_query(tables_dict, table_id)
            
            # Run each system
            for system_name, system in systems.items():
                try:
                    start_time = time.time()
                    
                    # Answer question
                    answer, metadata = system.answer_question(query, tables)
                    
                    elapsed_time = time.time() - start_time
                    
                    # Store results
                    all_predictions.append(answer)
                    all_timings.append(elapsed_time)
                    all_token_usage.append(
                        metadata.get("prompt_tokens", 0)
                        + metadata.get("completion_tokens", 0)
                    )
                    
                    # Log to W&B or MLflow
                    if run_id:
                        self._log_query_result(
                            run_id,
                            system_name,
                            i,
                            query,
                            answer,
                            metadata,
                        )
                    
                except Exception as e:
                    error_info = {
                        "system": system_name,
                        "query_id": i,
                        "query": query,
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    results.errors.append(error_info)
                    logger.error(f"Error in {system_name}: {e}")
                    continue
        
        results.predictions = all_predictions
        
        # Evaluate
        eval_results = self.evaluation_framework.evaluate_on_dataset(
            all_predictions, ground_truth, contexts=all_contexts
        )
        
        results.metrics = eval_results["overall_metrics"]
        results.per_query_metrics = eval_results["per_query_metrics"]
        results.timing = {
            "mean": float(np.mean(all_timings)),
            "median": float(np.median(all_timings)),
            "total": float(np.sum(all_timings)),
        }
        results.token_usage = {
            "total": int(np.sum(all_token_usage)),
            "mean": int(np.mean(all_token_usage)),
        }
        
        return results
    
    def _get_tables_for_query(
        self, tables_dict: Dict[str, Any], table_id: str
    ) -> List[TableObject]:
        """Get tables for a query."""
        # Placeholder - would load actual TableObject from tables_dict
        return []
    
    def _init_logging(self, config: ExperimentConfig) -> Optional[str]:
        """Initialize logging (W&B or MLflow)."""
        run_id = None
        
        if config.log_to_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_id,
                config=config.to_dict(),
            )
            run_id = wandb.run.id if wandb.run else None
        
        elif config.log_to_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_experiment(config.mlflow_experiment)
            run = mlflow.start_run(run_name=config.experiment_id)
            mlflow.log_params(config.to_dict())
            run_id = run.info.run_id
        
        return run_id
    
    def _log_query_result(
        self,
        run_id: str,
        system_name: str,
        query_id: int,
        query: str,
        answer: Answer,
        metadata: Dict[str, Any],
    ) -> None:
        """Log individual query result."""
        if WANDB_AVAILABLE and wandb.run:
            wandb.log(
                {
                    f"{system_name}/query_{query_id}/answer_length": len(
                        answer.answer_text
                    ),
                    f"{system_name}/query_{query_id}/tokens": (
                        metadata.get("prompt_tokens", 0)
                        + metadata.get("completion_tokens", 0)
                    ),
                    f"{system_name}/query_{query_id}/time": metadata.get(
                        "total_time", 0.0
                    ),
                }
            )
        
        elif MLFLOW_AVAILABLE:
            mlflow.log_metrics(
                {
                    f"{system_name}_query_{query_id}_answer_length": len(
                        answer.answer_text
                    ),
                    f"{system_name}_query_{query_id}_tokens": (
                        metadata.get("prompt_tokens", 0)
                        + metadata.get("completion_tokens", 0)
                    ),
                }
            )
    
    def _finalize_logging(
        self, run_id: Optional[str], results: ExperimentResults
    ) -> None:
        """Finalize logging."""
        if WANDB_AVAILABLE and wandb.run:
            wandb.log(results.metrics)
            wandb.finish()
        
        elif MLFLOW_AVAILABLE:
            mlflow.log_metrics(results.metrics)
            mlflow.end_run()
    
    def _get_checkpoint_path(self, config: ExperimentConfig) -> Path:
        """Get checkpoint path for configuration."""
        checkpoint_dir = Path(config.checkpoint_dir)
        config_hash = config.get_hash()
        return checkpoint_dir / f"{config.experiment_id}_{config_hash}.pkl"
    
    def generate_error_analysis(
        self, results: ExperimentResults
    ) -> Dict[str, Any]:
        """
        Generate error analysis report.
        
        Args:
            results: ExperimentResults object
            
        Returns:
            Error analysis report dictionary
        """
        logger.info("Generating error analysis")
        
        report = {
            "total_queries": len(results.ground_truth),
            "total_errors": len(results.errors),
            "error_rate": (
                len(results.errors) / len(results.ground_truth)
                if results.ground_truth
                else 0.0
            ),
            "errors_by_type": defaultdict(int),
            "errors_by_query_type": defaultdict(int),
            "errors_by_complexity": defaultdict(int),
            "error_details": [],
            "retrieval_failures": [],
            "reasoning_failures": [],
            "citation_errors": [],
        }
        
        # Analyze errors
        for error in results.errors:
            error_type = error.get("error", "unknown")
            report["errors_by_type"][error_type] += 1
            
            # Classify error type
            error_str = str(error_type).lower()
            if "retrieval" in error_str or "index" in error_str:
                report["retrieval_failures"].append(error)
            elif "reasoning" in error_str or "generation" in error_str:
                report["reasoning_failures"].append(error)
            elif "citation" in error_str or "cell" in error_str:
                report["citation_errors"].append(error)
            
            # Get query type from ground truth
            query_id = error.get("query_id", -1)
            if 0 <= query_id < len(results.ground_truth):
                qa = results.ground_truth[query_id]
                query_type = qa.query_type or "unknown"
                report["errors_by_query_type"][query_type] += 1
        
        # Analyze per-query metrics for failures
        for i, metrics in enumerate(results.per_query_metrics):
            if metrics.get("exact_match", 1.0) < 1.0:
                error_detail = {
                    "query_id": i,
                    "question": metrics.get("question", ""),
                    "exact_match": metrics.get("exact_match", 0.0),
                    "f1": metrics.get("f1", 0.0),
                    "structure_recall": metrics.get("structure_recall", 0.0),
                    "citation_precision": metrics.get("citation_precision", 0.0),
                    "query_type": results.ground_truth[i].query_type
                    if i < len(results.ground_truth)
                    else "unknown",
                }
                report["error_details"].append(error_detail)
        
        # Convert defaultdicts to dicts
        report["errors_by_type"] = dict(report["errors_by_type"])
        report["errors_by_query_type"] = dict(report["errors_by_query_type"])
        report["errors_by_complexity"] = dict(report["errors_by_complexity"])
        
        # Generate visualizations if matplotlib available
        if MATPLOTLIB_AVAILABLE:
            self._plot_error_analysis(report, results.config.experiment_id)
        
        return report
    
    def _plot_error_analysis(
        self, report: Dict[str, Any], experiment_id: str
    ) -> None:
        """Plot error analysis visualizations."""
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Error by type
        if report["errors_by_type"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            types = list(report["errors_by_type"].keys())
            counts = list(report["errors_by_type"].values())
            ax.bar(types, counts)
            ax.set_xlabel("Error Type")
            ax.set_ylabel("Count")
            ax.set_title("Errors by Type")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            output_path = figures_dir / f"{experiment_id}_errors_by_type.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Saved error analysis plot to {output_path}")
        
        # Error by query type
        if report["errors_by_query_type"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            types = list(report["errors_by_query_type"].keys())
            counts = list(report["errors_by_query_type"].values())
            ax.bar(types, counts)
            ax.set_xlabel("Query Type")
            ax.set_ylabel("Error Count")
            ax.set_title("Errors by Query Type")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            output_path = figures_dir / f"{experiment_id}_errors_by_query_type.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
    
    def generate_paper_figures(
        self, all_results: Dict[str, ExperimentResults]
    ) -> List[Path]:
        """
        Generate paper figures.
        
        Args:
            all_results: Dictionary mapping experiment IDs to results
            
        Returns:
            List of figure file paths
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping figure generation")
            return []
        
        logger.info("Generating paper figures")
        
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        figure_paths = []
        
        # 1. Bar chart comparing systems
        fig_path = self._plot_system_comparison(all_results, figures_dir)
        if fig_path:
            figure_paths.append(fig_path)
        
        # 2. Performance vs complexity scatter plot
        fig_path = self._plot_performance_vs_complexity(all_results, figures_dir)
        if fig_path:
            figure_paths.append(fig_path)
        
        # 3. Ablation study results
        fig_path = self._plot_ablation_study(all_results, figures_dir)
        if fig_path:
            figure_paths.append(fig_path)
        
        return figure_paths
    
    def _plot_system_comparison(
        self, all_results: Dict[str, ExperimentResults], output_dir: Path
    ) -> Optional[Path]:
        """Plot bar chart comparing systems."""
        systems = []
        em_scores = []
        f1_scores = []
        
        for exp_id, results in all_results.items():
            systems.append(exp_id)
            em_scores.append(results.metrics.get("exact_match", 0.0))
            f1_scores.append(results.metrics.get("f1", 0.0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.arange(len(systems))
        width = 0.35
        
        ax1.bar(x - width / 2, em_scores, width, label="Exact Match")
        ax1.bar(x + width / 2, f1_scores, width, label="F1")
        ax1.set_xlabel("System")
        ax1.set_ylabel("Score")
        ax1.set_title("System Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(systems, rotation=45, ha="right")
        ax1.legend()
        ax1.set_ylim(0, 1.0)
        
        ax2.bar(systems, em_scores, label="Exact Match")
        ax2.set_xlabel("System")
        ax2.set_ylabel("Exact Match Score")
        ax2.set_title("Exact Match by System")
        ax2.set_xticklabels(systems, rotation=45, ha="right")
        ax2.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        output_path = output_dir / "system_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved system comparison plot to {output_path}")
        return output_path
    
    def _plot_performance_vs_complexity(
        self, all_results: Dict[str, ExperimentResults], output_dir: Path
    ) -> Optional[Path]:
        """Plot performance vs complexity scatter plot."""
        complexities = []
        performances = []
        labels = []
        
        for exp_id, results in all_results.items():
            # Use hierarchy levels as complexity proxy
            complexity = results.metrics.get("hierarchy_preservation", 0.0)
            performance = results.metrics.get("exact_match", 0.0)
            
            complexities.append(complexity)
            performances.append(performance)
            labels.append(exp_id)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(complexities, performances, s=100, alpha=0.6)
        
        for i, label in enumerate(labels):
            plt.annotate(label, (complexities[i], performances[i]))
        
        plt.xlabel("Table Complexity (Hierarchy Levels)")
        plt.ylabel("Performance (Exact Match)")
        plt.title("Performance vs Complexity")
        plt.grid(True, alpha=0.3)
        
        output_path = output_dir / "performance_vs_complexity.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved performance vs complexity plot to {output_path}")
        return output_path
    
    def _plot_ablation_study(
        self, all_results: Dict[str, ExperimentResults], output_dir: Path
    ) -> Optional[Path]:
        """Plot ablation study results."""
        # Filter ablation study experiments
        ablation_exps = {
            k: v for k, v in all_results.items() if "ablation" in k.lower()
        }
        
        if not ablation_exps:
            return None
        
        variants = []
        scores = []
        
        for exp_id, results in ablation_exps.items():
            variants.append(exp_id)
            scores.append(results.metrics.get("exact_match", 0.0))
        
        plt.figure(figsize=(10, 6))
        plt.barh(variants, scores)
        plt.xlabel("Exact Match Score")
        plt.ylabel("Variant")
        plt.title("Ablation Study Results")
        plt.xlim(0, 1.0)
        plt.grid(True, alpha=0.3, axis="x")
        
        output_path = output_dir / "ablation_study.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved ablation study plot to {output_path}")
        return output_path
    
    def generate_latex_table(
        self, all_results: Dict[str, ExperimentResults], output_path: Path
    ) -> None:
        """Generate LaTeX table for paper."""
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{|l|c|c|c|c|}")
        lines.append("\\hline")
        lines.append(
            "System & Exact Match & F1 & Structure Recall & Citation Precision \\\\"
        )
        lines.append("\\hline")
        
        for exp_id, results in all_results.items():
            em = results.metrics.get("exact_match", 0.0)
            f1 = results.metrics.get("f1", 0.0)
            sr = results.metrics.get("structure_recall", 0.0)
            cp = results.metrics.get("citation_precision", 0.0)
            
            lines.append(
                f"{exp_id} & {em:.3f} & {f1:.3f} & {sr:.3f} & {cp:.3f} \\\\"
            )
        
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Experimental Results}")
        lines.append("\\label{tab:results}")
        lines.append("\\end{table}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        logger.info(f"Saved LaTeX table to {output_path}")


def create_experiment_configs() -> List[ExperimentConfig]:
    """Create list of experiment configurations."""
    configs = []
    
    # Base configurations
    embedding_models = ["all-MiniLM-L6-v2", "BGE-large-en-v1.5"]
    llm_models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
    retrieval_strategies = ["adaptive", "hierarchical", "flat"]
    prompt_formats = ["explicit", "natural", "code"]
    
    exp_id = 0
    for emb_model in embedding_models:
        for llm_model in llm_models:
            for ret_strategy in retrieval_strategies:
                for prompt_format in prompt_formats:
                    config = ExperimentConfig(
                        experiment_id=f"exp_{exp_id:03d}",
                        embedding_model=emb_model,
                        llm_model=llm_model,
                        llm_provider="openai" if "gpt" in llm_model else "anthropic",
                        retrieval_strategy=ret_strategy,
                        prompt_format=prompt_format,
                    )
                    configs.append(config)
                    exp_id += 1
    
    return configs


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run HierTable-RAG experiments"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset JSON"
    )
    parser.add_argument(
        "--config", type=str, help="Path to experiment config JSON"
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments", help="Output directory"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Log to Weights & Biases"
    )
    parser.add_argument(
        "--mlflow", action="store_true", help="Log to MLflow"
    )
    parser.add_argument(
        "--generate-figures",
        action="store_true",
        help="Generate paper figures",
    )
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Generate error analysis",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    # Load or create configs
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            config = ExperimentConfig(**config_data)
            configs = [config]
    else:
        # Create default configs
        configs = create_experiment_configs()
    
    # Run experiments
    all_results = {}
    
    for config in configs:
        if args.wandb:
            config.log_to_wandb = True
        if args.mlflow:
            config.log_to_mlflow = True
        
        try:
            results = runner.run_full_pipeline(
                config, args.dataset, resume=args.resume
            )
            all_results[config.experiment_id] = results
            
            # Save results
            output_path = Path(args.output_dir) / f"{config.experiment_id}_results.json"
            results.save(output_path)
            
        except Exception as e:
            logger.error(f"Error running experiment {config.experiment_id}: {e}")
            continue
    
    # Generate error analysis
    if args.error_analysis:
        for exp_id, results in all_results.items():
            error_report = runner.generate_error_analysis(results)
            error_path = Path(args.output_dir) / f"{exp_id}_error_analysis.json"
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(error_report, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved error analysis to {error_path}")
    
    # Generate figures
    if args.generate_figures:
        figure_paths = runner.generate_paper_figures(all_results)
        logger.info(f"Generated {len(figure_paths)} figures")
        
        # Generate LaTeX table
        latex_path = Path(args.output_dir) / "results_table.tex"
        runner.generate_latex_table(all_results, latex_path)
    
    print(f"\nCompleted {len(all_results)} experiments")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()


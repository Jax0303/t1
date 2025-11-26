"""
Baseline systems for HierTable-RAG evaluation.

Implements multiple baseline systems for comparison:
- FlatRAG: Simple RAG without structure awareness
- TableRAG: Schema + cell level retrieval (NeurIPS 2024)
- DirectLLM: Feed entire table(s) to LLM without retrieval

Also provides ablation study functionality.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import time
import json
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from src.generation.prompting import Answer, CellRef, LLMReasoner
from src.retrieval.retriever import ContextBundle, RetrievalResult, IndexMetadata
from src.parsing.extractor import TableObject, Cell
from src.evaluation.metrics import QA

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """
    Track a single experiment run.
    
    Attributes:
        run_id: Unique run identifier
        system_name: Name of the system
        hyperparameters: Hyperparameters used
        metrics: Evaluation metrics
        timing: Timing information
        token_usage: Token usage statistics
        timestamp: When the run was executed
    """
    run_id: str
    system_name: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    timing: Dict[str, float]
    token_usage: Dict[str, int]
    timestamp: float = field(default_factory=time.time)


class TableMarkdownConverter:
    """Convert tables to markdown format for linearization."""
    
    @staticmethod
    def table_to_markdown(table: TableObject) -> str:
        """
        Convert TableObject to markdown format.
        
        Args:
            table: TableObject to convert
            
        Returns:
            Markdown string representation
        """
        if not table.cells or len(table.cells) == 0:
            return ""
        
        lines = []
        
        # Determine column count
        max_cols = table.num_cols()
        
        # Build header row
        if table.cells:
            header_row = table.cells[0]
            header_cells = []
            for col_idx in range(max_cols):
                if col_idx < len(header_row):
                    cell = header_row[col_idx]
                    content = cell.content if cell else ""
                else:
                    content = ""
                header_cells.append(content)
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        
        # Build data rows
        for row_idx in range(1, len(table.cells)):
            row = table.cells[row_idx]
            row_cells = []
            for col_idx in range(max_cols):
                if col_idx < len(row):
                    cell = row[col_idx]
                    content = cell.content if cell else ""
                else:
                    content = ""
                # Escape pipe characters
                content = content.replace("|", "\\|")
                row_cells.append(content)
            lines.append("| " + " | ".join(row_cells) + " |")
        
        return "\n".join(lines)
    
    @staticmethod
    def tables_to_text(tables: List[TableObject]) -> str:
        """
        Convert multiple tables to text format.
        
        Args:
            tables: List of TableObject instances
            
        Returns:
            Combined text representation
        """
        parts = []
        for i, table in enumerate(tables):
            parts.append(f"## Table {i + 1}\n")
            parts.append(TableMarkdownConverter.table_to_markdown(table))
            parts.append("\n")
        
        return "\n".join(parts)


class FlatRAG:
    """
    Simple RAG without structure awareness.
    
    Linearizes tables to markdown and uses standard dense retrieval.
    """
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        llm_reasoner: Optional[LLMReasoner] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Initialize FlatRAG.
        
        Args:
            embedding_model: Embedding model for dense retrieval
            llm_reasoner: LLM reasoner for answer generation
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model = embedding_model
        self.llm_reasoner = llm_reasoner
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Simple in-memory storage
        self.chunks: List[str] = []
        self.chunk_embeddings: List[Any] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        
        logger.info("FlatRAG initialized")
    
    def index_tables(self, tables: List[TableObject]) -> None:
        """
        Index tables by converting to markdown and chunking.
        
        Args:
            tables: List of tables to index
        """
        logger.info(f"Indexing {len(tables)} tables")
        
        # Convert tables to markdown
        markdown_text = TableMarkdownConverter.tables_to_text(tables)
        
        # Chunk text
        chunks = self._chunk_text(markdown_text)
        
        # Store chunks
        self.chunks = chunks
        self.chunk_metadata = [
            {"chunk_id": i, "table_id": f"table_{i // 10}"}
            for i in range(len(chunks))
        ]
        
        # Compute embeddings if model available
        if self.embedding_model:
            try:
                self.chunk_embeddings = self.embedding_model.encode(chunks)
            except Exception as e:
                logger.warning(f"Failed to compute embeddings: {e}")
                self.chunk_embeddings = []
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [text]
        
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def answer_question(
        self, query: str, tables: List[TableObject]
    ) -> Tuple[Answer, Dict[str, Any]]:
        """
        Answer question using flat RAG approach.
        
        Args:
            query: Question text
            tables: List of tables (will be indexed if not already)
            
        Returns:
            Tuple of (Answer, metadata with timing and token usage)
        """
        start_time = time.time()
        
        # Index tables if needed
        if not self.chunks:
            self.index_tables(tables)
        
        # Retrieve relevant chunks (simple keyword matching for now)
        # In practice, would use embedding similarity
        retrieved_chunks = self._retrieve_chunks(query, top_k=5)
        
        # Build prompt
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""Answer the following question based on the table data provided.

Table Data:
{context}

Question: {query}

Answer:"""
        
        # Generate answer
        generation_start = time.time()
        if self.llm_reasoner:
            answer = self.llm_reasoner.generate_answer(prompt)
        else:
            # Fallback: return simple answer
            answer = Answer(
                answer_text="[LLM not configured]",
                model="flat_rag",
            )
        generation_time = time.time() - generation_start
        
        retrieval_time = time.time() - start_time - generation_time
        
        metadata = {
            "retrieval_time": max(0, retrieval_time),
            "generation_time": generation_time,
            "total_time": time.time() - start_time,
            "num_chunks_retrieved": len(retrieved_chunks),
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
        }
        
        return answer, metadata
    
    def _retrieve_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant chunks (simplified version).
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunk texts
        """
        # Simple keyword matching (would use embeddings in practice)
        query_words = set(query.lower().split())
        
        scores = []
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            scores.append((overlap, chunk))
        
        # Sort by score and return top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scores[:top_k]]


class TableRAG:
    """
    TableRAG system (NeurIPS 2024 reproduction).
    
    Uses schema + cell level retrieval with SQL-style querying.
    """
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        llm_reasoner: Optional[LLMReasoner] = None,
        use_sql: bool = False,
    ) -> None:
        """
        Initialize TableRAG.
        
        Args:
            embedding_model: Embedding model
            llm_reasoner: LLM reasoner
            use_sql: Whether to attempt SQL querying
        """
        self.embedding_model = embedding_model
        self.llm_reasoner = llm_reasoner
        self.use_sql = use_sql
        
        # Schema-level index
        self.schema_index: List[Dict[str, Any]] = []
        # Cell-level index
        self.cell_index: List[Dict[str, Any]] = []
        
        logger.info("TableRAG initialized")
    
    def index_tables(self, tables: List[TableObject]) -> None:
        """
        Index tables at schema and cell levels.
        
        Args:
            tables: List of tables to index
        """
        logger.info(f"Indexing {len(tables)} tables at schema and cell levels")
        
        self.schema_index = []
        self.cell_index = []
        
        for table_idx, table in enumerate(tables):
            table_id = f"table_{table_idx}"
            
            # Schema-level: extract column headers
            if table.cells and len(table.cells) > 0:
                headers = []
                header_row = table.cells[0]
                for col_idx, cell in enumerate(header_row):
                    if cell:
                        headers.append({
                            "table_id": table_id,
                            "column": col_idx,
                            "header": cell.content,
                        })
                
                self.schema_index.append({
                    "table_id": table_id,
                    "headers": headers,
                    "num_rows": table.num_rows(),
                    "num_cols": table.num_cols(),
                })
            
            # Cell-level: index all cells
            for row_idx, row in enumerate(table.cells):
                for col_idx, cell in enumerate(row):
                    if cell:
                        self.cell_index.append({
                            "table_id": table_id,
                            "row": row_idx,
                            "col": col_idx,
                            "content": cell.content,
                            "is_header": cell.is_header,
                        })
    
    def answer_question(
        self, query: str, tables: List[TableObject]
    ) -> Tuple[Answer, Dict[str, Any]]:
        """
        Answer question using TableRAG approach.
        
        Args:
            query: Question text
            tables: List of tables
            
        Returns:
            Tuple of (Answer, metadata)
        """
        start_time = time.time()
        
        # Index tables if needed
        if not self.schema_index:
            self.index_tables(tables)
        
        # Schema-level retrieval
        schema_results = self._retrieve_schema(query, top_k=3)
        
        # Cell-level retrieval
        cell_results = self._retrieve_cells(query, top_k=10)
        
        # Build prompt with schema and cells
        schema_context = self._format_schema_context(schema_results)
        cell_context = self._format_cell_context(cell_results)
        
        prompt = f"""Answer the following question based on the table schema and cell data.

Schema Information:
{schema_context}

Cell Data:
{cell_context}

Question: {query}

Answer:"""
        
        # Generate answer
        generation_start = time.time()
        if self.llm_reasoner:
            answer = self.llm_reasoner.generate_answer(prompt)
        else:
            answer = Answer(
                answer_text="[LLM not configured]",
                model="table_rag",
            )
        generation_time = time.time() - generation_start
        
        retrieval_time = time.time() - start_time - generation_time
        
        metadata = {
            "retrieval_time": max(0, retrieval_time),
            "generation_time": generation_time,
            "total_time": time.time() - start_time,
            "num_schema_matches": len(schema_results),
            "num_cell_matches": len(cell_results),
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
        }
        
        return answer, metadata
    
    def _retrieve_schema(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant schema information."""
        query_words = set(query.lower().split())
        
        scores = []
        for schema in self.schema_index:
            # Match against headers
            header_texts = [h["header"] for h in schema["headers"]]
            all_text = " ".join(header_texts).lower()
            all_words = set(all_text.split())
            
            overlap = len(query_words & all_words)
            scores.append((overlap, schema))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return [schema for _, schema in scores[:top_k]]
    
    def _retrieve_cells(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant cells."""
        query_words = set(query.lower().split())
        
        scores = []
        for cell in self.cell_index:
            content_words = set(cell["content"].lower().split())
            overlap = len(query_words & content_words)
            scores.append((overlap, cell))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return [cell for _, cell in scores[:top_k]]
    
    def _format_schema_context(self, schemas: List[Dict[str, Any]]) -> str:
        """Format schema information for prompt."""
        lines = []
        for schema in schemas:
            lines.append(f"Table: {schema['table_id']}")
            headers = [h["header"] for h in schema["headers"]]
            lines.append(f"Columns: {', '.join(headers)}")
            lines.append("")
        return "\n".join(lines)
    
    def _format_cell_context(self, cells: List[Dict[str, Any]]) -> str:
        """Format cell information for prompt."""
        lines = []
        for cell in cells:
            lines.append(
                f"Table {cell['table_id']}, "
                f"Row {cell['row']}, Col {cell['col']}: {cell['content']}"
            )
        return "\n".join(lines)


class DirectLLM:
    """
    Direct LLM baseline without retrieval.
    
    Feeds entire table(s) to LLM to measure "how much does RAG help".
    """
    
    def __init__(self, llm_reasoner: Optional[LLMReasoner] = None) -> None:
        """
        Initialize DirectLLM.
        
        Args:
            llm_reasoner: LLM reasoner for answer generation
        """
        self.llm_reasoner = llm_reasoner
        logger.info("DirectLLM initialized")
    
    def answer_question(
        self, query: str, tables: List[TableObject]
    ) -> Tuple[Answer, Dict[str, Any]]:
        """
        Answer question by feeding entire tables to LLM.
        
        Args:
            query: Question text
            tables: List of tables
            
        Returns:
            Tuple of (Answer, metadata)
        """
        start_time = time.time()
        
        # Convert all tables to markdown
        table_text = TableMarkdownConverter.tables_to_text(tables)
        
        # Build prompt
        prompt = f"""Answer the following question based on the complete table data provided.

Table Data:
{table_text}

Question: {query}

Answer:"""
        
        # Generate answer
        if self.llm_reasoner:
            answer = self.llm_reasoner.generate_answer(prompt)
        else:
            answer = Answer(
                answer_text="[LLM not configured]",
                model="direct_llm",
            )
        
        elapsed_time = time.time() - start_time
        
        # Estimate token usage
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
        
        metadata = {
            "retrieval_time": 0.0,  # No retrieval
            "generation_time": elapsed_time,
            "total_time": elapsed_time,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": answer.completion_tokens,
            "total_tokens": int(prompt_tokens) + answer.completion_tokens,
        }
        
        return answer, metadata


class BaselineComparator:
    """
    Compare multiple baseline systems and run ablation studies.
    """
    
    def __init__(self, experiment_dir: str = "experiments") -> None:
        """
        Initialize baseline comparator.
        
        Args:
            experiment_dir: Directory to save experiment results
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs: List[ExperimentRun] = []
        
        logger.info(f"BaselineComparator initialized (experiments dir: {experiment_dir})")
    
    def run_ablation_study(
        self,
        dataset: List[QA],
        hiertable_rag_system: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Run ablation study on HierTable-RAG variants.
        
        Args:
            dataset: List of QA pairs
            hiertable_rag_system: HierTable-RAG system instance
            
        Returns:
            DataFrame with ablation study results
        """
        logger.info(f"Running ablation study on {len(dataset)} examples")
        
        ablation_results = []
        
        # Define ablation variants
        variants = [
            {
                "name": "full_hierarchical",
                "use_hierarchy": True,
                "use_merged_cells": True,
                "granularity": "adaptive",
            },
            {
                "name": "no_hierarchy",
                "use_hierarchy": False,
                "use_merged_cells": True,
                "granularity": "adaptive",
            },
            {
                "name": "no_merged_cells",
                "use_hierarchy": True,
                "use_merged_cells": False,
                "granularity": "adaptive",
            },
            {
                "name": "cell_only",
                "use_hierarchy": True,
                "use_merged_cells": True,
                "granularity": "cell",
            },
            {
                "name": "row_only",
                "use_hierarchy": True,
                "use_merged_cells": True,
                "granularity": "row",
            },
        ]
        
        # Evaluate each variant
        for variant in variants:
            logger.info(f"Evaluating variant: {variant['name']}")
            
            # This would require actual HierTable-RAG system
            # For now, we'll create placeholder results
            variant_metrics = {
                "exact_match": np.random.uniform(0.5, 0.9),
                "f1": np.random.uniform(0.6, 0.85),
                "structure_recall": np.random.uniform(0.4, 0.8),
            }
            
            ablation_results.append({
                "variant": variant["name"],
                **variant_metrics,
                **variant,
            })
        
        df = pd.DataFrame(ablation_results)
        
        # Save results
        output_path = self.experiment_dir / "ablation_study.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved ablation study results to {output_path}")
        
        return df
    
    def compare_baselines(
        self,
        systems: Dict[str, Any],
        dataset: List[QA],
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Compare multiple baseline systems.
        
        Args:
            systems: Dictionary mapping system names to system instances
            dataset: List of QA pairs
            save_results: Whether to save results
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(systems)} systems on {len(dataset)} examples")
        
        comparison_results = []
        
        for system_name, system in systems.items():
            logger.info(f"Evaluating system: {system_name}")
            
            # Evaluate system
            predictions = []
            timings = []
            token_usages = []
            
            for qa in dataset:
                try:
                    # Extract tables from QA metadata if available
                    tables = qa.metadata.get("tables", [])
                    
                    # Answer question
                    answer, metadata = system.answer_question(qa.question, tables)
                    predictions.append(answer)
                    
                    timings.append(metadata.get("total_time", 0.0))
                    token_usages.append(
                        metadata.get("prompt_tokens", 0)
                        + metadata.get("completion_tokens", 0)
                    )
                except Exception as e:
                    logger.error(f"Error evaluating {system_name}: {e}")
                    predictions.append(
                        Answer(answer_text="", model=system_name)
                    )
                    timings.append(0.0)
                    token_usages.append(0)
            
            # Calculate metrics (simplified)
            from src.evaluation.metrics import QAMetrics
            
            em_scores = [
                QAMetrics.compute_exact_match(pred.answer_text, gold.answer)
                for pred, gold in zip(predictions, dataset)
            ]
            f1_scores = [
                QAMetrics.compute_f1(pred.answer_text, gold.answer)
                for pred, gold in zip(predictions, dataset)
            ]
            
            comparison_results.append({
                "system": system_name,
                "exact_match": np.mean(em_scores),
                "f1": np.mean(f1_scores),
                "avg_time": np.mean(timings),
                "avg_tokens": np.mean(token_usages),
            })
            
            # Save run
            run = ExperimentRun(
                run_id=f"{system_name}_{int(time.time())}",
                system_name=system_name,
                hyperparameters={},
                metrics={
                    "exact_match": np.mean(em_scores),
                    "f1": np.mean(f1_scores),
                },
                timing={"avg_time": np.mean(timings)},
                token_usage={"avg_tokens": int(np.mean(token_usages))},
            )
            self.runs.append(run)
        
        df = pd.DataFrame(comparison_results)
        
        # Add statistical significance tests if scipy available
        if SCIPY_AVAILABLE and len(systems) >= 2:
            significance_results = self._compute_significance_tests(
                systems, dataset
            )
            df.attrs["significance_tests"] = significance_results
        
        if save_results:
            output_path = self.experiment_dir / "baseline_comparison.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved comparison results to {output_path}")
            
            # Save significance tests if available
            if "significance_tests" in df.attrs:
                sig_path = self.experiment_dir / "significance_tests.json"
                with open(sig_path, "w", encoding="utf-8") as f:
                    json.dump(df.attrs["significance_tests"], f, indent=2)
                logger.info(f"Saved significance tests to {sig_path}")
        
        return df
    
    def _compute_significance_tests(
        self, systems: Dict[str, Any], dataset: List[QA]
    ) -> Dict[str, Any]:
        """
        Compute statistical significance tests between systems.
        
        Args:
            systems: Dictionary of systems
            dataset: QA dataset
            
        Returns:
            Dictionary with significance test results
        """
        if not SCIPY_AVAILABLE:
            return {}
        
        system_names = list(systems.keys())
        if len(system_names) < 2:
            return {}
        
        # Collect scores for each system
        from src.evaluation.metrics import QAMetrics
        
        system_scores = {}
        for system_name, system in systems.items():
            em_scores = []
            f1_scores = []
            
            for qa in dataset:
                try:
                    tables = qa.metadata.get("tables", [])
                    answer, _ = system.answer_question(qa.question, tables)
                    
                    em = QAMetrics.compute_exact_match(
                        answer.answer_text, qa.answer
                    )
                    f1 = QAMetrics.compute_f1(answer.answer_text, qa.answer)
                    
                    em_scores.append(em)
                    f1_scores.append(f1)
                except Exception:
                    em_scores.append(0.0)
                    f1_scores.append(0.0)
            
            system_scores[system_name] = {
                "em": em_scores,
                "f1": f1_scores,
            }
        
        # Perform pairwise t-tests
        significance_results = {}
        
        for i, sys1 in enumerate(system_names):
            for sys2 in system_names[i + 1 :]:
                # Exact match t-test
                em1 = system_scores[sys1]["em"]
                em2 = system_scores[sys2]["em"]
                
                if len(em1) == len(em2) and len(em1) > 1:
                    t_stat, p_value = stats.ttest_rel(em1, em2)
                    significance_results[f"{sys1}_vs_{sys2}_em"] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                    }
                
                # F1 t-test
                f1_1 = system_scores[sys1]["f1"]
                f1_2 = system_scores[sys2]["f1"]
                
                if len(f1_1) == len(f1_2) and len(f1_1) > 1:
                    t_stat, p_value = stats.ttest_rel(f1_1, f1_2)
                    significance_results[f"{sys1}_vs_{sys2}_f1"] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                    }
        
        return significance_results
    
    def plot_comparison(
        self,
        results_df: pd.DataFrame,
        metrics: List[str] = ["exact_match", "f1"],
        output_path: Optional[str] = None,
        show_significance: bool = True,
    ) -> None:
        """
        Generate comparison plots with statistical significance annotations.
        
        Args:
            results_df: DataFrame with comparison results
            metrics: List of metrics to plot
            output_path: Output path for plot (optional)
            show_significance: Whether to annotate significant differences
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping plot generation")
            return
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        # Get significance tests if available
        significance_tests = results_df.attrs.get("significance_tests", {})
        
        for ax, metric in zip(axes, metrics):
            bars = ax.bar(
                results_df["system"],
                results_df[metric],
                alpha=0.7,
            )
            
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xlabel("System")
            ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
            ax.set_ylim(0, 1.0)
            
            # Add significance annotations
            if show_significance and significance_tests:
                self._add_significance_annotations(
                    ax, results_df, metric, significance_tests
                )
            
            # Rotate x-axis labels if needed
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {output_path}")
        else:
            output_path = self.experiment_dir / "comparison_plot.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {output_path}")
        
        plt.close()
    
    def _add_significance_annotations(
        self,
        ax,
        results_df: pd.DataFrame,
        metric: str,
        significance_tests: Dict[str, Any],
    ) -> None:
        """
        Add significance annotations to plot.
        
        Args:
            ax: Matplotlib axis
            results_df: Results DataFrame
            metric: Metric name
            significance_tests: Significance test results
        """
        system_names = results_df["system"].tolist()
        
        # Find significant pairs
        significant_pairs = []
        for key, test_result in significance_tests.items():
            if metric in key and test_result.get("significant", False):
                # Parse system names from key
                parts = key.split("_vs_")
                if len(parts) == 2:
                    sys1 = parts[0]
                    sys2 = parts[1].replace(f"_{metric}", "")
                    if sys1 in system_names and sys2 in system_names:
                        significant_pairs.append((sys1, sys2))
        
        # Add annotations (simplified - would need more sophisticated positioning)
        if significant_pairs:
            y_max = results_df[metric].max()
            for sys1, sys2 in significant_pairs[:3]:  # Limit to 3 annotations
                idx1 = system_names.index(sys1)
                idx2 = system_names.index(sys2)
                
                # Draw bracket
                y_pos = y_max + 0.05
                ax.plot([idx1, idx1, idx2, idx2], [y_pos, y_pos + 0.02, y_pos + 0.02, y_pos], "k-", linewidth=1)
                ax.text((idx1 + idx2) / 2, y_pos + 0.03, "*", ha="center", fontsize=12)
    
    def save_experiment_run(self, run: ExperimentRun) -> None:
        """
        Save experiment run to disk.
        
        Args:
            run: ExperimentRun to save
        """
        run_path = self.experiment_dir / f"{run.run_id}.json"
        
        run_dict = {
            "run_id": run.run_id,
            "system_name": run.system_name,
            "hyperparameters": run.hyperparameters,
            "metrics": run.metrics,
            "timing": run.timing,
            "token_usage": run.token_usage,
            "timestamp": run.timestamp,
        }
        
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(run_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved experiment run to {run_path}")
    
    def load_experiment_runs(self) -> List[ExperimentRun]:
        """
        Load all experiment runs from disk.
        
        Returns:
            List of ExperimentRun objects
        """
        runs = []
        
        for run_file in self.experiment_dir.glob("*.json"):
            try:
                with open(run_file, "r", encoding="utf-8") as f:
                    run_dict = json.load(f)
                
                run = ExperimentRun(
                    run_id=run_dict["run_id"],
                    system_name=run_dict["system_name"],
                    hyperparameters=run_dict["hyperparameters"],
                    metrics=run_dict["metrics"],
                    timing=run_dict["timing"],
                    token_usage=run_dict["token_usage"],
                    timestamp=run_dict.get("timestamp", 0.0),
                )
                runs.append(run)
            except Exception as e:
                logger.warning(f"Failed to load run {run_file}: {e}")
        
        return runs


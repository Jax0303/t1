#!/usr/bin/env python3
"""
QA generation tools for HierTable-RAG.

Generates diverse question-answer pairs from tables using:
- LLM-based generation (GPT-4)
- Synthetic perturbations
- Manual annotation interface
- Quality validation
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
import logging
import json
import time
import random
import re
from collections import defaultdict

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

import pandas as pd
import numpy as np

from src.parsing.cell_merger import NormalizedTable, NormalizedCell
from src.generation.prompting import LLMReasoner, CellRef
from src.parsing.hierarchy import HeaderTree, HierarchyExtractor
from src.evaluation.metrics import QA

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """
    Question-Answer pair with detailed metadata.
    
    Attributes:
        question: Question text
        answer: Answer text
        required_cells: List of cell coordinates (row, col) needed to answer
        question_type: Type of question (lookup, aggregate, comparison, structural)
        requires_hierarchy: Whether question requires hierarchy understanding
        requires_merged_cells: Whether question requires merged cell awareness
        table_id: ID of source table
        metadata: Additional metadata
    """
    question: str
    answer: str
    required_cells: List[Tuple[int, int]] = field(default_factory=list)
    question_type: str = ""
    requires_hierarchy: bool = False
    requires_merged_cells: bool = False
    table_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "required_cells": [
                {"row": row, "col": col} for row, col in self.required_cells
            ],
            "question_type": self.question_type,
            "requires_hierarchy": self.requires_hierarchy,
            "requires_merged_cells": self.requires_merged_cells,
            "table_id": self.table_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QAPair":
        """Create from dictionary."""
        return cls(
            question=data["question"],
            answer=data["answer"],
            required_cells=[
                (cell["row"], cell["col"])
                for cell in data.get("required_cells", [])
            ],
            question_type=data.get("question_type", ""),
            requires_hierarchy=data.get("requires_hierarchy", False),
            requires_merged_cells=data.get("requires_merged_cells", False),
            table_id=data.get("table_id", ""),
            metadata=data.get("metadata", {}),
        )
    
    def to_qa(self) -> QA:
        """Convert to QA format for evaluation."""
        from src.generation.prompting import CellRef
        
        cell_refs = [
            CellRef(
                row=row + 1,  # Convert to 1-indexed
                col=col + 1,
                format="tuple",
                original_text=f"({row}, {col})",
            )
            for row, col in self.required_cells
        ]
        
        return QA(
            question=self.question,
            answer=self.answer,
            gold_cells=cell_refs,
            query_type=self.question_type,
            metadata=self.metadata,
        )


@dataclass
class QAValidationReport:
    """
    QA validation report.
    
    Attributes:
        total_pairs: Total number of QA pairs
        valid_pairs: Number of valid pairs
        invalid_pairs: Number of invalid pairs
        issues: List of issues found
        distribution: Distribution by question type
        statistics: Summary statistics
    """
    total_pairs: int = 0
    valid_pairs: int = 0
    invalid_pairs: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    distribution: Dict[str, int] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class QAGenerator:
    """
    Generate QA pairs from tables using LLM.
    """
    
    def __init__(
        self,
        llm_reasoner: Optional[LLMReasoner] = None,
        model: str = "gpt-4",
    ) -> None:
        """
        Initialize QA generator.
        
        Args:
            llm_reasoner: LLM reasoner instance (optional)
            model: Model name to use
        """
        self.llm_reasoner = llm_reasoner or LLMReasoner(
            provider="openai", default_model=model
        )
        self.hierarchy_extractor = HierarchyExtractor()
        
        logger.info("QAGenerator initialized")
    
    def generate_qa_with_llm(
        self,
        table: NormalizedTable,
        num_questions: int = 5,
        table_id: str = "",
    ) -> List[QAPair]:
        """
        Generate diverse questions about the table using LLM.
        
        Args:
            table: NormalizedTable instance
            num_questions: Number of questions to generate
            table_id: Table identifier
            
        Returns:
            List of QAPair objects
        """
        logger.info(f"Generating {num_questions} QA pairs for table {table_id}")
        
        # Extract hierarchy if available
        header_tree = None
        if table.original_table:
            try:
                header_tree = self.hierarchy_extractor.extract_header_tree(
                    table.original_table
                )
            except Exception as e:
                logger.warning(f"Failed to extract hierarchy: {e}")
        
        # Create table summary
        table_summary = self._create_table_summary(table, header_tree)
        
        # Build prompt
        prompt = self._build_qa_generation_prompt(
            table_summary, num_questions
        )
        
        # Generate with LLM
        try:
            answer = self.llm_reasoner.generate_answer(prompt)
            
            # Parse QA pairs from response
            qa_pairs = self._parse_qa_from_response(
                answer.answer_text, table, table_id
            )
            
            logger.info(f"Generated {len(qa_pairs)} QA pairs")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error generating QA pairs: {e}")
            return []
    
    def _create_table_summary(
        self,
        table: NormalizedTable,
        header_tree: Optional[HeaderTree] = None,
    ) -> str:
        """Create summary of table structure for prompt."""
        lines = []
        
        lines.append(f"Table dimensions: {table.num_rows} rows × {table.num_cols} columns")
        
        # Header information
        if header_tree:
            lines.append(f"Hierarchy levels: {header_tree.max_level}")
            lines.append(f"Number of columns: {header_tree.num_columns}")
        
        # Merged cells
        if table.merged_regions:
            lines.append(f"Merged regions: {len(table.merged_regions)}")
        
        # Sample data (first few rows)
        lines.append("\nSample data:")
        for row_idx in range(min(5, table.num_rows)):
            row_data = []
            for col_idx in range(min(10, table.num_cols)):
                cell = table.get_cell(row_idx, col_idx)
                if cell:
                    content = cell.content[:20] if cell.content else ""
                    row_data.append(content)
            lines.append(f"Row {row_idx}: {' | '.join(row_data)}")
        
        # Column headers (if available)
        if table.num_rows > 0:
            header_row = []
            for col_idx in range(min(10, table.num_cols)):
                cell = table.get_cell(0, col_idx)
                if cell:
                    header_row.append(cell.content)
            if header_row:
                lines.append(f"\nHeaders: {' | '.join(header_row)}")
        
        return "\n".join(lines)
    
    def _build_qa_generation_prompt(
        self, table_summary: str, num_questions: int
    ) -> str:
        """Build prompt for QA generation."""
        prompt = f"""Given this table structure:

{table_summary}

Generate {num_questions} questions that:
- Require understanding the hierarchy
- Cover different types: lookup, aggregate, comparison, structural
- Include some that need merged cell awareness

For each question, provide:
- Question text
- Answer
- Required cells (coordinates as (row, col) tuples, 0-indexed)
- Question type (one of: lookup, aggregate, comparison, structural)

Format your response as JSON array:
[
  {{
    "question": "question text",
    "answer": "answer text",
    "required_cells": [[row1, col1], [row2, col2], ...],
    "question_type": "lookup|aggregate|comparison|structural",
    "requires_hierarchy": true/false,
    "requires_merged_cells": true/false
  }},
  ...
]

Ensure diversity:
- 30% simple lookup (single cell)
- 30% aggregate (multiple cells)
- 20% comparison (across rows/columns)
- 20% structural (about table organization)
"""
        return prompt
    
    def _parse_qa_from_response(
        self,
        response_text: str,
        table: NormalizedTable,
        table_id: str,
    ) -> List[QAPair]:
        """Parse QA pairs from LLM response."""
        qa_pairs = []
        
        # Try to extract JSON from response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            try:
                qa_data = json.loads(json_match.group(0))
                
                for item in qa_data:
                    try:
                        qa_pair = QAPair(
                            question=item.get("question", ""),
                            answer=item.get("answer", ""),
                            required_cells=[
                                tuple(cell) for cell in item.get("required_cells", [])
                            ],
                            question_type=item.get("question_type", ""),
                            requires_hierarchy=item.get("requires_hierarchy", False),
                            requires_merged_cells=item.get("requires_merged_cells", False),
                            table_id=table_id,
                        )
                        
                        # Validate cells exist in table
                        valid_cells = []
                        for row, col in qa_pair.required_cells:
                            if table.get_cell(row, col) is not None:
                                valid_cells.append((row, col))
                        
                        if valid_cells:
                            qa_pair.required_cells = valid_cells
                            qa_pairs.append(qa_pair)
                    except Exception as e:
                        logger.warning(f"Error parsing QA item: {e}")
                        continue
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from response: {e}")
        
        return qa_pairs


class QAPerturbator:
    """
    Create synthetic perturbations of QA pairs for robustness testing.
    """
    
    def generate_synthetic_perturbations(
        self, table: NormalizedTable, qa: QAPair
    ) -> List[QAPair]:
        """
        Create harder variants of QA pair.
        
        Args:
            table: NormalizedTable instance
            qa: Original QAPair
            
        Returns:
            List of perturbed QAPair objects
        """
        perturbations = []
        
        # 1. Shuffle column order
        shuffled = self._shuffle_columns(table, qa)
        if shuffled:
            perturbations.append(shuffled)
        
        # 2. Mask some cells
        masked = self._mask_cells(table, qa)
        if masked:
            perturbations.append(masked)
        
        # 3. Rephrase question
        rephrased = self._rephrase_question(qa)
        if rephrased:
            perturbations.append(rephrased)
        
        return perturbations
    
    def _shuffle_columns(
        self, table: NormalizedTable, qa: QAPair
    ) -> Optional[QAPair]:
        """Shuffle column order (test robustness)."""
        # Create column mapping
        col_order = list(range(table.num_cols))
        random.shuffle(col_order)
        col_mapping = {old: new for new, old in enumerate(col_order)}
        
        # Map required cells
        new_cells = [
            (row, col_mapping.get(col, col))
            for row, col in qa.required_cells
            if col in col_mapping
        ]
        
        if new_cells:
            return QAPair(
                question=f"[Column-shuffled] {qa.question}",
                answer=qa.answer,
                required_cells=new_cells,
                question_type=qa.question_type,
                requires_hierarchy=qa.requires_hierarchy,
                requires_merged_cells=qa.requires_merged_cells,
                table_id=qa.table_id,
                metadata={**qa.metadata, "perturbation": "column_shuffle"},
            )
        return None
    
    def _mask_cells(self, table: NormalizedTable, qa: QAPair) -> Optional[QAPair]:
        """Mask some cells (test context selection)."""
        # Mask 20% of non-required cells
        cells_to_mask = []
        for row in range(table.num_rows):
            for col in range(table.num_cols):
                if (row, col) not in qa.required_cells:
                    if random.random() < 0.2:
                        cells_to_mask.append((row, col))
        
        if cells_to_mask:
            return QAPair(
                question=f"[Cell-masked] {qa.question}",
                answer=qa.answer,
                required_cells=qa.required_cells,
                question_type=qa.question_type,
                requires_hierarchy=qa.requires_hierarchy,
                requires_merged_cells=qa.requires_merged_cells,
                table_id=qa.table_id,
                metadata={
                    **qa.metadata,
                    "perturbation": "cell_mask",
                    "masked_cells": cells_to_mask,
                },
            )
        return None
    
    def _rephrase_question(self, qa: QAPair) -> Optional[QAPair]:
        """Rephrase question (test semantic matching)."""
        # Simple rephrasing patterns
        rephrased = qa.question
        
        # Pattern replacements
        patterns = [
            (r"What is (.+)\?", r"Tell me about \1."),
            (r"How many (.+)\?", r"What is the count of \1?"),
            (r"Which (.+) has (.+)\?", r"Find the \1 with \2."),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, qa.question, re.IGNORECASE):
                rephrased = re.sub(pattern, replacement, qa.question, flags=re.IGNORECASE)
                break
        
        if rephrased != qa.question:
            return QAPair(
                question=rephrased,
                answer=qa.answer,
                required_cells=qa.required_cells,
                question_type=qa.question_type,
                requires_hierarchy=qa.requires_hierarchy,
                requires_merged_cells=qa.requires_merged_cells,
                table_id=qa.table_id,
                metadata={**qa.metadata, "perturbation": "rephrase"},
            )
        return None


class QAValidator:
    """
    Validate QA pairs for quality control.
    """
    
    def validate_qa_pairs(
        self, qa_pairs: List[QAPair], tables: Dict[str, NormalizedTable]
    ) -> QAValidationReport:
        """
        Validate QA pairs and generate report.
        
        Args:
            qa_pairs: List of QAPair objects
            tables: Dictionary mapping table_id to NormalizedTable
            
        Returns:
            QAValidationReport
        """
        logger.info(f"Validating {len(qa_pairs)} QA pairs")
        
        report = QAValidationReport(total_pairs=len(qa_pairs))
        
        valid_count = 0
        distribution = defaultdict(int)
        
        for qa in qa_pairs:
            issues = []
            
            # Check if table exists
            if qa.table_id not in tables:
                issues.append({
                    "type": "missing_table",
                    "message": f"Table {qa.table_id} not found",
                })
            
            # Check if answer can be derived from cited cells
            if qa.table_id in tables:
                table = tables[qa.table_id]
                if not self._can_answer_from_cells(table, qa):
                    issues.append({
                        "type": "insufficient_cells",
                        "message": "Answer cannot be derived from cited cells",
                    })
            
            # Check if question is unambiguous
            if self._is_ambiguous(qa):
                issues.append({
                    "type": "ambiguous",
                    "message": "Question is ambiguous",
                })
            
            # Check for duplicates
            if self._is_duplicate(qa, qa_pairs):
                issues.append({
                    "type": "duplicate",
                    "message": "Duplicate question found",
                })
            
            # Check if trivial
            if self._is_trivial(qa):
                issues.append({
                    "type": "trivial",
                    "message": "Question is too trivial",
                })
            
            if not issues:
                valid_count += 1
                distribution[qa.question_type] += 1
            else:
                report.issues.append({
                    "qa_pair": qa.to_dict(),
                    "issues": issues,
                })
        
        report.valid_pairs = valid_count
        report.invalid_pairs = len(qa_pairs) - valid_count
        report.distribution = dict(distribution)
        
        # Statistics
        report.statistics = {
            "validity_rate": (
                valid_count / len(qa_pairs) if qa_pairs else 0.0
            ),
            "avg_cells_per_question": (
                np.mean([len(qa.required_cells) for qa in qa_pairs])
                if qa_pairs
                else 0.0
            ),
            "hierarchy_required_rate": (
                sum(1 for qa in qa_pairs if qa.requires_hierarchy) / len(qa_pairs)
                if qa_pairs
                else 0.0
            ),
        }
        
        logger.info(
            f"Validation complete: {valid_count}/{len(qa_pairs)} valid pairs"
        )
        
        return report
    
    def _can_answer_from_cells(
        self, table: NormalizedTable, qa: QAPair
    ) -> bool:
        """Check if answer can be derived from cited cells."""
        if not qa.required_cells:
            return False
        
        # Extract cell values
        cell_values = []
        for row, col in qa.required_cells:
            cell = table.get_cell(row, col)
            if cell:
                cell_values.append(cell.content)
        
        # Check if answer contains cell values or can be computed from them
        answer_lower = qa.answer.lower()
        
        # Check for direct matches
        for value in cell_values:
            if value and value.lower() in answer_lower:
                return True
        
        # Check for numeric computation
        try:
            numeric_values = [float(v) for v in cell_values if self._is_numeric(v)]
            if numeric_values:
                # Check if answer is a sum, average, etc.
                if "sum" in answer_lower or "total" in answer_lower:
                    return True
                if "average" in answer_lower or "mean" in answer_lower:
                    return True
        except ValueError:
            pass
        
        return len(cell_values) > 0
    
    def _is_ambiguous(self, qa: QAPair) -> bool:
        """Check if question is ambiguous."""
        ambiguous_patterns = [
            r"what.*\?",
            r"which.*\?",
            r"how.*\?",
        ]
        
        question_lower = qa.question.lower()
        
        # Check for vague questions
        vague_words = ["something", "anything", "whatever", "some"]
        if any(word in question_lower for word in vague_words):
            return True
        
        # Check if question is too short
        if len(qa.question.split()) < 3:
            return True
        
        return False
    
    def _is_duplicate(self, qa: QAPair, all_pairs: List[QAPair]) -> bool:
        """Check if question is duplicate."""
        question_lower = qa.question.lower().strip()
        
        for other_qa in all_pairs:
            if other_qa == qa:
                continue
            
            other_lower = other_qa.question.lower().strip()
            
            # Exact match
            if question_lower == other_lower:
                return True
            
            # Very similar (Levenshtein distance < 3)
            if self._levenshtein_distance(question_lower, other_lower) < 3:
                return True
        
        return False
    
    def _is_trivial(self, qa: QAPair) -> bool:
        """Check if question is too trivial."""
        # Check if answer is just "yes" or "no"
        answer_lower = qa.answer.lower().strip()
        if answer_lower in ["yes", "no", "true", "false"]:
            return True
        
        # Check if question is too simple
        if len(qa.required_cells) == 1 and len(qa.question.split()) < 5:
            return True
        
        return False
    
    def _is_numeric(self, value: str) -> bool:
        """Check if value is numeric."""
        try:
            float(value.replace(",", "").replace("%", ""))
            return True
        except ValueError:
            return False
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class ManualAnnotationInterface:
    """
    Manual annotation interface for QA pairs.
    """
    
    def __init__(self, output_dir: str = "data/annotations") -> None:
        """
        Initialize annotation interface.
        
        Args:
            output_dir: Directory to save annotations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.annotations: List[QAPair] = []
        
        logger.info(f"ManualAnnotationInterface initialized (output_dir: {output_dir})")
    
    def create_annotation_ui(
        self, tables: List[NormalizedTable], table_ids: Optional[List[str]] = None
    ) -> None:
        """
        Create annotation UI using Gradio or Streamlit.
        
        Args:
            tables: List of NormalizedTable objects
            table_ids: Optional list of table IDs
        """
        if GRADIO_AVAILABLE:
            self._create_gradio_ui(tables, table_ids)
        elif STREAMLIT_AVAILABLE:
            self._create_streamlit_ui(tables, table_ids)
        else:
            logger.error(
                "Neither Gradio nor Streamlit available. "
                "Install one: pip install gradio or pip install streamlit"
            )
    
    def _create_gradio_ui(
        self, tables: List[NormalizedTable], table_ids: Optional[List[str]]
    ) -> None:
        """Create Gradio UI."""
        if not GRADIO_AVAILABLE:
            return
        
        def annotate_table(table_idx: int, question: str, answer: str, selected_cells: str):
            """Process annotation."""
            if table_idx >= len(tables):
                return "Invalid table index"
            
            table = tables[table_idx]
            table_id = table_ids[table_idx] if table_ids else f"table_{table_idx}"
            
            # Parse selected cells
            cell_coords = self._parse_cell_selection(selected_cells)
            
            # Create QAPair
            qa_pair = QAPair(
                question=question,
                answer=answer,
                required_cells=cell_coords,
                table_id=table_id,
            )
            
            # Save
            self.annotations.append(qa_pair)
            self._save_annotations()
            
            return f"Saved annotation for table {table_id}"
        
        # Create UI
        with gr.Blocks() as demo:
            gr.Markdown("# Table QA Annotation Interface")
            
            table_idx = gr.Slider(
                minimum=0,
                maximum=len(tables) - 1,
                step=1,
                label="Table Index",
            )
            
            question = gr.Textbox(label="Question")
            answer = gr.Textbox(label="Answer")
            selected_cells = gr.Textbox(
                label="Selected Cells (format: (row,col) (row,col) ...)"
            )
            
            submit_btn = gr.Button("Save Annotation")
            output = gr.Textbox(label="Status")
            
            submit_btn.click(
                annotate_table,
                inputs=[table_idx, question, answer, selected_cells],
                outputs=output,
            )
        
        demo.launch()
    
    def _create_streamlit_ui(
        self, tables: List[NormalizedTable], table_ids: Optional[List[str]]
    ) -> None:
        """Create Streamlit UI."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.title("Table QA Annotation Interface")
        
        table_idx = st.slider("Table Index", 0, len(tables) - 1, 0)
        
        table = tables[table_idx]
        table_id = table_ids[table_idx] if table_ids else f"table_{table_idx}"
        
        # Display table
        st.subheader(f"Table: {table_id}")
        table_df = self._table_to_dataframe(table)
        st.dataframe(table_df)
        
        # Annotation form
        question = st.text_input("Question")
        answer = st.text_input("Answer")
        selected_cells = st.text_input(
            "Selected Cells (format: (row,col) (row,col) ...)"
        )
        
        if st.button("Save Annotation"):
            cell_coords = self._parse_cell_selection(selected_cells)
            
            qa_pair = QAPair(
                question=question,
                answer=answer,
                required_cells=cell_coords,
                table_id=table_id,
            )
            
            self.annotations.append(qa_pair)
            self._save_annotations()
            
            st.success(f"Saved annotation for table {table_id}")
    
    def _parse_cell_selection(self, selection: str) -> List[Tuple[int, int]]:
        """Parse cell selection string."""
        cells = []
        pattern = r"\((\d+),(\d+)\)"
        
        for match in re.finditer(pattern, selection):
            row = int(match.group(1))
            col = int(match.group(2))
            cells.append((row, col))
        
        return cells
    
    def _table_to_dataframe(self, table: NormalizedTable) -> pd.DataFrame:
        """Convert NormalizedTable to DataFrame for display."""
        data = []
        for row_idx in range(table.num_rows):
            row_data = []
            for col_idx in range(table.num_cols):
                cell = table.get_cell(row_idx, col_idx)
                if cell:
                    row_data.append(cell.content)
                else:
                    row_data.append("")
            data.append(row_data)
        
        return pd.DataFrame(data)
    
    def _save_annotations(self) -> None:
        """Save annotations to file."""
        output_path = self.output_dir / f"annotations_{int(time.time())}.json"
        
        annotations_data = [qa.to_dict() for qa in self.annotations]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotations_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.annotations)} annotations to {output_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="QA generation tools for HierTable-RAG"
    )
    parser.add_argument(
        "--tables", type=str, required=True, help="Path to tables JSON file"
    )
    parser.add_argument(
        "--output", type=str, default="data/qa_pairs.json", help="Output file"
    )
    parser.add_argument(
        "--num-questions", type=int, default=5, help="Questions per table"
    )
    parser.add_argument(
        "--generate-perturbations",
        action="store_true",
        help="Generate synthetic perturbations",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate QA pairs"
    )
    parser.add_argument(
        "--annotation-ui", action="store_true", help="Launch annotation UI"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load tables (placeholder - would need actual loading logic)
    logger.info("Loading tables...")
    tables = []  # Would load from JSON
    table_ids = []
    
    # Generate QA pairs
    generator = QAGenerator()
    all_qa_pairs = []
    
    iterator = (
        tqdm(zip(tables, table_ids), desc="Generating QA pairs")
        if TQDM_AVAILABLE
        else zip(tables, table_ids)
    )
    
    for table, table_id in iterator:
        qa_pairs = generator.generate_qa_with_llm(
            table, num_questions=args.num_questions, table_id=table_id
        )
        all_qa_pairs.extend(qa_pairs)
        
        # Generate perturbations if requested
        if args.generate_perturbations:
            perturbator = QAPerturbator()
            for qa in qa_pairs:
                perturbations = perturbator.generate_synthetic_perturbations(
                    table, qa
                )
                all_qa_pairs.extend(perturbations)
    
    # Validate if requested
    if args.validate:
        validator = QAValidator()
        tables_dict = {tid: table for tid, table in zip(table_ids, tables)}
        report = validator.validate_qa_pairs(all_qa_pairs, tables_dict)
        
        print(f"\nValidation Report:")
        print(f"  Valid pairs: {report.valid_pairs}/{report.total_pairs}")
        print(f"  Distribution: {report.distribution}")
        print(f"  Issues: {len(report.issues)}")
    
    # Save QA pairs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    qa_data = [qa.to_dict() for qa in all_qa_pairs]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(all_qa_pairs)} QA pairs to {output_path}")
    
    # Launch annotation UI if requested
    if args.annotation_ui:
        interface = ManualAnnotationInterface()
        interface.create_annotation_ui(tables, table_ids)


if __name__ == "__main__":
    main()



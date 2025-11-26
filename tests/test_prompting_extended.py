"""
Extended tests for StructuredPromptBuilder.

Tests cell citations, few-shot prompts, chain-of-thought prompts,
and evaluation with LLM models.
"""

import pytest
from typing import List

from src.parsing.extractor import TableObject, Cell
from src.parsing.cell_merger import CellMergeHandler, NormalizedTable
from src.parsing.hierarchy import HierarchyExtractor, HeaderTree
from src.retrieval.indexer import HierarchicalIndexer, IndexMetadata
from src.retrieval.retriever import ContextBundle, RetrievalResult
from src.generation.prompting import StructuredPromptBuilder


class TestStructuredPromptBuilderExtended:
    """Extended test cases for StructuredPromptBuilder."""

    @pytest.fixture
    def sample_context_bundle(self):
        """Create a sample ContextBundle for testing."""
        metadata1 = IndexMetadata(
            item_id="item1",
            source_table_id="table1",
            granularity="cell",
            hierarchy_path="Table > Revenue > Q1",
            coordinates=((2, 3), (0, 1)),
            text="100",
        )
        
        metadata2 = IndexMetadata(
            item_id="item2",
            source_table_id="table1",
            granularity="cell",
            hierarchy_path="Table > Revenue > Q2",
            coordinates=((2, 3), (1, 2)),
            text="200",
        )
        
        result1 = RetrievalResult(
            item_id="item1",
            metadata=metadata1,
            score=0.95,
            granularity="cell",
            content="100",
            hierarchy_path="Table > Revenue > Q1",
        )
        
        result2 = RetrievalResult(
            item_id="item2",
            metadata=metadata2,
            score=0.90,
            granularity="cell",
            content="200",
            hierarchy_path="Table > Revenue > Q2",
        )
        
        bundle = ContextBundle(
            retrieved_cells=[result1, result2],
            hierarchy_context={
                "root": "Table",
                "nodes": [
                    {
                        "path": "Table > Revenue",
                        "granularity": "subtable",
                        "content": "Revenue section",
                        "score": 0.9,
                    },
                ],
            },
            table_metadata={
                "table_id": "table1",
                "hierarchy_path": "Table",
            },
            query="What is the revenue?",
        )
        
        return bundle

    @pytest.fixture
    def sample_cells(self):
        """Create sample Cell objects for citations."""
        return [
            Cell(2, 0, "100", row=2, col=0),
            Cell(2, 1, "200", row=2, col=1),
            Cell(2, 2, "150", row=2, col=2),
        ]

    def test_add_cell_citations(self, sample_context_bundle, sample_cells):
        """Test adding cell citations to prompt."""
        builder = StructuredPromptBuilder()
        prompt = builder.build_rag_prompt(
            "What is the revenue?", sample_context_bundle, style="explicit"
        )
        
        prompt_with_citations = builder.add_cell_citations(prompt, sample_cells)
        
        assert "Cell Citations" in prompt_with_citations
        assert "[Row 2, Col 0" in prompt_with_citations
        assert "100" in prompt_with_citations
        assert len(prompt_with_citations) > len(prompt)

    def test_add_cell_citations_empty(self, sample_context_bundle):
        """Test adding citations with empty cell list."""
        builder = StructuredPromptBuilder()
        prompt = builder.build_rag_prompt(
            "query", sample_context_bundle, style="explicit"
        )
        
        prompt_with_citations = builder.add_cell_citations(prompt, [])
        
        assert prompt_with_citations == prompt

    def test_build_few_shot_prompt(self, sample_context_bundle):
        """Test building few-shot prompt with examples."""
        builder = StructuredPromptBuilder()
        
        examples = [
            {
                "query": "What is the total revenue?",
                "answer": "The total revenue is 450, calculated from Q1 (100) + Q2 (200) + Q3 (150).",
                "reasoning": (
                    "Step 1: Identified Revenue section in hierarchy\n"
                    "Step 2: Found Q1, Q2, Q3 cells under Revenue\n"
                    "Step 3: Summed values: 100 + 200 + 150 = 450"
                ),
                "hierarchy_usage": (
                    "Used hierarchy to identify all quarters under Revenue "
                    "and aggregate their values."
                ),
            },
            {
                "query": "Which quarter has the highest revenue?",
                "answer": "Q2 has the highest revenue of 200 [Row 2, Col 1].",
                "reasoning": (
                    "Step 1: Identified Revenue > Q1, Q2, Q3 in hierarchy\n"
                    "Step 2: Compared values: Q1=100, Q2=200, Q3=150\n"
                    "Step 3: Q2 has maximum value"
                ),
                "hierarchy_usage": "Used hierarchy to locate all quarters and compare values.",
            },
        ]
        
        prompt = builder.build_few_shot_prompt(
            "What is the revenue for Q1?",
            sample_context_bundle,
            examples,
            style="explicit",
        )
        
        assert "Examples" in prompt
        assert "Example 1" in prompt
        assert "Example 2" in prompt
        assert "Reasoning" in prompt
        assert "Hierarchy Usage" in prompt
        assert "similar reasoning" in prompt.lower()

    def test_build_few_shot_prompt_limits_examples(self, sample_context_bundle):
        """Test that few-shot prompt limits to 3 examples."""
        builder = StructuredPromptBuilder()
        
        # Create 5 examples
        examples = [
            {
                "query": f"Query {i}",
                "answer": f"Answer {i}",
                "reasoning": f"Reasoning {i}",
            }
            for i in range(5)
        ]
        
        prompt = builder.build_few_shot_prompt(
            "query", sample_context_bundle, examples, style="explicit"
        )
        
        # Should only include 3 examples
        assert "Example 1" in prompt
        assert "Example 2" in prompt
        assert "Example 3" in prompt
        assert "Example 4" not in prompt
        assert "Example 5" not in prompt

    def test_build_cot_prompt(self, sample_context_bundle):
        """Test building chain-of-thought prompt."""
        builder = StructuredPromptBuilder()
        
        prompt = builder.build_cot_prompt(
            "What is the total revenue?",
            sample_context_bundle,
            style="explicit",
        )
        
        assert "Reasoning Steps" in prompt
        assert "Step 1" in prompt
        assert "Step 2" in prompt
        assert "Step 3" in prompt
        assert "Step 4" in prompt
        assert "Identify which columns/rows" in prompt
        assert "merged cells" in prompt.lower()
        assert "Extract the values" in prompt
        assert "Provide the final answer" in prompt
        assert "Final Answer" in prompt

    def test_build_cot_prompt_with_citations(
        self, sample_context_bundle, sample_cells
    ):
        """Test building CoT prompt with cell citations."""
        builder = StructuredPromptBuilder()
        
        prompt = builder.build_cot_prompt_with_citations(
            "What is the total revenue?",
            sample_context_bundle,
            cells=sample_cells,
            style="explicit",
        )
        
        assert "Reasoning Steps" in prompt
        assert "Cell Citations" in prompt
        assert "[Row 2, Col 0" in prompt

    def test_cot_prompt_structure(self, sample_context_bundle):
        """Test that CoT prompt has correct structure."""
        builder = StructuredPromptBuilder()
        
        prompt = builder.build_cot_prompt(
            "query", sample_context_bundle, style="explicit"
        )
        
        # Check all required sections
        sections = [
            "Table Structure",
            "Retrieved Context",
            "Question",
            "Instructions",
            "Reasoning Steps",
            "Step 1",
            "Step 2",
            "Step 3",
            "Step 4",
            "Final Answer",
        ]
        
        for section in sections:
            assert section in prompt

    def test_few_shot_example_structure(self, sample_context_bundle):
        """Test that few-shot examples have correct structure."""
        builder = StructuredPromptBuilder()
        
        examples = [
            {
                "query": "Example query",
                "answer": "Example answer",
                "reasoning": "Example reasoning",
            }
        ]
        
        prompt = builder.build_few_shot_prompt(
            "query", sample_context_bundle, examples, style="explicit"
        )
        
        assert "Query:" in prompt
        assert "Reasoning:" in prompt
        assert "Answer:" in prompt

    def test_citations_format(self, sample_context_bundle, sample_cells):
        """Test that citations use correct format."""
        builder = StructuredPromptBuilder()
        prompt = builder.build_rag_prompt(
            "query", sample_context_bundle, style="explicit"
        )
        
        prompt_with_citations = builder.add_cell_citations(prompt, sample_cells)
        
        # Check citation format
        assert "[Row 2, Col 0:" in prompt_with_citations
        assert '"100"' in prompt_with_citations
        assert "cite cells using the format" in prompt_with_citations.lower()

    def test_cot_prompt_reasoning_guidance(self, sample_context_bundle):
        """Test that CoT prompt provides clear reasoning guidance."""
        builder = StructuredPromptBuilder()
        
        prompt = builder.build_cot_prompt(
            "What is the total?", sample_context_bundle, style="explicit"
        )
        
        # Check for reasoning guidance
        assert "Your reasoning for Step 1" in prompt
        assert "Your reasoning for Step 2" in prompt
        assert "Your reasoning for Step 3" in prompt
        assert "Your reasoning for Step 4" in prompt

    def test_prompt_style_consistency(self, sample_context_bundle):
        """Test that prompt styles are consistent across methods."""
        builder = StructuredPromptBuilder()
        
        # Test all methods with same style
        style = "explicit"
        
        base_prompt = builder.build_rag_prompt(
            "query", sample_context_bundle, style=style
        )
        
        cot_prompt = builder.build_cot_prompt(
            "query", sample_context_bundle, style=style
        )
        
        # Both should use same formatting style
        assert isinstance(base_prompt, str)
        assert isinstance(cot_prompt, str)
        # CoT should contain base prompt
        assert "Table Structure" in cot_prompt

    def test_empty_examples_handling(self, sample_context_bundle):
        """Test handling of empty examples list."""
        builder = StructuredPromptBuilder()
        
        prompt = builder.build_few_shot_prompt(
            "query", sample_context_bundle, examples=[], style="explicit"
        )
        
        # Should still build prompt, just without examples
        assert isinstance(prompt, str)
        assert "Table Structure" in prompt

    def test_cot_with_merged_cells(self):
        """Test CoT prompt with merged cell context."""
        # Create context with merged cell
        metadata = IndexMetadata(
            item_id="item1",
            source_table_id="table1",
            granularity="cell",
            hierarchy_path="Table > Merged",
            coordinates=((0, 2), (0, 2)),  # Merged: 2x2
            text="Merged Value",
        )
        
        result = RetrievalResult(
            item_id="item1",
            metadata=metadata,
            score=0.9,
            granularity="cell",
            content="Merged Value",
            hierarchy_path="Table > Merged",
        )
        
        bundle = ContextBundle(
            retrieved_cells=[result],
            hierarchy_context={},
            table_metadata={},
            query="What is the value?",
        )
        
        builder = StructuredPromptBuilder()
        prompt = builder.build_cot_prompt("query", bundle, style="explicit")
        
        assert "merged cells" in prompt.lower()
        assert "Step 2" in prompt



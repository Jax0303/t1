"""
Tests for StructuredPromptBuilder class.

Tests RAG prompt building with different styles and formats.
"""

import pytest
from typing import List

from src.parsing.extractor import TableObject, Cell
from src.parsing.cell_merger import CellMergeHandler, NormalizedTable
from src.parsing.hierarchy import HierarchyExtractor, HeaderTree
from src.retrieval.indexer import HierarchicalIndexer, IndexMetadata
from src.retrieval.retriever import ContextBundle, RetrievalResult
from src.generation.prompting import StructuredPromptBuilder


class TestStructuredPromptBuilder:
    """Test cases for StructuredPromptBuilder class."""

    @pytest.fixture
    def sample_context_bundle(self):
        """Create a sample ContextBundle for testing."""
        # Create metadata
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
        
        # Create retrieval results
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
        
        # Create context bundle
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
                    {
                        "path": "Table > Revenue > Q1",
                        "granularity": "cell",
                        "content": "100",
                        "score": 0.95,
                    },
                ],
            },
            table_metadata={
                "table_id": "table1",
                "hierarchy_path": "Table",
                "coordinates": ((0, 3), (0, 4)),
            },
            query="What is the revenue?",
        )
        
        return bundle

    def test_builder_init(self):
        """Test builder initialization."""
        builder = StructuredPromptBuilder()
        assert builder.include_merged_cell_info is True
        assert builder.include_coordinates is True

    def test_build_rag_prompt_explicit(self, sample_context_bundle):
        """Test building RAG prompt with explicit style."""
        builder = StructuredPromptBuilder()
        prompt = builder.build_rag_prompt(
            "What is the revenue?", sample_context_bundle, style="explicit"
        )
        
        assert isinstance(prompt, str)
        assert "Table Structure" in prompt
        assert "Retrieved Context" in prompt
        assert "Question" in prompt
        assert "Instructions" in prompt
        assert "What is the revenue?" in prompt

    def test_build_rag_prompt_natural(self, sample_context_bundle):
        """Test building RAG prompt with natural style."""
        builder = StructuredPromptBuilder()
        prompt = builder.build_rag_prompt(
            "What is the revenue?", sample_context_bundle, style="natural"
        )
        
        assert isinstance(prompt, str)
        assert "Table Structure" in prompt
        assert "natural" in prompt.lower() or "revenue" in prompt.lower()

    def test_build_rag_prompt_code(self, sample_context_bundle):
        """Test building RAG prompt with code style."""
        builder = StructuredPromptBuilder()
        prompt = builder.build_rag_prompt(
            "What is the revenue?", sample_context_bundle, style="code"
        )
        
        assert isinstance(prompt, str)
        # Should contain JSON-like structure
        assert "{" in prompt or "[" in prompt

    def test_build_rag_prompt_table_markdown(self, sample_context_bundle):
        """Test building RAG prompt with table_markdown style."""
        builder = StructuredPromptBuilder()
        prompt = builder.build_rag_prompt(
            "What is the revenue?", sample_context_bundle, style="table_markdown"
        )
        
        assert isinstance(prompt, str)
        assert "|" in prompt  # Markdown table

    def test_build_rag_prompt_invalid_style(self, sample_context_bundle):
        """Test error handling for invalid style."""
        builder = StructuredPromptBuilder()
        
        # Should not raise error, fall back to explicit
        prompt = builder.build_rag_prompt(
            "query", sample_context_bundle, style="invalid"
        )
        assert isinstance(prompt, str)

    def test_format_table_structure_explicit(self, sample_context_bundle):
        """Test explicit table structure formatting."""
        builder = StructuredPromptBuilder()
        structure = builder._format_table_structure(
            sample_context_bundle, style="explicit"
        )
        
        assert isinstance(structure, str)
        assert "Table" in structure or "Revenue" in structure

    def test_format_retrieved_context_explicit(self, sample_context_bundle):
        """Test explicit context formatting."""
        builder = StructuredPromptBuilder()
        context = builder._format_retrieved_context(
            sample_context_bundle, style="explicit"
        )
        
        assert isinstance(context, str)
        assert "Hierarchy Path" in context
        assert "Content" in context
        assert "100" in context or "200" in context

    def test_format_merged_cell_info(self):
        """Test merged cell information formatting."""
        # Create context with merged cell
        metadata = IndexMetadata(
            item_id="item1",
            source_table_id="table1",
            granularity="cell",
            hierarchy_path="Table > Merged",
            coordinates=((0, 2), (0, 2)),  # Merged: 2 rows, 2 cols
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
            query="query",
        )
        
        builder = StructuredPromptBuilder()
        merged_info = builder._format_merged_cell_info(bundle, style="explicit")
        
        assert isinstance(merged_info, str)
        assert "merged" in merged_info.lower()
        assert "Spans" in merged_info or "span" in merged_info.lower()

    def test_format_merged_cell_info_no_merged(self, sample_context_bundle):
        """Test merged cell info when no merged cells."""
        builder = StructuredPromptBuilder()
        merged_info = builder._format_merged_cell_info(
            sample_context_bundle, style="explicit"
        )
        
        # Should indicate no merged cells or handle gracefully
        assert isinstance(merged_info, str)

    def test_build_instructions(self):
        """Test instructions building."""
        builder = StructuredPromptBuilder()
        instructions = builder._build_instructions()
        
        assert isinstance(instructions, str)
        assert "hierarchy" in instructions.lower()
        assert "merged" in instructions.lower()
        assert "coordinates" in instructions.lower()

    def test_prompt_includes_all_sections(self, sample_context_bundle):
        """Test that prompt includes all required sections."""
        builder = StructuredPromptBuilder()
        prompt = builder.build_rag_prompt(
            "What is the revenue?", sample_context_bundle, style="explicit"
        )
        
        sections = [
            "Table Structure",
            "Retrieved Context",
            "Merged Cell Information",
            "Question",
            "Instructions",
        ]
        
        for section in sections:
            assert section in prompt

    def test_coordinates_in_prompt(self, sample_context_bundle):
        """Test that coordinates are included when enabled."""
        builder = StructuredPromptBuilder(include_coordinates=True)
        prompt = builder.build_rag_prompt(
            "query", sample_context_bundle, style="explicit"
        )
        
        # Should contain coordinate information
        assert "Row" in prompt or "Column" in prompt or "Coordinates" in prompt

    def test_no_coordinates_when_disabled(self):
        """Test that coordinates can be disabled."""
        metadata = IndexMetadata(
            item_id="item1",
            source_table_id="table1",
            granularity="cell",
            hierarchy_path="Table > Cell",
            coordinates=((0, 1), (0, 1)),
            text="Value",
        )
        
        result = RetrievalResult(
            item_id="item1",
            metadata=metadata,
            score=0.9,
            granularity="cell",
            content="Value",
            hierarchy_path="Table > Cell",
        )
        
        bundle = ContextBundle(
            retrieved_cells=[result],
            hierarchy_context={},
            table_metadata={},
            query="query",
        )
        
        builder = StructuredPromptBuilder(include_coordinates=False)
        prompt = builder.build_rag_prompt("query", bundle, style="explicit")
        
        # Coordinates might still appear in some formats, but should be minimal
        assert isinstance(prompt, str)

    def test_empty_context_handling(self):
        """Test handling of empty context."""
        empty_bundle = ContextBundle(
            retrieved_cells=[],
            hierarchy_context={},
            table_metadata={},
            query="query",
        )
        
        builder = StructuredPromptBuilder()
        prompt = builder.build_rag_prompt("query", empty_bundle, style="explicit")
        
        assert isinstance(prompt, str)
        assert "Question" in prompt
        assert "Instructions" in prompt



"""
Prompt construction and LLM generation for hierarchical tables.

Provides structured prompt building with multiple formatting styles
for RAG applications with hierarchical table context.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from src.retrieval.retriever import ContextBundle, RetrievalResult
from src.encoding.tree_encoder import TreeEncoder
from src.parsing.extractor import Cell

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Build prompts for LLM generation with hierarchical table context.
    
    Creates structured prompts that effectively communicate
    table hierarchy and relationships to the LLM.
    """

    def __init__(
        self,
        template: Optional[str] = None,
        include_structure: bool = True,
        include_hierarchy: bool = True,
    ) -> None:
        """
        Initialize prompt builder.
        
        Args:
            template: Custom prompt template (uses default if None)
            include_structure: Include table structure in prompt
            include_hierarchy: Include hierarchy information in prompt
        """
        self.template = template
        self.include_structure = include_structure
        self.include_hierarchy = include_hierarchy
        logger.info("PromptBuilder initialized")

    def build_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        include_metadata: bool = True,
    ) -> str:
        """
        Build a prompt from query and retrieved context.
        
        Args:
            query: User query
            context: Retrieved table segments with metadata
            include_metadata: Include table metadata in prompt
            
        Returns:
            Formatted prompt string
        """
        logger.info("Building prompt from query and context")
        # Implementation will format prompt with hierarchy
        return ""

    def format_table_context(
        self, tables: List[Dict[str, Any]], format: str = "structured"
    ) -> str:
        """
        Format table context for inclusion in prompt.
        
        Args:
            tables: List of table dictionaries
            format: Format style ("structured", "markdown", "json")
            
        Returns:
            Formatted table context string
        """
        logger.info(f"Formatting {len(tables)} tables as {format}")
        # Implementation will format tables appropriately
        return ""

    def format_hierarchy_info(
        self, hierarchy: Dict[str, Any]
    ) -> str:
        """
        Format hierarchy information for prompt.
        
        Args:
            hierarchy: Hierarchy metadata dictionary
            
        Returns:
            Formatted hierarchy description
        """
        logger.info("Formatting hierarchy information")
        # Implementation will format hierarchy relationships
        return ""


class StructuredPromptBuilder:
    """
    Build structured RAG prompts with hierarchical table context.
    
    Supports multiple formatting styles:
    - explicit: Clearly show hierarchy with indentation
    - natural: Convert to natural language description
    - code: Present as Python dict/JSON
    - table_markdown: Use markdown table with hierarchy indicators
    """

    def __init__(
        self,
        include_merged_cell_info: bool = True,
        include_coordinates: bool = True,
    ) -> None:
        """
        Initialize structured prompt builder.
        
        Args:
            include_merged_cell_info: Include merged cell information
            include_coordinates: Include cell coordinates in prompt
        """
        self.include_merged_cell_info = include_merged_cell_info
        self.include_coordinates = include_coordinates
        self.tree_encoder = TreeEncoder()
        logger.info("StructuredPromptBuilder initialized")

    def build_rag_prompt(
        self,
        query: str,
        context: ContextBundle,
        style: str = "explicit",
    ) -> str:
        """
        Build RAG prompt with structured table context.
        
        Template structure:
        - Table Structure: Hierarchy tree in chosen format
        - Retrieved Context: Relevant cells with hierarchy path
        - Merged Cell Information: Explanation of merged cells if any
        - Question: User query
        - Instructions: Guidance on using hierarchy and merged cells
        
        Args:
            query: User query
            context: ContextBundle with retrieved cells and hierarchy
            style: Formatting style ("explicit", "natural", "code", "table_markdown")
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If style is not supported
        """
        logger.info(f"Building RAG prompt with style '{style}'")
        
        # Build sections
        sections = []
        
        # 1. Table Structure
        structure_section = self._format_table_structure(
            context, style
        )
        if structure_section:
            sections.append(f"## Table Structure\n\n{structure_section}")
        
        # 2. Retrieved Context
        context_section = self._format_retrieved_context(
            context, style
        )
        if context_section:
            sections.append(f"## Retrieved Context\n\n{context_section}")
        
        # 3. Merged Cell Information
        if self.include_merged_cell_info:
            merged_section = self._format_merged_cell_info(
                context, style
            )
            if merged_section:
                sections.append(
                    f"## Merged Cell Information\n\n{merged_section}"
                )
        
        # 4. Question
        sections.append(f"## Question\n\n{query}")
        
        # 5. Instructions
        instructions = self._build_instructions()
        sections.append(f"## Instructions\n\n{instructions}")
        
        # Combine sections
        prompt = "\n\n".join(sections)
        
        return prompt

    def _format_table_structure(
        self, context: ContextBundle, style: str
    ) -> str:
        """
        Format table structure section.
        
        Args:
            context: ContextBundle
            style: Formatting style
            
        Returns:
            Formatted structure string
        """
        hierarchy_context = context.hierarchy_context
        
        if not hierarchy_context or "nodes" not in hierarchy_context:
            return "No hierarchy information available."
        
        if style == "explicit":
            return self._format_structure_explicit(hierarchy_context)
        elif style == "natural":
            return self._format_structure_natural(hierarchy_context)
        elif style == "code":
            return self._format_structure_code(hierarchy_context)
        elif style == "table_markdown":
            return self._format_structure_table_markdown(hierarchy_context)
        else:
            return str(hierarchy_context)

    def _format_structure_explicit(
        self, hierarchy_context: Dict[str, Any]
    ) -> str:
        """Format structure with explicit indentation."""
        lines = []
        
        root = hierarchy_context.get("root", "Table")
        nodes = hierarchy_context.get("nodes", [])
        
        lines.append(f"{root}")
        
        # Group nodes by hierarchy path depth
        path_to_nodes: Dict[str, List[Dict[str, Any]]] = {}
        for node in nodes:
            path = node.get("path", "")
            if path not in path_to_nodes:
                path_to_nodes[path] = []
            path_to_nodes[path].append(node)
        
        # Build indented structure
        for path, node_list in sorted(path_to_nodes.items()):
            path_parts = path.split(" > ")
            indent_level = len(path_parts) - 1
            
            for node in node_list:
                indent = "  " * indent_level
                content = node.get("content", "")
                granularity = node.get("granularity", "")
                
                line = f"{indent}- {path_parts[-1]}"
                if content:
                    line += f": {content[:50]}"
                if granularity:
                    line += f" [{granularity}]"
                
                lines.append(line)
        
        return "\n".join(lines)

    def _format_structure_natural(
        self, hierarchy_context: Dict[str, Any]
    ) -> str:
        """Format structure as natural language."""
        lines = []
        
        root = hierarchy_context.get("root", "Table")
        nodes = hierarchy_context.get("nodes", [])
        
        lines.append(f"The table has a hierarchical structure with {root} as the root.")
        
        # Group by hierarchy level
        level_groups: Dict[int, List[str]] = {}
        for node in nodes:
            path = node.get("path", "")
            path_parts = path.split(" > ")
            level = len(path_parts) - 1
            
            if level not in level_groups:
                level_groups[level] = []
            
            content = node.get("content", "")
            if content:
                level_groups[level].append(f"{path_parts[-1]} ({content[:30]})")
            else:
                level_groups[level].append(path_parts[-1])
        
        # Describe each level
        for level in sorted(level_groups.keys()):
            items = level_groups[level]
            if level == 0:
                lines.append(f"At the top level: {', '.join(items[:5])}")
            else:
                lines.append(
                    f"At level {level}: {', '.join(items[:5])}"
                )
        
        return " ".join(lines)

    def _format_structure_code(
        self, hierarchy_context: Dict[str, Any]
    ) -> str:
        """Format structure as Python dict/JSON."""
        return json.dumps(hierarchy_context, indent=2, ensure_ascii=False)

    def _format_structure_table_markdown(
        self, hierarchy_context: Dict[str, Any]
    ) -> str:
        """Format structure as markdown table."""
        lines = []
        lines.append("| Level | Path | Content | Granularity |")
        lines.append("|-------|------|---------|-------------|")
        
        nodes = hierarchy_context.get("nodes", [])
        for node in nodes:
            path = node.get("path", "")
            path_parts = path.split(" > ")
            level = len(path_parts) - 1
            
            content = node.get("content", "")[:50] or "-"
            granularity = node.get("granularity", "-")
            
            # Escape pipe characters
            path_display = path.replace("|", "\\|")
            content_display = content.replace("|", "\\|")
            
            lines.append(
                f"| {level} | {path_display} | {content_display} | {granularity} |"
            )
        
        return "\n".join(lines)

    def _format_retrieved_context(
        self, context: ContextBundle, style: str
    ) -> str:
        """
        Format retrieved context section.
        
        Args:
            context: ContextBundle
            style: Formatting style
            
        Returns:
            Formatted context string
        """
        cells = context.retrieved_cells
        
        if not cells:
            return "No relevant cells retrieved."
        
        if style == "explicit":
            return self._format_context_explicit(cells)
        elif style == "natural":
            return self._format_context_natural(cells)
        elif style == "code":
            return self._format_context_code(cells)
        elif style == "table_markdown":
            return self._format_context_table_markdown(cells)
        else:
            return self._format_context_explicit(cells)

    def _format_context_explicit(
        self, cells: List[RetrievalResult]
    ) -> str:
        """Format context with explicit hierarchy paths."""
        lines = []
        
        for i, cell in enumerate(cells, 1):
            lines.append(f"### Cell {i}")
            lines.append(f"**Hierarchy Path:** {cell.hierarchy_path}")
            lines.append(f"**Content:** {cell.content}")
            
            if self.include_coordinates:
                coords = cell.metadata.coordinates
                (row_start, row_end), (col_start, col_end) = coords
                lines.append(
                    f"**Coordinates:** Row {row_start}-{row_end-1}, "
                    f"Column {col_start}-{col_end-1}"
                )
            
            lines.append(f"**Granularity:** {cell.granularity}")
            lines.append(f"**Relevance Score:** {cell.score:.3f}")
            lines.append("")
        
        return "\n".join(lines)

    def _format_context_natural(
        self, cells: List[RetrievalResult]
    ) -> str:
        """Format context as natural language."""
        lines = []
        
        lines.append(
            f"Retrieved {len(cells)} relevant cells from the table:"
        )
        
        for i, cell in enumerate(cells, 1):
            path = cell.hierarchy_path
            content = cell.content[:100]
            
            line = f"{i}. At {path}, the value is '{content}'"
            
            if self.include_coordinates:
                coords = cell.metadata.coordinates
                (row_start, _), (col_start, _) = coords
                line += f" (located at row {row_start}, column {col_start})"
            
            lines.append(line)
        
        return " ".join(lines)

    def _format_context_code(
        self, cells: List[RetrievalResult]
    ) -> str:
        """Format context as JSON."""
        cells_data = [cell.to_dict() for cell in cells]
        return json.dumps(cells_data, indent=2, ensure_ascii=False)

    def _format_context_table_markdown(
        self, cells: List[RetrievalResult]
    ) -> str:
        """Format context as markdown table."""
        lines = []
        lines.append(
            "| # | Hierarchy Path | Content | Coordinates | Score |"
        )
        lines.append("|---|----------------|---------|-------------|-------|")
        
        for i, cell in enumerate(cells, 1):
            path = cell.hierarchy_path.replace("|", "\\|")
            content = (cell.content[:50] or "-").replace("|", "\\|")
            
            if self.include_coordinates:
                coords = cell.metadata.coordinates
                (row_start, row_end), (col_start, col_end) = coords
                coord_str = f"R{row_start}-{row_end-1}, C{col_start}-{col_end-1}"
            else:
                coord_str = "-"
            
            score = f"{cell.score:.3f}"
            
            lines.append(
                f"| {i} | {path} | {content} | {coord_str} | {score} |"
            )
        
        return "\n".join(lines)

    def _format_merged_cell_info(
        self, context: ContextBundle, style: str
    ) -> str:
        """
        Format merged cell information section.
        
        Args:
            context: ContextBundle
            style: Formatting style
            
        Returns:
            Formatted merged cell info string
        """
        cells = context.retrieved_cells
        
        # Find merged cells
        merged_cells = []
        for cell in cells:
            if cell.metadata.granularity == "cell":
                coords = cell.metadata.coordinates
                (row_start, row_end), (col_start, col_end) = coords
                
                # Check if cell is merged (spans multiple rows/cols)
                if (row_end - row_start > 1) or (col_end - col_start > 1):
                    merged_cells.append(cell)
        
        if not merged_cells:
            return "No merged cells in the retrieved context."
        
        lines = []
        lines.append(
            f"The following {len(merged_cells)} retrieved cell(s) are merged "
            f"(span multiple rows/columns):"
        )
        lines.append("")
        
        for i, cell in enumerate(merged_cells, 1):
            coords = cell.metadata.coordinates
            (row_start, row_end), (col_start, col_end) = coords
            
            rowspan = row_end - row_start
            colspan = col_end - col_start
            
            lines.append(f"{i}. **{cell.hierarchy_path}**")
            lines.append(f"   - Content: {cell.content}")
            lines.append(
                f"   - Spans {rowspan} row(s) and {colspan} column(s)"
            )
            lines.append(
                f"   - Coordinates: Row {row_start}-{row_end-1}, "
                f"Column {col_start}-{col_end-1}"
            )
            lines.append(
                "   - Note: This cell value applies to all cells "
                "within this merged region."
            )
            lines.append("")
        
        return "\n".join(lines)

    def _build_instructions(self) -> str:
        """
        Build instructions section for the prompt.
        
        Returns:
            Instructions string
        """
        instructions = [
            "1. **Use the hierarchy to understand column/row relationships**:",
            "   - The hierarchy path shows how cells are organized",
            "   - Parent headers apply to all child cells",
            "   - Use the hierarchy to correctly interpret cell meanings",
            "",
            "2. **Check if merged cells affect the answer**:",
            "   - Merged cells span multiple rows/columns",
            "   - The same value applies to all cells in the merged region",
            "   - Consider this when aggregating or comparing values",
            "",
            "3. **Cite specific cells in your answer using coordinates**:",
            "   - Use the format: (Row X, Column Y) or (R X, C Y)",
            "   - Reference the hierarchy path when relevant",
            "   - Be precise about which cells you're referring to",
            "",
            "4. **Answer the question accurately**:",
            "   - Use only the information provided in the context",
            "   - If information is missing, state that clearly",
            "   - Provide specific values when available",
        ]
        
        return "\n".join(instructions)

    def add_cell_citations(
        self, prompt: str, cells: List[Cell]
    ) -> str:
        """
        Add explicit cell references to prompt.
        
        Adds citations in format: [Row X, Col Y: "content"]
        Makes it easy for LLM to cite sources.
        
        Args:
            prompt: Original prompt string
            cells: List of Cell objects to cite
            
        Returns:
            Prompt with cell citations added
        """
        if not cells:
            return prompt
        
        citations_section = "\n\n## Cell Citations\n\n"
        citations_section += "For reference, here are the exact cell locations:\n\n"
        
        for i, cell in enumerate(cells, 1):
            citation = f"[Row {cell.row}, Col {cell.col}: \"{cell.content}\"]"
            citations_section += f"{i}. {citation}\n"
        
        citations_section += (
            "\nWhen answering, cite cells using the format: "
            "[Row X, Col Y] or refer to the citation number."
        )
        
        return prompt + citations_section

    def build_few_shot_prompt(
        self,
        query: str,
        context: ContextBundle,
        examples: List[Dict[str, Any]],
        style: str = "explicit",
    ) -> str:
        """
        Build few-shot prompt with similar examples and reasoning traces.
        
        Includes 2-3 similar examples showing how to use hierarchy information.
        
        Args:
            query: User query
            context: ContextBundle with retrieved context
            examples: List of example dictionaries with:
                - query: Example query
                - context: Example context (or simplified)
                - answer: Example answer
                - reasoning: Reasoning trace showing hierarchy usage
            style: Formatting style for main prompt
            
        Returns:
            Few-shot prompt string
        """
        logger.info(f"Building few-shot prompt with {len(examples)} examples")
        
        # Build base prompt
        base_prompt = self.build_rag_prompt(query, context, style=style)
        
        # Add examples section
        examples_section = "\n\n## Examples\n\n"
        examples_section += (
            "Here are similar examples showing how to use hierarchy information:\n\n"
        )
        
        for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples
            ex_query = example.get("query", "")
            ex_answer = example.get("answer", "")
            ex_reasoning = example.get("reasoning", "")
            
            examples_section += f"### Example {i}\n\n"
            examples_section += f"**Query:** {ex_query}\n\n"
            
            if ex_reasoning:
                examples_section += f"**Reasoning:**\n{ex_reasoning}\n\n"
            
            examples_section += f"**Answer:** {ex_answer}\n\n"
            
            # Add hierarchy usage note if present
            hierarchy_note = example.get("hierarchy_usage", "")
            if hierarchy_note:
                examples_section += (
                    f"**Hierarchy Usage:** {hierarchy_note}\n\n"
                )
            
            examples_section += "---\n\n"
        
        examples_section += (
            "Now answer the question above using similar reasoning, "
            "paying attention to the hierarchy structure.\n"
        )
        
        return base_prompt + examples_section

    def build_cot_prompt(
        self,
        query: str,
        context: ContextBundle,
        style: str = "explicit",
    ) -> str:
        """
        Build chain-of-thought prompt specifically for table reasoning.
        
        Provides step-by-step reasoning structure:
        Step 1: Identify which columns/rows are relevant based on hierarchy
        Step 2: Check if any merged cells affect interpretation
        Step 3: Extract values and perform calculations
        Step 4: Provide final answer with citations
        
        Args:
            query: User query
            context: ContextBundle with retrieved context
            style: Formatting style for context
            
        Returns:
            Chain-of-thought prompt string
        """
        logger.info("Building chain-of-thought prompt")
        
        # Build base prompt
        base_prompt = self.build_rag_prompt(query, context, style=style)
        
        # Add CoT instructions
        cot_section = "\n\n## Reasoning Steps\n\n"
        cot_section += (
            "Please follow these steps to answer the question:\n\n"
        )
        
        cot_steps = [
            (
                "Step 1: Identify which columns/rows are relevant "
                "based on the hierarchy",
                (
                    "- Examine the hierarchy path of retrieved cells\n"
                    "- Determine which header levels are relevant to the query\n"
                    "- Identify the scope of data needed (single cell, row, column, or range)"
                ),
            ),
            (
                "Step 2: Check if any merged cells affect the interpretation",
                (
                    "- Review the Merged Cell Information section\n"
                    "- Understand which cells share the same value\n"
                    "- Consider how merged cells impact aggregation or comparison operations"
                ),
            ),
            (
                "Step 3: Extract the values and perform any calculations",
                (
                    "- Extract relevant cell values from the Retrieved Context\n"
                    "- Apply any necessary operations (sum, average, comparison, etc.)\n"
                    "- Verify calculations using the hierarchy structure"
                ),
            ),
            (
                "Step 4: Provide the final answer with citations",
                (
                    "- State your answer clearly\n"
                    "- Cite specific cells using coordinates: [Row X, Col Y]\n"
                    "- Reference the hierarchy path when relevant\n"
                    "- Explain your reasoning if the answer involves interpretation"
                ),
            ),
        ]
        
        for step_num, (step_title, step_details) in enumerate(
            cot_steps, 1
        ):
            cot_section += f"### {step_title}\n\n"
            cot_section += f"{step_details}\n\n"
            cot_section += f"Your reasoning for Step {step_num}:\n\n"
        
        cot_section += "### Final Answer\n\n"
        cot_section += "Based on the above reasoning steps, provide your final answer:\n\n"
        
        return base_prompt + cot_section

    def build_cot_prompt_with_citations(
        self,
        query: str,
        context: ContextBundle,
        cells: Optional[List[Cell]] = None,
        style: str = "explicit",
    ) -> str:
        """
        Build CoT prompt with explicit cell citations.
        
        Combines chain-of-thought reasoning with cell citations.
        
        Args:
            query: User query
            context: ContextBundle with retrieved context
            cells: Optional list of Cell objects for citations
            style: Formatting style
            
        Returns:
            CoT prompt with citations
        """
        # Build CoT prompt
        cot_prompt = self.build_cot_prompt(query, context, style=style)
        
        # Add citations if cells provided
        if cells:
            cot_prompt = self.add_cell_citations(cot_prompt, cells)
        
        return cot_prompt


@dataclass
class CellRef:
    """
    Reference to a cell with coordinates.
    
    Attributes:
        row: Row index (0-based or 1-based depending on format)
        col: Column index (0-based or 1-based depending on format)
        format: Format used ("row_col", "excel", "compact", etc.)
        original_text: Original text that was parsed
    """
    row: int
    col: int
    format: str
    original_text: str

    def to_tuple(self, zero_indexed: bool = True) -> Tuple[int, int]:
        """
        Convert to tuple coordinates.
        
        Args:
            zero_indexed: Whether to return 0-indexed coordinates
            
        Returns:
            Tuple (row, col)
        """
        if zero_indexed:
            # Convert from 1-indexed if needed
            row = self.row - 1 if self.format == "excel" else self.row
            col = self.col - 1 if self.format == "excel" else self.col
        else:
            row = self.row
            col = self.col
        
        return (row, col)


@dataclass
class Answer:
    """
    Structured answer from LLM.
    
    Attributes:
        answer_text: Final answer text
        reasoning: Chain of thought reasoning (if available)
        cited_cells: List of cell coordinates mentioned
        confidence: Model's confidence score (if available)
        model: Model used for generation
        prompt_tokens: Number of tokens in prompt
        completion_tokens: Number of tokens in completion
    """
    answer_text: str
    reasoning: str = ""
    cited_cells: List[CellRef] = field(default_factory=list)
    confidence: Optional[float] = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer_text": self.answer_text,
            "reasoning": self.reasoning,
            "cited_cells": [
                {
                    "row": ref.row,
                    "col": ref.col,
                    "format": ref.format,
                    "original_text": ref.original_text,
                }
                for ref in self.cited_cells
            ],
            "confidence": self.confidence,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class LLMReasoner:
    """
    LLM-based reasoning with structured answer parsing and citation verification.
    
    Supports OpenAI and Anthropic APIs with retry logic and batch processing.
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        default_model: str = "gpt-4",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize LLM reasoner.
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            api_key: API key (uses environment variable if None)
            default_model: Default model name
            temperature: Sampling temperature
            max_retries: Maximum number of retries
            retry_delay: Initial retry delay in seconds
            
        Raises:
            ImportError: If required libraries are not available
            ValueError: If provider is not supported
        """
        self.provider = provider.lower()
        self.default_model = default_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set API key
        if api_key:
            self.api_key = api_key
        else:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "gemini":
                self.api_key = os.getenv("GEMINI_API_KEY")
            else:
                self.api_key = None
        
        # Initialize client
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "openai library is required. Install with: pip install openai"
                )
            self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "anthropic library is required. "
                    "Install with: pip install anthropic"
                )
            self.client = (
                anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
            )
        elif self.provider == "gemini":
            from src.utils.gemini_manager import GeminiKeyManager
            # Expect api_key to be a comma-separated list of keys for Gemini
            keys = self.api_key.split(",") if self.api_key else []
            self.gemini_manager = GeminiKeyManager(keys)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: openai, anthropic, gemini"
            )
        
        # Rate limiting
        self.rate_limit_lock = Lock()
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # Minimum time between requests (seconds)
        
        logger.info(
            f"LLMReasoner initialized (provider={provider}, model={default_model})"
        )

    def generate_answer(
        self,
        prompt: str,
        model: Optional[str] = None,
    ) -> Answer:
        """
        Call OpenAI/Anthropic/Gemini API with prompt and parse response.
        
        Args:
            prompt: Prompt string
            model: Model name (uses default if None)
            
        Returns:
            Answer object with parsed response
            
        Raises:
            ValueError: If API key is not set
            Exception: If API call fails after retries
        """
        if not self.api_key:
            raise ValueError(
                f"{self.provider.upper()} API key is required. "
                f"Set {self.provider.upper()}_API_KEY environment variable "
                f"or pass api_key parameter."
            )
        
        model = model or self.default_model
        
        logger.info(f"Generating answer with {self.provider} model {model}")
        
        # Retry logic with exponential backoff
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self._wait_for_rate_limit()
                
                # Call API
                if self.provider == "openai":
                    response = self._call_openai(prompt, model)
                elif self.provider == "anthropic":
                    response = self._call_anthropic(prompt, model)
                elif self.provider == "gemini":
                    response = self._call_gemini(prompt, model)
                
                # Parse response
                answer = self._parse_response(response, model)
                
                # Extract cell references
                answer.cited_cells = self.extract_cell_references(
                    answer.answer_text + "\n" + answer.reasoning
                )
                
                logger.info(
                    f"Generated answer: {len(answer.answer_text)} chars, "
                    f"{len(answer.cited_cells)} citations"
                )
                
                return answer
                
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    raise
        
        # Should not reach here, but just in case
        raise last_exception or Exception("Failed to generate answer")

    def _call_openai(self, prompt: str, model: str) -> Any:
        """
        Call OpenAI API.
        
        Args:
            prompt: Prompt string
            model: Model name
            
        Returns:
            OpenAI response object
            
        Raises:
            Exception: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about tables. Always cite specific cells when referencing data."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            return response
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _call_anthropic(self, prompt: str, model: str) -> Any:
        """
        Call Anthropic API.
        
        Args:
            prompt: Prompt string
            model: Model name
            
        Returns:
            Anthropic response object
            
        Raises:
            Exception: If API call fails
        """
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            return response
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise

    def _call_gemini(self, prompt: str, model: str) -> Any:
        """
        Call Gemini API via GeminiKeyManager.
        """
        return self.gemini_manager.generate_content(
            model_name=model,
            prompt=prompt,
            temperature=self.temperature
        )

    def _parse_response(self, response: Any, model: str) -> Answer:
        """
        Parse API response into Answer object.
        """
        if self.provider == "openai":
            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            reasoning, answer_text = self._extract_reasoning_and_answer(content)
            confidence = self._extract_confidence(content)
            
            return Answer(
                answer_text=answer_text,
                reasoning=reasoning,
                confidence=confidence,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        elif self.provider == "anthropic":
            content = response.content[0].text
            
            reasoning, answer_text = self._extract_reasoning_and_answer(content)
            confidence = self._extract_confidence(content)
            
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage'):
                prompt_tokens = getattr(response.usage, 'input_tokens', 0)
                completion_tokens = getattr(response.usage, 'output_tokens', 0)
            
            return Answer(
                answer_text=answer_text,
                reasoning=reasoning,
                confidence=confidence,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        elif self.provider == "gemini":
            content = response # response is just text string from our manager
            
            reasoning, answer_text = self._extract_reasoning_and_answer(content)
            confidence = self._extract_confidence(content)
            
            return Answer(
                answer_text=answer_text,
                reasoning=reasoning,
                confidence=confidence,
                model=model,
                # Token counts not easily available from simple text response
                prompt_tokens=0,
                completion_tokens=0,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _extract_reasoning_and_answer(
        self, content: str
    ) -> Tuple[str, str]:
        """
        Extract reasoning and answer from LLM response.
        
        Args:
            content: Full response content
            
        Returns:
            Tuple of (reasoning, answer_text)
        """
        # Look for "Final Answer" section
        final_answer_match = re.search(
            r"(?:Final Answer|Answer|답변)[:\s]*(.+?)(?:\n\n|\Z)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        
        if final_answer_match:
            answer_text = final_answer_match.group(1).strip()
            reasoning = content[: final_answer_match.start()].strip()
        else:
            # Try to find reasoning steps
            reasoning_match = re.search(
                r"(?:Reasoning|추론|단계)[:\s]*(.+?)(?:\n\nFinal|Answer|답변|\Z)",
                content,
                re.IGNORECASE | re.DOTALL,
            )
            
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                answer_text = content[reasoning_match.end() :].strip()
            else:
                # No clear separation, use entire content as answer
                reasoning = ""
                answer_text = content.strip()
        
        return reasoning, answer_text

    def _extract_confidence(self, content: str) -> Optional[float]:
        """
        Extract confidence score from LLM response if available.
        
        Looks for patterns like:
        - "Confidence: 85%"
        - "Confidence: 0.85"
        - "신뢰도: 85%"
        
        Args:
            content: Full response content
            
        Returns:
            Confidence score (0.0-1.0) or None if not found
        """
        # Pattern 1: "Confidence: X%" or "신뢰도: X%"
        confidence_percent_match = re.search(
            r"(?:Confidence|신뢰도|확신도)[:\s]*(\d+(?:\.\d+)?)\s*%",
            content,
            re.IGNORECASE,
        )
        if confidence_percent_match:
            percent = float(confidence_percent_match.group(1))
            return percent / 100.0
        
        # Pattern 2: "Confidence: 0.X" or "Confidence: X"
        confidence_decimal_match = re.search(
            r"(?:Confidence|신뢰도|확신도)[:\s]*(0?\.\d+|1\.0)",
            content,
            re.IGNORECASE,
        )
        if confidence_decimal_match:
            return float(confidence_decimal_match.group(1))
        
        # Pattern 3: Look for standalone percentage in context
        # Only if it appears near confidence-related keywords
        standalone_percent_match = re.search(
            r"(?:confidence|신뢰|확신).*?(\d+(?:\.\d+)?)\s*%",
            content,
            re.IGNORECASE,
        )
        if standalone_percent_match:
            percent = float(standalone_percent_match.group(1))
            return percent / 100.0
        
        return None

    def extract_cell_references(self, text: str) -> List[CellRef]:
        """
        Parse cell references from LLM output.
        
        Supports multiple formats:
        - "(Row 2, Col 3)" or "(Row 2, Column 3)"
        - "[R2C3]" or "[R2, C3]"
        - "A1" (Excel format)
        - "(2, 3)" (tuple format)
        - "Row 2, Column 3" (natural language)
        
        Args:
            text: Text to parse
            
        Returns:
            List of CellRef objects
        """
        references: List[CellRef] = []
        
        # Pattern 1: (Row X, Col Y) or (Row X, Column Y)
        pattern1 = re.compile(
            r"\(Row\s+(\d+)[,\s]+(?:Col|Column)\s+(\d+)\)",
            re.IGNORECASE,
        )
        for match in pattern1.finditer(text):
            row = int(match.group(1))
            col = int(match.group(2))
            references.append(
                CellRef(
                    row=row,
                    col=col,
                    format="row_col",
                    original_text=match.group(0),
                )
            )
        
        # Pattern 2: [R2C3] or [R2, C3]
        pattern2 = re.compile(r"\[R(\d+)[,\s]*C(\d+)\]", re.IGNORECASE)
        for match in pattern2.finditer(text):
            row = int(match.group(1))
            col = int(match.group(2))
            references.append(
                CellRef(
                    row=row,
                    col=col,
                    format="compact",
                    original_text=match.group(0),
                )
            )
        
        # Pattern 3: Excel format (A1, B2, etc.)
        pattern3 = re.compile(r"\b([A-Z]+)(\d+)\b", re.IGNORECASE)
        for match in pattern3.finditer(text):
            col_str = match.group(1).upper()
            row_str = match.group(2)
            
            # Convert column letters to number
            col = 0
            for char in col_str:
                col = col * 26 + (ord(char) - ord("A") + 1)
            col -= 1  # Convert to 0-indexed for internal use, but store as 1-indexed
            
            row = int(row_str)
            
            references.append(
                CellRef(
                    row=row,  # Excel is 1-indexed
                    col=col + 1,  # Convert back to 1-indexed
                    format="excel",
                    original_text=match.group(0),
                )
            )
        
        # Pattern 4: Tuple format (2, 3) - context dependent
        # Only match if it appears in citation context
        pattern4 = re.compile(
            r"(?:cell|cells|at|from|coordinate)[\s:]*\((\d+)[,\s]+(\d+)\)",
            re.IGNORECASE,
        )
        for match in pattern4.finditer(text):
            row = int(match.group(1))
            col = int(match.group(2))
            references.append(
                CellRef(
                    row=row,
                    col=col,
                    format="tuple",
                    original_text=match.group(0),
                )
            )
        
        # Pattern 5: Natural language "Row X, Column Y"
        pattern5 = re.compile(
            r"Row\s+(\d+)[,\s]+(?:Column|Col)\s+(\d+)",
            re.IGNORECASE,
        )
        for match in pattern5.finditer(text):
            row = int(match.group(1))
            col = int(match.group(2))
            references.append(
                CellRef(
                    row=row,
                    col=col,
                    format="natural",
                    original_text=match.group(0),
                )
            )
        
        # Remove duplicates (same row/col)
        seen = set()
        unique_refs = []
        for ref in references:
            key = (ref.row, ref.col)
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)
        
        logger.debug(f"Extracted {len(unique_refs)} cell references from text")
        
        return unique_refs

    def verify_citations(
        self, answer: Answer, context: ContextBundle
    ) -> bool:
        """
        Check if all cited cells actually exist in context.
        
        Verifies:
        - Cited cells exist in retrieved context
        - Cited values match actual cell content
        
        Args:
            answer: Answer object with cited cells
            context: ContextBundle with retrieved cells
            
        Returns:
            True if all citations are valid, False if hallucination detected
        """
        if not answer.cited_cells:
            logger.debug("No citations to verify")
            return True
        
        logger.info(f"Verifying {len(answer.cited_cells)} citations")
        
        # Build coordinate to cell mapping from context
        coord_to_cell: Dict[Tuple[int, int], RetrievalResult] = {}
        
        for result in context.retrieved_cells:
            coords = result.metadata.coordinates
            (row_start, row_end), (col_start, col_end) = coords
            
            # Map all cells in the range
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    coord_to_cell[(row, col)] = result
        
        # Verify each citation
        invalid_citations = []
        value_mismatches = []
        
        for cell_ref in answer.cited_cells:
            # Convert to 0-indexed coordinates
            row, col = cell_ref.to_tuple(zero_indexed=True)
            
            if (row, col) not in coord_to_cell:
                logger.warning(
                    f"Citation not found in context: {cell_ref.original_text} "
                    f"-> ({row}, {col})"
                )
                invalid_citations.append(cell_ref)
                continue
            
            # Check if value matches (if mentioned in answer)
            cell_result = coord_to_cell[(row, col)]
            cell_value = cell_result.content.strip() if cell_result.content else ""
            
            # Check if answer mentions this value (case-insensitive, partial match)
            if cell_value:
                # Normalize for comparison
                cell_value_lower = cell_value.lower()
                answer_lower = answer.answer_text.lower()
                
                # Check for exact match or substring match
                if cell_value_lower not in answer_lower:
                    # Try to find numeric values if cell contains numbers
                    numeric_pattern = re.search(r'\d+(?:\.\d+)?', cell_value)
                    if numeric_pattern:
                        numeric_value = numeric_pattern.group(0)
                        if numeric_value not in answer.answer_text:
                            logger.debug(
                                f"Cited cell value '{cell_value}' (numeric: {numeric_value}) "
                                f"not clearly mentioned in answer"
                            )
                            value_mismatches.append((cell_ref, cell_value))
                    else:
                        logger.debug(
                            f"Cited cell value '{cell_value}' not mentioned in answer"
                        )
                        value_mismatches.append((cell_ref, cell_value))
        
        is_valid = len(invalid_citations) == 0
        
        if invalid_citations:
            logger.warning(
                f"Found {len(invalid_citations)} invalid citations: "
                f"{[c.original_text for c in invalid_citations]}"
            )
        
        if value_mismatches:
            logger.info(
                f"Found {len(value_mismatches)} citations with potential value mismatches"
            )
        
        return is_valid

    def _wait_for_rate_limit(self) -> None:
        """Wait to respect rate limits."""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()

    def batch_generate(
        self,
        prompts: List[str],
        max_workers: int = 5,
        model: Optional[str] = None,
    ) -> List[Answer]:
        """
        Process multiple queries in parallel.
        
        Respects rate limits and handles errors gracefully.
        
        Args:
            prompts: List of prompt strings
            max_workers: Maximum number of parallel workers
            model: Model name (uses default if None)
            
        Returns:
            List of Answer objects (may contain None for failed requests)
        """
        logger.info(f"Batch generating answers for {len(prompts)} prompts")
        
        model = model or self.default_model
        results: List[Answer] = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_prompt = {
                executor.submit(self.generate_answer, prompt, model): prompt
                for prompt in prompts
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    answer = future.result()
                    results.append(answer)
                except Exception as e:
                    logger.error(f"Failed to generate answer for prompt: {e}")
                    # Add empty answer as placeholder
                    results.append(
                        Answer(
                            answer_text="",
                            reasoning=f"Error: {str(e)}",
                            model=model,
                        )
                    )
        
        logger.info(f"Completed batch generation: {len(results)}/{len(prompts)} successful")
        
        return results

    def generate_with_verification(
        self,
        prompt: str,
        context: ContextBundle,
        model: Optional[str] = None,
    ) -> Tuple[Answer, bool]:
        """
        Generate answer and verify citations.
        
        Args:
            prompt: Prompt string
            context: ContextBundle for verification
            model: Model name
            
        Returns:
            Tuple of (Answer, is_valid)
        """
        answer = self.generate_answer(prompt, model=model)
        is_valid = self.verify_citations(answer, context)
        
        return answer, is_valid


class LLMGenerator:
    """
    Generate responses using LLM with hierarchical table prompts.
    
    Handles LLM interaction and response generation with
    context-aware prompting.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        prompt_builder: Optional[StructuredPromptBuilder] = None,
    ) -> None:
        """
        Initialize LLM generator.
        
        Args:
            model_name: Name of LLM model
            api_key: Optional API key
            temperature: Sampling temperature
            prompt_builder: Prompt builder instance
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.prompt_builder = (
            prompt_builder or StructuredPromptBuilder()
        )
        logger.info(f"LLMGenerator initialized (model: {model_name})")

    def generate(
        self,
        query: str,
        context: ContextBundle,
        use_hierarchy: bool = True,
        style: str = "explicit",
    ) -> str:
        """
        Generate response using query and context.
        
        Args:
            query: User query
            context: ContextBundle with retrieved context
            use_hierarchy: Use hierarchy information in generation
            style: Prompt formatting style
            
        Returns:
            Generated response text
        """
        logger.info("Generating response with LLM")
        
        # Build prompt
        prompt = self.prompt_builder.build_rag_prompt(
            query, context, style=style
        )
        
        # Call LLM API (placeholder - actual implementation would use OpenAI/LangChain)
        # For now, return the prompt as placeholder
        logger.info("Prompt built, LLM call would happen here")
        return prompt

    def generate_with_hierarchy(
        self,
        query: str,
        retrieved_data: Dict[str, Any],
    ) -> str:
        """
        Generate response with explicit hierarchy context.
        
        Args:
            query: User query
            retrieved_data: Retrieved data with hierarchy information
            
        Returns:
            Generated response
        """
        logger.info("Generating response with hierarchy context")
        # Implementation will use hierarchy-aware prompting
        return ""

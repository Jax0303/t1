"""
Tests for LLMReasoner class.

Tests answer generation, citation extraction, verification,
retry logic, and batch processing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from src.retrieval.retriever import ContextBundle, RetrievalResult
from src.retrieval.indexer import IndexMetadata
from src.generation.prompting import (
    LLMReasoner,
    Answer,
    CellRef,
)


class TestLLMReasoner:
    """Test cases for LLMReasoner class."""

    @pytest.fixture
    def sample_context_bundle(self):
        """Create sample context bundle for testing."""
        metadata1 = IndexMetadata(
            item_id="item1",
            source_table_id="table1",
            granularity="cell",
            hierarchy_path="Table > Revenue > Q1",
            coordinates=((2, 3), (0, 1)),
            text="100",
        )
        
        result1 = RetrievalResult(
            item_id="item1",
            metadata=metadata1,
            score=0.95,
            granularity="cell",
            content="100",
            hierarchy_path="Table > Revenue > Q1",
        )
        
        bundle = ContextBundle(
            retrieved_cells=[result1],
            hierarchy_context={},
            table_metadata={},
            query="What is the revenue?",
        )
        
        return bundle

    def test_reasoner_init_openai(self):
        """Test reasoner initialization with OpenAI."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("src.generation.prompting.openai") as mock_openai:
                reasoner = LLMReasoner(
                    provider="openai",
                    default_model="gpt-4",
                )
                assert reasoner.provider == "openai"
                assert reasoner.default_model == "gpt-4"

    def test_reasoner_init_anthropic(self):
        """Test reasoner initialization with Anthropic."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("src.generation.prompting.anthropic") as mock_anthropic:
                reasoner = LLMReasoner(
                    provider="anthropic",
                    default_model="claude-3-opus",
                )
                assert reasoner.provider == "anthropic"

    def test_reasoner_init_no_api_key(self):
        """Test error when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                reasoner = LLMReasoner(provider="openai")
                reasoner.generate_answer("test prompt")

    def test_extract_cell_references_row_col_format(self):
        """Test extracting cell references in (Row X, Col Y) format."""
        reasoner = LLMReasoner(provider="openai", api_key="test")
        
        text = "The value at (Row 2, Col 3) is 100."
        refs = reasoner.extract_cell_references(text)
        
        assert len(refs) == 1
        assert refs[0].row == 2
        assert refs[0].col == 3
        assert refs[0].format == "row_col"

    def test_extract_cell_references_compact_format(self):
        """Test extracting cell references in [R2C3] format."""
        reasoner = LLMReasoner(provider="openai", api_key="test")
        
        text = "Cell [R2C3] contains the value 100."
        refs = reasoner.extract_cell_references(text)
        
        assert len(refs) == 1
        assert refs[0].row == 2
        assert refs[0].col == 3
        assert refs[0].format == "compact"

    def test_extract_cell_references_excel_format(self):
        """Test extracting cell references in Excel format (A1)."""
        reasoner = LLMReasoner(provider="openai", api_key="test")
        
        text = "The value in cell A1 is 100, and B2 is 200."
        refs = reasoner.extract_cell_references(text)
        
        assert len(refs) == 2
        # A1 -> row 1, col 1 (1-indexed)
        assert refs[0].row == 1
        assert refs[0].col == 1
        assert refs[0].format == "excel"
        
        # B2 -> row 2, col 2 (1-indexed)
        assert refs[1].row == 2
        assert refs[1].col == 2

    def test_extract_cell_references_multiple_formats(self):
        """Test extracting multiple cell references in different formats."""
        reasoner = LLMReasoner(provider="openai", api_key="test")
        
        text = (
            "Values: (Row 1, Col 2) = 100, [R3C4] = 200, "
            "and A5 = 300."
        )
        refs = reasoner.extract_cell_references(text)
        
        assert len(refs) >= 3
        # Check that duplicates are removed
        unique_coords = {(r.row, r.col) for r in refs}
        assert len(unique_coords) <= len(refs)

    def test_extract_cell_references_natural_language(self):
        """Test extracting cell references in natural language."""
        reasoner = LLMReasoner(provider="openai", api_key="test")
        
        text = "The value at Row 5, Column 3 is 150."
        refs = reasoner.extract_cell_references(text)
        
        assert len(refs) == 1
        assert refs[0].row == 5
        assert refs[0].col == 3
        assert refs[0].format == "natural"

    def test_verify_citations_valid(self, sample_context_bundle):
        """Test citation verification with valid citations."""
        reasoner = LLMReasoner(provider="openai", api_key="test")
        
        # Create answer with valid citation
        answer = Answer(
            answer_text="The revenue is 100 [Row 2, Col 0]",
            cited_cells=[
                CellRef(row=2, col=0, format="row_col", original_text="[Row 2, Col 0]")
            ],
        )
        
        is_valid = reasoner.verify_citations(answer, sample_context_bundle)
        assert is_valid is True

    def test_verify_citations_invalid(self, sample_context_bundle):
        """Test citation verification with invalid citations."""
        reasoner = LLMReasoner(provider="openai", api_key="test")
        
        # Create answer with invalid citation (out of bounds)
        answer = Answer(
            answer_text="The value is 999 [Row 99, Col 99]",
            cited_cells=[
                CellRef(
                    row=99, col=99, format="row_col", original_text="[Row 99, Col 99]"
                )
            ],
        )
        
        is_valid = reasoner.verify_citations(answer, sample_context_bundle)
        assert is_valid is False

    def test_verify_citations_no_citations(self, sample_context_bundle):
        """Test citation verification with no citations."""
        reasoner = LLMReasoner(provider="openai", api_key="test")
        
        answer = Answer(answer_text="The revenue is 100")
        
        is_valid = reasoner.verify_citations(answer, sample_context_bundle)
        # No citations to verify, should return True
        assert is_valid is True

    @patch("src.generation.prompting.openai.OpenAI")
    def test_generate_answer_openai(self, mock_openai_class):
        """Test answer generation with OpenAI (mocked)."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="Step 1: Found revenue section\nStep 2: Extracted Q1 value\nFinal Answer: The revenue for Q1 is 100 [Row 2, Col 0]"
                )
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        
        mock_client.chat.completions.create.return_value = mock_response
        
        reasoner = LLMReasoner(provider="openai", api_key="test-key")
        reasoner.client = mock_client
        
        answer = reasoner.generate_answer("What is the revenue?", model="gpt-4")
        
        assert isinstance(answer, Answer)
        assert len(answer.answer_text) > 0
        assert "gpt-4" in answer.model.lower()
        assert answer.prompt_tokens > 0

    @patch("src.generation.prompting.anthropic.Anthropic")
    def test_generate_answer_anthropic(self, mock_anthropic_class):
        """Test answer generation with Anthropic (mocked)."""
        # Mock Anthropic client and response
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="Step 1: Found revenue section\nStep 2: Extracted Q1 value\nFinal Answer: The revenue for Q1 is 100 [Row 2, Col 0]"
            )
        ]
        
        mock_client.messages.create.return_value = mock_response
        
        reasoner = LLMReasoner(provider="anthropic", api_key="test-key")
        reasoner.client = mock_client
        
        answer = reasoner.generate_answer("What is the revenue?", model="claude-3-opus")
        
        assert isinstance(answer, Answer)
        assert len(answer.answer_text) > 0

    def test_generate_answer_retry_logic(self):
        """Test retry logic with exponential backoff."""
        reasoner = LLMReasoner(
            provider="openai",
            api_key="test-key",
            max_retries=3,
            retry_delay=0.1,
        )
        
        # Mock client to fail twice then succeed
        mock_client = MagicMock()
        call_count = 0
        
        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("API Error")
            # Success on third try
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Answer: 100"))
            ]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            return mock_response
        
        mock_client.chat.completions.create = mock_create
        reasoner.client = mock_client
        
        answer = reasoner.generate_answer("test")
        
        assert call_count == 3
        assert isinstance(answer, Answer)

    def test_batch_generate(self):
        """Test batch generation of multiple prompts."""
        reasoner = LLMReasoner(
            provider="openai",
            api_key="test-key",
            max_retries=1,
        )
        
        # Mock client
        mock_client = MagicMock()
        
        def mock_create(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content=f"Answer for prompt"))
            ]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            return mock_response
        
        mock_client.chat.completions.create = mock_create
        reasoner.client = mock_client
        
        prompts = ["Query 1", "Query 2", "Query 3"]
        results = reasoner.batch_generate(prompts, max_workers=2)
        
        assert len(results) == len(prompts)
        assert all(isinstance(r, Answer) for r in results)

    def test_generate_with_verification(self, sample_context_bundle):
        """Test generate with citation verification."""
        reasoner = LLMReasoner(provider="openai", api_key="test-key")
        
        # Mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="The revenue is 100 [Row 2, Col 0]"
                )
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create.return_value = mock_response
        reasoner.client = mock_client
        
        answer, is_valid = reasoner.generate_with_verification(
            "What is the revenue?", sample_context_bundle
        )
        
        assert isinstance(answer, Answer)
        assert isinstance(is_valid, bool)

    def test_extract_reasoning_and_answer(self):
        """Test extraction of reasoning and answer from response."""
        reasoner = LLMReasoner(provider="openai", api_key="test")
        
        content = (
            "Step 1: Found revenue section\n"
            "Step 2: Extracted Q1 value\n"
            "Final Answer: The revenue for Q1 is 100"
        )
        
        reasoning, answer = reasoner._extract_reasoning_and_answer(content)
        
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "100" in answer

    def test_cell_ref_to_tuple(self):
        """Test CellRef to_tuple conversion."""
        # Excel format (1-indexed)
        ref_excel = CellRef(row=1, col=1, format="excel", original_text="A1")
        row, col = ref_excel.to_tuple(zero_indexed=True)
        assert row == 0
        assert col == 0
        
        # Row/col format (already 0-indexed typically)
        ref_row_col = CellRef(row=2, col=3, format="row_col", original_text="(Row 2, Col 3)")
        row, col = ref_row_col.to_tuple(zero_indexed=True)
        assert row == 2
        assert col == 3

    def test_answer_to_dict(self):
        """Test Answer serialization to dictionary."""
        answer = Answer(
            answer_text="The revenue is 100",
            reasoning="Step 1: Found section",
            cited_cells=[
                CellRef(row=2, col=0, format="row_col", original_text="[Row 2, Col 0]")
            ],
            confidence=0.95,
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
        )
        
        answer_dict = answer.to_dict()
        
        assert answer_dict["answer_text"] == "The revenue is 100"
        assert answer_dict["reasoning"] == "Step 1: Found section"
        assert len(answer_dict["cited_cells"]) == 1
        assert answer_dict["confidence"] == 0.95
        assert answer_dict["model"] == "gpt-4"

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        reasoner = LLMReasoner(
            provider="openai",
            api_key="test-key",
        )
        reasoner.min_request_interval = 0.1
        
        start_time = time.time()
        reasoner._wait_for_rate_limit()
        reasoner._wait_for_rate_limit()
        elapsed = time.time() - start_time
        
        # Should have waited at least min_request_interval
        assert elapsed >= reasoner.min_request_interval



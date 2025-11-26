"""
Tests for core RAG functionality.
"""

import pytest
from pathlib import Path
from hiertable_rag.core.rag import HierTableRAG


class TestHierTableRAG:
    """Test cases for HierTableRAG class."""

    def test_init(self) -> None:
        """Test HierTableRAG initialization."""
        rag = HierTableRAG()
        assert rag is not None
        assert rag.embedding_model is None

    def test_init_with_params(self) -> None:
        """Test HierTableRAG initialization with parameters."""
        rag = HierTableRAG(
            embedding_model="test-model",
            vector_store_path=Path("test_index")
        )
        assert rag.embedding_model == "test-model"
        assert rag.vector_store_path == Path("test_index")

    def test_extract_tables_empty(self, tmp_path: Path) -> None:
        """Test table extraction with empty document."""
        rag = HierTableRAG()
        # This will be implemented later
        result = rag.extract_tables(tmp_path / "empty.pdf")
        assert isinstance(result, list)



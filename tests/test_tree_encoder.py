"""
Tests for TreeEncoder class.

Tests multiple serialization formats and round-trip encoding/decoding.
"""

import pytest
from typing import List

from src.parsing.extractor import TableObject, Cell
from src.parsing.hierarchy import HierarchyExtractor, HeaderTree, HeaderNode
from src.encoding.tree_encoder import TreeEncoder


class TestTreeEncoder:
    """Test cases for TreeEncoder class."""

    def test_encoder_init(self) -> None:
        """Test encoder initialization."""
        encoder = TreeEncoder()
        assert encoder.indent_size == 2
        assert encoder.include_metadata is True

    def test_to_indented_text_simple(self) -> None:
        """Test indented text format with simple tree."""
        encoder = TreeEncoder()
        
        # Create simple tree
        root = HeaderNode(text="Root", level=0, span=4, col_start=0, col_end=4)
        cat_a = HeaderNode(text="Category A", level=1, span=2, col_start=0, col_end=2)
        cat_b = HeaderNode(text="Category B", level=1, span=2, col_start=2, col_end=4)
        sub_a1 = HeaderNode(text="Subcategory A1", level=2, span=1, col_start=0, col_end=1)
        sub_a2 = HeaderNode(text="Subcategory A2", level=2, span=1, col_start=1, col_end=2)
        
        root.add_child(cat_a)
        root.add_child(cat_b)
        cat_a.add_child(sub_a1)
        cat_a.add_child(sub_a2)
        
        tree = HeaderTree(root=root)
        text = encoder.to_indented_text(tree)
        
        assert isinstance(text, str)
        assert "Root" in text
        assert "Category A" in text
        assert "Category B" in text
        assert "Subcategory A1" in text
        assert "Subcategory A2" in text
        assert "(span=2)" in text or "span=2" in text

    def test_to_indented_text_custom_indent(self) -> None:
        """Test indented text with custom indent size."""
        encoder = TreeEncoder(indent_size=4)
        
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=1, span=1, col_start=0, col_end=1)
        root.add_child(child)
        
        tree = HeaderTree(root=root)
        text = encoder.to_indented_text(tree)
        
        # Check that indentation uses 4 spaces
        lines = text.split("\n")
        if len(lines) > 1:
            assert lines[1].startswith("    ")  # 4 spaces

    def test_to_json_ld_simple(self) -> None:
        """Test JSON-LD format conversion."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=1, span=1, col_start=0, col_end=1)
        root.add_child(child)
        
        tree = HeaderTree(root=root)
        json_ld = encoder.to_json_ld(tree)
        
        assert isinstance(json_ld, dict)
        assert "@context" in json_ld
        assert "@type" in json_ld
        assert "nodes" in json_ld
        assert isinstance(json_ld["nodes"], list)
        assert len(json_ld["nodes"]) == 2  # Root and child
        
        # Check node structure
        node = json_ld["nodes"][0]
        assert "label" in node
        assert "level" in node
        assert "span" in node
        assert "children" in node

    def test_to_json_ld_complex(self) -> None:
        """Test JSON-LD format with complex tree."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=4, col_start=0, col_end=4)
        cat_a = HeaderNode(text="Category A", level=1, span=2, col_start=0, col_end=2)
        cat_b = HeaderNode(text="Category B", level=1, span=2, col_start=2, col_end=4)
        sub_a1 = HeaderNode(text="Sub A1", level=2, span=1, col_start=0, col_end=1)
        sub_a2 = HeaderNode(text="Sub A2", level=2, span=1, col_start=1, col_end=2)
        
        root.add_child(cat_a)
        root.add_child(cat_b)
        cat_a.add_child(sub_a1)
        cat_a.add_child(sub_a2)
        
        tree = HeaderTree(root=root)
        json_ld = encoder.to_json_ld(tree)
        
        assert len(json_ld["nodes"]) == 4
        assert json_ld["max_level"] == 2
        assert json_ld["num_columns"] == 2

    def test_to_llm_prompt_format_natural(self) -> None:
        """Test natural language format."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        cat_a = HeaderNode(text="Category A", level=1, span=1, col_start=0, col_end=1)
        cat_b = HeaderNode(text="Category B", level=1, span=1, col_start=1, col_end=2)
        
        root.add_child(cat_a)
        root.add_child(cat_b)
        
        tree = HeaderTree(root=root)
        text = encoder.to_llm_prompt_format(tree, style="natural")
        
        assert isinstance(text, str)
        assert "table" in text.lower() or "Root" in text
        assert "Category A" in text or "category" in text.lower()

    def test_to_llm_prompt_format_structured(self) -> None:
        """Test structured markdown format."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=1, span=1, col_start=0, col_end=1)
        root.add_child(child)
        
        tree = HeaderTree(root=root)
        text = encoder.to_llm_prompt_format(tree, style="structured")
        
        assert isinstance(text, str)
        assert "#" in text  # Markdown header
        assert "Root" in text
        assert "Child" in text

    def test_to_llm_prompt_format_code(self) -> None:
        """Test code format."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=1, span=1, col_start=0, col_end=1)
        root.add_child(child)
        
        tree = HeaderTree(root=root)
        text = encoder.to_llm_prompt_format(tree, style="code")
        
        assert isinstance(text, str)
        assert "text" in text
        assert "Root" in text
        assert "Child" in text
        assert "level" in text
        assert "span" in text

    def test_to_llm_prompt_format_invalid_style(self) -> None:
        """Test error handling for invalid style."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=1, col_start=0, col_end=1)
        tree = HeaderTree(root=root)
        
        with pytest.raises(ValueError, match="Unsupported style"):
            encoder.to_llm_prompt_format(tree, style="invalid")

    def test_encode_for_embedding_with_tokens(self) -> None:
        """Test embedding encoding with special tokens."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=1, span=1, col_start=0, col_end=1)
        root.add_child(child)
        
        tree = HeaderTree(root=root)
        text = encoder.encode_for_embedding(tree, include_tokens=True)
        
        assert isinstance(text, str)
        assert "[LEVEL_0]" in text or "[LEVEL_1]" in text
        assert "[PARENT]" in text or "[CHILD]" in text
        assert "Root" in text
        assert "Child" in text

    def test_encode_for_embedding_without_tokens(self) -> None:
        """Test embedding encoding without special tokens."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=1, span=1, col_start=0, col_end=1)
        root.add_child(child)
        
        tree = HeaderTree(root=root)
        text = encoder.encode_for_embedding(tree, include_tokens=False)
        
        assert isinstance(text, str)
        assert "[LEVEL" not in text
        assert "[PARENT]" not in text
        assert "Root" in text
        assert "Child" in text

    def test_decode_from_indented_text(self) -> None:
        """Test decoding from indented text."""
        encoder = TreeEncoder()
        
        # Create original tree
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=1, span=1, col_start=0, col_end=1)
        root.add_child(child)
        
        original_tree = HeaderTree(root=root)
        
        # Encode
        encoded = encoder.to_indented_text(original_tree)
        
        # Decode
        decoded_tree = encoder.decode_from_indented_text(encoded)
        
        assert decoded_tree is not None
        assert decoded_tree.root.text == "Root"
        assert len(decoded_tree.root.children) == 1
        assert decoded_tree.root.children[0].text == "Child"

    def test_round_trip_test_success(self) -> None:
        """Test successful round-trip encoding/decoding."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        cat_a = HeaderNode(text="Category A", level=1, span=1, col_start=0, col_end=1)
        cat_b = HeaderNode(text="Category B", level=1, span=1, col_start=1, col_end=2)
        
        root.add_child(cat_a)
        root.add_child(cat_b)
        
        tree = HeaderTree(root=root)
        
        # Test round-trip
        success = encoder.round_trip_test(tree, format_type="indented")
        assert success is True

    def test_round_trip_test_complex(self) -> None:
        """Test round-trip with complex tree."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=4, col_start=0, col_end=4)
        cat_a = HeaderNode(text="Category A", level=1, span=2, col_start=0, col_end=2)
        cat_b = HeaderNode(text="Category B", level=1, span=2, col_start=2, col_end=4)
        sub_a1 = HeaderNode(text="Sub A1", level=2, span=1, col_start=0, col_end=1)
        sub_a2 = HeaderNode(text="Sub A2", level=2, span=1, col_start=1, col_end=2)
        sub_b1 = HeaderNode(text="Sub B1", level=2, span=1, col_start=2, col_end=3)
        sub_b2 = HeaderNode(text="Sub B2", level=2, span=1, col_start=3, col_end=4)
        
        root.add_child(cat_a)
        root.add_child(cat_b)
        cat_a.add_child(sub_a1)
        cat_a.add_child(sub_a2)
        cat_b.add_child(sub_b1)
        cat_b.add_child(sub_b2)
        
        tree = HeaderTree(root=root)
        
        # Test round-trip
        success = encoder.round_trip_test(tree, format_type="indented")
        assert success is True

    def test_encode_with_max_depth(self) -> None:
        """Test encoding with max depth limit."""
        encoder = TreeEncoder(max_depth=1)
        
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=1, span=1, col_start=0, col_end=1)
        grandchild = HeaderNode(text="Grandchild", level=2, span=1, col_start=0, col_end=1)
        
        root.add_child(child)
        child.add_child(grandchild)
        
        tree = HeaderTree(root=root)
        text = encoder.to_indented_text(tree)
        
        # Grandchild should not appear (max_depth=1)
        assert "Grandchild" not in text
        assert "Child" in text

    def test_json_ld_serialization(self) -> None:
        """Test that JSON-LD can be serialized to JSON."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=2, col_start=0, col_end=2)
        child = HeaderNode(text="Child", level=1, span=1, col_start=0, col_end=1)
        root.add_child(child)
        
        tree = HeaderTree(root=root)
        json_ld = encoder.to_json_ld(tree)
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(json_ld, ensure_ascii=False)
        assert isinstance(json_str, str)
        
        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed["@type"] == "table"

    def test_embedding_encoding_preserves_hierarchy(self) -> None:
        """Test that embedding encoding preserves hierarchy information."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=3, col_start=0, col_end=3)
        cat_a = HeaderNode(text="Category A", level=1, span=2, col_start=0, col_end=2)
        cat_b = HeaderNode(text="Category B", level=1, span=1, col_start=2, col_end=3)
        sub_a1 = HeaderNode(text="Sub A1", level=2, span=1, col_start=0, col_end=1)
        sub_a2 = HeaderNode(text="Sub A2", level=2, span=1, col_start=1, col_end=2)
        
        root.add_child(cat_a)
        root.add_child(cat_b)
        cat_a.add_child(sub_a1)
        cat_a.add_child(sub_a2)
        
        tree = HeaderTree(root=root)
        text = encoder.encode_for_embedding(tree, include_tokens=True)
        
        # Should contain all nodes
        assert "Root" in text
        assert "Category A" in text
        assert "Category B" in text
        assert "Sub A1" in text
        assert "Sub A2" in text
        
        # Should contain level information
        assert "[LEVEL_0]" in text or "[LEVEL_1]" in text or "[LEVEL_2]" in text
        
        # Should contain structure information
        assert "[MAX_LEVEL=" in text
        assert "[NUM_COLUMNS=" in text

    def test_indented_text_with_spans(self) -> None:
        """Test indented text includes span information."""
        encoder = TreeEncoder()
        
        root = HeaderNode(text="Root", level=0, span=4, col_start=0, col_end=4)
        cat_a = HeaderNode(text="Category A", level=1, span=3, col_start=0, col_end=3)
        cat_b = HeaderNode(text="Category B", level=1, span=1, col_start=3, col_end=4)
        
        root.add_child(cat_a)
        root.add_child(cat_b)
        
        tree = HeaderTree(root=root)
        text = encoder.to_indented_text(tree)
        
        # Should show span for Category A (span > 1)
        assert "span=3" in text or "(span=3)" in text



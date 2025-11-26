"""
Tree-based encoding for hierarchical table structures.

Provides multiple serialization strategies for HeaderTree objects including
indented text, JSON-LD, LLM prompt formats, and embedding-friendly encodings.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import logging

from src.parsing.hierarchy import HeaderTree, HeaderNode

logger = logging.getLogger(__name__)


class TreeEncoder:
    """
    Encode hierarchical table structures as trees with multiple formats.
    
    Supports various serialization strategies:
    - Indented text format
    - JSON-LD format
    - LLM-friendly prompt formats
    - Embedding-friendly text encodings
    """

    def __init__(
        self,
        indent_size: int = 2,
        include_metadata: bool = True,
        max_depth: Optional[int] = None,
    ) -> None:
        """
        Initialize tree encoder.
        
        Args:
            indent_size: Number of spaces for indentation (default: 2)
            include_metadata: Include table metadata in encoding
            max_depth: Maximum tree depth to encode (None for unlimited)
        """
        self.indent_size = indent_size
        self.include_metadata = include_metadata
        self.max_depth = max_depth
        logger.info("TreeEncoder initialized")

    def to_indented_text(
        self, header_tree: HeaderTree, indent_size: Optional[int] = None
    ) -> str:
        """
        Convert tree to indented text format.
        
        Example output:
        ```
        Root
          - Category A (span=2)
            - Subcategory A1 (span=1)
            - Subcategory A2 (span=1)
          - Category B (span=2)
        ```
        
        Args:
            header_tree: HeaderTree to convert
            indent_size: Override default indent size
            
        Returns:
            Indented text representation
        """
        indent = indent_size if indent_size is not None else self.indent_size
        lines: List[str] = []
        
        self._to_indented_text_recursive(
            header_tree.root, lines, 0, indent, self.max_depth
        )
        
        return "\n".join(lines)

    def _to_indented_text_recursive(
        self,
        node: HeaderNode,
        lines: List[str],
        current_depth: int,
        indent_size: int,
        max_depth: Optional[int],
    ) -> None:
        """
        Recursively convert node to indented text.
        
        Args:
            node: Current HeaderNode
            lines: List to append lines to
            current_depth: Current depth in tree
            indent_size: Size of indentation
            max_depth: Maximum depth to traverse
        """
        if max_depth is not None and current_depth > max_depth:
            return
        
        # Create indentation
        indent_str = " " * (current_depth * indent_size)
        
        # Format node text with span information
        if node.span > 1:
            node_text = f"{node.text} (span={node.span})"
        else:
            node_text = node.text
        
        # Add bullet point for non-root nodes
        if current_depth > 0:
            node_text = f"- {node_text}"
        
        lines.append(f"{indent_str}{node_text}")
        
        # Process children
        for child in node.children:
            self._to_indented_text_recursive(
                child, lines, current_depth + 1, indent_size, max_depth
            )

    def to_json_ld(self, header_tree: HeaderTree) -> Dict[str, Any]:
        """
        Convert to JSON-LD format with explicit parent-child relationships.
        
        Schema includes:
        - @context for semantic meaning
        - nodes with id, label, children, level, span
        
        Args:
            header_tree: HeaderTree to convert
            
        Returns:
            JSON-LD dictionary representation
        """
        nodes: List[Dict[str, Any]] = []
        node_id_map: Dict[HeaderNode, str] = {}
        
        # Assign IDs to all nodes
        self._assign_node_ids(header_tree.root, node_id_map, 0)
        
        # Build node list
        self._build_json_ld_nodes(
            header_tree.root, nodes, node_id_map, self.max_depth
        )
        
        # Build JSON-LD structure
        json_ld = {
            "@context": {
                "@vocab": "https://schema.org/",
                "table": "https://schema.org/Table",
                "header": "https://schema.org/TableHeader",
                "level": "https://schema.org/position",
                "span": "https://schema.org/width",
            },
            "@type": "table",
            "nodes": nodes,
            "max_level": header_tree.max_level,
            "num_columns": header_tree.num_columns,
        }
        
        if self.include_metadata and header_tree.root.cell_ref:
            json_ld["metadata"] = {
                "col_start": header_tree.root.col_start,
                "col_end": header_tree.root.col_end,
                "row_start": header_tree.root.row_start,
                "row_end": header_tree.root.row_end,
            }
        
        return json_ld

    def _assign_node_ids(
        self, node: HeaderNode, node_id_map: Dict[HeaderNode, str], counter: int
    ) -> int:
        """
        Assign unique IDs to nodes.
        
        Args:
            node: Current node
            node_id_map: Mapping from node to ID
            counter: Current counter value
            
        Returns:
            Updated counter value
        """
        node_id = f"node_{counter}"
        node_id_map[node] = node_id
        counter += 1
        
        for child in node.children:
            counter = self._assign_node_ids(child, node_id_map, counter)
        
        return counter

    def _build_json_ld_nodes(
        self,
        node: HeaderNode,
        nodes: List[Dict[str, Any]],
        node_id_map: Dict[HeaderNode, str],
        max_depth: Optional[int],
        current_depth: int = 0,
    ) -> None:
        """
        Build JSON-LD node list recursively.
        
        Args:
            node: Current HeaderNode
            nodes: List to append nodes to
            node_id_map: Mapping from node to ID
            max_depth: Maximum depth to traverse
            current_depth: Current depth
        """
        if max_depth is not None and current_depth > max_depth:
            return
        
        node_id = node_id_map[node]
        children_ids = [node_id_map[child] for child in node.children]
        
        node_dict = {
            "@id": node_id,
            "@type": "header",
            "label": node.text,
            "level": node.level,
            "span": node.span,
            "row_span": node.row_span,
            "col_start": node.col_start,
            "col_end": node.col_end,
            "children": children_ids,
        }
        
        if node.parent is not None:
            node_dict["parent"] = node_id_map[node.parent]
        
        nodes.append(node_dict)
        
        # Process children
        for child in node.children:
            self._build_json_ld_nodes(
                child, nodes, node_id_map, max_depth, current_depth + 1
            )

    def to_llm_prompt_format(
        self, header_tree: HeaderTree, style: str = "natural"
    ) -> str:
        """
        Generate LLM-friendly representations.
        
        Styles:
        - "natural": Natural language description
        - "structured": Markdown nested lists
        - "code": Python dict-like representation
        
        Args:
            header_tree: HeaderTree to convert
            style: Output style ("natural", "structured", "code")
            
        Returns:
            LLM-friendly text representation
            
        Raises:
            ValueError: If style is not supported
        """
        if style == "natural":
            return self._to_natural_language(header_tree)
        elif style == "structured":
            return self._to_structured_markdown(header_tree)
        elif style == "code":
            return self._to_code_format(header_tree)
        else:
            raise ValueError(
                f"Unsupported style: {style}. "
                f"Supported styles: natural, structured, code"
            )

    def _to_natural_language(self, header_tree: HeaderTree) -> str:
        """
        Convert to natural language description.
        
        Example: "The table has Category A with subcategories A1 and A2..."
        
        Args:
            header_tree: HeaderTree to convert
            
        Returns:
            Natural language description
        """
        parts: List[str] = []
        
        def describe_node(node: HeaderNode, is_root: bool = False) -> str:
            """Recursively describe node."""
            if is_root:
                desc = f"The table has {len(node.children)} main categories"
            else:
                desc = node.text
                if node.span > 1:
                    desc += f" (spanning {node.span} columns)"
            
            if node.children:
                child_descs = []
                for child in node.children:
                    child_descs.append(describe_node(child))
                
                if len(child_descs) == 1:
                    desc += f" with subcategory {child_descs[0]}"
                elif len(child_descs) == 2:
                    desc += f" with subcategories {child_descs[0]} and {child_descs[1]}"
                elif len(child_descs) > 2:
                    desc += (
                        f" with subcategories {', '.join(child_descs[:-1])}, "
                        f"and {child_descs[-1]}"
                    )
            
            return desc
        
        description = describe_node(header_tree.root, is_root=True)
        parts.append(description)
        
        # Add details about structure
        parts.append(
            f"The table has {header_tree.max_level} levels of hierarchy "
            f"with {header_tree.num_columns} data columns."
        )
        
        return ". ".join(parts) + "."

    def _to_structured_markdown(self, header_tree: HeaderTree) -> str:
        """
        Convert to markdown nested lists.
        
        Args:
            header_tree: HeaderTree to convert
            
        Returns:
            Markdown formatted text
        """
        lines: List[str] = []
        lines.append("# Table Header Structure\n")
        
        def add_markdown_node(node: HeaderNode, depth: int = 0) -> None:
            """Recursively add markdown node."""
            indent = "  " * depth
            marker = "-" if depth > 0 else "#"
            
            if depth == 0:
                lines.append(f"{marker} {node.text}")
            else:
                text = node.text
                if node.span > 1:
                    text += f" (span={node.span})"
                lines.append(f"{indent}{marker} {text}")
            
            for child in node.children:
                add_markdown_node(child, depth + 1)
        
        add_markdown_node(header_tree.root)
        
        return "\n".join(lines)

    def _to_code_format(self, header_tree: HeaderTree) -> str:
        """
        Convert to Python dict-like representation.
        
        Args:
            header_tree: HeaderTree to convert
            
        Returns:
            Code-formatted string
        """
        def node_to_dict(node: HeaderNode) -> Dict[str, Any]:
            """Convert node to dictionary."""
            node_dict = {
                "text": node.text,
                "level": node.level,
                "span": node.span,
            }
            
            if node.children:
                node_dict["children"] = [
                    node_to_dict(child) for child in node.children
                ]
            
            return node_dict
        
        tree_dict = node_to_dict(header_tree.root)
        
        # Format as Python dict string
        return json.dumps(tree_dict, indent=2, ensure_ascii=False)

    def encode_for_embedding(
        self, header_tree: HeaderTree, include_tokens: bool = True
    ) -> str:
        """
        Create embedding-friendly text that preserves semantic hierarchy.
        
        Includes both structure and content with special tokens like
        [PARENT], [CHILD], [LEVEL_1], [LEVEL_2] for better embedding quality.
        
        Args:
            header_tree: HeaderTree to encode
            include_tokens: Whether to include special tokens
            
        Returns:
            Embedding-friendly text string
        """
        segments: List[str] = []
        
        def encode_node(node: HeaderNode, level: int = 0) -> None:
            """Recursively encode node."""
            # Add level token
            if include_tokens:
                level_token = f"[LEVEL_{level}]"
                segments.append(level_token)
            
            # Add node text
            segments.append(node.text)
            
            # Add span information
            if node.span > 1:
                segments.append(f"(span={node.span})")
            
            # Add parent token if not root
            if include_tokens and node.parent is not None:
                segments.append("[CHILD]")
                segments.append(node.parent.text)
                segments.append("[PARENT]")
            
            # Process children
            if include_tokens and node.children:
                segments.append("[PARENT]")
            
            for child in node.children:
                encode_node(child, level + 1)
            
            if include_tokens and node.children:
                segments.append("[/PARENT]")
        
        encode_node(header_tree.root)
        
        # Add metadata
        if self.include_metadata:
            segments.append(f"[MAX_LEVEL={header_tree.max_level}]")
            segments.append(f"[NUM_COLUMNS={header_tree.num_columns}]")
        
        return " ".join(segments)

    def decode_from_indented_text(self, text: str) -> Optional[HeaderTree]:
        """
        Decode HeaderTree from indented text format.
        
        This is a basic implementation - may not handle all edge cases.
        
        Args:
            text: Indented text representation
            
        Returns:
            HeaderTree or None if parsing fails
        """
        lines = text.strip().split("\n")
        if not lines:
            return None
        
        # Parse root
        root_line = lines[0].strip()
        if root_line.startswith("-"):
            root_line = root_line[1:].strip()
        
        # Extract root text and span
        root_text, root_span = self._parse_node_text(root_line)
        
        root = HeaderNode(
            text=root_text,
            level=0,
            span=root_span,
            col_start=0,
            col_end=root_span,
        )
        
        # Parse children recursively
        self._parse_indented_children(lines, 1, root, 0)
        
        return HeaderTree(root=root)

    def _parse_node_text(self, text: str) -> Tuple[str, int]:
        """
        Parse node text and extract span information.
        
        Args:
            text: Node text line
            
        Returns:
            Tuple of (text, span)
        """
        # Remove leading dash if present
        if text.startswith("-"):
            text = text[1:].strip()
        
        # Check for span in parentheses
        span_match = None
        if "(span=" in text:
            import re
            span_match = re.search(r"\(span=(\d+)\)", text)
            if span_match:
                span = int(span_match.group(1))
                text = re.sub(r"\s*\(span=\d+\)", "", text).strip()
                return (text, span)
        
        return (text, 1)

    def _parse_indented_children(
        self,
        lines: List[str],
        start_idx: int,
        parent: HeaderNode,
        parent_depth: int,
    ) -> int:
        """
        Parse indented children recursively.
        
        Args:
            lines: All lines to parse
            start_idx: Starting index in lines
            parent: Parent HeaderNode
            parent_depth: Depth of parent node
            
        Returns:
            Next index to process
        """
        idx = start_idx
        col_counter = parent.col_start
        
        while idx < len(lines):
            line = lines[idx]
            
            # Calculate depth
            stripped = line.lstrip()
            depth = (len(line) - len(stripped)) // self.indent_size
            
            # If depth is not greater than parent, we're done with this level
            if depth <= parent_depth:
                break
            
            # This is a child
            if depth == parent_depth + 1:
                node_text, node_span = self._parse_node_text(stripped)
                
                node = HeaderNode(
                    text=node_text,
                    level=parent.level + 1,
                    span=node_span,
                    col_start=col_counter,
                    col_end=col_counter + node_span,
                )
                parent.add_child(node)
                col_counter += node_span
                
                # Check for grandchildren
                idx += 1
                if idx < len(lines):
                    next_line = lines[idx]
                    next_stripped = next_line.lstrip()
                    next_depth = (len(next_line) - len(next_stripped)) // self.indent_size
                    
                    if next_depth > depth:
                        idx = self._parse_indented_children(
                            lines, idx, node, depth
                        )
                        continue
                
                idx += 1
            else:
                idx += 1
        
        return idx

    def round_trip_test(
        self, header_tree: HeaderTree, format_type: str = "indented"
    ) -> bool:
        """
        Test round-trip encoding/decoding.
        
        Args:
            header_tree: Original HeaderTree
            format_type: Format to test ("indented" only supported currently)
            
        Returns:
            True if round-trip successful, False otherwise
        """
        if format_type == "indented":
            # Encode
            encoded = self.to_indented_text(header_tree)
            
            # Decode
            decoded = self.decode_from_indented_text(encoded)
            
            if decoded is None:
                return False
            
            # Compare structure (basic comparison)
            return self._compare_trees(header_tree.root, decoded.root)
        else:
            logger.warning(f"Round-trip test not supported for format: {format_type}")
            return False

    def _compare_trees(self, node1: HeaderNode, node2: HeaderNode) -> bool:
        """
        Compare two tree structures.
        
        Args:
            node1: First HeaderNode
            node2: Second HeaderNode
            
        Returns:
            True if trees are equivalent
        """
        # Compare node properties
        if node1.text != node2.text:
            return False
        
        if node1.span != node2.span:
            return False
        
        if len(node1.children) != len(node2.children):
            return False
        
        # Compare children recursively
        for child1, child2 in zip(node1.children, node2.children):
            if not self._compare_trees(child1, child2):
                return False
        
        return True

from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class SchemaNode:
    """Represents a header or hierarchical structure element."""
    id: str
    text: str
    parent_id: Optional[str] = None
    level: int = 0
    children_ids: List[str] = field(default_factory=list)

@dataclass
class CellNode:
    """Represents a data cell with value and its structural context."""
    id: str
    text: str
    row_header_ids: List[str] = field(default_factory=list)
    col_header_ids: List[str] = field(default_factory=list)
    table_id: str = ""

class TableGraph:
    """
    Constructs and manages a graph representation of a table.
    Includes SchemaNodes (headers) and CellNodes (values).
    """
    
    def __init__(self, table_id: str):
        self.table_id = table_id
        self.schema_nodes: Dict[str, SchemaNode] = {}
        self.cell_nodes: Dict[str, CellNode] = {}
        self.triples: List[Tuple[str, str, str]] = []

    def build_from_dict(self, table_data: Dict[str, Any]):
        """
        Builds the graph from a structured dictionary.
        
        Expected table_data format:
        {
            "id": "table_1",
            "headers": [{"id": "h1", "text": "Header 1", "parent": None}, ...],
            "cells": [{"id": "c1", "text": "Value", "row_headers": ["h1"], "col_headers": ["h2"]}, ...]
        }
        """
        # 1. Build Schema Nodes
        headers = table_data.get("headers", [])
        for h in headers:
            node_id = str(h.get("id", f"{self.table_id}:h_{h['text']}"))
            node = SchemaNode(
                id=node_id,
                text=h["text"],
                parent_id=h.get("parent")
            )
            self.schema_nodes[node_id] = node
            
            if node.parent_id:
                self.triples.append((str(node.parent_id), "has_child", node.id))
        
        # Update children_ids
        for node_id, node in self.schema_nodes.items():
            if node.parent_id and str(node.parent_id) in self.schema_nodes:
                self.schema_nodes[str(node.parent_id)].children_ids.append(node_id)

        # 2. Build Cell Nodes
        cells = table_data.get("cells", [])
        for i, cell in enumerate(cells):
            cell_id = str(cell.get("id", f"{self.table_id}:c_{i}"))
            row_headers = [str(rh) for rh in cell.get("row_headers", [cell.get("row_header")] if cell.get("row_header") else [])]
            col_headers = [str(ch) for ch in cell.get("col_headers", [cell.get("col_header")] if cell.get("col_header") else [])]
            
            node = CellNode(
                id=cell_id,
                text=str(cell["text"]),
                row_header_ids=row_headers,
                col_header_ids=col_headers,
                table_id=self.table_id
            )
            self.cell_nodes[cell_id] = node
            
            for rh in row_headers:
                self.triples.append((rh, "has_value", cell_id))
            for ch in col_headers:
                self.triples.append((ch, "contains", cell_id))
                
        return self.triples

    def get_context_for_cell(self, cell_id: str) -> str:
        """Returns string representation of cell and its headers."""
        if cell_id not in self.cell_nodes:
            return ""
        cell = self.cell_nodes[cell_id]
        headers = []
        for h_id in cell.row_header_ids + cell.col_header_ids:
            if h_id in self.schema_nodes:
                headers.append(self.schema_nodes[h_id].text)
        return f"Cell: {cell.text} | Context: {' > '.join(headers)}"

class KnowledgeGraphBuilder:
    """Legacy wrapper for backward compatibility if needed, though we prefer TableGraph."""
    def build_graph(self, table_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        tg = TableGraph(table_data.get("id", "unknown"))
        return tg.build_from_dict(table_data)

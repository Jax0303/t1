"""
Knowledge Graph Builder for Table Structure.

Converts hierarchical table data into Graph triples:
(Header) -> [has_child] -> (Sub-Header)
(Header) -> [has_value] -> (Cell)
(Row_Entity) -> [has_attribute] -> (Cell)
"""

from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """
    Constructs KG from table structure.
    """
    
    def build_graph(self, table_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """
        Build triples from table.
        
        Args:
            table_data: Processed table dictionary
            
        Returns:
            List of (Subject, Predicate, Object) triples
        """
        triples = []
        table_id = table_data.get("id", "unknown")
        
        # 1. Header Hierarchy
        headers = table_data.get("headers", [])
        for h in headers:
            # Parent-Child
            if h.get("parent"):
                triples.append((h["parent"], "has_child", h["text"]))
                
        # 2. Cell Values
        cells = table_data.get("cells", [])
        for cell in cells:
            row_header = cell.get("row_header")
            col_header = cell.get("col_header")
            value = cell.get("text")
            
            if row_header and value:
                triples.append((row_header, "has_value", value))
                
            if col_header and value:
                triples.append((col_header, "contains", value))
                
            # Link row and col headers
            if row_header and col_header:
                triples.append((row_header, "related_to", col_header))
                
        return triples

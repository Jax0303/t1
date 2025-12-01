"""
Multi-granularity Indexer for Structure-Aware RAG.

Manages 3 levels of indices:
1. Table Level: Global table summaries
2. Text Level: Flattened text segments
3. Cell Level: Individual cell values with context
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MultiGranularityIndexer:
    """
    Indexer supporting Table, Text, and Cell granularity.
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        
        # In-memory stores for demonstration
        # In production, use FAISS or ChromaDB
        self.table_index: List[Dict[str, Any]] = []
        self.text_index: List[Dict[str, Any]] = []
        self.cell_index: List[Dict[str, Any]] = []
        
    def add_table(self, table_data: Dict[str, Any], embedding: List[float]):
        """Add table-level entry."""
        self.table_index.append({
            "data": table_data,
            "embedding": embedding,
            "id": table_data.get("id", len(self.table_index))
        })
        
    def add_text_chunk(self, text: str, source_table_id: str, embedding: List[float]):
        """Add text-level entry."""
        self.text_index.append({
            "text": text,
            "source_id": source_table_id,
            "embedding": embedding
        })
        
    def add_cell(self, cell_data: Dict[str, Any], embedding: List[float]):
        """Add cell-level entry."""
        self.cell_index.append({
            "data": cell_data,
            "embedding": embedding
        })
        
    def search(self, query_embedding: List[float], level: str = "table", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search in specific index level.
        
        Args:
            query_embedding: Query vector
            level: 'table', 'text', or 'cell'
            top_k: Number of results
            
        Returns:
            List of matches with scores
        """
        if level == "table":
            index = self.table_index
        elif level == "text":
            index = self.text_index
        elif level == "cell":
            index = self.cell_index
        else:
            raise ValueError(f"Unknown level: {level}")
            
        if not index:
            return []
            
        # Simple cosine similarity search
        scores = []
        q_vec = np.array(query_embedding)
        norm_q = np.linalg.norm(q_vec)
        
        for item in index:
            d_vec = np.array(item["embedding"])
            norm_d = np.linalg.norm(d_vec)
            
            if norm_q == 0 or norm_d == 0:
                score = 0.0
            else:
                score = np.dot(q_vec, d_vec) / (norm_q * norm_d)
                
            scores.append((score, item))
            
        # Sort and return top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, "item": i} for s, i in scores[:top_k]]

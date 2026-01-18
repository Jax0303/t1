from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MultiGranularityIndexer:
    """
    Indexer supporting Table, Schema, and Cell granularity.
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        
        # In-memory stores for demonstration
        self.table_index: List[Dict[str, Any]] = []
        self.schema_index: List[Dict[str, Any]] = []
        self.cell_index: List[Dict[str, Any]] = []
        
    def add_table(self, table_data: Dict[str, Any], embedding: List[float], confidence: float = 1.0):
        """Add table-level entry with confidence metadata."""
        self.table_index.append({
            "data": table_data,
            "embedding": embedding,
            "id": table_data.get("id", len(self.table_index)),
            "confidence": confidence
        })
        
    def add_schema(self, schema_data: Dict[str, Any], embedding: List[float], confidence: float = 1.0):
        """Add schema-level entry (headers) with confidence metadata."""
        self.schema_index.append({
            "data": schema_data,
            "embedding": embedding,
            "id": schema_data.get("id", len(self.schema_index)),
            "confidence": confidence
        })
        
    def add_cell(self, cell_data: Dict[str, Any], embedding: List[float], confidence: float = 1.0):
        """Add cell-level entry with confidence metadata."""
        self.cell_index.append({
            "data": cell_data,
            "embedding": embedding,
            "id": cell_data.get("id", len(self.cell_index)),
            "confidence": confidence
        })
        
    def search(
        self, 
        query_embedding: List[float], 
        level: str = "table", 
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search in specific index level with confidence-aware weighting.
        """
        if level == "table":
            index = self.table_index
        elif level == "schema":
            index = self.schema_index
        elif level == "cell":
            index = self.cell_index
        else:
            raise ValueError(f"Unknown level: {level}")
            
        if not index:
            return []
            
        scores = []
        # In this stub, we simulate matching.
        # If no real embedding logic, we return top elements with simulated scores.
        for i, item in enumerate(index):
            # Simulate a score that includes confidence
            conf = item.get("confidence", 1.0)
            # Higher index = more 'recent' or 'relevant' for this test
            base_score = 0.5 + (i / (len(index) + 1)) * 0.4 
            weighted_score = base_score * (alpha + (1.0 - alpha) * conf)
            scores.append((weighted_score, item))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"score": float(s), "item": i} for s, i in scores[:top_k]]

class MultiGranularityRetriever:
    """
    Retriever that selects context from multiple granularities.
    """
    def __init__(self, indexer: MultiGranularityIndexer):
        self.indexer = indexer

    def retrieve(self, query_embedding: List[float], levels: List[str], top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve context from multiple levels.
        """
        results = {}
        for level in levels:
            results[level] = self.indexer.search(query_embedding, level=level, top_k=top_k)
        return results

    def bundle_evidence(self, retrieved_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Consolidates results into an 'Evidence Pack'.
        """
        evidence_pack = []
        for level, results in retrieved_results.items():
            for res in results:
                item = res["item"]
                evidence_pack.append({
                    "id": item.get("id"),
                    "level": level,
                    "content": item.get("data", {}).get("text", str(item.get("data"))),
                    "score": res["score"]
                })
        return evidence_pack

"""
Adaptive Router for Query Classification.

Classifies queries into:
1. EXACT_VALUE: Specific cell lookup
2. HEADER_MATCH: Concept/Header search
3. AGGREGATION: Multi-hop/Aggregation reasoning
"""

from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)

class AdaptiveRouter:
    """
    Routes queries to appropriate retrieval strategy.
    """
    
    def __init__(self):
        # Simple keyword-based heuristics for demonstration
        # In production, use a trained classifier (BERT/LLM)
        self.agg_keywords = ["total", "sum", "average", "mean", "count", "compare", "difference", "ratio", "percentage", "growth"]
        self.header_keywords = ["column", "row", "header", "category", "type", "list", "show"]
        
    def route(self, query: str) -> str:
        """
        Classify query.
        
        Returns:
            'EXACT_VALUE', 'HEADER_MATCH', or 'AGGREGATION'
        """
        query_lower = query.lower()
        
        # Check for aggregation
        if any(k in query_lower for k in self.agg_keywords):
            return "AGGREGATION"
            
        # Check for header/schema questions
        if any(k in query_lower for k in self.header_keywords):
            return "HEADER_MATCH"
            
        # Default to exact value lookup (most common for specific data points)
        # Or if it looks like "What is X of Y?"
        return "EXACT_VALUE"

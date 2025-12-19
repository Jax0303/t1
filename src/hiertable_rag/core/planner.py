from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Planner(ABC):
    """
    Abstract interface for Query Planner.
    Decides retrieval granularity (Table/Schema/Cell).
    """
    @abstractmethod
    def plan(self, query: str) -> List[str]:
        """
        Returns a list of levels to retrieve from.
        e.g., ["table"], ["cell", "schema"]
        """
        pass

class RuleBasedPlanner(Planner):
    """
    Simple heuristic-based planner.
    """
    def __init__(self):
        self.agg_keywords = ["total", "sum", "average", "mean", "count", "compare", "difference", "ratio", "percentage", "growth", "all"]
        self.schema_keywords = ["column", "row", "header", "category", "type", "list", "show", "what is the structure"]
        
    def plan(self, query: str) -> List[str]:
        query_lower = query.lower()
        
        # If it's an aggregation or broad question, look at table/schema level
        if any(k in query_lower for k in self.agg_keywords):
            return ["table", "schema"]
            
        # If it's about structure/headers
        if any(k in query_lower for k in self.schema_keywords):
            return ["schema"]
            
        # Default to cell-level lookup for specific values
        return ["cell", "schema"] # Usually need schema for context anyway

class TrainablePlanner(Planner):
    """
    Interface for a model-based planner (e.g., using LLM or BERT).
    """
    def __init__(self, model_client: Any = None):
        self.model_client = model_client

    def plan(self, query: str) -> List[str]:
        # In a real implementation, this would call an LLM to classify the query
        # For now, placeholder calling RuleBasedPlanner
        logger.info("Using TrainablePlanner placeholder (falling back to RuleBased)")
        return RuleBasedPlanner().plan(query)

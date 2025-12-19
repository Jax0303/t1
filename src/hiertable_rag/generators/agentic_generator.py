import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AgenticGenerator:
    """
    Generator that enforces citation and JSON output format.
    """
    def __init__(self, model_client: Any = None):
        self.model_client = model_client

    def generate(self, query: str, evidence_pack: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a JSON response based on the evidence pack.
        """
        if not evidence_pack:
            return self._refuse("No evidence provided.")

        prompt = self._build_prompt(query, evidence_pack)
        
        # In a real implementation, this would call the LLM
        # For this prototype, we'll simulate the response
        logger.info(f"Generating citation-grounded response for query: {query}")
        
        # Mocking LLM response
        try:
            # Simple simulation: just pick the first cell if it exists
            cells = [e for e in evidence_pack if e["level"] == "cell"]
            if not cells:
                return self._refuse("INSUFFICIENT_EVIDENCE")
                
            top_cell = cells[0]
            simulated_response = {
                "answer": f"The value for {query} is {top_cell['content']}.",
                "cited_cell_ids": [top_cell["id"]],
                "confidence": 0.95
            }
            return simulated_response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._refuse("GENERATION_ERROR")

    def _build_prompt(self, query: str, evidence_pack: List[Dict[str, Any]]) -> str:
        evidence_str = "\n".join([f"[{e['id']}] ({e['level']}): {e['content']}" for e in evidence_pack])
        return f"""
        Query: {query}
        Evidence:
        {evidence_str}
        
        Output format: JSON with "answer", "cited_cell_ids", and "confidence".
        Rules:
        1. Mandatory citation for every claim using cited_cell_ids.
        2. If evidence is insufficient, answer exactly "INSUFFICIENT_EVIDENCE".
        """

    def _refuse(self, reason: str) -> Dict[str, Any]:
        return {
            "answer": reason,
            "cited_cell_ids": [],
            "confidence": 0.0
        }

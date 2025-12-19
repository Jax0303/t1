from typing import List, Dict, Any, Set
import time
import logging

logger = logging.getLogger(__name__)

class AgenticEvaluator:
    """
    Evaluates grounding, retrieval, efficiency, and latency.
    """
    
    def calculate_citation_f1(self, predicted_ids: List[str], gold_ids: List[str]) -> float:
        """
        Calculates F1 score based on cited cell IDs.
        Formula from image: Citation F1 = 2PR / (P + R)
        """
        pred_set = set(predicted_ids)
        gold_set = set(gold_ids)
        
        if not pred_set and not gold_set:
            return 1.0
        if not pred_set or not gold_set:
            return 0.0
            
        intersection = pred_set.intersection(gold_set)
        precision = len(intersection) / len(pred_set)
        recall = len(intersection) / len(gold_set)
        
        if precision + recall == 0:
            return 0.0
            
        f1 = (2 * precision * recall) / (precision + recall)
        return float(f1)

    def calculate_recall_at_k(self, retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
        """Calculates Recall@K."""
        top_k = set(retrieved_ids[:k])
        gold_set = set(gold_ids)
        
        if not gold_set:
            return 1.0
            
        intersection = top_k.intersection(gold_set)
        return float(len(intersection) / len(gold_set))

    def evaluate_response(self, response: Dict[str, Any], gold_evidence: List[str], latency: float, tokens: int) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single response.
        """
        cited_ids = response.get("cited_cell_ids", [])
        citation_f1 = self.calculate_citation_f1(cited_ids, gold_evidence)
        
        return {
            "citation_f1": citation_f1,
            "latency_ms": latency,
            "tokens": tokens,
            "answer": response.get("answer")
        }

class LatencyTracker:
    """Helper to track timing of different stages."""
    def __init__(self):
        self.start_times = {}
        self.durations = {}

    def start(self, name: str):
        self.start_times[name] = time.time()

    def stop(self, name: str):
        if name in self.start_times:
            duration = (time.time() - self.start_times[name]) * 1000 # ms
            self.durations[name] = duration
            return duration
        return 0.0

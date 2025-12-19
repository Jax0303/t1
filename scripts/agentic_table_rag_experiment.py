import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from src.hiertable_rag.core.rag import HierTableRAG
from src.hiertable_rag.core.evaluator import AgenticEvaluator

logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, dataset_name: str = "hitab"):
        self.dataset_name = dataset_name
        self.rag = HierTableRAG(planner_type="rule_based")
        self.evaluator = AgenticEvaluator()

    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        """Loads dataset from file."""
        if not Path(path).exists():
            logger.warning(f"Dataset path {path} not found. Using mock data.")
            return self._get_mock_dataset()
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_mock_dataset(self) -> List[Dict[str, Any]]:
        return [{
            "id": "q1",
            "question": "What is the revenue for North America in Q1?",
            "table": {
                "id": "t1",
                "headers": [
                    {"id": "h1", "text": "Region", "parent": None},
                    {"id": "h2", "text": "North America", "parent": "h1"},
                    {"id": "h3", "text": "Revenue", "parent": None},
                    {"id": "h4", "text": "Q1", "parent": "h3"},
                ],
                "cells": [
                    {"id": "c1", "text": "120M", "row_headers": ["h2"], "col_headers": ["h4"]},
                ]
            },
            "gold_evidence": ["c1"]
        }]

    def run(self, dataset_path: str, output_path: str):
        dataset = self.load_dataset(dataset_path)
        all_results = []
        
        print(f"Running experiment on {len(dataset)} samples...")
        
        for entry in dataset:
            # 1. Ingest table
            table_data = entry["table"]
            self.rag.ingest_table(table_data)
            
            # 2. Query
            query = entry["question"]
            start_time = time.time()
            result = self.rag.query(query)
            end_time = time.time()
            
            # 3. Evaluate
            gold_ids = entry.get("gold_evidence", [])
            latency = (end_time - start_time) * 1000
            
            # Simulated tokens (e.g., query length + context length)
            tokens = len(query.split()) + result["evidence_count"] * 50 
            
            metrics = self.evaluator.evaluate_response(
                result["response"],
                gold_ids,
                latency,
                tokens
            )
            
            all_results.append({
                "sample_id": entry.get("id"),
                "query": query,
                "response": result["response"],
                "metrics": metrics,
                "latency_breakdown": result["latency_ms"]
            })

        # Calculate Averages
        summary = self._calculate_summary(all_results)
        
        final_output = {
            "summary": summary,
            "results": all_results
        }
        
        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=2)
            
        print(f"Results saved to {output_path}")
        print("--- Summary ---")
        for k, v in summary.items():
            print(f"{k}: {v}")

    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        citation_f1s = [r["metrics"]["citation_f1"] for r in results]
        latencies = [r["metrics"]["latency_ms"] for r in results]
        tokens = [r["metrics"]["tokens"] for r in results]
        
        return {
            "avg_citation_f1": float(np.mean(citation_f1s)),
            "avg_latency_ms": float(np.mean(latencies)),
            "avg_tokens": float(np.mean(tokens)),
            "total_samples": len(results)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hitab")
    parser.add_argument("--input", type=str, default="data/sample_dataset.json")
    parser.add_argument("--output", type=str, default="outputs/agentic_rag_results.json")
    args = parser.parse_args()
    
    runner = ExperimentRunner(dataset_name=args.dataset)
    runner.run(args.input, args.output)

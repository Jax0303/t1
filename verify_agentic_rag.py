import sys
import os
from pprint import pprint

# Add src to path
sys.path.append(os.path.abspath("."))

from src.hiertable_rag.core.rag import HierTableRAG
from src.hiertable_rag.core.evaluator import AgenticEvaluator

def test_agentic_rag():
    # 1. Setup RAG
    rag = HierTableRAG(planner_type="rule_based")
    evaluator = AgenticEvaluator()
    
    # 2. Mock Table Data (HiTab style)
    table_data = {
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
    }
    
    # 3. Ingest
    print("Ingesting table...")
    rag.ingest_table(table_data)
    
    # 4. Query
    query = "What is the Q1 revenue for North America?"
    print(f"Querying: {query}")
    result = rag.query(query)
    
    pprint(result)
    
    # 5. Evaluate
    gold_evidence = ["c1"]
    eval_metrics = evaluator.evaluate_response(
        result["response"], 
        gold_evidence, 
        result["latency_ms"]["total"], 
        0 # mock tokens
    )
    
    print("\nEvaluation Metrics:")
    pprint(eval_metrics)
    
    # 6. Test Refusal
    print("\nTesting Refusal (Query without evidence):")
    refusal_result = rag.query("What is the Q1 revenue for Europe?")
    pprint(refusal_result)

if __name__ == "__main__":
    test_agentic_rag()

"""
Evaluation metrics for Table RAG.
Includes TEDS (Table Edit Distance Score), NDCG, and Retrieval metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional

def calculate_hit_at_k(retrieved_ids: List[str], ground_truth_id: str, k: int) -> float:
    """Calculates Hit@k (1.0 if GT is in top-k, else 0.0)."""
    return 1.0 if ground_truth_id in retrieved_ids[:k] else 0.0

def calculate_ndcg_at_k(retrieved_ids: List[str], ground_truth_id: str, k: int) -> float:
    """Calculates NDCG@k for a single relevant item."""
    if ground_truth_id not in retrieved_ids[:k]:
        return 0.0
    rank = retrieved_ids.index(ground_truth_id) + 1
    return 1.0 / np.log2(rank + 1)

def calculate_teds_struct(pred_table: Dict[str, Any], gt_table: Dict[str, Any]) -> float:
    """
    Simplified TEDS (Table Edit Distance Score) focusing on structure.
    Compares the row/col span consistency and layout.
    """
    # Simplified version: Compare cell counts and basic grid alignment
    p_rows = pred_table.get("rows", [])
    g_rows = gt_table.get("rows", [])
    
    if not p_rows or not g_rows:
        return 0.0
        
    # Structural similarity based on row/column counts
    row_sim = 1.0 - abs(len(p_rows) - len(g_rows)) / max(len(p_rows), len(g_rows))
    
    p_cols = max(len(r) for r in p_rows) if p_rows else 0
    g_cols = max(len(r) for r in g_rows) if g_rows else 0
    col_sim = 1.0 - abs(p_cols - g_cols) / max(p_cols, g_cols) if max(p_cols, g_cols) > 0 else 0
    
    return (row_sim + col_sim) / 2.0

def calculate_em_score(pred_answer: str, gt_answer: str) -> float:
    """Exact Match score."""
    return 1.0 if str(pred_answer).strip().lower() == str(gt_answer).strip().lower() else 0.0

def calculate_f1_score(pred_answer: str, gt_answer: str) -> float:
    """Token-level F1 score for answers."""
    pred_tokens = str(pred_answer).lower().split()
    gt_tokens = str(gt_answer).lower().split()
    
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
        
    common = set(pred_tokens) & set(gt_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0.0
        
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

"""
Self-Consistency Metrics (no GT required)
==========================================
All metrics operate solely on the normalized JSON output of a single model.

Metrics
-------
col_x_align_score    : 1/(1+std(x_left)+std(x_right)) per col, then average
row_y_align_score    : 1/(1+std(y_top)+std(y_bottom)) per row, then average
col_monotonicity_violations : # adjacent col pairs where x_left[i+1] < x_left[i]
row_monotonicity_violations : # adjacent row pairs where y_top[i+1] < y_top[i]
col_overlap_ratio    : fraction of adjacent col pairs with x overlap
col_gap_ratio        : fraction of adjacent col pairs with a gap > threshold
row_overlap_ratio    : fraction of adjacent row pairs with y overlap
row_gap_ratio        : fraction of adjacent row pairs with a gap > threshold
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_self_metrics(
    normalized: Dict[str, Any],
    col_gap_threshold: float = 5.0,
) -> Dict[str, Any]:
    """
    Compute all self-consistency metrics for a normalized JSON doc.

    Parameters
    ----------
    normalized        : Output of normalizer.normalize()
    col_gap_threshold : Minimum gap (px) treated as a real gap (not rounding noise).

    Returns
    -------
    dict with all metric values and per-col/per-row breakdown lists.
    """
    tables = normalized.get("tables", [])
    if not tables:
        return _empty_metrics()

    table = tables[0]   # evaluate first (and usually only) table
    cells = table.get("cells", [])
    rows = table.get("rows", [])
    cols = table.get("cols", [])

    metrics: Dict[str, Any] = {}

    # ── Column alignment ────────────────────────────────────────────────────
    col_align = _col_alignment(cells)
    metrics["col_x_align_per_col"] = col_align["per_col"]
    metrics["col_x_align_score"] = col_align["avg_score"]

    # ── Row alignment ───────────────────────────────────────────────────────
    row_align = _row_alignment(cells)
    metrics["row_y_align_per_row"] = row_align["per_row"]
    metrics["row_y_align_score"] = row_align["avg_score"]

    # ── Monotonicity ─────────────────────────────────────────────────────────
    mono = _monotonicity(rows, cols)
    metrics["col_monotonicity_violations"] = mono["col_violations"]
    metrics["row_monotonicity_violations"] = mono["row_violations"]
    metrics["col_monotonicity_detail"] = mono["col_detail"]
    metrics["row_monotonicity_detail"] = mono["row_detail"]

    # ── Overlap / Gap ─────────────────────────────────────────────────────────
    og = _overlap_gap(rows, cols, col_gap_threshold)
    metrics["col_overlap_ratio"] = og["col_overlap_ratio"]
    metrics["col_gap_ratio"] = og["col_gap_ratio"]
    metrics["row_overlap_ratio"] = og["row_overlap_ratio"]
    metrics["row_gap_ratio"] = og["row_gap_ratio"]
    metrics["col_overlap_pairs"] = og["col_overlap_pairs"]
    metrics["col_gap_pairs"] = og["col_gap_pairs"]
    metrics["row_overlap_pairs"] = og["row_overlap_pairs"]
    metrics["row_gap_pairs"] = og["row_gap_pairs"]

    # ── Summary score ─────────────────────────────────────────────────────────
    # Higher = better structured.  Violations/overlaps/gaps penalize score.
    penalty = (
        (metrics["col_monotonicity_violations"] + metrics["row_monotonicity_violations"]) * 0.05
        + metrics["col_overlap_ratio"] * 0.5
        + metrics["row_overlap_ratio"] * 0.5
    )
    metrics["summary_score"] = max(0.0, min(1.0,
        0.5 * metrics["col_x_align_score"]
        + 0.5 * metrics["row_y_align_score"]
        - penalty
    ))

    return metrics


# ── Column alignment ──────────────────────────────────────────────────────────

def _col_alignment(cells: List[Dict]) -> Dict:
    """Per-col std of x_left and x_right across all cells in that col."""
    col_map: Dict[int, List] = {}
    for c in cells:
        cidx = c.get("col_idx", 0)
        bbox = c.get("bbox", [0, 0, 0, 0])
        col_map.setdefault(cidx, []).append((bbox[0], bbox[2]))  # (x_left, x_right)

    per_col = []
    scores = []
    for cidx in sorted(col_map.keys()):
        xs = col_map[cidx]
        x_lefts = [x[0] for x in xs]
        x_rights = [x[1] for x in xs]
        std_l = float(np.std(x_lefts)) if len(x_lefts) > 1 else 0.0
        std_r = float(np.std(x_rights)) if len(x_rights) > 1 else 0.0
        score = 1.0 / (1.0 + std_l + std_r)
        per_col.append({
            "col_idx": cidx,
            "std_x_left": round(std_l, 3),
            "std_x_right": round(std_r, 3),
            "align_score": round(score, 4),
            "n_cells": len(xs),
        })
        scores.append(score)

    avg_score = float(np.mean(scores)) if scores else 0.0
    return {"per_col": per_col, "avg_score": round(avg_score, 4)}


# ── Row alignment ─────────────────────────────────────────────────────────────

def _row_alignment(cells: List[Dict]) -> Dict:
    """Per-row std of y_top and y_bottom."""
    row_map: Dict[int, List] = {}
    for c in cells:
        ridx = c.get("row_idx", 0)
        bbox = c.get("bbox", [0, 0, 0, 0])
        row_map.setdefault(ridx, []).append((bbox[1], bbox[3]))  # (y_top, y_bottom)

    per_row = []
    scores = []
    for ridx in sorted(row_map.keys()):
        ys = row_map[ridx]
        y_tops = [y[0] for y in ys]
        y_bots = [y[1] for y in ys]
        std_t = float(np.std(y_tops)) if len(y_tops) > 1 else 0.0
        std_b = float(np.std(y_bots)) if len(y_bots) > 1 else 0.0
        score = 1.0 / (1.0 + std_t + std_b)
        per_row.append({
            "row_idx": ridx,
            "std_y_top": round(std_t, 3),
            "std_y_bottom": round(std_b, 3),
            "align_score": round(score, 4),
            "n_cells": len(ys),
        })
        scores.append(score)

    avg_score = float(np.mean(scores)) if scores else 0.0
    return {"per_row": per_row, "avg_score": round(avg_score, 4)}


# ── Monotonicity ──────────────────────────────────────────────────────────────

def _monotonicity(rows: List[Dict], cols: List[Dict]) -> Dict:
    """Count monotonicity violations in row/col ordering."""
    col_detail = []
    col_violations = 0
    sorted_cols = sorted(cols, key=lambda c: c.get("idx", 0))
    for i in range(len(sorted_cols) - 1):
        b0 = sorted_cols[i].get("bbox", [0, 0, 0, 0])
        b1 = sorted_cols[i + 1].get("bbox", [0, 0, 0, 0])
        x_left0, x_left1 = b0[0], b1[0]
        x_right0, x_right1 = b0[2], b1[2]
        viol = (x_left1 < x_left0) or (x_right1 < x_right0)
        if viol:
            col_violations += 1
        col_detail.append({
            "col_pair": (sorted_cols[i].get("idx"), sorted_cols[i + 1].get("idx")),
            "x_left": (round(x_left0, 1), round(x_left1, 1)),
            "x_right": (round(x_right0, 1), round(x_right1, 1)),
            "violation": viol,
        })

    row_detail = []
    row_violations = 0
    sorted_rows = sorted(rows, key=lambda r: r.get("idx", 0))
    for i in range(len(sorted_rows) - 1):
        b0 = sorted_rows[i].get("bbox", [0, 0, 0, 0])
        b1 = sorted_rows[i + 1].get("bbox", [0, 0, 0, 0])
        y_top0, y_top1 = b0[1], b1[1]
        y_bot0, y_bot1 = b0[3], b1[3]
        viol = (y_top1 < y_top0) or (y_bot1 < y_bot0)
        if viol:
            row_violations += 1
        row_detail.append({
            "row_pair": (sorted_rows[i].get("idx"), sorted_rows[i + 1].get("idx")),
            "y_top": (round(y_top0, 1), round(y_top1, 1)),
            "y_bottom": (round(y_bot0, 1), round(y_bot1, 1)),
            "violation": viol,
        })

    return {
        "col_violations": col_violations,
        "row_violations": row_violations,
        "col_detail": col_detail,
        "row_detail": row_detail,
    }


# ── Overlap / Gap ─────────────────────────────────────────────────────────────

def _overlap_gap(
    rows: List[Dict],
    cols: List[Dict],
    gap_threshold: float,
) -> Dict:
    """Compute overlap and gap ratios for adjacent row/col pairs."""

    def _check_pairs(items: List[Dict], axis0: int, axis1: int):
        """axis0/axis1 = start/end indices into bbox (0=x0,1=y0,2=x1,3=y1)."""
        sorted_items = sorted(items, key=lambda x: x.get("bbox", [0, 0, 0, 0])[axis0])
        overlap_pairs = []
        gap_pairs = []
        n = len(sorted_items)
        for i in range(n - 1):
            b0 = sorted_items[i].get("bbox", [0, 0, 0, 0])
            b1 = sorted_items[i + 1].get("bbox", [0, 0, 0, 0])
            end0 = b0[axis1]
            start1 = b1[axis0]
            diff = start1 - end0  # >0 gap, <0 overlap
            pair_info = {
                "pair": (sorted_items[i].get("idx", i), sorted_items[i + 1].get("idx", i + 1)),
                "diff_px": round(diff, 1),
            }
            if diff < 0:
                overlap_pairs.append(pair_info)
            elif diff > gap_threshold:
                gap_pairs.append(pair_info)
        n_pairs = max(n - 1, 1)
        return (
            len(overlap_pairs) / n_pairs,
            len(gap_pairs) / n_pairs,
            overlap_pairs,
            gap_pairs,
        )

    col_ov, col_gap, col_ov_pairs, col_gap_pairs = _check_pairs(cols, 0, 2)
    row_ov, row_gap, row_ov_pairs, row_gap_pairs = _check_pairs(rows, 1, 3)

    return {
        "col_overlap_ratio": round(col_ov, 4),
        "col_gap_ratio": round(col_gap, 4),
        "row_overlap_ratio": round(row_ov, 4),
        "row_gap_ratio": round(row_gap, 4),
        "col_overlap_pairs": col_ov_pairs,
        "col_gap_pairs": col_gap_pairs,
        "row_overlap_pairs": row_ov_pairs,
        "row_gap_pairs": row_gap_pairs,
    }


def _empty_metrics() -> Dict:
    return {
        "col_x_align_per_col": [],
        "col_x_align_score": 0.0,
        "row_y_align_per_row": [],
        "row_y_align_score": 0.0,
        "col_monotonicity_violations": 0,
        "row_monotonicity_violations": 0,
        "col_monotonicity_detail": [],
        "row_monotonicity_detail": [],
        "col_overlap_ratio": 0.0,
        "col_gap_ratio": 0.0,
        "row_overlap_ratio": 0.0,
        "row_gap_ratio": 0.0,
        "col_overlap_pairs": [],
        "col_gap_pairs": [],
        "row_overlap_pairs": [],
        "row_gap_pairs": [],
        "summary_score": 0.0,
    }

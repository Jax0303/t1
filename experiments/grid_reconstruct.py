#!/usr/bin/env python3
"""
Grid Reconstruction — Baseline
================================
Detection-based TSR Stage 2: row/col/span bbox → cells with logical indices.

Baseline 방식: span bbox vs 기본 셀(row×col 교차) IoU > threshold로 할당.
"""
from __future__ import annotations
from typing import List, Tuple


def _bbox_iou(a: list, b: list) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1-x0) * max(0, y1-y0)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)


class GridReconstructor:
    """
    Baseline Grid Reconstruction.

    Parameters
    ----------
    iou_thresh : float
        Cell IoU threshold for span-to-cell assignment (default 0.3).
    min_span_cells : int
        Minimum rows OR cols for a valid span (default 2).
    """

    def __init__(self, iou_thresh: float = 0.3, min_span_cells: int = 2):
        self.iou_thresh = iou_thresh
        self.min_span_cells = min_span_cells

    def reconstruct(self,
                    rows_bbox: List[list],
                    cols_bbox: List[list],
                    spans_bbox: List[list]) -> dict:
        if not rows_bbox or not cols_bbox:
            return {"cells": [], "n_rows": 0, "n_cols": 0}

        base_cells = self._build_base_grid(rows_bbox, cols_bbox)

        if not spans_bbox:
            return {"cells": base_cells,
                    "n_rows": len(rows_bbox),
                    "n_cols": len(cols_bbox)}

        cells = self._assign_spans(base_cells, spans_bbox)

        return {"cells": cells,
                "n_rows": len(rows_bbox),
                "n_cols": len(cols_bbox)}

    def _build_base_grid(self,
                          rows_bbox: List[list],
                          cols_bbox: List[list]) -> List[dict]:
        cells = []
        for ri, rb in enumerate(rows_bbox):
            for ci, cb in enumerate(cols_bbox):
                x0 = max(cb[0], rb[0]); y0 = max(rb[1], cb[1])
                x1 = min(cb[2], rb[2]); y1 = min(rb[3], cb[3])
                if x1 > x0 and y1 > y0:
                    cells.append({
                        "start_row": ri, "end_row": ri,
                        "start_col": ci, "end_col": ci,
                        "row_idx": ri, "col_idx": ci,
                        "row_span": 1, "col_span": 1,
                        "bbox": [x0, y0, x1, y1],
                    })
        return cells

    def _assign_spans(self,
                       base_cells: List[dict],
                       spans_bbox: List[list]) -> List[dict]:
        merged_idx = set()
        merged = []

        for sb in spans_bbox:
            ov = [(i, c) for i, c in enumerate(base_cells)
                  if i not in merged_idx
                  and _bbox_iou(sb, c["bbox"]) > self.iou_thresh]

            if not ov:
                continue

            row_ids = [c["start_row"] for _, c in ov]
            col_ids = [c["start_col"] for _, c in ov]
            r0, r1 = min(row_ids), max(row_ids)
            c0, c1 = min(col_ids), max(col_ids)

            n_rows_span = r1 - r0 + 1
            n_cols_span = c1 - c0 + 1
            if n_rows_span < self.min_span_cells and n_cols_span < self.min_span_cells:
                continue

            merged.append({
                "start_row": r0, "end_row": r1,
                "start_col": c0, "end_col": c1,
                "row_idx": r0, "col_idx": c0,
                "row_span": r1 - r0 + 1,
                "col_span": c1 - c0 + 1,
                "bbox": sb,
            })
            for i, _ in ov:
                merged_idx.add(i)

        remaining = [c for i, c in enumerate(base_cells) if i not in merged_idx]
        return merged + remaining


def make_baseline(iou_thresh: float = 0.3) -> GridReconstructor:
    """Baseline grid reconstruction (IoU-based span assignment)."""
    return GridReconstructor(iou_thresh=iou_thresh, min_span_cells=2)

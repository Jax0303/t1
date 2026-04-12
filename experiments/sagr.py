#!/usr/bin/env python3
"""
SAGR — Span-Aware Grid Reconstruction
======================================
Detection-based TSR Stage 2의 병목(grid reconstruction)을 개선하는 알고리즘.

## 문제 (Baseline Stage 2의 실패 원인)
Baseline: span bbox vs 기본 셀(row×col 교차) IoU > 0.3으로 할당.
  - Type 1 실패(16/171, 9%): span bbox와 기본 셀의 IoU가 0.3 미만 → 할당 자체 실패
  - Type 2 실패(58/171, 34%): 할당됐지만 off-by-1 오류 (88%가 off-by-1)
    → span bbox가 경계를 살짝 넘어 인접 행/열의 셀 IoU가 threshold를 넘음

## SAGR의 핵심 개선
Cell level IoU 대신 Row/Column level Coverage 사용:
  - 행 i에 대해: coverage(span, row_i) = overlap_y / min(span_height, row_height)
  - 열 j에 대해: coverage(span, col_j) = overlap_x / min(span_width, col_width)
  - threshold 이상이면 해당 행/열에 span이 속한다고 판정
  - 할당 후 contiguity 강제 (중간 빈 행/열 채우기)

장점:
  - span bbox의 절대 크기에 민감하지 않음 (상대적 coverage 비율 사용)
  - off-by-1 오류: 인접 행에 span이 조금 걸려도 min() 기준이므로 낮은 coverage
  - backbone 학습 불필요, TATR에 plug-in 가능

## Ablation 구성
  SAGR-full : coverage + contiguity 모두 적용
  SAGR-cov  : coverage만 (contiguity 없음)
  SAGR-con  : baseline IoU + contiguity만
  Baseline  : 기존 IoU > 0.3
"""
from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
def _interval_coverage(s0: float, s1: float,
                        r0: float, r1: float) -> float:
    """
    1D coverage: overlap / min(span_length, range_length).
    "span [s0,s1]가 구간 [r0,r1]을 얼마나 커버하는가"의 양방향 척도.

    min()을 분모로 쓰는 이유:
      - span이 row보다 작아도 row가 span보다 작아도 동일하게 동작
      - 두 구간 중 짧은 쪽이 긴 쪽에 많이 겹칠수록 높은 값
      → off-by-1 문제: 인접 행에 span이 2~3px 걸려도 min_len은 그 행의 높이
                      → coverage 비율이 매우 낮아져 threshold 통과 안 됨
    """
    inter = max(0.0, min(s1, r1) - max(s0, r0))
    min_len = min(s1 - s0, r1 - r0)
    if min_len <= 0:
        return 0.0
    return inter / min_len


def _bbox_iou(a: list, b: list) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1-x0) * max(0, y1-y0)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)


# ─────────────────────────────────────────────
# Core SAGR variants
# ─────────────────────────────────────────────
class SpanAwareGridReconstruction:
    """
    SAGR: Span-Aware Grid Reconstruction.

    Parameters
    ----------
    coverage_thresh : float
        Row/Col coverage threshold (default 0.3).
        min-interval-overlap / min(span_dim, row/col_dim) > this → include.
    enforce_contiguous : bool
        True  → fill index gaps in assigned rows/cols (contiguity constraint).
        False → allow non-contiguous assignment (coverage only).
    use_coverage : bool
        True  → use coverage-based assignment (SAGR).
        False → fall back to cell IoU > iou_thresh (baseline).
    iou_thresh : float
        Cell IoU threshold for baseline mode (default 0.3).
    min_span_cells : int
        Minimum number of rows OR cols for a valid span (default 2).
    """

    def __init__(self,
                 coverage_thresh: float = 0.3,
                 enforce_contiguous: bool = True,
                 use_coverage: bool = True,
                 iou_thresh: float = 0.3,
                 min_span_cells: int = 2):
        self.coverage_thresh   = coverage_thresh
        self.enforce_contiguous = enforce_contiguous
        self.use_coverage      = use_coverage
        self.iou_thresh        = iou_thresh
        self.min_span_cells    = min_span_cells

    # ── Public API ──────────────────────────────
    def reconstruct(self,
                    rows_bbox: List[list],
                    cols_bbox: List[list],
                    spans_bbox: List[list]) -> dict:
        """
        rows_bbox, cols_bbox, spans_bbox → cells + n_rows + n_cols.
        Replaces the baseline reconstruct_grid() function.
        """
        if not rows_bbox or not cols_bbox:
            return {"cells": [], "n_rows": 0, "n_cols": 0}

        # Step 1: 기본 1×1 셀 그리드 생성 (row×col 교차)
        base_cells = self._build_base_grid(rows_bbox, cols_bbox)

        if not spans_bbox:
            return {"cells": base_cells,
                    "n_rows": len(rows_bbox),
                    "n_cols": len(cols_bbox)}

        # Step 2: 각 span을 grid에 할당
        cells = self._assign_spans(base_cells, rows_bbox, cols_bbox, spans_bbox)

        return {"cells": cells,
                "n_rows": len(rows_bbox),
                "n_cols": len(cols_bbox)}

    # ── Private helpers ─────────────────────────
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
                        "row_span": 1, "col_span": 1,
                        "bbox": [x0, y0, x1, y1],
                    })
        return cells

    def _assign_spans(self,
                       base_cells: List[dict],
                       rows_bbox: List[list],
                       cols_bbox: List[list],
                       spans_bbox: List[list]) -> List[dict]:
        merged_idx = set()
        merged     = []

        for sb in spans_bbox:
            if self.use_coverage:
                row_ids, col_ids = self._coverage_assign(sb, rows_bbox, cols_bbox)
            else:
                row_ids, col_ids = self._iou_assign(sb, base_cells)

            if not row_ids or not col_ids:
                continue

            # Contiguity: min~max 사이 빈 인덱스 채우기
            if self.enforce_contiguous:
                row_ids = list(range(min(row_ids), max(row_ids) + 1))
                col_ids = list(range(min(col_ids), max(col_ids) + 1))

            # 최소 spanning 조건
            n_rows_span = max(row_ids) - min(row_ids) + 1
            n_cols_span = max(col_ids) - min(col_ids) + 1
            if n_rows_span < self.min_span_cells and n_cols_span < self.min_span_cells:
                continue  # 1×1 → spanning 아님

            r0, r1 = min(row_ids), max(row_ids)
            c0, c1 = min(col_ids), max(col_ids)
            row_id_set = set(row_ids)
            col_id_set = set(col_ids)

            ov = [(i, c) for i, c in enumerate(base_cells)
                  if i not in merged_idx
                  and c["start_row"] in row_id_set
                  and c["start_col"] in col_id_set]

            if not ov:
                continue

            merged.append({
                "start_row": r0, "end_row": r1,
                "start_col": c0, "end_col": c1,
                "row_span": r1 - r0 + 1,
                "col_span": c1 - c0 + 1,
                "bbox": sb,
            })
            for i, _ in ov:
                merged_idx.add(i)

        remaining = [c for i, c in enumerate(base_cells) if i not in merged_idx]
        return merged + remaining

    def _coverage_assign(self,
                          sb: list,
                          rows_bbox: List[list],
                          cols_bbox: List[list]) -> Tuple[List[int], List[int]]:
        """
        Row/Col coverage-based assignment.
        _interval_coverage(span_y, row_y) > threshold → include row.
        """
        row_ids = [
            ri for ri, rb in enumerate(rows_bbox)
            if _interval_coverage(sb[1], sb[3], rb[1], rb[3]) > self.coverage_thresh
        ]
        col_ids = [
            ci for ci, cb in enumerate(cols_bbox)
            if _interval_coverage(sb[0], sb[2], cb[0], cb[2]) > self.coverage_thresh
        ]
        return row_ids, col_ids

    def _iou_assign(self,
                     sb: list,
                     base_cells: List[dict]) -> Tuple[List[int], List[int]]:
        """Baseline: span bbox vs base cell IoU > threshold."""
        ov = [c for c in base_cells if _bbox_iou(sb, c["bbox"]) > self.iou_thresh]
        if not ov:
            return [], []
        row_ids = list({c["start_row"] for c in ov})
        col_ids = list({c["start_col"] for c in ov})
        return row_ids, col_ids


# ─────────────────────────────────────────────
# Ablation variants (편의 함수)
# ─────────────────────────────────────────────
def make_baseline(iou_thresh: float = 0.3) -> SpanAwareGridReconstruction:
    """기존 방식 (coverage 없음, contiguity 없음)."""
    return SpanAwareGridReconstruction(
        use_coverage=False,
        enforce_contiguous=False,
        iou_thresh=iou_thresh,
        min_span_cells=2,
    )


def make_sagr_cov(coverage_thresh: float = 0.3) -> SpanAwareGridReconstruction:
    """SAGR-Cov: coverage만 적용, contiguity 없음."""
    return SpanAwareGridReconstruction(
        use_coverage=True,
        enforce_contiguous=False,
        coverage_thresh=coverage_thresh,
        min_span_cells=2,
    )


def make_sagr_con(iou_thresh: float = 0.3) -> SpanAwareGridReconstruction:
    """SAGR-Con: 기존 IoU + contiguity만 추가."""
    return SpanAwareGridReconstruction(
        use_coverage=False,
        enforce_contiguous=True,
        iou_thresh=iou_thresh,
        min_span_cells=2,
    )


def make_sagr_full(coverage_thresh: float = 0.3) -> SpanAwareGridReconstruction:
    """SAGR-Full: coverage + contiguity 모두 적용."""
    return SpanAwareGridReconstruction(
        use_coverage=True,
        enforce_contiguous=True,
        coverage_thresh=coverage_thresh,
        min_span_cells=2,
    )


# ─────────────────────────────────────────────
# Threshold sweep helper
# ─────────────────────────────────────────────
def sweep_coverage_thresh(thresholds: List[float] = None) -> List[SpanAwareGridReconstruction]:
    """coverage_thresh 감도 분석용 variants."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    return [make_sagr_full(t) for t in thresholds]

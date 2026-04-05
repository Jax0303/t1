#!/usr/bin/env python3
"""
Logical-Index Span Evaluation (hole 1 해결)
============================================
IoU 기반 span recovery 의 측정 artifact 를 제거한 evaluation.

동기 (motivation):
  기존 span_recovery_rate() 는 chunk 파일로부터 재구성한 GT pixel bbox
  와 예측 bbox 사이의 IoU≥0.5 를 쓴다. chunk 파일은 OCR 토큰 기준이므로
  실제 표 셀 경계선과 ~10% 오프셋이 발생하여 IoU 값이 체계적으로
  저평가된다. 20-cycle diagnostic 에서 IoU≥0.1 에서 93% 가 맞는데
  IoU≥0.5 에서 1.6% 로 급락하는 것이 그 증거.

접근:
  GT cell 픽셀 bbox 에 의존하지 않는 logical-index exact matching.
  1. TATR 의 row/col bbox 중심을 GT row/col 중심에 nearest-neighbor 로 매핑
     (row/col 경계는 cell bbox 보다 훨씬 robust — 텍스트 행 간격만 필요)
  2. 매핑 후 TATR merged cell 의 (sr, er, sc, ec) 를 GT 좌표계로 변환
  3. GT spanning cell 의 (sr, er, sc, ec) 튜플과 exact match 여부 확인
  4. Recovery = exact match 갯수 / GT spanning 총 수

이 metric 은 pixel-IoU 에 전혀 의존하지 않으며,
"detection paradigm 이 logical structure 를 올바르게 재구성하는가"
라는 질문에 직접 답한다.

출력:
  experiments/results_logical/logical_summary.json
  experiments/results_logical/logical_per_table.json
  experiments/results_logical/logical_eval.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

sys.path.insert(0, str(Path(__file__).parent))
from _eval_utils import (
    load_valid_ids, load_gt, load_image,
    CHUNK_DIR, bbox_iou,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results_logical"
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5
IOU_MERGE = 0.3


# ─────────────────────────────────────────────
# GT row/col center 추정 (chunk 기반, 경계만 사용)
# ─────────────────────────────────────────────
def infer_gt_row_col_centers(
    gt: dict, table_id: str, img_size: Tuple[int, int]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    GT row / col 중심을 이미지 픽셀 좌표로 반환 (pixel-cell-bbox 미사용).

    chunk 파일의 y-center / x-center 분포를 1D 클러스터링하여
    n_rows / n_cols 그룹으로 나눈 뒤 각 그룹 평균을 중심으로 쓴다.

    주의: 셀 pixel bbox 재구성 (infer_gt_cell_bboxes) 과 다른 점은
          여기서는 경계(break) 가 아닌 각 row/col 의 "중심값" 만
          뽑는다는 것. 중심값은 gap clustering 보다 훨씬 stable.

    Returns:
      (row_centers, col_centers) in image pixel space.
      chunk 파일 없거나 실패 시 None.
    """
    chunk_path = CHUNK_DIR / f"{table_id}.chunk"
    if not chunk_path.exists():
        return None
    try:
        with open(chunk_path) as f:
            data = json.load(f)
        chunks = data.get("chunks", [])
        if not chunks:
            return None
    except Exception:
        return None

    n_rows, n_cols = gt["n_rows"], gt["n_cols"]
    if n_rows == 0 or n_cols == 0:
        return None

    all_x1 = [c["pos"][0] for c in chunks]
    all_y1 = [c["pos"][1] for c in chunks]
    all_x2 = [c["pos"][2] for c in chunks]
    all_y2 = [c["pos"][3] for c in chunks]
    tx1, ty1 = min(all_x1), min(all_y1)
    tx2, ty2 = max(all_x2), max(all_y2)

    cy = np.array([(c["pos"][1] + c["pos"][3]) / 2 for c in chunks])
    cx = np.array([(c["pos"][0] + c["pos"][2]) / 2 for c in chunks])

    def kmeans_1d(vals: np.ndarray, k: int) -> np.ndarray:
        """gap-based k-partition → 각 bin 중앙값."""
        if len(vals) == 0 or k <= 0:
            return np.array([])
        vals_s = np.sort(vals)
        if len(vals_s) <= k:
            return vals_s if len(vals_s) == k else np.linspace(vals_s[0], vals_s[-1], k)
        gaps = np.diff(vals_s)
        # k-1 largest gaps → split points
        split_idx = np.argsort(gaps)[::-1][: k - 1]
        split_idx_sorted = sorted(split_idx.tolist())
        groups = np.split(vals_s, [i + 1 for i in split_idx_sorted])
        centers = np.array([float(np.mean(g)) if len(g) else 0.0 for g in groups])
        if len(centers) != k:  # fallback
            centers = np.linspace(vals_s[0], vals_s[-1], k)
        return centers

    row_centers_chunk = kmeans_1d(cy, n_rows)
    col_centers_chunk = kmeans_1d(cx, n_cols)

    # → image pixel space
    img_w, img_h = float(img_size[0]), float(img_size[1])
    chunk_w = tx2 - tx1 if tx2 > tx1 else 1.0
    chunk_h = ty2 - ty1 if ty2 > ty1 else 1.0
    sx = img_w / chunk_w
    sy = img_h / chunk_h
    row_centers = (row_centers_chunk - ty1) * sy
    col_centers = (col_centers_chunk - tx1) * sx
    row_centers = np.sort(np.clip(row_centers, 0, img_h))
    col_centers = np.sort(np.clip(col_centers, 0, img_w))
    return row_centers, col_centers


# ─────────────────────────────────────────────
# TATR → cells (with raw row/col bboxes exposed)
# ─────────────────────────────────────────────
def _load_tatr(model_id: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    if hasattr(processor, "size") and isinstance(processor.size, dict):
        if "longest_edge" in processor.size and "shortest_edge" not in processor.size:
            le = processor.size["longest_edge"]
            processor.size = {"shortest_edge": le, "longest_edge": le}
    model = TableTransformerForObjectDetection.from_pretrained(model_id)
    model = model.to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def _run_tatr(processor, model, image: Image.Image) -> dict:
    enc = processor(images=image, return_tensors="pt")
    model_keys = {"pixel_values", "pixel_mask"}
    model_enc = {k: v.to(DEVICE) for k, v in enc.items() if k in model_keys}
    outputs = model(**model_enc)
    target_sizes = [(image.size[1], image.size[0])]
    results = processor.post_process_object_detection(
        outputs, threshold=CONF_THRESH, target_sizes=target_sizes
    )[0]
    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    id2label = model.config.id2label
    rows, cols, spans = [], [], []
    for box, lid, sc in zip(boxes, labels, scores):
        lname = id2label[lid].lower()
        entry = {"box": box.tolist(), "score": float(sc), "label": lname}
        if "column" in lname:
            cols.append(entry)
        elif "spanning" in lname or "projected" in lname:
            spans.append(entry)
        else:
            rows.append(entry)
    rows.sort(key=lambda r: (r["box"][1] + r["box"][3]) / 2)
    cols.sort(key=lambda c: (c["box"][0] + c["box"][2]) / 2)

    cells = []
    for ri, rdata in enumerate(rows):
        for ci, cdata in enumerate(cols):
            rb, cb = rdata["box"], cdata["box"]
            x0, y0 = max(cb[0], rb[0]), max(rb[1], cb[1])
            x1, y1 = min(cb[2], rb[2]), min(rb[3], cb[3])
            if x1 > x0 and y1 > y0:
                cells.append({
                    "start_row": ri, "end_row": ri,
                    "start_col": ci, "end_col": ci,
                    "row_span": 1, "col_span": 1,
                    "bbox": [x0, y0, x1, y1],
                })

    if spans:
        merged_idx = set()
        merged_cells = []
        for sdata in spans:
            sbox = sdata["box"]
            overlaps = [
                (idx, c) for idx, c in enumerate(cells)
                if idx not in merged_idx and bbox_iou(sbox, c["bbox"]) > IOU_MERGE
            ]
            if overlaps:
                rs = [c["start_row"] for _, c in overlaps]
                cs = [c["start_col"] for _, c in overlaps]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                merged_cells.append({
                    "start_row": r0, "end_row": r1,
                    "start_col": c0, "end_col": c1,
                    "row_span": r1 - r0 + 1, "col_span": c1 - c0 + 1,
                    "bbox": sbox,
                })
                for idx, _ in overlaps:
                    merged_idx.add(idx)
        cells = merged_cells + [c for i, c in enumerate(cells) if i not in merged_idx]

    return {
        "cells": cells,
        "rows_bbox": [r["box"] for r in rows],
        "cols_bbox": [c["box"] for c in cols],
        "n_rows": len(rows), "n_cols": len(cols),
        "n_spans_detected": len(spans),
        "spans_bbox": [s["box"] for s in spans],
    }


# ─────────────────────────────────────────────
# TATR 좌표계 → GT 좌표계 매핑
# ─────────────────────────────────────────────
def align_indices(
    tatr_centers: np.ndarray, gt_centers: np.ndarray
) -> np.ndarray:
    """
    TATR row/col 중심 배열을 GT row/col 중심 배열에 nearest-neighbor 매핑.

    Args:
      tatr_centers: shape (R',)  TATR 에서 검출한 row(or col) 중심들
      gt_centers:   shape (R,)   GT row(or col) 중심들 (chunk 기반)

    Returns:
      mapping: shape (R',) — mapping[i] = j  (TATR i 번째 → GT j 번째)

    주의: 다 대 일 매핑 허용 (TATR 가 GT row 를 과분할한 경우).
    """
    mapping = np.zeros(len(tatr_centers), dtype=np.int64)
    if len(gt_centers) == 0:
        return mapping
    for i, c in enumerate(tatr_centers):
        mapping[i] = int(np.argmin(np.abs(gt_centers - c)))
    return mapping


def logical_span_recovery(
    gt: dict, tatr_out: dict, row_centers_gt: np.ndarray, col_centers_gt: np.ndarray
) -> dict:
    """
    GT 좌표계에서 TATR merged cell 의 logical span 일치 여부 검사.

    절차:
      1. TATR row bbox y-center → GT row idx 매핑 (nearest neighbor)
      2. TATR col bbox x-center → GT col idx 매핑
      3. 각 TATR merged cell (sr', er', sc', ec') 을 GT 공간으로 변환
         → (map_r[sr'], map_r[er'], map_c[sc'], map_c[ec'])
      4. GT spanning cell 집합과 exact tuple match 확인

    Returns:
      dict with:
        gt_spanning: int
        recovered_exact: int
        rate_exact: float
        pred_spans_in_gt_space: list of tuples
    """
    rows_bbox = tatr_out.get("rows_bbox", [])
    cols_bbox = tatr_out.get("cols_bbox", [])
    if len(rows_bbox) == 0 or len(cols_bbox) == 0:
        return {"gt_spanning": 0, "recovered_exact": 0, "rate_exact": None,
                "pred_spans_in_gt_space": []}
    tatr_row_cy = np.array([(b[1] + b[3]) / 2 for b in rows_bbox])
    tatr_col_cx = np.array([(b[0] + b[2]) / 2 for b in cols_bbox])
    map_r = align_indices(tatr_row_cy, row_centers_gt)
    map_c = align_indices(tatr_col_cx, col_centers_gt)

    gt_span_tuples = set()
    for c in gt["cells"]:
        if c["row_span"] > 1 or c["col_span"] > 1:
            gt_span_tuples.add(
                (c["start_row"], c["end_row"], c["start_col"], c["end_col"])
            )

    pred_span_tuples = []
    for c in tatr_out["cells"]:
        if c["row_span"] > 1 or c["col_span"] > 1:
            sr_p, er_p = c["start_row"], c["end_row"]
            sc_p, ec_p = c["start_col"], c["end_col"]
            # clamp in range
            if sr_p >= len(map_r) or er_p >= len(map_r):
                continue
            if sc_p >= len(map_c) or ec_p >= len(map_c):
                continue
            sr_g = int(map_r[sr_p]); er_g = int(map_r[er_p])
            sc_g = int(map_c[sc_p]); ec_g = int(map_c[ec_p])
            if sr_g > er_g: sr_g, er_g = er_g, sr_g
            if sc_g > ec_g: sc_g, ec_g = ec_g, sc_g
            pred_span_tuples.append((sr_g, er_g, sc_g, ec_g))

    pred_set = set(pred_span_tuples)
    recovered = sum(1 for t in gt_span_tuples if t in pred_set)
    total = len(gt_span_tuples)

    return {
        "gt_spanning": total,
        "recovered_exact": recovered,
        "rate_exact": recovered / total if total > 0 else None,
        "pred_spans_in_gt_space": list(pred_set),
    }


# ─────────────────────────────────────────────
# 추가 지표: span distribution match + span class detection rate
# ─────────────────────────────────────────────
def span_distribution_match(gt: dict, tatr_out: dict) -> float:
    """
    GT / TATR 에서 (row_span, col_span) multiset 비교 (Jaccard).
    """
    gt_multi = {}
    for c in gt["cells"]:
        if c["row_span"] > 1 or c["col_span"] > 1:
            k = (c["row_span"], c["col_span"])
            gt_multi[k] = gt_multi.get(k, 0) + 1
    pr_multi = {}
    for c in tatr_out["cells"]:
        if c["row_span"] > 1 or c["col_span"] > 1:
            k = (c["row_span"], c["col_span"])
            pr_multi[k] = pr_multi.get(k, 0) + 1
    keys = set(gt_multi) | set(pr_multi)
    if not keys:
        return 1.0
    inter = sum(min(gt_multi.get(k, 0), pr_multi.get(k, 0)) for k in keys)
    union = sum(max(gt_multi.get(k, 0), pr_multi.get(k, 0)) for k in keys)
    return inter / union if union else 1.0


def span_class_detection_rate(
    gt: dict, tatr_out: dict, gt_cells_bbox: Optional[list] = None
) -> Optional[float]:
    """
    GT spanning cell 영역 위에 TATR 의 'spanning cell' class 검출이
    최소 IoU 0.1 로 hit 하는 비율 (loose recall).

    gt_cells_bbox: infer_gt_cell_bboxes 결과 (있을 때만 계산 가능).
    """
    if not gt_cells_bbox:
        return None
    gt_spans = [c for c in gt_cells_bbox if c["row_span"] > 1 or c["col_span"] > 1]
    if not gt_spans:
        return None
    hits = 0
    for gs in gt_spans:
        gb = gs["bbox"]
        hit = any(bbox_iou(gb, sb) >= 0.1 for sb in tatr_out["spans_bbox"])
        if hit:
            hits += 1
    return hits / len(gt_spans)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
MODELS = [
    {"id": "microsoft/table-transformer-structure-recognition",           "label": "TATR v1.0"},
    {"id": "microsoft/table-transformer-structure-recognition-v1.1-pub",  "label": "TATR v1.1-pub"},
    {"id": "microsoft/table-transformer-structure-recognition-v1.1-all",  "label": "TATR v1.1-all"},
]


def run():
    from _eval_utils import infer_gt_cell_bboxes
    valid_ids = load_valid_ids()
    print(f"[logical_eval] evaluating on {len(valid_ids)} SciTSR-COMP tables", flush=True)
    all_results: Dict[str, List[dict]] = {}
    summary: Dict[str, dict] = {}

    for m in MODELS:
        print(f"\n=== {m['label']} ===", flush=True)
        processor, model = _load_tatr(m["id"])
        records = []
        err_count = 0
        for tid in tqdm(valid_ids, desc=m["label"]):
            try:
                gt = load_gt(tid)
                img = load_image(tid)
                centers = infer_gt_row_col_centers(gt, tid, img.size)
                if centers is None:
                    continue
                row_c, col_c = centers
                tatr = _run_tatr(processor, model, img)
                log = logical_span_recovery(gt, tatr, row_c, col_c)
                gt_cells_bbox = infer_gt_cell_bboxes(gt, tid, img_size=img.size)
                recall_cls = span_class_detection_rate(gt, tatr, gt_cells_bbox)
                dist_j = span_distribution_match(gt, tatr)
                records.append({
                    "table_id": tid,
                    "gt_n_rows": gt["n_rows"], "gt_n_cols": gt["n_cols"],
                    "gt_spanning": log["gt_spanning"],
                    "pred_n_rows": tatr["n_rows"], "pred_n_cols": tatr["n_cols"],
                    "pred_n_spans_det": tatr["n_spans_detected"],
                    "recovered_exact": log["recovered_exact"],
                    "rate_exact": log["rate_exact"],
                    "span_class_recall": recall_cls,
                    "span_dist_jaccard": dist_j,
                })
            except Exception as e:
                err_count += 1
                continue
        print(f"  processed={len(records)}  errors={err_count}", flush=True)
        all_results[m["label"]] = records

        # aggregate
        rates = [r["rate_exact"] for r in records if r["rate_exact"] is not None]
        recalls = [r["span_class_recall"] for r in records if r["span_class_recall"] is not None]
        dists = [r["span_dist_jaccard"] for r in records]
        total_sp = sum(r["gt_spanning"] for r in records)
        total_rec = sum(r["recovered_exact"] for r in records)
        summary[m["label"]] = {
            "n_tables": len(records),
            "macro_logical_exact": float(np.mean(rates)) if rates else None,
            "micro_logical_exact": (total_rec / total_sp) if total_sp > 0 else None,
            "n_spans_total": total_sp,
            "n_spans_recovered_exact": total_rec,
            "mean_span_class_recall_loose": float(np.mean(recalls)) if recalls else None,
            "mean_span_dist_jaccard": float(np.mean(dists)) if dists else None,
        }

    # save
    (RESULTS_DIR / "logical_per_table.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "logical_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 64)
    print("LOGICAL-INDEX SPAN EVALUATION SUMMARY")
    print("=" * 64)
    for lbl, s in summary.items():
        print(f"\n{lbl}  (n_tables={s['n_tables']}, n_spans={s['n_spans_total']})")
        print(f"  macro  logical-exact : {s['macro_logical_exact']}")
        print(f"  micro  logical-exact : {s['micro_logical_exact']}")
        print(f"  class-recall (loose) : {s['mean_span_class_recall_loose']}")
        print(f"  span-dist Jaccard    : {s['mean_span_dist_jaccard']}")

    make_figure(summary, all_results)


def make_figure(summary: dict, all_results: dict):
    labels = list(summary.keys())
    micro = [summary[l]["micro_logical_exact"] or 0.0 for l in labels]
    macro = [summary[l]["macro_logical_exact"] or 0.0 for l in labels]
    recall = [summary[l]["mean_span_class_recall_loose"] or 0.0 for l in labels]
    jac = [summary[l]["mean_span_dist_jaccard"] or 0.0 for l in labels]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(len(labels))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    titles = [
        "Logical Exact-Match (micro)",
        "Logical Exact-Match (macro)",
        "Span-class Recall @ IoU≥0.1 (loose)",
        "Span-Distribution Jaccard",
    ]
    vals = [micro, macro, recall, jac]
    for ax, title, v in zip(axes, titles, vals):
        ax.bar(x, v, color=colors[: len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, fontsize=8)
        ax.set_ylim(0, max(0.05, max(v) * 1.2))
        ax.set_title(title, fontsize=10)
        for xi, vi in zip(x, v):
            ax.text(xi, vi, f"{vi:.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("Logical-Index Span Evaluation (pixel-IoU free)", fontsize=12)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "logical_eval.png", dpi=150)
    plt.close()
    print(f"\n[saved] {RESULTS_DIR / 'logical_eval.png'}")


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
GriTS Evaluation (hole 3 해결)
=============================
Smock et al. CVPR 2023 "GriTS: Grid table similarity metric for TSR"
커뮤니티 표준 metric 구현.

구현 범위:
  GriTS-Top : cell 의 (row_span, col_span) shape 유사도 기반
              → "논리 구조(위상)" 정합성
  GriTS-Con : cell text 의 LCS 기반 유사도
              → "내용" 정합성

알고리즘 (factored 2D-MSS):
  1. 각 table 을 2D grid matrix M[r][c] 로 펼치기 (span 은 반복 기입)
  2. Row-level Needleman-Wunsch 정렬
     — row similarity = column-level NW alignment score of that row
  3. 각 match 된 (row_p, row_g) 쌍에서 cell-wise 합산
  4. precision = total_sim / |pred grid|, recall = total_sim / |gt grid|
  5. F1 = 2PR/(P+R)

GriTS-Top similarity (논문 Eq. 5):
  f_top(a,b) = |shape_intersect| / |shape_union|
             = min(rs_a,rs_b)·min(cs_a,cs_b)
               / (rs_a·cs_a + rs_b·cs_b − min(rs_a,rs_b)·min(cs_a,cs_b))

GriTS-Con similarity:
  f_con(a,b) = 2·LCS(t_a,t_b) / (|t_a|+|t_b|)

출력:
  experiments/results_grits/grits_summary.json
  experiments/results_grits/grits_per_table.json
  experiments/results_grits/grits_eval.png
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
from _eval_utils import load_valid_ids, load_gt, load_image, bbox_iou

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results_grits"
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5
IOU_MERGE = 0.3


# ─────────────────────────────────────────────
# Cell similarity functions
# ─────────────────────────────────────────────
def sim_top(a: dict, b: dict) -> float:
    """Span-shape IoU between two cells (논문 Eq. 5)."""
    rs_a = a.get("row_span", 1); cs_a = a.get("col_span", 1)
    rs_b = b.get("row_span", 1); cs_b = b.get("col_span", 1)
    inter = min(rs_a, rs_b) * min(cs_a, cs_b)
    union = rs_a * cs_a + rs_b * cs_b - inter
    return inter / union if union else 0.0


def _lcs_len(s: str, t: str) -> int:
    m, n = len(s), len(t)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ci = s[i - 1]
        for j in range(1, n + 1):
            if ci == t[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n]


def sim_con(a: dict, b: dict) -> float:
    ta = (a.get("text") or "").strip()
    tb = (b.get("text") or "").strip()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    lcs = _lcs_len(ta, tb)
    return 2.0 * lcs / (len(ta) + len(tb))


# ─────────────────────────────────────────────
# Build 2D grid matrix (r, c) → cell reference
# ─────────────────────────────────────────────
def build_grid(cells: List[dict], n_rows: int, n_cols: int) -> List[List[dict]]:
    default = {"row_span": 1, "col_span": 1, "text": ""}
    M = [[default] * n_cols for _ in range(n_rows)]
    for c in cells:
        sr = max(0, min(c["start_row"], n_rows - 1))
        er = max(0, min(c["end_row"], n_rows - 1))
        sc = max(0, min(c["start_col"], n_cols - 1))
        ec = max(0, min(c["end_col"], n_cols - 1))
        for r in range(sr, er + 1):
            row = list(M[r])
            for cc in range(sc, ec + 1):
                row[cc] = c
            M[r] = row
    return M


# ─────────────────────────────────────────────
# 1D Needleman-Wunsch sequence alignment
# (skip cost = 0, match = sim_fn)
# ─────────────────────────────────────────────
def nw_alignment_score(
    seq_a: List[dict], seq_b: List[dict],
    sim_fn,
) -> float:
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        row_prev = dp[i - 1]
        row_cur = dp[i]
        ai = seq_a[i - 1]
        for j in range(1, n + 1):
            m_ = row_prev[j - 1] + sim_fn(ai, seq_b[j - 1])
            s1 = row_prev[j]
            s2 = row_cur[j - 1]
            row_cur[j] = m_ if m_ >= s1 and m_ >= s2 else (s1 if s1 >= s2 else s2)
    return dp[m][n]


# ─────────────────────────────────────────────
# Factored 2D-MSS GriTS
# ─────────────────────────────────────────────
def grits_factored(
    pred_cells: List[dict], gt_cells: List[dict],
    nr_p: int, nc_p: int, nr_g: int, nc_g: int,
    sim_fn,
) -> Tuple[float, float, float]:
    """
    Factored GriTS: row-level NW where match score = col-level NW on that row pair.

    Returns (precision, recall, f1).
    """
    if nr_p == 0 or nc_p == 0 or nr_g == 0 or nc_g == 0:
        return 0.0, 0.0, 0.0

    grid_p = build_grid(pred_cells, nr_p, nc_p)
    grid_g = build_grid(gt_cells, nr_g, nc_g)

    # precompute row-pair similarity via column NW
    row_pair_sim = np.zeros((nr_p, nr_g), dtype=np.float64)
    for i in range(nr_p):
        for j in range(nr_g):
            row_pair_sim[i, j] = nw_alignment_score(grid_p[i], grid_g[j], sim_fn)

    # row-level NW
    dp = np.zeros((nr_p + 1, nr_g + 1), dtype=np.float64)
    for i in range(1, nr_p + 1):
        for j in range(1, nr_g + 1):
            m_ = dp[i - 1, j - 1] + row_pair_sim[i - 1, j - 1]
            s1 = dp[i - 1, j]
            s2 = dp[i, j - 1]
            dp[i, j] = max(m_, s1, s2)

    total_sim = float(dp[nr_p, nr_g])
    pred_n = nr_p * nc_p
    gt_n = nr_g * nc_g
    precision = total_sim / pred_n if pred_n else 0.0
    recall = total_sim / gt_n if gt_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ─────────────────────────────────────────────
# TATR inference (동일 로직)
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
    mk = {"pixel_values", "pixel_mask"}
    model_enc = {k: v.to(DEVICE) for k, v in enc.items() if k in mk}
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
        entry = {"box": box.tolist(), "score": float(sc)}
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
                    "text": "",
                })
    if spans:
        merged_idx = set()
        merged = []
        for sdata in spans:
            sb = sdata["box"]
            ov = [(idx, c) for idx, c in enumerate(cells)
                  if idx not in merged_idx and bbox_iou(sb, c["bbox"]) > IOU_MERGE]
            if ov:
                rs = [c["start_row"] for _, c in ov]
                cs = [c["start_col"] for _, c in ov]
                r0, r1 = min(rs), max(rs); c0, c1 = min(cs), max(cs)
                merged.append({
                    "start_row": r0, "end_row": r1,
                    "start_col": c0, "end_col": c1,
                    "row_span": r1 - r0 + 1, "col_span": c1 - c0 + 1,
                    "bbox": sb, "text": "",
                })
                for idx, _ in ov:
                    merged_idx.add(idx)
        cells = merged + [c for i, c in enumerate(cells) if i not in merged_idx]
    return {"cells": cells, "n_rows": len(rows), "n_cols": len(cols)}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
MODELS = [
    {"id": "microsoft/table-transformer-structure-recognition",           "label": "TATR v1.0"},
    {"id": "microsoft/table-transformer-structure-recognition-v1.1-pub",  "label": "TATR v1.1-pub"},
    {"id": "microsoft/table-transformer-structure-recognition-v1.1-all",  "label": "TATR v1.1-all"},
]


def _span_only_grits_top(pred_cells: List[dict], gt_cells: List[dict]) -> Optional[float]:
    """
    GT/pred spanning cell 만 대상으로 GriTS-Top 계산.
    GT에 spanning cell이 없으면 None, pred에 없으면 0.0 반환.
    """
    gt_spans = [c for c in gt_cells if c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1]
    pr_spans = [c for c in pred_cells if c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1]
    if not gt_spans:
        return None
    if not pr_spans:
        return 0.0
    gt_r = max(c["end_row"] for c in gt_spans) + 1
    gt_c = max(c["end_col"] for c in gt_spans) + 1
    pr_r = max(c["end_row"] for c in pr_spans) + 1
    pr_c = max(c["end_col"] for c in pr_spans) + 1
    _, _, f1 = grits_factored(pr_spans, gt_spans, pr_r, pr_c, gt_r, gt_c, sim_top)
    return f1


def _trivial_grits_top(gt_cells: List[dict], nr_g: int, nc_g: int,
                       nr_p: int, nc_p: int) -> float:
    """
    Trivial baseline: pred grid 크기로 모든 cell을 1×1로 채웠을 때 GriTS-Top.
    (GriTS-Top이 span이 아닌 grid skeleton 정렬로 얼마나 올라가는지 기준선)
    """
    trivial = [
        {"start_row": r, "end_row": r, "start_col": c, "end_col": c,
         "row_span": 1, "col_span": 1, "text": ""}
        for r in range(nr_p) for c in range(nc_p)
    ]
    _, _, f1 = grits_factored(trivial, gt_cells, nr_p, nc_p, nr_g, nc_g, sim_top)
    return f1


def run():
    valid_ids = load_valid_ids()
    print(f"[grits_eval] evaluating on {len(valid_ids)} SciTSR-COMP tables", flush=True)

    # Span fraction 통계 (GT 기준)
    span_fracs = []
    for tid in valid_ids:
        try:
            gt = load_gt(tid)
            grid_total = sum(c["row_span"] * c["col_span"] for c in gt["cells"])
            grid_span = sum(c["row_span"] * c["col_span"]
                           for c in gt["cells"] if c["row_span"] > 1 or c["col_span"] > 1)
            if grid_total > 0:
                span_fracs.append(grid_span / grid_total)
        except Exception:
            pass
    mean_span_frac = float(np.mean(span_fracs)) if span_fracs else 0.0
    print(f"  GT span fraction (avg): {mean_span_frac*100:.1f}%", flush=True)

    all_results: Dict[str, List[dict]] = {}
    summary: Dict[str, dict] = {}

    for m in MODELS:
        print(f"\n=== {m['label']} ===", flush=True)
        processor, model = _load_tatr(m["id"])
        records = []
        err = 0
        for tid in tqdm(valid_ids, desc=m["label"]):
            try:
                gt = load_gt(tid)
                img = load_image(tid)
                pred = _run_tatr(processor, model, img)
                nr_p, nc_p = pred["n_rows"], pred["n_cols"]
                nr_g, nc_g = gt["n_rows"], gt["n_cols"]
                if nr_p == 0 or nc_p == 0 or nr_g == 0 or nc_g == 0:
                    continue
                p_t, r_t, f_t = grits_factored(
                    pred["cells"], gt["cells"], nr_p, nc_p, nr_g, nc_g, sim_top)
                f_trivial = _trivial_grits_top(gt["cells"], nr_g, nc_g, nr_p, nc_p)
                f_span = _span_only_grits_top(pred["cells"], gt["cells"])
                # GriTS-Con은 OCR 텍스트 없는 구조 전용 모델에는 N/A
                records.append({
                    "table_id": tid,
                    "gt_n_rows": nr_g, "gt_n_cols": nc_g,
                    "pred_n_rows": nr_p, "pred_n_cols": nc_p,
                    "grits_top_p": p_t, "grits_top_r": r_t, "grits_top_f1": f_t,
                    "grits_top_f1_trivial": f_trivial,
                    "grits_top_f1_span_only": f_span,  # None if no GT spans
                    "grits_con_f1": None,  # N/A: structural-only model has no text output
                })
            except Exception:
                err += 1
                continue
        print(f"  processed={len(records)}  errors={err}", flush=True)
        all_results[m["label"]] = records

        if records:
            top_f1s    = [r["grits_top_f1"] for r in records]
            triv_f1s   = [r["grits_top_f1_trivial"] for r in records]
            span_f1s   = [r["grits_top_f1_span_only"] for r in records if r["grits_top_f1_span_only"] is not None]
            summary[m["label"]] = {
                "n_tables": len(records),
                "grits_top_p":              float(np.mean([r["grits_top_p"] for r in records])),
                "grits_top_r":              float(np.mean([r["grits_top_r"] for r in records])),
                "grits_top_f1":             float(np.mean(top_f1s)),
                "grits_top_f1_trivial":     float(np.mean(triv_f1s)),
                "grits_top_f1_span_only":   float(np.mean(span_f1s)) if span_f1s else None,
                "grits_top_delta_vs_trivial": float(np.mean(top_f1s) - np.mean(triv_f1s)),
                "grits_con_f1":             None,  # N/A for structural-only models
                "gt_mean_span_fraction":    mean_span_frac,
            }

    (RESULTS_DIR / "grits_per_table.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8")
    (RESULTS_DIR / "grits_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("GriTS EVALUATION SUMMARY")
    print("=" * 64)
    print(f"  GT span fraction: {mean_span_frac*100:.1f}%")
    for lbl, s in summary.items():
        delta = s["grits_top_delta_vs_trivial"]
        sign = "+" if delta >= 0 else ""
        print(f"\n{lbl}  (n={s['n_tables']})")
        print(f"  GriTS-Top F1        : {s['grits_top_f1']:.4f}")
        print(f"  GriTS-Top trivial   : {s['grits_top_f1_trivial']:.4f}  (Δ={sign}{delta:.4f})")
        print(f"  GriTS-Top span-only : {s['grits_top_f1_span_only']}")
        print(f"  GriTS-Con           : N/A (no text output)")

    make_figure(summary)


def make_figure(summary: dict):
    labels = list(summary.keys())
    top_f1   = [summary[l]["grits_top_f1"] for l in labels]
    triv_f1  = [summary[l]["grits_top_f1_trivial"] for l in labels]
    span_f1  = [summary[l]["grits_top_f1_span_only"] or 0.0 for l in labels]
    top_p    = [summary[l]["grits_top_p"] for l in labels]
    top_r    = [summary[l]["grits_top_r"] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(labels)); w = 0.28
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # Panel 1: full vs trivial baseline
    ax = axes[0]
    ax.bar(x - w/2, top_f1,  width=w, label="GriTS-Top (model)", color="#4C72B0")
    ax.bar(x + w/2, triv_f1, width=w, label="GriTS-Top (trivial all-1×1)", color="#AAAAAA")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_title("Full GriTS-Top vs Trivial Baseline"); ax.legend(fontsize=8)
    for xi, v in zip(x - w/2, top_f1):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w/2, triv_f1):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # Panel 2: span-only GriTS-Top
    ax = axes[1]
    ax.bar(x, span_f1, color=colors[:len(labels)])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_title("GriTS-Top (Spanning Cells Only)")
    for xi, v in zip(x, span_f1):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # Panel 3: P/R breakdown
    ax = axes[2]
    ax.bar(x - w/2, top_p, width=w, label="P", color="#55A868")
    ax.bar(x + w/2, top_r, width=w, label="R", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_title("GriTS-Top P/R"); ax.legend(fontsize=8)

    plt.suptitle("GriTS (Smock CVPR'23) — TATR on SciTSR-COMP\n"
                 "(GriTS-Con N/A: models output structure only, no text)", fontsize=11)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "grits_eval.png", dpi=150)
    plt.close()
    print(f"\n[saved] {RESULTS_DIR / 'grits_eval.png'}")


if __name__ == "__main__":
    run()

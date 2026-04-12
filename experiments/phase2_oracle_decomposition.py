#!/usr/bin/env python3
"""
Phase 2 — Oracle Decomposition
================================
"Stage 1 (detection) vs Stage 2 (grid reconstruction) — 어디가 병목인가?"

데이터: FinTabNet complex 68개
  → XML에 GT row/col/span bbox 모두 포함 → 올바른 Oracle 실험 가능
  SciTSR-COMP 716개는 Logical-Exact Rate 보조 측정에 사용

Oracle 실험 (FinTabNet complex):
  A. Full TATR       : 현재 파이프라인 (detect row/col/span → Stage 2)
  B. Oracle Row/Col  : GT row/col bbox + TATR span bbox
     → "Stage 2만 바꿨을 때" 성능 → Stage 2 병목 증명
  C. Full Oracle     : GT row/col/span bbox (상한선)
  D. Oracle Span     : TATR row/col bbox + GT span bbox
     → Stage 1의 span detection 기여 격리

추가 Logical-Exact Rate 분석 (SciTSR COMP):
  - Full TATR의 span 논리 인덱스 복원 실패율 측정

Noise Sensitivity (FinTabNet):
  GT row/col bbox에 가우시안 노이즈 σ=0~30px 주입
  → Stage 2의 구조적 민감도 정량화

출력:
  experiments/results_phase2/
    oracle_summary.json
    fig5_oracle_decomposition.png   ← 4개 Oracle 비교
    fig6_noise_sensitivity.png
    fig7_logical_exact_scitsr.png   ← SciTSR Logical-Exact
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import xml.etree.ElementTree as ET

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
    load_comp_ids, load_gt, load_image,
    load_fintabnet_gt, load_fintabnet_image,
    FINTAB_ANN, bbox_iou,
)
from phase1_problem_proof import (
    load_tatr, run_tatr, grits_top, grits_top_span_only,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results_phase2"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "microsoft/table-transformer-structure-recognition-v1.1-all"
MODEL_LABEL = "TATR v1.1-all"
IOU_MERGE   = 0.3
IOU_ASSIGN  = 0.1  # span-to-cell 할당 threshold


# ─────────────────────────────────────────────
# FinTabNet XML → GT row/col/span bbox 파싱
# ─────────────────────────────────────────────
def parse_fintabnet_xml_bboxes(xml_path: Path) -> Optional[dict]:
    """
    FinTabNet XML → {rows_bbox, cols_bbox, spans_bbox, is_complex}
    원본 GT bbox 그대로 사용, 가공 없음.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None

    rows_b, cols_b, spans_b = [], [], []
    for o in root.findall("object"):
        name = o.findtext("name", "").strip().lower()
        bb = o.find("bndbox")
        if bb is None:
            continue
        box = [
            float(bb.findtext("xmin", 0)),
            float(bb.findtext("ymin", 0)),
            float(bb.findtext("xmax", 0)),
            float(bb.findtext("ymax", 0)),
        ]
        if name == "table row":
            rows_b.append(box)
        elif name == "table column":
            cols_b.append(box)
        elif name == "table spanning cell":
            spans_b.append(box)

    if not rows_b or not cols_b:
        return None

    rows_b.sort(key=lambda b: (b[1]+b[3])/2)
    cols_b.sort(key=lambda b: (b[0]+b[2])/2)
    return {
        "rows_bbox":  rows_b,
        "cols_bbox":  cols_b,
        "spans_bbox": spans_b,
        "is_complex": len(spans_b) > 0,
    }


# ─────────────────────────────────────────────
# Grid reconstruction (Stage 2)
# ─────────────────────────────────────────────
def reconstruct_grid(rows_bbox: List[list], cols_bbox: List[list],
                     spans_bbox: List[list]) -> dict:
    """rows/cols/spans bbox → cells with logical indices."""
    cells = []
    for ri, rb in enumerate(rows_bbox):
        for ci, cb in enumerate(cols_bbox):
            x0 = max(cb[0], rb[0]); y0 = max(rb[1], cb[1])
            x1 = min(cb[2], rb[2]); y1 = min(rb[3], cb[3])
            if x1 > x0 and y1 > y0:
                cells.append({"start_row": ri, "end_row": ri,
                               "start_col": ci, "end_col": ci,
                               "row_span": 1, "col_span": 1,
                               "bbox": [x0, y0, x1, y1]})
    if spans_bbox:
        merged_idx = set()
        merged = []
        for sb in spans_bbox:
            ov = [(i, c) for i, c in enumerate(cells)
                  if i not in merged_idx and bbox_iou(sb, c["bbox"]) > IOU_MERGE]
            if ov:
                rs = [c["start_row"] for _, c in ov]
                cs = [c["start_col"] for _, c in ov]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                merged.append({"start_row": r0, "end_row": r1,
                                "start_col": c0, "end_col": c1,
                                "row_span": r1-r0+1, "col_span": c1-c0+1,
                                "bbox": sb})
                for i, _ in ov:
                    merged_idx.add(i)
        cells = merged + [c for i, c in enumerate(cells) if i not in merged_idx]

    return {"cells": cells, "n_rows": len(rows_bbox), "n_cols": len(cols_bbox)}


# ─────────────────────────────────────────────
# Logical-Exact Rate (FinTabNet)
# GT 논리 인덱스: XML GT bbox 기반으로 도출
# ─────────────────────────────────────────────
def derive_gt_logical_spans(gt_xml: dict) -> List[Tuple[int,int,int,int]]:
    """
    GT XML row/col/span bbox → GT spanning cell의 (sr,er,sc,ec).
    GT row/col bbox 기준 nearest-neighbor로 각 span의 논리 인덱스 결정.
    """
    rows_b = gt_xml["rows_bbox"]
    cols_b = gt_xml["cols_bbox"]
    spans_b = gt_xml["spans_bbox"]
    if not spans_b:
        return []

    gt_spans = []
    for sb in spans_b:
        # span bbox와 각 row/col bbox의 overlap 비율로 할당
        row_overlaps = []
        for ri, rb in enumerate(rows_b):
            y0 = max(sb[1], rb[1]); y1 = min(sb[3], rb[3])
            h_inter = max(0, y1 - y0)
            h_span = sb[3] - sb[1]
            if h_span > 0 and h_inter / h_span > IOU_ASSIGN:
                row_overlaps.append(ri)

        col_overlaps = []
        for ci, cb in enumerate(cols_b):
            x0 = max(sb[0], cb[0]); x1 = min(sb[2], cb[2])
            w_inter = max(0, x1 - x0)
            w_span = sb[2] - sb[0]
            if w_span > 0 and w_inter / w_span > IOU_ASSIGN:
                col_overlaps.append(ci)

        if row_overlaps and col_overlaps:
            sr, er = min(row_overlaps), max(row_overlaps)
            sc, ec = min(col_overlaps), max(col_overlaps)
            if er > sr or ec > sc:  # 실제 spanning
                gt_spans.append((sr, er, sc, ec))
    return gt_spans


def logical_exact_rate(pred_cells: List[dict],
                       gt_spans: List[Tuple[int,int,int,int]]) -> Optional[float]:
    """
    GT spanning cell (sr,er,sc,ec)을 pred에서 정확히 맞춘 비율.
    GT spans 없으면 None.
    """
    if not gt_spans:
        return None
    pred_spans = {(c["start_row"], c["end_row"], c["start_col"], c["end_col"])
                  for c in pred_cells
                  if c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1}
    matched = sum(1 for s in gt_spans if s in pred_spans)
    return matched / len(gt_spans)


# ─────────────────────────────────────────────
# Oracle variants
# ─────────────────────────────────────────────
def oracle_full_tatr(proc, mdl, image: Image.Image,
                     gt_xml: dict, gt_struct: dict) -> Optional[dict]:
    """Exp A: Full TATR."""
    pred = run_tatr(proc, mdl, image)
    nr_p, nc_p = pred["n_rows"], pred["n_cols"]
    nr_g, nc_g = gt_struct["n_rows"], gt_struct["n_cols"]
    if nr_p == 0 or nc_p == 0 or nr_g == 0 or nc_g == 0:
        return None
    _, _, f_full = grits_top(pred["cells"], gt_struct["cells"], nr_p, nc_p, nr_g, nc_g)
    f_span = grits_top_span_only(pred["cells"], gt_struct["cells"])
    gt_spans = derive_gt_logical_spans(gt_xml)
    le = logical_exact_rate(pred["cells"], gt_spans)
    return {"grits_top_f1": f_full, "grits_top_span_only": f_span, "logical_exact": le}


def oracle_gt_rowcol(proc, mdl, image: Image.Image,
                     gt_xml: dict, gt_struct: dict,
                     noise_sigma: float = 0.0) -> Optional[dict]:
    """Exp B: GT row/col + TATR span (± noise)."""
    rows_b = [list(b) for b in gt_xml["rows_bbox"]]
    cols_b = [list(b) for b in gt_xml["cols_bbox"]]

    if noise_sigma > 0:
        rng = np.random.default_rng(42)
        for b in rows_b:
            b[1] += float(rng.normal(0, noise_sigma))
            b[3] += float(rng.normal(0, noise_sigma))
        for b in cols_b:
            b[0] += float(rng.normal(0, noise_sigma))
            b[2] += float(rng.normal(0, noise_sigma))

    tatr_out = run_tatr(proc, mdl, image)
    spans_bbox = tatr_out["spans_bbox"]

    pred = reconstruct_grid(rows_b, cols_b, spans_bbox)
    nr_p, nc_p = pred["n_rows"], pred["n_cols"]
    nr_g, nc_g = gt_struct["n_rows"], gt_struct["n_cols"]
    if nr_p == 0 or nc_p == 0 or nr_g == 0 or nc_g == 0:
        return None

    _, _, f_full = grits_top(pred["cells"], gt_struct["cells"], nr_p, nc_p, nr_g, nc_g)
    f_span = grits_top_span_only(pred["cells"], gt_struct["cells"])
    gt_spans = derive_gt_logical_spans(gt_xml)
    le = logical_exact_rate(pred["cells"], gt_spans)
    return {"grits_top_f1": f_full, "grits_top_span_only": f_span,
            "logical_exact": le, "noise_sigma": noise_sigma}


def oracle_full_gt(gt_xml: dict, gt_struct: dict) -> Optional[dict]:
    """Exp C: GT row/col/span (전부 GT — 상한선)."""
    rows_b = gt_xml["rows_bbox"]
    cols_b = gt_xml["cols_bbox"]
    spans_b = gt_xml["spans_bbox"]

    pred = reconstruct_grid(rows_b, cols_b, spans_b)
    nr_p, nc_p = pred["n_rows"], pred["n_cols"]
    nr_g, nc_g = gt_struct["n_rows"], gt_struct["n_cols"]
    if nr_p == 0 or nc_p == 0 or nr_g == 0 or nc_g == 0:
        return None

    _, _, f_full = grits_top(pred["cells"], gt_struct["cells"], nr_p, nc_p, nr_g, nc_g)
    f_span = grits_top_span_only(pred["cells"], gt_struct["cells"])
    gt_spans = derive_gt_logical_spans(gt_xml)
    le = logical_exact_rate(pred["cells"], gt_spans)
    return {"grits_top_f1": f_full, "grits_top_span_only": f_span, "logical_exact": le}


def oracle_gt_span(proc, mdl, image: Image.Image,
                   gt_xml: dict, gt_struct: dict) -> Optional[dict]:
    """Exp D: TATR row/col + GT span."""
    tatr_out = run_tatr(proc, mdl, image)
    # TATR의 row/col bbox 추출
    tatr_rows, tatr_cols = _extract_tatr_rowcol(proc, mdl, image)
    spans_b = gt_xml["spans_bbox"]

    pred = reconstruct_grid(tatr_rows, tatr_cols, spans_b)
    nr_p, nc_p = pred["n_rows"], pred["n_cols"]
    nr_g, nc_g = gt_struct["n_rows"], gt_struct["n_cols"]
    if nr_p == 0 or nc_p == 0 or nr_g == 0 or nc_g == 0:
        return None

    _, _, f_full = grits_top(pred["cells"], gt_struct["cells"], nr_p, nc_p, nr_g, nc_g)
    f_span = grits_top_span_only(pred["cells"], gt_struct["cells"])
    gt_spans = derive_gt_logical_spans(gt_xml)
    le = logical_exact_rate(pred["cells"], gt_spans)
    return {"grits_top_f1": f_full, "grits_top_span_only": f_span, "logical_exact": le}


@torch.no_grad()
def _extract_tatr_rowcol(proc, mdl, image: Image.Image):
    """TATR inference에서 row/col bbox만 추출."""
    enc = proc(images=image, return_tensors="pt")
    out = mdl(**{k: v.to(DEVICE) for k, v in enc.items()
                 if k in {"pixel_values", "pixel_mask"}})
    res = proc.post_process_object_detection(
        out, threshold=0.5, target_sizes=[(image.size[1], image.size[0])])[0]
    boxes  = res["boxes"].cpu().numpy()
    labels = res["labels"].cpu().numpy()
    id2l   = mdl.config.id2label
    rows, cols = [], []
    for box, lid in zip(boxes, labels):
        lname = id2l[lid].lower()
        if "column" in lname:
            cols.append(box.tolist())
        elif "spanning" not in lname and "projected" not in lname:
            rows.append(box.tolist())
    rows.sort(key=lambda b: (b[1]+b[3])/2)
    cols.sort(key=lambda b: (b[0]+b[2])/2)
    return rows, cols


# ─────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────
def agg(records: List[dict]) -> dict:
    if not records:
        return {"n": 0}
    f1s  = [r["grits_top_f1"] for r in records]
    spans = [r["grits_top_span_only"] for r in records
             if r.get("grits_top_span_only") is not None]
    lexs  = [r["logical_exact"] for r in records
             if r.get("logical_exact") is not None]
    return {
        "n":                   len(records),
        "grits_top_f1":        float(np.mean(f1s)),
        "grits_top_span_only": float(np.mean(spans)) if spans else None,
        "logical_exact":       float(np.mean(lexs))  if lexs  else None,
        "logical_exact_std":   float(np.std(lexs))   if lexs  else None,
    }


# ─────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────
def fig5_oracle_decomposition(results: Dict[str, dict]):
    """
    Fig 5: 4개 Oracle 조건 비교.
    A=Full TATR / B=Oracle Row/Col / C=Full GT / D=Oracle Span
    """
    conditions = [
        ("A: Full TATR\n(Stage1+Stage2)",         "#CC4444"),
        ("B: Oracle Row/Col\n(GT row/col+TATR span)", "#55A868"),
        ("C: Full Oracle\n(All GT bbox)",           "#2C5F8A"),
        ("D: Oracle Span\n(TATR row/col+GT span)",  "#DD8452"),
    ]
    keys = ["exp_A", "exp_B_sigma0", "exp_C", "exp_D"]
    metrics = ["grits_top_f1", "grits_top_span_only", "logical_exact"]
    metric_labels = ["GriTS-Top F1", "Span-Only F1", "Logical-Exact Rate"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Oracle Decomposition: Stage 1 vs Stage 2 Bottleneck Analysis\n"
                 f"(FinTabNet complex, n={results.get('exp_A',{}).get('n','?')}, {MODEL_LABEL})",
                 fontsize=12, fontweight="bold")

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        vals = []
        colors = []
        labels = []
        for key, (label, color) in zip(keys, conditions):
            r = results.get(key, {})
            v = r.get(metric)
            vals.append(v if v is not None else 0)
            colors.append(color)
            labels.append(label)

        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(mlabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        # A→B 향상 화살표
        if len(vals) >= 2 and vals[1] > vals[0] + 0.01:
            ax.annotate("", xy=(1, vals[1]+0.04), xytext=(0, vals[0]+0.04),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
            ax.text(0.5, max(vals[0],vals[1])+0.07,
                    f"+{vals[1]-vals[0]:.3f}", ha="center", color="red",
                    fontsize=9, fontweight="bold")

    plt.tight_layout()
    p = RESULTS_DIR / "fig5_oracle_decomposition.png"
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"[saved] {p}")


def fig6_noise_sensitivity(noise_results: Dict[int, dict]):
    """Fig 6: bbox 노이즈 σ vs 성능 저하."""
    sigmas = sorted(noise_results.keys())
    logical_means = [noise_results[s].get("logical_exact") or 0 for s in sigmas]
    logical_stds  = [noise_results[s].get("logical_exact_std") or 0 for s in sigmas]
    span_f1s      = [noise_results[s].get("grits_top_span_only") or 0 for s in sigmas]
    grits_f1s     = [noise_results[s].get("grits_top_f1") or 0 for s in sigmas]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Grid Reconstruction Sensitivity to Row/Col BBox Noise\n"
                 f"(FinTabNet complex, Oracle Row/Col + TATR span, {MODEL_LABEL})",
                 fontsize=12, fontweight="bold")

    for ax, vals, stds, title, color in [
        (axes[0], logical_means, logical_stds, "Logical-Exact Rate", "#CC0000"),
        (axes[1], span_f1s,      [0]*len(sigmas), "Span-Only F1",    "#4C72B0"),
        (axes[2], grits_f1s,     [0]*len(sigmas), "GriTS-Top F1",    "#55A868"),
    ]:
        ax.errorbar(sigmas, vals, yerr=stds, marker="o", color=color,
                    linewidth=2, capsize=4, markersize=6)
        if any(s > 0 for s in stds):
            ax.fill_between(sigmas,
                            [m-s for m,s in zip(vals,stds)],
                            [m+s for m,s in zip(vals,stds)],
                            alpha=0.15, color=color)
        for sigma, val in zip(sigmas, vals):
            ax.text(sigma, val+0.02, f"{val:.3f}", ha="center", fontsize=8)
        ax.set_xlabel("Gaussian Noise σ (pixels)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    p = RESULTS_DIR / "fig6_noise_sensitivity.png"
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"[saved] {p}")


def fig7_logical_exact_scitsr(results_scitsr: dict):
    """Fig 7: SciTSR COMP — Logical-Exact Rate distribution (per-table histogram)."""
    lexs = [r["logical_exact"] for r in results_scitsr if r.get("logical_exact") is not None]
    if not lexs:
        print("[fig7] No logical_exact data for SciTSR")
        return

    mean_le = np.mean(lexs)
    zero_frac = sum(1 for v in lexs if v == 0) / len(lexs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"SciTSR-COMP: Logical-Exact Rate Distribution\n"
                 f"({MODEL_LABEL}, n={len(lexs)} complex tables)",
                 fontsize=12, fontweight="bold")

    ax1.hist(lexs, bins=20, color="#CC4444", alpha=0.75, edgecolor="white")
    ax1.axvline(mean_le, color="black", linestyle="--", linewidth=1.5,
                label=f"Mean={mean_le:.4f}")
    ax1.set_xlabel("Logical-Exact Rate (per table)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of per-table Logical-Exact Rate")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # zero vs non-zero pie
    ax2.pie([zero_frac, 1-zero_frac],
            labels=[f"0.00 (complete failure)\n{zero_frac:.1%}",
                    f">0.00 (partial success)\n{1-zero_frac:.1%}"],
            colors=["#CC4444", "#55A868"], autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10})
    ax2.set_title(f"Logical-Exact Rate = 0 (full failure)\nvs >0 (any span correct)")

    plt.tight_layout()
    p = RESULTS_DIR / "fig7_logical_exact_scitsr.png"
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run():
    print("=" * 60)
    print("Phase 2: Oracle Decomposition Experiments")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_LABEL}")
    print("=" * 60)

    # ── FinTabNet complex 로드 ──────────────────
    print("\n[Data] Loading FinTabNet complex tables...")
    ft_items = []
    for xml_f in sorted(FINTAB_ANN.glob("*.xml")):
        img = load_fintabnet_image(xml_f)
        if img is None:
            continue
        gt_struct = load_fintabnet_gt(xml_f)
        if gt_struct is None or not gt_struct["is_complex"]:
            continue
        gt_xml = parse_fintabnet_xml_bboxes(xml_f)
        if gt_xml is None or not gt_xml["is_complex"]:
            continue
        ft_items.append({"gt_struct": gt_struct, "gt_xml": gt_xml,
                          "image": img, "xml_f": xml_f})
    print(f"  FinTabNet complex: {len(ft_items)} tables")

    proc, mdl = load_tatr(MODEL_ID)

    # ── Exp A: Full TATR ────────────────────────
    print("\n[Exp A] Full TATR (baseline)...")
    recs_A = []
    for item in tqdm(ft_items, desc="Exp A"):
        try:
            r = oracle_full_tatr(proc, mdl, item["image"],
                                 item["gt_xml"], item["gt_struct"])
            if r: recs_A.append(r)
        except Exception:
            continue
    res_A = agg(recs_A)
    print(f"  n={res_A['n']}, "
          f"F1={res_A['grits_top_f1']:.4f}, "
          f"SpanF1={res_A.get('grits_top_span_only') or 0:.4f}, "
          f"LogicalExact={res_A.get('logical_exact') or 0:.4f}")

    # ── Exp B: Oracle Row/Col (σ=0) ────────────
    print("\n[Exp B] Oracle Row/Col (GT bbox, σ=0)...")
    recs_B0 = []
    for item in tqdm(ft_items, desc="Exp B σ=0"):
        try:
            r = oracle_gt_rowcol(proc, mdl, item["image"],
                                 item["gt_xml"], item["gt_struct"],
                                 noise_sigma=0.0)
            if r: recs_B0.append(r)
        except Exception:
            continue
    res_B0 = agg(recs_B0)
    print(f"  n={res_B0['n']}, "
          f"F1={res_B0['grits_top_f1']:.4f}, "
          f"SpanF1={res_B0.get('grits_top_span_only') or 0:.4f}, "
          f"LogicalExact={res_B0.get('logical_exact') or 0:.4f}")

    # ── Exp C: Full Oracle (GT row/col/span) ────
    print("\n[Exp C] Full Oracle (all GT bbox)...")
    recs_C = []
    for item in tqdm(ft_items, desc="Exp C"):
        try:
            r = oracle_full_gt(item["gt_xml"], item["gt_struct"])
            if r: recs_C.append(r)
        except Exception:
            continue
    res_C = agg(recs_C)
    print(f"  n={res_C['n']}, "
          f"F1={res_C['grits_top_f1']:.4f}, "
          f"SpanF1={res_C.get('grits_top_span_only') or 0:.4f}, "
          f"LogicalExact={res_C.get('logical_exact') or 0:.4f}")

    # ── Exp D: Oracle Span ──────────────────────
    print("\n[Exp D] Oracle Span (TATR row/col + GT span)...")
    recs_D = []
    for item in tqdm(ft_items, desc="Exp D"):
        try:
            r = oracle_gt_span(proc, mdl, item["image"],
                               item["gt_xml"], item["gt_struct"])
            if r: recs_D.append(r)
        except Exception:
            continue
    res_D = agg(recs_D)
    print(f"  n={res_D['n']}, "
          f"F1={res_D['grits_top_f1']:.4f}, "
          f"SpanF1={res_D.get('grits_top_span_only') or 0:.4f}, "
          f"LogicalExact={res_D.get('logical_exact') or 0:.4f}")

    # ── Exp C: Noise Sensitivity ────────────────
    print("\n[Exp Noise] Oracle Row/Col + noise σ = 0, 5, 10, 15, 20, 30 px...")
    sigmas = [0, 5, 10, 15, 20, 30]
    noise_results: Dict[int, dict] = {}
    for sigma in sigmas:
        recs_n = []
        for item in tqdm(ft_items, desc=f"  σ={sigma}px"):
            try:
                r = oracle_gt_rowcol(proc, mdl, item["image"],
                                     item["gt_xml"], item["gt_struct"],
                                     noise_sigma=float(sigma))
                if r: recs_n.append(r)
            except Exception:
                continue
        noise_results[sigma] = agg(recs_n)
        le_str = f"{noise_results[sigma].get('logical_exact'):.4f}" \
                 if noise_results[sigma].get("logical_exact") is not None else "N/A"
        print(f"  σ={sigma:2d}px → LogicalExact={le_str}, "
              f"SpanF1={noise_results[sigma].get('grits_top_span_only') or 0:.4f}")

    # ── SciTSR Logical-Exact (보조) ─────────────
    print("\n[SciTSR] Logical-Exact Rate (COMP complex tables)...")
    comp_ids = load_comp_ids()
    scitsr_le_records = []
    for tid in tqdm(comp_ids, desc="SciTSR Logical-Exact"):
        try:
            gt = load_gt(tid)
            img = load_image(tid)
            # GT spans (logical indices only, no pixel bbox)
            gt_spans_logical = [(c["start_row"], c["end_row"],
                                  c["start_col"], c["end_col"])
                                 for c in gt["cells"]
                                 if c.get("row_span",1)>1 or c.get("col_span",1)>1]
            if not gt_spans_logical:
                continue
            tatr_out = run_tatr(proc, mdl, img)
            le = logical_exact_rate(tatr_out["cells"], gt_spans_logical)
            scitsr_le_records.append({"table_id": tid, "logical_exact": le,
                                       "n_gt_spans": len(gt_spans_logical)})
        except Exception:
            continue

    scitsr_le_mean = float(np.mean([r["logical_exact"] for r in scitsr_le_records
                                    if r["logical_exact"] is not None]))
    print(f"  SciTSR COMP LogicalExact mean: {scitsr_le_mean:.4f} (n={len(scitsr_le_records)})")

    # 저장
    summary = {
        "exp_A":          res_A,
        "exp_B_sigma0":   res_B0,
        "exp_C":          res_C,
        "exp_D":          res_D,
        "exp_noise":      {str(k): v for k, v in noise_results.items()},
        "scitsr_logical_exact": scitsr_le_mean,
        "bottleneck_summary": {
            "A_logical_exact":  res_A.get("logical_exact"),
            "B_logical_exact":  res_B0.get("logical_exact"),
            "C_logical_exact":  res_C.get("logical_exact"),
            "D_logical_exact":  res_D.get("logical_exact"),
            "B_vs_A_improvement": (res_B0.get("logical_exact") or 0)
                                   - (res_A.get("logical_exact") or 0),
            "C_vs_A_improvement": (res_C.get("logical_exact") or 0)
                                   - (res_A.get("logical_exact") or 0),
        }
    }
    with open(RESULTS_DIR / "oracle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {RESULTS_DIR / 'oracle_summary.json'}")

    # Figures
    print("\n[Figures] Generating...")
    fig5_oracle_decomposition(summary)
    fig6_noise_sensitivity(noise_results)
    fig7_logical_exact_scitsr(scitsr_le_records)

    del proc, mdl
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Phase 2 DONE.")
    print("Bottleneck Summary:")
    bs = summary["bottleneck_summary"]
    print(f"  Full TATR LogicalExact    : {bs['A_logical_exact']:.4f}")
    print(f"  Oracle Row/Col            : {bs['B_logical_exact']:.4f}  (Δ={bs['B_vs_A_improvement']:+.4f})")
    print(f"  Full Oracle (GT all)      : {bs['C_logical_exact']:.4f}  (Δ={bs['C_vs_A_improvement']:+.4f})")
    print(f"  Oracle Span (TATR rc+GT sp): {bs['D_logical_exact']:.4f}")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run()

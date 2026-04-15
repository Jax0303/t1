#!/usr/bin/env python3
"""
Phase 1 — Problem Proof
========================
"Detection-based TSR fails structurally on spanning cells."

실험 설계:
  - 데이터 가공 없음. 있는 그대로 전수 평가.
  - 공식 simple/complex 스플릿 사용:
      SciTSR  : SciTSR-COMP.list = complex (716), 나머지 = simple (2284)
      FinTabNet: 'table spanning cell' in XML = complex, 없으면 simple

  - 모델: TATR v1.0 / v1.1-pub / v1.1-all
  - Metric:
      GriTS-Top F1 (full)    : 전체 구조 정합성
      GriTS-Top F1 (span-only): spanning cell만 격리 평가
      Trivial-Delta          : trivial baseline(모든 셀 1×1) 대비 향상폭
      Span-Class Recall      : span bbox를 detect했는가 (IoU ≥ 0.1)

출력:
  experiments/results_phase1/
    summary.json
    per_table_scitsr.json
    per_table_fintabnet.json
    fig1_grits_simple_vs_complex.png  ← 핵심 figure
    fig2_trivial_delta.png
    fig3_span_class_recall.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

sys.path.insert(0, str(Path(__file__).parent))
from _eval_utils import (
    load_all_scitsr_ids, load_comp_ids, load_gt, load_image, scitsr_split,
    load_fintabnet_gt, load_fintabnet_image,
    FINTAB_ANN, bbox_iou,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results_phase1"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5
IOU_MERGE   = 0.3

MODELS = [
    {"id": "microsoft/table-transformer-structure-recognition",          "label": "TATR v1.0"},
    {"id": "microsoft/table-transformer-structure-recognition-v1.1-pub", "label": "TATR v1.1-pub"},
    {"id": "microsoft/table-transformer-structure-recognition-v1.1-all", "label": "TATR v1.1-all"},
]

COLORS = {
    "TATR v1.0":     "#4C72B0",
    "TATR v1.1-pub": "#DD8452",
    "TATR v1.1-all": "#55A868",
}


# ─────────────────────────────────────────────
# GriTS implementation (Smock CVPR'23)
# ─────────────────────────────────────────────
def sim_top(a: dict, b: dict) -> float:
    rs_a = a.get("row_span", 1); cs_a = a.get("col_span", 1)
    rs_b = b.get("row_span", 1); cs_b = b.get("col_span", 1)
    inter = min(rs_a, rs_b) * min(cs_a, cs_b)
    union = rs_a * cs_a + rs_b * cs_b - inter
    return inter / union if union else 0.0


def build_grid(cells: List[dict], n_rows: int, n_cols: int) -> List[List[dict]]:
    default = {"row_span": 1, "col_span": 1}
    M = [[default] * n_cols for _ in range(n_rows)]
    for c in cells:
        sr = max(0, min(c["start_row"], n_rows - 1))
        er = max(0, min(c["end_row"],   n_rows - 1))
        sc = max(0, min(c["start_col"], n_cols - 1))
        ec = max(0, min(c["end_col"],   n_cols - 1))
        for r in range(sr, er + 1):
            row = list(M[r])
            for cc in range(sc, ec + 1):
                row[cc] = c
            M[r] = row
    return M


def nw_score(seq_a: List[dict], seq_b: List[dict], sim_fn) -> float:
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i-1][j-1] + sim_fn(seq_a[i-1], seq_b[j-1])
            dp[i][j] = max(match, dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def grits_top(pred_cells, gt_cells, nr_p, nc_p, nr_g, nc_g) -> Tuple[float, float, float]:
    if nr_p == 0 or nc_p == 0 or nr_g == 0 or nc_g == 0:
        return 0.0, 0.0, 0.0
    gp = build_grid(pred_cells, nr_p, nc_p)
    gg = build_grid(gt_cells,   nr_g, nc_g)
    rps = np.zeros((nr_p, nr_g))
    for i in range(nr_p):
        for j in range(nr_g):
            rps[i, j] = nw_score(gp[i], gg[j], sim_top)
    dp = np.zeros((nr_p + 1, nr_g + 1))
    for i in range(1, nr_p + 1):
        for j in range(1, nr_g + 1):
            dp[i, j] = max(dp[i-1, j-1] + rps[i-1, j-1], dp[i-1, j], dp[i, j-1])
    total = float(dp[nr_p, nr_g])
    prec = total / (nr_p * nc_p) if nr_p * nc_p else 0.0
    rec  = total / (nr_g * nc_g) if nr_g * nc_g else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def grits_top_span_only(pred_cells, gt_cells) -> Optional[float]:
    """GT spanning cell만 격리 평가. GT에 span 없으면 None."""
    gt_sp = [c for c in gt_cells  if c.get("row_span",1)>1 or c.get("col_span",1)>1]
    pr_sp = [c for c in pred_cells if c.get("row_span",1)>1 or c.get("col_span",1)>1]
    if not gt_sp:
        return None
    if not pr_sp:
        return 0.0
    nr_g = max(c["end_row"] for c in gt_sp) + 1
    nc_g = max(c["end_col"] for c in gt_sp) + 1
    nr_p = max(c["end_row"] for c in pr_sp) + 1
    nc_p = max(c["end_col"] for c in pr_sp) + 1
    _, _, f1 = grits_top(pr_sp, gt_sp, nr_p, nc_p, nr_g, nc_g)
    return f1


def trivial_grits_top(gt_cells, nr_g, nc_g, nr_p, nc_p) -> float:
    """모든 셀을 1×1로 예측했을 때 GriTS-Top (trivial baseline)."""
    trivial = [
        {"start_row": r, "end_row": r, "start_col": c, "end_col": c,
         "row_span": 1, "col_span": 1}
        for r in range(nr_p) for c in range(nc_p)
    ]
    _, _, f1 = grits_top(trivial, gt_cells, nr_p, nc_p, nr_g, nc_g)
    return f1


# ─────────────────────────────────────────────
# TATR inference
# ─────────────────────────────────────────────
def load_tatr(model_id: str):
    import os
    import json
    import tempfile
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection, TableTransformerConfig

    proc = AutoImageProcessor.from_pretrained(model_id)
    # [FIX] HuggingFace image processor size validation
    if hasattr(proc, "size"):
        # Force both edges to be present if one is found
        if "longest_edge" in proc.size and ("shortest_edge" not in proc.size or proc.size["shortest_edge"] is None):
            le = proc.size["longest_edge"]
            proc.size["shortest_edge"] = le
            proc.size["longest_edge"] = le
        elif "shortest_edge" in proc.size and ("longest_edge" not in proc.size or proc.size["longest_edge"] is None):
            se = proc.size["shortest_edge"]
            proc.size["shortest_edge"] = se
            proc.size["longest_edge"] = se
    
    # [FIX] HuggingFace dilation validation error
    try:
        mdl = TableTransformerForObjectDetection.from_pretrained(model_id).to(DEVICE).eval()
    except Exception:
        # Manual Patch: download config.json, fix dilation, save to temp dir
        config_file = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        
        if "backbone_config" in config_dict and config_dict["backbone_config"] is not None:
             if config_dict["backbone_config"].get("dilation") is None:
                 config_dict["backbone_config"]["dilation"] = False
        if config_dict.get("dilation") is None:
            config_dict["dilation"] = False

        tmp_dir = tempfile.mkdtemp()
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        
        mdl = TableTransformerForObjectDetection.from_pretrained(model_id, config=tmp_dir).to(DEVICE).eval()
        
    return proc, mdl


@torch.no_grad()
def run_tatr(proc, mdl, image: Image.Image) -> dict:
    enc = proc(images=image, return_tensors="pt")
    out = mdl(**{k: v.to(DEVICE) for k, v in enc.items()
                 if k in {"pixel_values", "pixel_mask"}})
    res = proc.post_process_object_detection(
        out, threshold=CONF_THRESH, target_sizes=[(image.size[1], image.size[0])])[0]
    boxes  = res["boxes"].cpu().numpy()
    labels = res["labels"].cpu().numpy()
    id2l   = mdl.config.id2label
    rows, cols, spans = [], [], []
    for box, lid in zip(boxes, labels):
        lname = id2l[lid].lower()
        e = {"box": box.tolist()}
        if "column" in lname:          cols.append(e)
        elif "spanning" in lname or "projected" in lname: spans.append(e)
        else:                          rows.append(e)
    rows.sort(key=lambda r: (r["box"][1]+r["box"][3])/2)
    cols.sort(key=lambda c: (c["box"][0]+c["box"][2])/2)

    cells = []
    for ri, rd in enumerate(rows):
        for ci, cd in enumerate(cols):
            rb, cb = rd["box"], cd["box"]
            x0,y0 = max(cb[0],rb[0]), max(rb[1],cb[1])
            x1,y1 = min(cb[2],rb[2]), min(rb[3],cb[3])
            if x1>x0 and y1>y0:
                cells.append({"start_row":ri,"end_row":ri,
                               "start_col":ci,"end_col":ci,
                               "row_span":1,"col_span":1,
                               "bbox":[x0,y0,x1,y1]})
    if spans:
        merged_idx = set()
        merged = []
        for sd in spans:
            sb = sd["box"]
            ov = [(i,c) for i,c in enumerate(cells)
                  if i not in merged_idx and bbox_iou(sb, c["bbox"]) > IOU_MERGE]
            if ov:
                rs=[c["start_row"] for _,c in ov]; cs=[c["start_col"] for _,c in ov]
                r0,r1=min(rs),max(rs); c0,c1=min(cs),max(cs)
                merged.append({"start_row":r0,"end_row":r1,
                                "start_col":c0,"end_col":c1,
                                "row_span":r1-r0+1,"col_span":c1-c0+1,
                                "bbox":sb})
                for i,_ in ov: merged_idx.add(i)
        cells = merged + [c for i,c in enumerate(cells) if i not in merged_idx]

    return {"cells":cells, "n_rows":len(rows), "n_cols":len(cols),
            "spans_bbox":[s["box"] for s in spans]}


# ─────────────────────────────────────────────
# Span-Class Recall @IoU≥0.1
# ─────────────────────────────────────────────
def span_class_recall(pred_spans_bbox: List[list], gt_cells: List[dict],
                      image_size: Tuple[int,int], iou_thresh: float = 0.1) -> Optional[float]:
    """
    GT spanning cell bbox를 예측된 span bbox로 얼마나 커버하는가.
    GT에 span 없으면 None.
    FinTabNet GT: bbox 있음. SciTSR GT: bbox 없음 → None.
    """
    gt_spans_with_bbox = [c for c in gt_cells
                          if (c.get("row_span",1)>1 or c.get("col_span",1)>1)
                          and "bbox" in c]
    if not gt_spans_with_bbox:
        return None
    if not pred_spans_bbox:
        return 0.0
    matched = 0
    for gc in gt_spans_with_bbox:
        if any(bbox_iou(gc["bbox"], pb) >= iou_thresh for pb in pred_spans_bbox):
            matched += 1
    return matched / len(gt_spans_with_bbox)


# ─────────────────────────────────────────────
# Single table evaluation
# ─────────────────────────────────────────────
def eval_table(proc, mdl, gt: dict, image: Image.Image) -> dict:
    pred = run_tatr(proc, mdl, image)
    nr_p, nc_p = pred["n_rows"], pred["n_cols"]
    nr_g, nc_g = gt["n_rows"],  gt["n_cols"]
    if nr_p == 0 or nc_p == 0 or nr_g == 0 or nc_g == 0:
        return None

    _, _, f_full = grits_top(pred["cells"], gt["cells"], nr_p, nc_p, nr_g, nc_g)
    f_triv       = trivial_grits_top(gt["cells"], nr_g, nc_g, nr_p, nc_p)
    f_span       = grits_top_span_only(pred["cells"], gt["cells"])
    s_recall     = span_class_recall(pred["spans_bbox"], gt["cells"],
                                     (image.width, image.height))

    gt_n_spans   = sum(1 for c in gt["cells"]
                       if c.get("row_span",1)>1 or c.get("col_span",1)>1)
    pred_n_spans = sum(1 for c in pred["cells"]
                       if c.get("row_span",1)>1 or c.get("col_span",1)>1)

    return {
        "table_id":    gt.get("table_id",""),
        "gt_n_rows":   nr_g, "gt_n_cols": nc_g,
        "pred_n_rows": nr_p, "pred_n_cols": nc_p,
        "gt_n_spans":  gt_n_spans,
        "pred_n_spans":pred_n_spans,
        "grits_top_f1":        f_full,
        "grits_top_trivial":   f_triv,
        "trivial_delta":       f_full - f_triv,
        "grits_top_span_only": f_span,   # None if no GT spans
        "span_class_recall":   s_recall, # None if no GT span bbox
    }


# ─────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────
def load_scitsr_dataset():
    """SciTSR test 전체 → list of (gt, image, split)."""
    all_ids   = load_all_scitsr_ids()
    comp_set  = set(load_comp_ids())
    items = []
    for tid in all_ids:
        try:
            gt  = load_gt(tid)
            img = load_image(tid)
            split = scitsr_split(tid, comp_set)
            items.append({"gt": gt, "image": img, "split": split, "dataset": "SciTSR"})
        except Exception:
            continue
    return items


def load_fintabnet_dataset():
    """FinTabNet 보유 120개 → list of (gt, image, split)."""
    items = []
    for xml_f in sorted(FINTAB_ANN.glob("*.xml")):
        img = load_fintabnet_image(xml_f)
        if img is None:
            continue
        gt = load_fintabnet_gt(xml_f)
        if gt is None:
            continue
        split = "complex" if gt["is_complex"] else "simple"
        items.append({"gt": gt, "image": img, "split": split, "dataset": "FinTabNet"})
    return items


# ─────────────────────────────────────────────
# Aggregation helper
# ─────────────────────────────────────────────
def aggregate(records: List[dict]) -> dict:
    """records 리스트 → 집계 통계."""
    if not records:
        return {}
    f1s      = [r["grits_top_f1"]        for r in records]
    trivs    = [r["trivial_delta"]        for r in records]
    f1_trivs = [r["grits_top_trivial"]    for r in records]
    spans    = [r["grits_top_span_only"]  for r in records
                if r["grits_top_span_only"] is not None]
    recalls  = [r["span_class_recall"]    for r in records
                if r["span_class_recall"]  is not None]
    return {
        "n":                   len(records),
        "grits_top_f1":        float(np.mean(f1s)),
        "grits_top_f1_std":    float(np.std(f1s)),
        "grits_top_trivial":   float(np.mean(f1_trivs)),
        "trivial_delta":       float(np.mean(trivs)),
        "grits_top_span_only": float(np.mean(spans))  if spans   else None,
        "span_class_recall":   float(np.mean(recalls)) if recalls else None,
    }


# ─────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────
def fig1_grits_simple_vs_complex(summary: dict):
    """
    Fig 1: GriTS-Top F1 — Simple vs Complex, 각 모델별.
    각 데이터셋(SciTSR, FinTabNet) 패널.
    """
    datasets = ["SciTSR", "FinTabNet"]
    model_labels = [m["label"] for m in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GriTS-Top F1: Simple vs Complex Tables\n(Official dataset splits, no manual curation)",
                 fontsize=13, fontweight="bold")

    for ax, ds in zip(axes, datasets):
        x = np.arange(len(model_labels))
        w = 0.3
        for i, split in enumerate(["simple", "complex"]):
            vals = []
            errs = []
            for m in MODELS:
                key = f"{ds}/{m['label']}/{split}"
                s = summary.get(key, {})
                vals.append(s.get("grits_top_f1", 0))
                errs.append(s.get("grits_top_f1_std", 0))
            color = "#4C9BE8" if split == "simple" else "#E84C4C"
            bars = ax.bar(x + (i-0.5)*w, vals, w, label=split.capitalize(),
                          color=color, alpha=0.85, yerr=errs, capsize=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(ds, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=10, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("GriTS-Top F1")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = RESULTS_DIR / "fig1_grits_simple_vs_complex.png"
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"[saved] {p}")


def fig2_trivial_delta(summary: dict):
    """
    Fig 2: Trivial-Delta (GriTS-Top - Trivial) — Simple vs Complex.
    핵심 smoking gun figure.
    """
    datasets = ["SciTSR", "FinTabNet"]
    model_labels = [m["label"] for m in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Trivial-Delta (GriTS-Top F1 − Trivial Baseline F1)\n"
                 "Near-zero delta → model adds no value over predicting all 1×1 cells",
                 fontsize=12, fontweight="bold")

    for ax, ds in zip(axes, datasets):
        x = np.arange(len(model_labels))
        w = 0.3
        for i, split in enumerate(["simple", "complex"]):
            vals = []
            for m in MODELS:
                key = f"{ds}/{m['label']}/{split}"
                s = summary.get(key, {})
                vals.append(s.get("trivial_delta", 0))
            color = "#4C9BE8" if split == "simple" else "#E84C4C"
            bars = ax.bar(x + (i-0.5)*w, vals, w, label=split.capitalize(),
                          color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.001 if v >= 0 else bar.get_height() - 0.005,
                        f"{v:+.4f}", ha="center", va="bottom", fontsize=7.5)

        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(ds, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=10, fontsize=9)
        ax.set_ylabel("Trivial-Delta")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = RESULTS_DIR / "fig2_trivial_delta.png"
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"[saved] {p}")


def fig3_span_only_f1(summary: dict):
    """
    Fig 3: GriTS-Top Span-Only F1 — Simple vs Complex.
    span 셀만 격리한 F1.
    """
    datasets = ["SciTSR", "FinTabNet"]
    model_labels = [m["label"] for m in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GriTS-Top Span-Only F1\n(Only spanning cells evaluated; None if no GT spans)",
                 fontsize=12, fontweight="bold")

    for ax, ds in zip(axes, datasets):
        x = np.arange(len(model_labels))
        w = 0.3
        for i, split in enumerate(["simple", "complex"]):
            vals = []
            for m in MODELS:
                key = f"{ds}/{m['label']}/{split}"
                s = summary.get(key, {})
                v = s.get("grits_top_span_only")
                vals.append(v if v is not None else 0)
            color = "#4C9BE8" if split == "simple" else "#E84C4C"
            bars = ax.bar(x + (i-0.5)*w, vals, w, label=split.capitalize(),
                          color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(ds, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=10, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Span-Only F1")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = RESULTS_DIR / "fig3_span_only_f1.png"
    fig.savefig(p, dpi=150)
    plt.close()
    print(f"[saved] {p}")


def fig4_full_summary_table(summary: dict):
    """
    Fig 4: 논문용 숫자 테이블 시각화.
    """
    model_labels = [m["label"] for m in MODELS]
    rows_data = []
    col_headers = ["Dataset", "Split", "N",
                   "GriTS-Top F1", "Trivial F1", "Trivial-Δ", "Span-Only F1"]
    for ds in ["SciTSR", "FinTabNet"]:
        for split in ["simple", "complex", "all"]:
            row = [ds, split]
            first_model = MODELS[0]["label"]
            key = f"{ds}/{first_model}/{split}"
            s = summary.get(key, {})
            row.append(str(s.get("n", "")))
            for m in [MODELS[0]]:  # 대표 모델 (v1.1-all)
                key2 = f"{ds}/{MODELS[2]['label']}/{split}"
                s2 = summary.get(key2, {})
                row.append(f"{s2.get('grits_top_f1',0):.4f}")
                row.append(f"{s2.get('grits_top_trivial',0):.4f}")
                row.append(f"{s2.get('trivial_delta',0):+.4f}")
                v = s2.get("grits_top_span_only")
                row.append(f"{v:.4f}" if v is not None else "N/A")
            rows_data.append(row)

    fig, ax = plt.subplots(figsize=(14, len(rows_data)*0.55 + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows_data, colLabels=col_headers,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    # 헤더 색
    for j in range(len(col_headers)):
        tbl[(0, j)].set_facecolor("#2C5F8A")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # complex 행 강조
    for i, row in enumerate(rows_data):
        if row[1] == "complex":
            for j in range(len(col_headers)):
                tbl[(i+1, j)].set_facecolor("#FFF0F0")

    ax.set_title("Phase 1 Results Summary (TATR v1.1-all, representative model)",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    p = RESULTS_DIR / "fig4_summary_table.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run():
    print("=" * 60)
    print("Phase 1: Problem Proof Experiments")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # 데이터 로드
    print("\n[Data] Loading SciTSR test (full 3000)...")
    scitsr_items = load_scitsr_dataset()
    scitsr_simple  = [x for x in scitsr_items if x["split"] == "simple"]
    scitsr_complex = [x for x in scitsr_items if x["split"] == "complex"]
    print(f"  SciTSR total: {len(scitsr_items)} "
          f"(simple={len(scitsr_simple)}, complex={len(scitsr_complex)})")

    print("\n[Data] Loading FinTabNet (120 tables)...")
    fintab_items = load_fintabnet_dataset()
    fintab_simple  = [x for x in fintab_items if x["split"] == "simple"]
    fintab_complex = [x for x in fintab_items if x["split"] == "complex"]
    print(f"  FinTabNet total: {len(fintab_items)} "
          f"(simple={len(fintab_simple)}, complex={len(fintab_complex)})")

    all_records: Dict[str, Dict[str, List[dict]]] = {}
    summary: Dict[str, dict] = {}

    datasets = [
        ("SciTSR",   scitsr_items,   scitsr_simple,  scitsr_complex),
        ("FinTabNet", fintab_items,   fintab_simple,  fintab_complex),
    ]

    for ds_name, all_items, simple_items, complex_items in datasets:
        all_records[ds_name] = {}
        for m_cfg in MODELS:
            mlabel = m_cfg["label"]
            print(f"\n[Eval] {ds_name} × {mlabel} ({len(all_items)} tables)...")
            proc, mdl = load_tatr(m_cfg["id"])

            records = []
            for item in tqdm(all_items, desc=f"{ds_name}/{mlabel}"):
                try:
                    r = eval_table(proc, mdl, item["gt"], item["image"])
                    if r is not None:
                        r["split"] = item["split"]
                        r["dataset"] = ds_name
                        records.append(r)
                except Exception as e:
                    continue

            all_records[ds_name][mlabel] = records

            # 집계
            recs_simple  = [r for r in records if r["split"] == "simple"]
            recs_complex = [r for r in records if r["split"] == "complex"]

            for split_name, recs in [("all", records),
                                      ("simple",  recs_simple),
                                      ("complex", recs_complex)]:
                key = f"{ds_name}/{mlabel}/{split_name}"
                summary[key] = aggregate(recs)

            # 즉시 출력
            def fmt(s):
                if not s: return "N/A"
                span = s.get('grits_top_span_only')
                return (f"  n={s['n']}, "
                        f"F1={s['grits_top_f1']:.4f}, "
                        f"Trivial={s['grits_top_trivial']:.4f}, "
                        f"Δ={s['trivial_delta']:+.4f}, "
                        f"SpanF1={f'{span:.4f}' if span is not None else 'N/A'}")
            print(f"  ALL    :{fmt(summary.get(f'{ds_name}/{mlabel}/all', {}))}")
            print(f"  SIMPLE :{fmt(summary.get(f'{ds_name}/{mlabel}/simple', {}))}")
            print(f"  COMPLEX:{fmt(summary.get(f'{ds_name}/{mlabel}/complex', {}))}")

            # GPU 메모리 정리
            del proc, mdl
            torch.cuda.empty_cache()

        # per-table JSON 저장
        fname = f"per_table_{ds_name.lower()}.json"
        with open(RESULTS_DIR / fname, "w") as f:
            json.dump(
                {ml: recs for ml, recs in all_records[ds_name].items()},
                f, indent=2)
        print(f"[saved] {RESULTS_DIR / fname}")

    # summary JSON
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {RESULTS_DIR / 'summary.json'}")

    # Figures
    print("\n[Figures] Generating...")
    fig1_grits_simple_vs_complex(summary)
    fig2_trivial_delta(summary)
    fig3_span_only_f1(summary)
    fig4_full_summary_table(summary)

    print("\n" + "=" * 60)
    print("Phase 1 DONE.")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run()

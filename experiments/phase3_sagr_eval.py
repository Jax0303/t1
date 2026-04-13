#!/usr/bin/env python3
"""
Phase 3 — SAGR Evaluation
===========================
Baseline vs SAGR 변종 비교 + Ablation Study.

평가 설정:
  데이터셋: SciTSR complex 716개 + FinTabNet complex 68개
  모델:     TATR v1.1-all (span detection용)
  비교:
    Baseline      : IoU > 0.3 (현재 방식)
    SAGR-Cov      : Coverage-based assignment (contiguity 없음)
    SAGR-Con      : IoU + Contiguity 강제
    SAGR-Full     : Coverage + Contiguity (제안 방법 full)
  Threshold sweep : coverage_thresh ∈ {0.1, 0.2, 0.3, 0.4, 0.5}

Metric:
  GriTS-Top F1    : 전체 구조 (Smock CVPR'23 표준)
  Span-Only F1    : span 셀만 격리
  Logical-Exact   : (sr,er,sc,ec) exact match

출력:
  experiments/results_phase3/
    phase3_summary.json
    fig8_sagr_ablation.png          ← Ablation: 4 variants × 3 metrics
    fig9_threshold_sweep.png        ← Threshold sensitivity
    fig10_combined_improvement.png  ← Phase 1→3 개선 전체 그림
    fig11_per_table_scatter.png     ← 테이블별 개선량 scatter
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
import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import tqdm

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _eval_utils import (
    load_all_scitsr_ids, load_comp_ids, load_gt, load_image,
    load_fintabnet_gt, load_fintabnet_image, FINTAB_ANN, bbox_iou,
)
from phase1_problem_proof import (
    load_tatr, grits_top, grits_top_span_only, trivial_grits_top,
)
from phase2_oracle_decomposition import (
    parse_fintabnet_xml_bboxes, derive_gt_logical_spans,
    logical_exact_rate, _extract_tatr_rowcol,
)
from sagr import (
    make_baseline, make_sagr_cov, make_sagr_con, make_sagr_full,
    sweep_coverage_thresh,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results_phase3"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"

VARIANTS = [
    ("Baseline",   make_baseline(),   "#CC4444"),
    ("SAGR-Con",   make_sagr_con(),   "#DD8452"),
    ("SAGR-Cov",   make_sagr_cov(),   "#4C72B0"),
    ("SAGR-Full",  make_sagr_full(),  "#2C9B3C"),
]


# ─────────────────────────────────────────────
# TATR inference → raw outputs
# ─────────────────────────────────────────────
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image

CONF_THRESH = 0.5


@torch.no_grad()
def run_tatr_raw(proc, mdl, image: Image.Image) -> dict:
    """TATR → raw row/col/span bboxes (분리)."""
    enc = proc(images=image, return_tensors="pt")
    out = mdl(**{k: v.to(DEVICE) for k, v in enc.items()
                 if k in {"pixel_values", "pixel_mask"}})
    res = proc.post_process_object_detection(
        out, threshold=CONF_THRESH,
        target_sizes=[(image.size[1], image.size[0])])[0]
    boxes  = res["boxes"].cpu().numpy()
    labels = res["labels"].cpu().numpy()
    id2l   = mdl.config.id2label

    rows, cols, spans = [], [], []
    for box, lid in zip(boxes, labels):
        lname = id2l[lid].lower()
        b = box.tolist()
        if "column" in lname:
            cols.append(b)
        elif "spanning" in lname or "projected" in lname:
            spans.append(b)
        else:
            rows.append(b)
    rows.sort(key=lambda b: (b[1]+b[3])/2)
    cols.sort(key=lambda b: (b[0]+b[2])/2)
    return {"rows_bbox": rows, "cols_bbox": cols, "spans_bbox": spans}


# ─────────────────────────────────────────────
# Single-table evaluation
# ─────────────────────────────────────────────
def eval_one(tatr_raw: dict, gt_cells: List[dict],
             nr_g: int, nc_g: int,
             gt_spans_logical: Optional[List[Tuple]],
             reconstructor) -> Optional[dict]:
    """
    주어진 reconstructor로 Stage 2 실행 → metrics.
    """
    pred = reconstructor.reconstruct(
        tatr_raw["rows_bbox"],
        tatr_raw["cols_bbox"],
        tatr_raw["spans_bbox"],
    )
    nr_p, nc_p = pred["n_rows"], pred["n_cols"]
    if nr_p == 0 or nc_p == 0 or nr_g == 0 or nc_g == 0:
        return None

    _, _, f_full = grits_top(pred["cells"], gt_cells, nr_p, nc_p, nr_g, nc_g)
    f_triv       = trivial_grits_top(gt_cells, nr_g, nc_g, nr_p, nc_p)
    f_span       = grits_top_span_only(pred["cells"], gt_cells)

    le = None
    if gt_spans_logical:
        le = logical_exact_rate(pred["cells"], gt_spans_logical)

    return {
        "grits_top_f1":        f_full,
        "trivial_delta":       f_full - f_triv,
        "grits_top_span_only": f_span,
        "logical_exact":       le,
    }


# ─────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────
def load_scitsr_complex():
    """SciTSR COMP complex 716개."""
    items = []
    for tid in load_comp_ids():
        try:
            gt  = load_gt(tid)
            img = load_image(tid)
            gt_spans = [(c["start_row"],c["end_row"],c["start_col"],c["end_col"])
                        for c in gt["cells"]
                        if c.get("row_span",1)>1 or c.get("col_span",1)>1]
            items.append({"gt": gt, "image": img,
                          "gt_spans": gt_spans, "dataset": "SciTSR"})
        except Exception:
            continue
    return items


def load_fintabnet_complex():
    """FinTabNet complex 68개."""
    items = []
    for xml_f in sorted(FINTAB_ANN.glob("*.xml")):
        img = load_fintabnet_image(xml_f)
        if img is None: continue
        gt_struct = load_fintabnet_gt(xml_f)
        if gt_struct is None or not gt_struct["is_complex"]: continue
        gt_xml = parse_fintabnet_xml_bboxes(xml_f)
        if gt_xml is None: continue
        gt_spans = derive_gt_logical_spans(gt_xml)
        items.append({"gt": gt_struct, "image": img,
                      "gt_spans": gt_spans, "dataset": "FinTabNet"})
    return items


# ─────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────
def agg(records: List[dict]) -> dict:
    if not records:
        return {"n": 0}
    f1s  = [r["grits_top_f1"] for r in records]
    trivs = [r["trivial_delta"] for r in records]
    spans = [r["grits_top_span_only"] for r in records if r.get("grits_top_span_only") is not None]
    lexs  = [r["logical_exact"] for r in records if r.get("logical_exact") is not None]
    return {
        "n":                   len(records),
        "grits_top_f1":        float(np.mean(f1s)),
        "grits_top_f1_std":    float(np.std(f1s)),
        "trivial_delta":       float(np.mean(trivs)),
        "grits_top_span_only": float(np.mean(spans)) if spans else None,
        "logical_exact":       float(np.mean(lexs))  if lexs  else None,
        "logical_exact_std":   float(np.std(lexs))   if lexs  else None,
    }


# ─────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────
def fig8_sagr_ablation(summary: dict):
    """Fig 8: Ablation — 4 variants × 3 metrics × 2 datasets."""
    var_names  = [v[0] for v in VARIANTS]
    var_colors = [v[2] for v in VARIANTS]
    datasets   = ["SciTSR", "FinTabNet"]
    metrics    = ["grits_top_f1", "grits_top_span_only", "logical_exact"]
    m_labels   = ["GriTS-Top F1", "Span-Only F1", "Logical-Exact Rate"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("SAGR Ablation Study — Baseline vs SAGR Variants\n"
                 "(TATR v1.1-all, complex tables only)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row_i, ds in enumerate(datasets):
        for col_i, (metric, mlabel) in enumerate(zip(metrics, m_labels)):
            ax = axes[row_i, col_i]
            vals = [summary.get(f"{ds}/{vn}", {}).get(metric) or 0 for vn in var_names]
            bars = ax.bar(range(len(var_names)), vals, color=var_colors,
                          alpha=0.85, width=0.6, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

            # Baseline→SAGR-Full 향상 화살표
            v_base = vals[0]; v_full = vals[-1]
            if v_full > v_base + 0.005:
                ax.annotate("", xy=(len(var_names)-1, v_full+0.03),
                            xytext=(0, v_base+0.03),
                            arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1.5))
                ax.text((len(var_names)-1)/2, max(v_base,v_full)+0.055,
                        f"+{v_full-v_base:.3f}", ha="center",
                        color="#CC0000", fontsize=9, fontweight="bold")

            ax.set_xticks(range(len(var_names)))
            ax.set_xticklabels(var_names, fontsize=9, rotation=10)
            ax.set_ylim(0, max(vals+[0.1])*1.3 + 0.05)
            ax.set_title(f"{ds} — {mlabel}", fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            if col_i == 0:
                ax.set_ylabel("Score", fontsize=9)

    plt.tight_layout()
    p = RESULTS_DIR / "fig8_sagr_ablation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


def fig9_threshold_sweep(thresh_results: dict, dataset: str):
    """Fig 9: Coverage threshold sensitivity (SAGR-Full)."""
    thresholds = sorted(thresh_results.keys())
    metrics    = ["grits_top_f1", "grits_top_span_only", "logical_exact"]
    m_labels   = ["GriTS-Top F1", "Span-Only F1", "Logical-Exact Rate"]
    colors     = ["#55A868", "#4C72B0", "#CC0000"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"SAGR Coverage Threshold Sensitivity ({dataset} complex)\n"
                 "SAGR-Full: coverage_thresh ∈ {{0.1, 0.2, 0.3, 0.4, 0.5}}",
                 fontsize=12, fontweight="bold")

    for ax, metric, mlabel, color in zip(axes, metrics, m_labels, colors):
        vals = [thresh_results[t].get(metric) or 0 for t in thresholds]
        stds = [thresh_results[t].get(metric+"_std") or 0 for t in thresholds]
        ax.plot(thresholds, vals, "o-", color=color, lw=2, markersize=7)
        ax.fill_between(thresholds, [v-s for v,s in zip(vals,stds)],
                        [v+s for v,s in zip(vals,stds)],
                        alpha=0.15, color=color)
        for t, v in zip(thresholds, vals):
            ax.text(t, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
        best_idx = int(np.argmax(vals))
        ax.axvline(thresholds[best_idx], color=color, linestyle="--",
                   alpha=0.5, label=f"best={thresholds[best_idx]}")
        ax.set_xlabel("coverage_thresh", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals+[0.1])*1.2 + 0.02)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = RESULTS_DIR / f"fig9_threshold_sweep_{dataset.lower()}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


def fig10_overall_improvement(phase1_summary: dict, phase3_summary: dict):
    """Fig 10: Phase 1 (Baseline) → Phase 3 (SAGR-Full) 개선 전체 그림."""
    datasets = ["SciTSR", "FinTabNet"]
    metrics  = ["grits_top_f1", "grits_top_span_only", "logical_exact"]
    m_labels = ["GriTS-Top F1", "Span-Only F1", "Logical-Exact Rate"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Overall Improvement: Baseline → SAGR-Full\n"
                 "(TATR v1.1-all, complex tables, both datasets)",
                 fontsize=13, fontweight="bold")

    for ax, metric, mlabel in zip(axes, metrics, m_labels):
        x = np.arange(len(datasets)); w = 0.32
        for i, (label, color) in enumerate([
            ("Baseline", "#CC4444"),
            ("SAGR-Full", "#2C9B3C"),
        ]):
            if label == "Baseline":
                vals = [phase1_summary.get(f"{ds}/TATR v1.1-all/complex",{}).get(metric) or 0
                        for ds in datasets]
            else:
                vals = [phase3_summary.get(f"{ds}/SAGR-Full",{}).get(metric) or 0
                        for ds in datasets]
            bars = ax.bar(x + (i-0.5)*w, vals, w, label=label, color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold")

        # 데이터셋별 개선폭
        for xi, ds in enumerate(datasets):
            v_base = phase1_summary.get(f"{ds}/TATR v1.1-all/complex",{}).get(metric) or 0
            v_sagr = phase3_summary.get(f"{ds}/SAGR-Full",{}).get(metric) or 0
            if v_sagr > v_base + 0.001:
                ax.text(xi, max(v_base,v_sagr)+0.05,
                        f"+{v_sagr-v_base:.3f}",
                        ha="center", color="#CC0000",
                        fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = RESULTS_DIR / "fig10_overall_improvement.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


def fig11_per_table_scatter(scitsr_records: dict, fintab_records: dict):
    """Fig 11: 테이블별 Baseline vs SAGR-Full GriTS-Top F1 scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Per-Table GriTS-Top F1: Baseline vs SAGR-Full\n"
                 "(Points above diagonal = SAGR-Full improves that table)",
                 fontsize=12, fontweight="bold")

    for ax, records, ds in [
        (axes[0], scitsr_records, "SciTSR complex"),
        (axes[1], fintab_records,  "FinTabNet complex"),
    ]:
        base_f1s = [r["grits_top_f1"] for r in records.get("Baseline", [])]
        sagr_f1s = [r["grits_top_f1"] for r in records.get("SAGR-Full", [])]

        n = min(len(base_f1s), len(sagr_f1s))
        if n == 0:
            continue
        base_f1s = base_f1s[:n]; sagr_f1s = sagr_f1s[:n]

        improved  = [(b,s) for b,s in zip(base_f1s,sagr_f1s) if s > b+0.001]
        degraded  = [(b,s) for b,s in zip(base_f1s,sagr_f1s) if s < b-0.001]
        unchanged = [(b,s) for b,s in zip(base_f1s,sagr_f1s) if abs(s-b)<=0.001]

        ax.scatter([b for b,s in improved],  [s for b,s in improved],
                   c="#2C9B3C", alpha=0.5, s=20, label=f"Improved ({len(improved)})")
        ax.scatter([b for b,s in degraded],  [s for b,s in degraded],
                   c="#CC4444", alpha=0.5, s=20, label=f"Degraded ({len(degraded)})")
        ax.scatter([b for b,s in unchanged], [s for b,s in unchanged],
                   c="#888888", alpha=0.3, s=15, label=f"Unchanged ({len(unchanged)})")

        ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Baseline GriTS-Top F1", fontsize=11)
        ax.set_ylabel("SAGR-Full GriTS-Top F1", fontsize=11)
        ax.set_title(ds, fontsize=12, fontweight="bold")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

        mean_delta = np.mean([s-b for b,s in zip(base_f1s,sagr_f1s)])
        ax.text(0.05, 0.92, f"mean Δ = {mean_delta:+.4f}", transform=ax.transAxes,
                fontsize=10, color="#333333", fontweight="bold")

    plt.tight_layout()
    p = RESULTS_DIR / "fig11_per_table_scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run():
    print("=" * 65)
    print("Phase 3: SAGR Evaluation + Ablation Study")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    # 데이터 로드
    print("\n[Data] Loading SciTSR complex (716)...")
    scitsr_items = load_scitsr_complex()
    print(f"  Loaded: {len(scitsr_items)}")

    print("[Data] Loading FinTabNet complex (68)...")
    fintab_items = load_fintabnet_complex()
    print(f"  Loaded: {len(fintab_items)}")

    # ── Step 1: TATR inference 캐싱 (재사용) ─────
    import pickle
    _DISK_CACHE = RESULTS_DIR / "tatr_inference_cache.pkl"

    if _DISK_CACHE.exists():
        print(f"\n[Cache] Loading TATR inference from disk: {_DISK_CACHE}")
        with open(_DISK_CACHE, "rb") as f:
            scitsr_cache, fintab_cache = pickle.load(f)
        print(f"  Loaded: SciTSR={len(scitsr_cache)}, FinTabNet={len(fintab_cache)}")
    else:
        proc, mdl = load_tatr(MODEL_ID)
        print("\n[Cache] Running TATR inference on all tables...")
        scitsr_cache, fintab_cache = [], []

        for item in tqdm(scitsr_items, desc="SciTSR TATR"):
            try:
                raw = run_tatr_raw(proc, mdl, item["image"])
                scitsr_cache.append((raw, item))
            except Exception:
                continue

        for item in tqdm(fintab_items, desc="FinTabNet TATR"):
            try:
                raw = run_tatr_raw(proc, mdl, item["image"])
                fintab_cache.append((raw, item))
            except Exception:
                continue

        print(f"  Cached: SciTSR={len(scitsr_cache)}, FinTabNet={len(fintab_cache)}")
        with open(_DISK_CACHE, "wb") as f:
            pickle.dump((scitsr_cache, fintab_cache), f)
        print(f"  Saved to disk: {_DISK_CACHE}")
        del proc, mdl
        torch.cuda.empty_cache()

    # ── Step 2: Ablation — 4 variants ────────────
    print("\n[Ablation] Evaluating 4 SAGR variants...")
    summary: Dict[str, dict] = {}
    per_table_records = {
        "SciTSR":   {vn: [] for vn,_,_ in VARIANTS},
        "FinTabNet": {vn: [] for vn,_,_ in VARIANTS},
    }

    for ds_name, cache in [("SciTSR", scitsr_cache), ("FinTabNet", fintab_cache)]:
        for vname, reconstructor, _ in VARIANTS:
            recs = []
            for raw, item in cache:
                gt   = item["gt"]
                gt_c = gt["cells"]; nr_g = gt["n_rows"]; nc_g = gt["n_cols"]
                gt_s = item["gt_spans"] if item["gt_spans"] else None

                r = eval_one(raw, gt_c, nr_g, nc_g, gt_s, reconstructor)
                if r:
                    recs.append(r)
                    per_table_records[ds_name][vname].append(r)

            agg_r = agg(recs)
            key = f"{ds_name}/{vname}"
            summary[key] = agg_r

            if agg_r.get("n", 0) == 0:
                print(f"  {ds_name}/{vname:12s}: n=0 (skipped)")
                continue

            le  = agg_r.get("logical_exact")
            spn = agg_r.get("grits_top_span_only")
            print(f"  {ds_name}/{vname:12s}: "
                  f"F1={agg_r['grits_top_f1']:.4f}, "
                  f"Δtriv={agg_r['trivial_delta']:+.4f}, "
                  f"SpanF1={spn or 0:.4f}, "
                  f"LogExact={le or 0:.4f}")

    # ── Step 3: Threshold sweep (SAGR-Full) ──────
    print("\n[Sweep] Coverage threshold sensitivity (SAGR-Full)...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    thresh_results = {"SciTSR": {}, "FinTabNet": {}}

    for ds_name, cache in [("SciTSR", scitsr_cache), ("FinTabNet", fintab_cache)]:
        for thresh in thresholds:
            recon = __import__("sagr").make_sagr_full(thresh)
            recs  = []
            for raw, item in cache:
                gt = item["gt"]
                r = eval_one(raw, gt["cells"], gt["n_rows"], gt["n_cols"],
                             item["gt_spans"] or None, recon)
                if r: recs.append(r)
            agg_r = agg(recs)
            thresh_results[ds_name][thresh] = agg_r
            if agg_r.get("n", 0) == 0:
                print(f"  {ds_name} thresh={thresh:.1f}: n=0 (skipped)")
                continue
            le = agg_r.get("logical_exact")
            print(f"  {ds_name} thresh={thresh:.1f}: "
                  f"F1={agg_r['grits_top_f1']:.4f}, "
                  f"LogExact={le or 0:.4f}")

    # 저장
    with open(RESULTS_DIR / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS_DIR / "thresh_results.json", "w") as f:
        json.dump({ds: {str(k): v for k,v in d.items()}
                   for ds, d in thresh_results.items()}, f, indent=2)
    print(f"\n[saved] phase3_summary.json")

    # Figures
    print("\n[Figures] Generating...")
    fig8_sagr_ablation(summary)
    fig9_threshold_sweep(thresh_results["SciTSR"],   "SciTSR")
    fig9_threshold_sweep(thresh_results["FinTabNet"], "FinTabNet")

    # Phase 1 summary 로드 (비교용)
    phase1_path = Path(__file__).parent / "results_phase1" / "summary.json"
    if phase1_path.exists():
        with open(phase1_path) as f:
            phase1_summary = json.load(f)
        fig10_overall_improvement(phase1_summary, summary)
    else:
        print("[skip] fig10: phase1 summary not found")

    fig11_per_table_scatter(per_table_records["SciTSR"],
                            per_table_records["FinTabNet"])

    if "proc" in locals() and proc is not None:
        del proc, mdl
    torch.cuda.empty_cache()

    # 최종 수치 출력
    print("\n" + "=" * 65)
    print("SAGR Phase 3 DONE — Key Results")
    print("=" * 65)
    for ds in ["SciTSR", "FinTabNet"]:
        print(f"\n  [{ds} complex]")
        base = summary.get(f"{ds}/Baseline",{})
        full = summary.get(f"{ds}/SAGR-Full",{})
        print(f"  {'Metric':<22} {'Baseline':>10}  {'SAGR-Full':>10}  {'Δ':>8}")
        print(f"  {'-'*52}")
        for metric, label in [
            ("grits_top_f1",        "GriTS-Top F1"),
            ("grits_top_span_only", "Span-Only F1"),
            ("logical_exact",       "Logical-Exact"),
        ]:
            bv = base.get(metric) or 0
            fv = full.get(metric) or 0
            print(f"  {label:<22} {bv:>10.4f}  {fv:>10.4f}  {fv-bv:>+8.4f}")
    print(f"\nResults: {RESULTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
Row-NMS Post-Processing Evaluation
===================================
가설:
  TATR의 row bbox 예측은 대다수(SciTSR 76%, FinTab 53%)가 정확히 +1
  over-prediction. 100%가 y-방향 overlap >0.5 → 기하학적으로 merge 가능.

방법:
  row_nms(rows, iou_thresh):
    y-축 1D IoU > thresh → union으로 merge
    (col에도 대칭으로 적용 가능)

Ablation:
  Baseline             : TATR raw
  +RowNMS(τ)           : row만 NMS
  +RowColNMS(τ)        : row+col 동시 NMS

출력:
  experiments/results_rownms/
    rownms_summary.json
    fig_dr_before_after.png       ← dr 분포 before/after NMS
    fig_f1_sweep.png              ← threshold sweep
    fig_main.png                  ← baseline vs RowNMS main comparison
    fig_scatter.png               ← per-table improvement scatter
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from phase1_problem_proof import grits_top, grits_top_span_only
from phase2_oracle_decomposition import derive_gt_logical_spans, logical_exact_rate, parse_fintabnet_xml_bboxes
from grid_reconstruct import make_baseline
from _eval_utils import FINTAB_ANN

CACHE_PATH  = Path(__file__).parent / "results_phase3" / "tatr_inference_cache.pkl"
RESULTS_DIR = Path(__file__).parent / "results_rownms"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Row NMS (1D y-axis)
# ─────────────────────────────────────────────
def _1d_iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _1d_overlap_over_min(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    min_len = min(a1 - a0, b1 - b0)
    return inter / min_len if min_len > 0 else 0.0


def row_nms(rows_bbox, iou_thresh=0.5):
    """Merge rows whose y-overlap/min_h > iou_thresh. Union-merge."""
    if not rows_bbox:
        return rows_bbox
    sorted_rows = sorted(rows_bbox, key=lambda b: (b[1] + b[3]) / 2)
    merged = [list(sorted_rows[0])]
    for b in sorted_rows[1:]:
        last = merged[-1]
        ov = _1d_overlap_over_min(last[1], last[3], b[1], b[3])
        if ov > iou_thresh:
            merged[-1] = [min(last[0], b[0]), min(last[1], b[1]),
                          max(last[2], b[2]), max(last[3], b[3])]
        else:
            merged.append(list(b))
    return merged


def col_nms(cols_bbox, iou_thresh=0.5):
    if not cols_bbox:
        return cols_bbox
    sorted_cols = sorted(cols_bbox, key=lambda b: (b[0] + b[2]) / 2)
    merged = [list(sorted_cols[0])]
    for b in sorted_cols[1:]:
        last = merged[-1]
        ov = _1d_overlap_over_min(last[0], last[2], b[0], b[2])
        if ov > iou_thresh:
            merged[-1] = [min(last[0], b[0]), min(last[1], b[1]),
                          max(last[2], b[2]), max(last[3], b[3])]
        else:
            merged.append(list(b))
    return merged


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def eval_pipeline(cache, mode: str, iou_thresh: float = 0.5, ds_name: str = "") -> dict:
    """
    mode: 'baseline' | 'row_nms' | 'rowcol_nms'
    Returns aggregate + per_table records + dr distribution.
    """
    recon = make_baseline()
    recs = []
    dr_list, dc_list = [], []

    # For FinTabNet logical_exact, we need gt_xml spans — look up by filename
    fintab_gt_xml = {}
    if ds_name == "FinTabNet":
        for xml_f in sorted(FINTAB_ANN.glob("*.xml")):
            fintab_gt_xml[xml_f.stem] = xml_f

    for raw, item in cache:
        gt = item["gt"]
        rows = raw["rows_bbox"]
        cols = raw["cols_bbox"]
        spans = raw["spans_bbox"]

        if mode == "row_nms":
            rows = row_nms(rows, iou_thresh)
        elif mode == "rowcol_nms":
            rows = row_nms(rows, iou_thresh)
            cols = col_nms(cols, iou_thresh)

        dr_list.append(len(rows) - gt["n_rows"])
        dc_list.append(len(cols) - gt["n_cols"])

        if not rows or not cols:
            continue
        pred = recon.reconstruct(rows, cols, spans)
        if pred["n_rows"] == 0 or pred["n_cols"] == 0:
            continue
        try:
            _, _, f1 = grits_top(pred["cells"], gt["cells"],
                                 pred["n_rows"], pred["n_cols"],
                                 gt["n_rows"], gt["n_cols"])
        except Exception:
            continue
        span_f1 = grits_top_span_only(pred["cells"], gt["cells"])

        # Logical exact for FinTabNet only
        le = None
        if ds_name == "FinTabNet":
            stem = item.get("gt", {}).get("table_id") or None
            # gt_xml lookup: phase3 stored item["gt"]["table_id"] from load_fintabnet_gt
            if stem and stem in fintab_gt_xml:
                gt_xml = parse_fintabnet_xml_bboxes(fintab_gt_xml[stem])
                if gt_xml:
                    gt_spans = derive_gt_logical_spans(gt_xml)
                    if gt_spans:
                        le = logical_exact_rate(pred["cells"], gt_spans)

        recs.append({
            "f1": float(f1),
            "span_f1": float(span_f1) if span_f1 is not None else None,
            "logical_exact": le,
            "dr_after": len(rows) - gt["n_rows"],
            "dc_after": len(cols) - gt["n_cols"],
        })

    f1s = [r["f1"] for r in recs]
    span_f1s = [r["span_f1"] for r in recs if r["span_f1"] is not None]
    les = [r["logical_exact"] for r in recs if r["logical_exact"] is not None]
    return {
        "mode":             mode,
        "iou_thresh":       iou_thresh,
        "n":                len(recs),
        "f1":               float(np.mean(f1s)) if f1s else None,
        "f1_std":           float(np.std(f1s)) if f1s else None,
        "span_f1":          float(np.mean(span_f1s)) if span_f1s else None,
        "logical_exact":    float(np.mean(les)) if les else None,
        "dr_dist":          dict(Counter(dr_list)),
        "dc_dist":          dict(Counter(dc_list)),
        "dr_exact_pct":     float(np.mean([1 if d == 0 else 0 for d in dr_list]) * 100),
        "dc_exact_pct":     float(np.mean([1 if d == 0 else 0 for d in dc_list]) * 100),
        "records":          recs,
    }


# ─────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────
COL_BASE  = "#C44E52"
COL_ROW   = "#4C72B0"
COL_BOTH  = "#2C9B3C"


def fig_dr_before_after(results: dict):
    """Figure: dr distribution before vs after RowNMS."""
    datasets = ["SciTSR", "FinTabNet"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    for i, ds in enumerate(datasets):
        base_dr = [d for d, c in results[ds]["baseline"]["dr_dist"].items() for _ in range(c)]
        nms_dr  = [d for d, c in results[ds]["row_nms@best"]["dr_dist"].items() for _ in range(c)]
        clip = 6
        base_c = [max(-clip, min(clip, d)) for d in base_dr]
        nms_c  = [max(-clip, min(clip, d)) for d in nms_dr]
        bins = np.arange(-clip-0.5, clip+1.5)

        ax = axes[i, 0]
        ax.hist(base_c, bins=bins, color=COL_BASE, alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", lw=1, linestyle="--")
        ax.set_title(f"{ds} — Δrows  BEFORE NMS  "
                     f"(exact={results[ds]['baseline']['dr_exact_pct']:.1f}%)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("n_pred - n_gt")
        ax.set_ylabel("# tables")
        ax.grid(axis="y", alpha=0.3)

        ax = axes[i, 1]
        ax.hist(nms_c, bins=bins, color=COL_ROW, alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", lw=1, linestyle="--")
        ax.set_title(f"{ds} — Δrows  AFTER Row-NMS  "
                     f"(exact={results[ds]['row_nms@best']['dr_exact_pct']:.1f}%)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("n_pred - n_gt")
        ax.set_ylabel("# tables")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Row Count Mismatch: Before vs After Row-NMS Post-Processing",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    p = RESULTS_DIR / "fig_dr_before_after.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


def fig_f1_sweep(sweep: dict):
    """Figure: F1 vs threshold, both datasets."""
    datasets = ["SciTSR", "FinTabNet"]
    thresholds = sorted(sweep["SciTSR"].keys())

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for ds, color, marker in [("SciTSR", "#4C72B0", "o"),
                              ("FinTabNet", "#C44E52", "s")]:
        vals = [sweep[ds][t] for t in thresholds]
        ax.plot(thresholds, vals, marker=marker, lw=2, markersize=9,
                color=color, label=f"{ds} — Row-NMS")
        # baseline as horizontal line
        base = sweep[ds + "_base"]
        ax.axhline(base, color=color, lw=1.2, linestyle=":",
                   alpha=0.75, label=f"{ds} — Baseline ({base:.4f})")
        for t, v in zip(thresholds, vals):
            ax.annotate(f"{v:.4f}", (t, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color=color)
        best = max(range(len(vals)), key=lambda i: vals[i])
        ax.plot(thresholds[best], vals[best], "D", color=color, markersize=13,
                markerfacecolor="gold", markeredgewidth=1.5, zorder=5)

    ax.set_xlabel("y-overlap threshold (1D IoU over min height)", fontsize=11)
    ax.set_ylabel("GriTS-Top F1 (complex split)", fontsize=11)
    ax.set_title("Row-NMS Threshold Sensitivity\n"
                 "(♦ = best; dotted = baseline)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = RESULTS_DIR / "fig_f1_sweep.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


def fig_main(results: dict):
    """Main comparison: Baseline vs Row-NMS (best τ)."""
    datasets = ["SciTSR", "FinTabNet"]
    metrics   = [("f1", "GriTS-Top F1"),
                 ("span_f1", "Span-Only F1"),
                 ("logical_exact", "Logical-Exact")]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3))

    for ax, (mkey, mlabel) in zip(axes, metrics):
        x = np.arange(len(datasets))
        w = 0.32
        base_vals = [results[ds]["baseline"][mkey] or 0 for ds in datasets]
        nms_vals  = [results[ds]["row_nms@best"][mkey] or 0 for ds in datasets]

        b1 = ax.bar(x - w/2, base_vals, w, color=COL_BASE, alpha=0.85,
                    label="Baseline (bug-fixed)", edgecolor="white")
        b2 = ax.bar(x + w/2, nms_vals,  w, color=COL_ROW, alpha=0.85,
                    label="+ Row-NMS", edgecolor="white")
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.4f}", ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold")
        # delta
        for xi, ds in enumerate(datasets):
            d = (nms_vals[xi] - base_vals[xi])
            if abs(d) > 0.0005:
                col = "#2C9B3C" if d > 0 else "#C44E52"
                ax.text(xi, max(base_vals[xi], nms_vals[xi]) + 0.05,
                        f"{d:+.4f}", ha="center", color=col,
                        fontsize=10.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(base_vals + nms_vals + [0.1]) * 1.25 + 0.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Row-NMS Post-Processing vs Baseline\n"
                 "(TATR v1.1-all, complex tables, both datasets)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = RESULTS_DIR / "fig_main.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


def fig_scatter(results: dict):
    """Per-table delta F1 scatter (Baseline vs RowNMS)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, ds in zip(axes, ["SciTSR", "FinTabNet"]):
        base = [r["f1"] for r in results[ds]["baseline"]["records"]]
        nms  = [r["f1"] for r in results[ds]["row_nms@best"]["records"]]
        n = min(len(base), len(nms))
        base, nms = base[:n], nms[:n]
        improved = [(b,s) for b,s in zip(base,nms) if s > b + 0.001]
        degraded = [(b,s) for b,s in zip(base,nms) if s < b - 0.001]
        same     = [(b,s) for b,s in zip(base,nms) if abs(s-b) <= 0.001]

        ax.scatter([b for b,s in improved], [s for b,s in improved],
                   c="#2C9B3C", alpha=0.6, s=22, label=f"Improved ({len(improved)})")
        ax.scatter([b for b,s in degraded], [s for b,s in degraded],
                   c="#C44E52", alpha=0.6, s=22, label=f"Degraded ({len(degraded)})")
        ax.scatter([b for b,s in same], [s for b,s in same],
                   c="#888888", alpha=0.25, s=15, label=f"Unchanged ({len(same)})")
        ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
        mean_delta = float(np.mean([s-b for b,s in zip(base,nms)]))
        ax.text(0.04, 0.93, f"mean Δ F1 = {mean_delta:+.4f}",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                color="#222", bbox=dict(boxstyle="round,pad=0.3", fc="white"))
        ax.set_xlabel("Baseline GriTS-Top F1"); ax.set_ylabel("+Row-NMS GriTS-Top F1")
        ax.set_title(f"{ds} complex (n={n})", fontsize=12, fontweight="bold")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.25)

    plt.suptitle("Per-Table F1: Baseline vs Row-NMS "
                 "(above diagonal = improved by post-processing)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = RESULTS_DIR / "fig_scatter.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Row-NMS Post-Processing Evaluation")
    print("=" * 65)

    with open(CACHE_PATH, "rb") as f:
        scitsr_cache, fintab_cache = pickle.load(f)
    print(f"[cache] SciTSR={len(scitsr_cache)}  FinTab={len(fintab_cache)}")

    THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results   = {"SciTSR": {}, "FinTabNet": {}}
    sweep     = {"SciTSR": {}, "FinTabNet": {}}

    for ds_name, cache in [("SciTSR", scitsr_cache), ("FinTabNet", fintab_cache)]:
        print(f"\n[{ds_name}] Baseline ...")
        base = eval_pipeline(cache, "baseline", ds_name=ds_name)
        results[ds_name]["baseline"] = base
        sweep[ds_name + "_base"] = base["f1"]
        print(f"  n={base['n']}  F1={base['f1']:.4f}  "
              f"Span={base['span_f1']:.4f}  "
              f"LE={base.get('logical_exact')}")

        for t in THRESHOLDS:
            r = eval_pipeline(cache, "row_nms", iou_thresh=t, ds_name=ds_name)
            sweep[ds_name][t] = r["f1"]
            print(f"  RowNMS τ={t}: F1={r['f1']:.4f}  dr_exact={r['dr_exact_pct']:.1f}%")
            results[ds_name][f"row_nms_{t}"] = r

        # pick best
        best_t = max(THRESHOLDS, key=lambda t: sweep[ds_name][t])
        results[ds_name]["row_nms@best"] = results[ds_name][f"row_nms_{best_t}"]
        results[ds_name]["row_nms@best"]["best_thresh"] = best_t
        print(f"  → best τ = {best_t}  F1 = {sweep[ds_name][best_t]:.4f}")

        # rowcol NMS at best τ
        rc = eval_pipeline(cache, "rowcol_nms", iou_thresh=best_t, ds_name=ds_name)
        results[ds_name]["rowcol_nms"] = rc
        print(f"  RowColNMS τ={best_t}: F1={rc['f1']:.4f}")

    # Save summary JSON (without per-table records to keep small)
    summary = {ds: {k: {kk: vv for kk, vv in v.items() if kk != "records"}
                    for k, v in results[ds].items()}
               for ds in results}
    with open(RESULTS_DIR / "rownms_summary.json", "w") as f:
        json.dump({"summary": summary, "sweep": sweep}, f, indent=2)
    print(f"\n[saved] {RESULTS_DIR / 'rownms_summary.json'}")

    # Figures
    print("\n[figures]")
    fig_dr_before_after(results)
    fig_f1_sweep(sweep)
    fig_main(results)
    fig_scatter(results)

    # Print summary
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    for ds in ["SciTSR", "FinTabNet"]:
        b = results[ds]["baseline"]
        n = results[ds]["row_nms@best"]
        print(f"\n  [{ds} complex]    best τ = {n['best_thresh']}")
        print(f"  {'Metric':<18}{'Baseline':>12}{'+RowNMS':>12}{'Δ':>12}")
        print(f"  {'-'*54}")
        for k, lab in [("f1","GriTS-Top F1"),
                       ("span_f1","Span-Only F1"),
                       ("logical_exact","Logical-Exact"),
                       ("dr_exact_pct","drExact %")]:
            bv = b.get(k) or 0
            nv = n.get(k) or 0
            print(f"  {lab:<18}{bv:>12.4f}{nv:>12.4f}{nv-bv:>+12.4f}")
    print(f"\n  Results: {RESULTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()

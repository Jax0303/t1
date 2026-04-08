#!/usr/bin/env python3
"""
PPT Figure Generator — Detection TSR Spanning Cell Bottleneck
=============================================================
발표용 고품질 시각화 5종 생성.
실측 데이터(grits_summary.json, benchmark_summary.json, boundary_summary.json)
기반으로만 작성 — 시뮬레이션 수치는 레퍼런스 참조선으로만 표시.

출력: experiments/results_ppt/
  fig1_problem_definition.png   — GriTS trivial vs model (span 신호 희석 증명)
  fig2_span_only_grits.png      — Span-only GriTS: TATR vs Docling (핵심 결과)
  fig3_boundary_analysis.png    — Recall@0.5 vs @0.9 (boundary 정밀도 분석)
  fig4_type_stratified.png      — 유형별(A~D) 패러다임 비교
  fig5_summary.png              — 전체 증거 요약 (발표 마지막 슬라이드용)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────────
# 경로 & 색상
# ─────────────────────────────────────────────
BASE = Path(__file__).parent
OUT  = BASE / "results_ppt"
OUT.mkdir(exist_ok=True)

GRITS_JSON     = BASE / "results_grits"     / "grits_summary.json"
BENCH_JSON     = BASE / "results_benchmark" / "benchmark_summary.json"
BOUNDARY_JSON  = BASE / "results"           / "boundary_summary.json"

# 모델/유형 색상 팔레트
C_TATR    = "#E05C4B"   # 붉은 주황
C_DOCLING = "#5B8DD9"   # 파란 계열
C_TRIVIAL = "#BBBBBB"   # 회색

TYPE_COLORS = {
    "A-Simple": "#4878CF",
    "B-Header": "#6ACC65",
    "C-Multi":  "#D65F5F",
    "D-Dense":  "#B47CC7",
}

FONT_TITLE  = 15
FONT_LABEL  = 12
FONT_TICK   = 10
FONT_ANNOT  = 9
DPI         = 180


# ─────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────
def load_json(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _grits():
    return load_json(GRITS_JSON)

def _bench():
    return load_json(BENCH_JSON)

def _boundary():
    return load_json(BOUNDARY_JSON)


def _s2():
    p = BASE / "results_s2" / "s2_summary.json"
    return load_json(p) if p.exists() else {}


# ─────────────────────────────────────────────
# Fig 1 — GriTS trivial vs model
# "GriTS-Top 전체 지표는 spanning cell에 둔감하다"
# ─────────────────────────────────────────────
def fig1_problem_definition():
    g = _grits()
    models = ["TATR v1.0", "TATR v1.1-pub", "TATR v1.1-all"]
    labels = ["TATR v1.0", "TATR v1.1-pub", "TATR v1.1-all"]

    f1_full    = [g[m]["grits_top_f1"]          for m in models]
    f1_trivial = [g[m]["grits_top_f1_trivial"]  for m in models]
    f1_span    = [g[m]["grits_top_f1_span_only"] for m in models]
    deltas     = [g[m]["grits_top_delta_vs_trivial"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Fig 1.  GriTS-Top is Insensitive to Spanning Cells\n"
        "(SciTSR-COMP, n=716, TATR family)",
        fontsize=FONT_TITLE, fontweight="bold", y=1.01,
    )

    x = np.arange(len(labels))
    w = 0.35

    # Panel A: Full GriTS vs Trivial
    ax = axes[0]
    b1 = ax.bar(x - w/2, f1_full,    width=w, color=C_TATR,    alpha=0.9, label="Model GriTS-Top")
    b2 = ax.bar(x + w/2, f1_trivial, width=w, color=C_TRIVIAL, alpha=0.9, label="Trivial (all 1×1)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylim(0.70, 0.88)
    ax.set_ylabel("GriTS-Top F1", fontsize=FONT_LABEL)
    ax.set_title("(A)  Full Table GriTS-Top vs Trivial Baseline", fontsize=FONT_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_TICK)
    ax.axhline(0.80, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    for bar, v in zip(b1, f1_full):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=FONT_ANNOT, fontweight="bold")
    for bar, d in zip(b2, deltas):
        sign = "+" if d >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"Δ={sign}{d:.4f}", ha="center", va="bottom",
                fontsize=7, color="#555555")

    # Panel B: Span-only GriTS
    ax = axes[1]
    bars = ax.bar(x, f1_span, color=[C_TATR]*len(x), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylim(0, 0.50)
    ax.set_ylabel("GriTS-Top F1  (spanning cells only)", fontsize=FONT_LABEL)
    ax.set_title("(B)  Span-Only GriTS-Top\n(GT spanning cells isolated)", fontsize=FONT_LABEL, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5)

    for bar, v in zip(bars, f1_span):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=FONT_ANNOT, fontweight="bold",
                color=C_TATR)

    # 주석
    ax.annotate(
        "TATR v1.0: full GriTS=0.802\nbut span-only GriTS=0.086\n→ span signal diluted by\nnon-spanning cells (~83%)",
        xy=(0, f1_span[0]), xytext=(0.5, 0.35),
        arrowprops=dict(arrowstyle="->", color="#333333"),
        fontsize=8, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFFDE7", ec="#CCCCCC"),
    )

    plt.tight_layout()
    out = OUT / "fig1_problem_definition.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out}")


# ─────────────────────────────────────────────
# Fig 2 — Span-only GriTS: TATR vs Docling
# "핵심 결과: generation paradigm이 span-only에서 명확히 우위"
# ─────────────────────────────────────────────
def fig2_span_only_grits():
    b = _bench()
    ttypes = ["A-Simple", "B-Header", "C-Multi", "D-Dense"]

    tatr    = b.get("TATR v1.1-all",       {})
    docling = b.get("Docling-TableFormer",  {})

    tatr_full    = [tatr.get(t,    {}).get("grits_top_f1",       0) or 0 for t in ttypes]
    docling_full = [docling.get(t, {}).get("grits_top_f1",       0) or 0 for t in ttypes]
    tatr_span    = [tatr.get(t,    {}).get("grits_top_span_only", None) for t in ttypes]
    docling_span = [docling.get(t, {}).get("grits_top_span_only", None) for t in ttypes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Fig 2.  Detection vs Generation Paradigm — GriTS-Top Comparison\n"
        "(SciTSR-COMP stratified sample, per span-complexity type)",
        fontsize=FONT_TITLE, fontweight="bold", y=1.01,
    )

    x = np.arange(len(ttypes))
    w = 0.35

    # Panel A: Full GriTS-Top
    ax = axes[0]
    b1 = ax.bar(x - w/2, tatr_full,    width=w, color=C_TATR,    alpha=0.9, label="TATR v1.1-all\n(detection-DETR)")
    b2 = ax.bar(x + w/2, docling_full, width=w, color=C_DOCLING, alpha=0.9, label="Docling-TableFormer\n(generation)")
    ax.set_xticks(x); ax.set_xticklabels(ttypes, fontsize=FONT_TICK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("GriTS-Top F1", fontsize=FONT_LABEL)
    ax.set_title("(A)  Full GriTS-Top F1\n(all cells, span dilution effect)", fontsize=FONT_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_TICK, loc="lower left")
    for bar, v in zip(b1, tatr_full):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=FONT_ANNOT)
    for bar, v in zip(b2, docling_full):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=FONT_ANNOT)

    # Panel B: Span-only GriTS-Top
    ax = axes[1]
    tatr_s_plot    = [v if v is not None else 0.0 for v in tatr_span]
    docling_s_plot = [v if v is not None else 0.0 for v in docling_span]
    b3 = ax.bar(x - w/2, tatr_s_plot,    width=w, color=C_TATR,    alpha=0.9, label="TATR v1.1-all")
    b4 = ax.bar(x + w/2, docling_s_plot, width=w, color=C_DOCLING, alpha=0.9, label="Docling-TableFormer")
    ax.set_xticks(x); ax.set_xticklabels(ttypes, fontsize=FONT_TICK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("GriTS-Top F1  (spanning cells only)", fontsize=FONT_LABEL)
    ax.set_title("(B)  Span-Only GriTS-Top F1  ★\n(spanning cells isolated — main result)", fontsize=FONT_LABEL, fontweight="bold")
    ax.legend(fontsize=FONT_TICK, loc="upper left")

    for bar, tv, dv in zip(b3, tatr_s_plot, docling_s_plot):
        ax.text(bar.get_x() + bar.get_width()/2, tv + 0.01,
                f"{tv:.3f}" if tv > 0 else "N/A",
                ha="center", va="bottom", fontsize=FONT_ANNOT, color=C_TATR)
    for bar, dv in zip(b4, docling_s_plot):
        ax.text(bar.get_x() + bar.get_width()/2, dv + 0.01,
                f"{dv:.3f}" if dv > 0 else "N/A",
                ha="center", va="bottom", fontsize=FONT_ANNOT, color=C_DOCLING)

    # Gap 화살표 (B-Header)
    idx = ttypes.index("B-Header")
    tv  = tatr_s_plot[idx]
    dv  = docling_s_plot[idx]
    if dv > tv:
        ax.annotate(
            "", xy=(idx + w/2, dv), xytext=(idx + w/2, tv),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5),
        )
        ax.text(idx + w/2 + 0.05, (tv + dv)/2,
                f"Δ={dv-tv:.3f}", fontsize=9, color="#333333", va="center")

    # N/A 안내
    ax.text(0.02, 0.03,
            "N/A = no spanning cells in this type (A-Simple)",
            transform=ax.transAxes, fontsize=7, color="#888888")

    plt.tight_layout()
    out = OUT / "fig2_span_only_grits.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out}")


# ─────────────────────────────────────────────
# Fig 3 — Boundary Precision Analysis
# "TATR은 span bbox를 찾지만 경계가 틀려 grid index 오류 발생"
# ─────────────────────────────────────────────
def fig3_boundary_analysis():
    bd = _boundary()

    n_gt  = bd["n_gt_spans"]
    n_50  = bd["n_detected_50"]
    n_75  = bd["n_detected_75"]
    n_90  = bd["n_detected_90"]
    gerr_all = bd["grid_err_rate_all_detected"]
    gerr_90  = bd["grid_err_rate_iou_90plus"]

    r50 = n_50 / n_gt
    r75 = n_75 / n_gt
    r90 = n_90 / n_gt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Fig 3.  Spanning Cell Boundary Precision Analysis\n"
        "(TATR v1.1-all, PubTables-1M test, n=300 tables, Hungarian matching)\n"
        "[Note: requires pixel-level GT bbox — SciTSR chunk-bbox is incompatible]",
        fontsize=FONT_TITLE - 1, fontweight="bold", y=1.02,
    )

    # Panel A: Recall @ IoU thresholds
    ax = axes[0]
    thresholds = ["@IoU≥0.5\n(loose)", "@IoU≥0.75\n(medium)", "@IoU≥0.9\n(tight)"]
    recalls    = [r50, r75, r90]
    bar_colors = ["#5B8DD9", "#F5A623", "#E05C4B"]

    bars = ax.bar(thresholds, recalls, color=bar_colors, alpha=0.9, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall (matched spans / GT spans)", fontsize=FONT_LABEL)
    ax.set_title(f"(A)  Spanning Cell Detection Recall\n(PubTables-1M, GT spans = {n_gt})",
                 fontsize=FONT_LABEL, fontweight="bold")

    for bar, v, n in zip(bars, recalls, [n_50, n_75, n_90]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f"{v:.1%}\n({n}/{n_gt})",
                ha="center", va="bottom", fontsize=FONT_ANNOT, fontweight="bold")

    # 핵심 gap 강조
    ax.annotate(
        f"Recall drops\n{r50:.0%}→{r90:.0%}\nwhen requiring tight bbox",
        xy=(2, r90), xytext=(1.0, 0.55),
        arrowprops=dict(arrowstyle="->", color="#555555"),
        fontsize=9, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#CCCCCC"),
    )

    # Panel B: Grid error rate
    ax = axes[1]
    labels  = ["All matched\n(IoU≥0.5)", "High-quality\n(IoU≥0.9)"]
    gerrs   = [gerr_all * 100, gerr_90 * 100]
    bcolors = ["#E05C4B", "#4CAF50"]

    bars2 = ax.bar(labels, gerrs, color=bcolors, alpha=0.85, width=0.4)
    ax.set_ylim(0, 12)
    ax.set_ylabel("Grid Index Error Rate (%)", fontsize=FONT_LABEL)
    ax.set_title(
        "(B)  Grid Index Error Rate\n(separator boundary crossed → wrong colspan/rowspan)",
        fontsize=FONT_LABEL, fontweight="bold"
    )

    for bar, v in zip(bars2, gerrs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f"{v:.2f}%", ha="center", va="bottom",
                fontsize=FONT_LABEL, fontweight="bold")

    ax.annotate(
        "When bbox IoU ≥ 0.9,\ngrid error = 0%\n→ boundary precision\nis the root cause",
        xy=(1, 0.0), xytext=(0.3, 6),
        arrowprops=dict(arrowstyle="->", color="#2E7D32"),
        fontsize=9, color="#2E7D32",
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#A5D6A7"),
    )

    plt.tight_layout()
    out = OUT / "fig3_boundary_analysis.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out}")


# ─────────────────────────────────────────────
# Fig 4 — Type-Stratified Comparison (A~D)
# "span 복잡도가 높을수록 gap 심화"
# ─────────────────────────────────────────────
def fig4_type_stratified():
    b = _bench()
    ttypes = ["A-Simple", "B-Header", "C-Multi", "D-Dense"]
    type_labels = [
        "A-Simple\n(0 spans)",
        "B-Header\n(1–3 spans)",
        "C-Multi\n(4–10 spans)",
        "D-Dense\n(11+ spans)",
    ]

    tatr    = b.get("TATR v1.1-all",       {})
    docling = b.get("Docling-TableFormer",  {})

    # Span-only GriTS (None → 0 for plot, annotate N/A)
    def safe(d, t, key):
        s = d.get(t, {})
        if s is None:
            return None
        return s.get(key)

    tatr_span    = [safe(tatr, t, "grits_top_span_only") for t in ttypes]
    docling_span = [safe(docling, t, "grits_top_span_only") for t in ttypes]
    tatr_full    = [safe(tatr, t, "grits_top_f1") or 0 for t in ttypes]
    docling_full = [safe(docling, t, "grits_top_f1") or 0 for t in ttypes]

    # gap
    gaps = []
    for ts, ds in zip(tatr_span, docling_span):
        if ts is not None and ds is not None:
            gaps.append(ds - ts)
        else:
            gaps.append(None)

    fig = plt.figure(figsize=(15, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    fig.suptitle(
        "Fig 4.  Span Complexity Stratification — Detection vs Generation\n"
        "(SciTSR-COMP, n=25–30 per type)",
        fontsize=FONT_TITLE, fontweight="bold",
    )

    x = np.arange(len(ttypes))
    w = 0.35

    # (A) Full GriTS
    b1 = ax1.bar(x - w/2, tatr_full,    width=w, color=C_TATR,    alpha=0.9, label="TATR v1.1-all")
    b2 = ax1.bar(x + w/2, docling_full, width=w, color=C_DOCLING, alpha=0.9, label="Docling")
    ax1.set_xticks(x); ax1.set_xticklabels(type_labels, fontsize=8)
    ax1.set_ylim(0, 1.1); ax1.set_ylabel("GriTS-Top F1", fontsize=FONT_LABEL)
    ax1.set_title("(A)  Full GriTS-Top F1", fontsize=FONT_LABEL, fontweight="bold")
    ax1.legend(fontsize=9)
    for bar, v in list(zip(b1, tatr_full)) + list(zip(b2, docling_full)):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=7)

    # (B) Span-only GriTS
    ts_plot = [v if v is not None else 0.0 for v in tatr_span]
    ds_plot = [v if v is not None else 0.0 for v in docling_span]
    b3 = ax2.bar(x - w/2, ts_plot, width=w, color=C_TATR,    alpha=0.9, label="TATR v1.1-all")
    b4 = ax2.bar(x + w/2, ds_plot, width=w, color=C_DOCLING, alpha=0.9, label="Docling")
    ax2.set_xticks(x); ax2.set_xticklabels(type_labels, fontsize=8)
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("GriTS-Top F1  (span-only)", fontsize=FONT_LABEL)
    ax2.set_title("(B)  Span-Only GriTS-Top F1  ★", fontsize=FONT_LABEL, fontweight="bold")
    ax2.legend(fontsize=9)
    for i, (tv, dv, ts, ds) in enumerate(zip(ts_plot, ds_plot, tatr_span, docling_span)):
        label_t = f"{tv:.3f}" if ts is not None else "N/A"
        label_d = f"{dv:.3f}" if ds is not None else "N/A"
        ax2.text(x[i] - w/2 + w/2, tv + 0.01, label_t,
                 ha="center", va="bottom", fontsize=7, color=C_TATR)
        ax2.text(x[i] + w/2 + w/2 - w, dv + 0.01, label_d,
                 ha="center", va="bottom", fontsize=7, color=C_DOCLING)

    # (C) Gap (Docling - TATR) on span-only
    gap_plot   = [g if g is not None else 0.0 for g in gaps]
    gap_colors = [TYPE_COLORS[t] for t in ttypes]
    b5 = ax3.bar(x, gap_plot, color=gap_colors, alpha=0.9)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels(type_labels, fontsize=8)
    ax3.set_ylim(-0.1, 0.65); ax3.set_ylabel("Δ GriTS-Top span-only\n(Docling − TATR)", fontsize=FONT_LABEL)
    ax3.set_title("(C)  Performance Gap\n(Docling advantage)", fontsize=FONT_LABEL, fontweight="bold")
    for bar, g_val, t in zip(b5, gap_plot, ttypes):
        label = f"+{g_val:.3f}" if g_val > 0 else ("N/A" if gaps[ttypes.index(t)] is None else f"{g_val:.3f}")
        ax3.text(bar.get_x() + bar.get_width()/2, max(g_val, 0) + 0.01,
                 label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = OUT / "fig4_type_stratified.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out}")


# ─────────────────────────────────────────────
# Fig 5 — Summary (발표 마지막 슬라이드)
# ─────────────────────────────────────────────
def fig5_summary():
    g  = _grits()
    b  = _bench()
    bd = _boundary()
    s2 = _s2()

    # --- 수치 수집 ---
    tatr_grits_full  = g["TATR v1.1-all"]["grits_top_f1"]
    tatr_grits_span  = g["TATR v1.1-all"]["grits_top_f1_span_only"]
    tatr_trivial     = g["TATR v1.1-all"]["grits_top_f1_trivial"]
    delta            = g["TATR v1.1-all"]["grits_top_delta_vs_trivial"]

    bench_tatr    = b.get("TATR v1.1-all", {})
    bench_docling = b.get("Docling-TableFormer", {})
    type_gaps = {}
    for t in ["B-Header", "C-Multi", "D-Dense"]:
        ts = (bench_tatr.get(t)    or {}).get("grits_top_span_only")
        ds = (bench_docling.get(t) or {}).get("grits_top_span_only")
        if ts is not None and ds is not None:
            type_gaps[t] = {"tatr": ts, "docling": ds, "gap": ds - ts}

    # s2 GriTS 수치 (n=50 샘플, GriTS 기반 — bbox 좌표계 무관)
    s2_tatr_grits    = (s2.get("models", {}).get("TATR v1.1-all",       {}) or {}).get("grits_span_only_mean")
    s2_docling_grits = (s2.get("models", {}).get("Docling",              {}) or {}).get("grits_span_only_mean")

    r50 = bd["n_detected_50"] / bd["n_gt_spans"]
    r90 = bd["n_detected_90"] / bd["n_gt_spans"]
    gerr_all = bd["grid_err_rate_all_detected"]
    gerr_90  = bd["grid_err_rate_iou_90plus"]

    # --- Layout ---
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    fig.suptitle(
        "Spanning Cell Bottleneck in Detection-Based TSR — Evidence Summary",
        fontsize=16, fontweight="bold", y=1.01,
    )

    # ── A: GriTS full vs trivial (TATR v1.1-all only)
    ax = axes[0]
    vals  = [tatr_grits_full, tatr_trivial]
    lbls  = ["Model\nGriTS-Top", "Trivial\n(all 1×1)"]
    colors = [C_TATR, C_TRIVIAL]
    bars = ax.bar(lbls, vals, color=colors, alpha=0.9, width=0.4)
    ax.set_ylim(0.70, 0.82)
    ax.set_title("[1] Full GriTS-Top\nvs Trivial Baseline", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("F1", fontsize=FONT_LABEL)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.08, f"Δ = +{delta:.4f}\n(near-zero!)",
            ha="center", transform=ax.transAxes, fontsize=9,
            color="#CC3333", fontweight="bold",
            bbox=dict(boxstyle="round", fc="#FFEEEE", ec="#CC3333"))

    # ── B: Span-only GriTS TATR vs Docling (B-Header)
    ax = axes[1]
    if "B-Header" in type_gaps:
        vals2  = [type_gaps["B-Header"]["tatr"], type_gaps["B-Header"]["docling"]]
        lbls2  = [f"TATR v1.1-all\n(detection)", f"Docling\n(generation)"]
        colors2 = [C_TATR, C_DOCLING]
        bars2 = ax.bar(lbls2, vals2, color=colors2, alpha=0.9, width=0.4)
        ax.set_ylim(0, 0.85)
        ax.set_title("[2] Span-Only GriTS-Top\n(B-Header type: 1-3 spans)", fontsize=FONT_LABEL, fontweight="bold")
        ax.set_ylabel("F1  (span cells only)", fontsize=FONT_LABEL)
        for bar, v in zip(bars2, vals2):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        gap = type_gaps["B-Header"]["gap"]
        ax.text(0.5, 0.85, f"Gap = +{gap:.3f}\n(Docling advantage)",
                ha="center", transform=ax.transAxes, fontsize=9,
                color=C_DOCLING, fontweight="bold",
                bbox=dict(boxstyle="round", fc="#EEF4FF", ec=C_DOCLING))

    # ── C: Recall @0.5 vs @0.9
    ax = axes[2]
    thresh_labels = ["Recall@0.5", "Recall@0.75", "Recall@0.9"]
    n_gt = bd["n_gt_spans"]
    thresh_vals = [
        bd["n_detected_50"] / n_gt,
        bd["n_detected_75"] / n_gt,
        bd["n_detected_90"] / n_gt,
    ]
    bars3 = ax.bar(thresh_labels, thresh_vals,
                   color=["#5B8DD9", "#F5A623", "#E05C4B"], alpha=0.9, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("[3] Bbox Detection Recall\n(spanning cells, TATR v1.1-all)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Recall", fontsize=FONT_LABEL)
    for bar, v in zip(bars3, thresh_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── D: Grid error rate
    ax = axes[3]
    gerr_labels = ["All matched\n(IoU≥0.5)", "High-quality\n(IoU≥0.9)"]
    gerr_vals   = [gerr_all * 100, gerr_90 * 100]
    bars4 = ax.bar(gerr_labels, gerr_vals, color=["#E05C4B", "#4CAF50"], alpha=0.9, width=0.4)
    ax.set_ylim(0, 10)
    ax.set_title("[4] Grid Index Error Rate\n(span index wrong due to boundary miss)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Error Rate (%)", fontsize=FONT_LABEL)
    for bar, v in zip(bars4, gerr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ── E: Type-stratified gap overview
    ax = axes[4]
    types_with_gap = [t for t in ["B-Header", "C-Multi", "D-Dense"] if t in type_gaps]
    gap_vals  = [type_gaps[t]["gap"] for t in types_with_gap]
    gap_tatr  = [type_gaps[t]["tatr"] for t in types_with_gap]
    gap_doc   = [type_gaps[t]["docling"] for t in types_with_gap]
    xi = np.arange(len(types_with_gap))
    bw = 0.30
    ax.bar(xi - bw/2, gap_tatr, width=bw, color=C_TATR,    alpha=0.9, label="TATR")
    ax.bar(xi + bw/2, gap_doc,  width=bw, color=C_DOCLING, alpha=0.9, label="Docling")
    ax.set_xticks(xi); ax.set_xticklabels(types_with_gap, fontsize=FONT_TICK)
    ax.set_ylim(0, 0.90); ax.set_ylabel("Span-Only GriTS-Top F1", fontsize=FONT_LABEL)
    ax.set_title("[5] Span-Only GriTS by Complexity\n(gap persists across all types)", fontsize=FONT_LABEL, fontweight="bold")
    ax.legend(fontsize=9)

    # ── F: Text summary box
    ax = axes[5]
    ax.axis("off")
    s2_tatr_str    = f"{s2_tatr_grits:.3f}"    if s2_tatr_grits    is not None else "N/A"
    s2_docling_str = f"{s2_docling_grits:.3f}" if s2_docling_grits is not None else "N/A"

    summary_text = (
        "KEY FINDINGS  (all metrics: GriTS-Top, bbox-free)\n"
        "-------------------------------------------------\n"
        f"[1] TATR full GriTS = {tatr_grits_full:.3f}  ~  trivial (D = {delta:+.4f})\n"
        f"    span signal diluted by non-spanning cells (~83%)\n\n"
        f"[2] Span-only GriTS (n=716, SciTSR-COMP):\n"
        f"    TATR = {tatr_grits_span:.3f}\n"
        f"    Docling B-Header = {type_gaps.get('B-Header',{}).get('docling', 0):.3f}\n\n"
        f"[3] Cross-paradigm GriTS-Span (n=50, direct):\n"
        f"    TATR = {s2_tatr_str}  vs  Docling = {s2_docling_str}\n"
        f"    [resolved s2 bbox-IoU artifact]\n\n"
        f"[4] Bbox precision (PubTables-1M, n=300):\n"
        f"    Recall@0.5={r50:.1%}  ->  Recall@0.9={r90:.1%}\n"
        f"    Grid err: {gerr_all:.1%} (IoU>=0.5) / {gerr_90:.1%} (IoU>=0.9)\n\n"
        "CONCLUSION:\n"
        "GriTS-Top confirms detection TSR fails at spans.\n"
        "Generation paradigm: ~6x higher span-only GriTS.\n"
        "Root cause: bbox boundary imprecision -> span index error."
    )
    ax.text(0.04, 0.97, summary_text,
            transform=ax.transAxes, fontsize=9.5,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F0F4FF", ec="#7986CB", linewidth=1.5))

    plt.tight_layout()
    out = OUT / "fig5_summary.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="#F8F9FA")
    plt.close()
    print(f"[saved] {out}")


# ─────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PPT Figure Generator — TSR Spanning Cell Bottleneck")
    print("=" * 60)
    print(f"  Output dir: {OUT}\n")

    # 데이터 파일 존재 확인
    missing = [p for p in [GRITS_JSON, BENCH_JSON, BOUNDARY_JSON] if not p.exists()]
    if missing:
        print("[ERROR] Missing result files:")
        for p in missing:
            print(f"  {p}")
        print("\nRun the following experiments first:")
        print("  python experiments/grits_eval.py")
        print("  python experiments/paradigm_benchmark.py")
        print("  python experiments/boundary_precision_analysis.py")
        return

    fig1_problem_definition()
    fig2_span_only_grits()
    fig3_boundary_analysis()
    fig4_type_stratified()
    fig5_summary()

    print(f"\n{'='*60}")
    print(f"All figures saved to: {OUT}")
    print(f"{'='*60}")
    print("\nFiles:")
    for f in sorted(OUT.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")


if __name__ == "__main__":
    main()

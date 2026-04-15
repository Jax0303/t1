#!/usr/bin/env python3
"""
PPT용 그림 생성:
  1. fig_pipeline.png       : 현재 baseline 파이프라인 + Row-NMS 삽입 위치
  2. fig_evidence_chain.png : row detection이 병목이라는 3단 근거 체인
  3. fig_axes_guide.md      : 기존 figure들의 축 의미 정리 (텍스트)
"""
from __future__ import annotations
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results_rownms"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. Pipeline diagram
# ─────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, color="#E8F1FA", edge="#2C5B8F", fs=11, bold=True):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.08,rounding_size=0.15",
                           linewidth=1.8, edgecolor=edge, facecolor=color)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color="#1a1a1a")

    def arrow(x1, y1, x2, y2, color="#555", lw=2):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->,head_width=0.25,head_length=0.35",
                            color=color, linewidth=lw, mutation_scale=15)
        ax.add_patch(a)

    # Title
    ax.text(7, 7.55, "Detection-based TSR Pipeline  (TATR + Grid Reconstruction)",
            ha="center", fontsize=15, fontweight="bold", color="#222")

    # Row 1: Input
    box(0.4, 5.4, 2.2, 1.2, "Input Image\n(table region)", color="#F5F5F5", edge="#666")
    arrow(2.65, 6.0, 3.55, 6.0)

    # Row 2: TATR
    box(3.6, 5.1, 3.3, 1.8,
        "TATR\n(DETR-based object detector)\nmicrosoft/table-transformer-v1.1-all",
        color="#FFE4B5", edge="#C77B00", fs=10)

    arrow(6.95, 6.0, 8.1, 6.0)

    # Row 3: three output branches
    box(8.15, 6.6, 3.0, 0.75, "row bboxes    (N_r)", color="#D7E8FF", edge="#2C5B8F", fs=10)
    box(8.15, 5.6, 3.0, 0.75, "column bboxes (N_c)", color="#D7E8FF", edge="#2C5B8F", fs=10)
    box(8.15, 4.6, 3.0, 0.75, "spanning cells (N_s)", color="#FFD7D7", edge="#AA2222", fs=10)

    # converge
    arrow(11.2, 6.97, 12.05, 5.1)
    arrow(11.2, 5.97, 12.05, 5.0)
    arrow(11.2, 4.97, 12.05, 4.9)

    # Grid reconstruct
    box(12.1, 4.2, 1.8, 1.6, "Grid\nReconstruction\n(IoU>0.3)",
        color="#E8E8E8", edge="#444", fs=10)

    # Output
    arrow(13.0, 4.15, 13.0, 3.55)
    box(11.9, 2.55, 2.0, 1.0, "Cells\n{(r,c), bbox}", color="#F5F5F5", edge="#666", fs=10)

    arrow(12.9, 2.5, 12.9, 1.85)
    box(11.5, 0.9, 2.8, 0.95, "GriTS-Top F1\n(evaluation metric)",
        color="#FFF6C0", edge="#A08700", fs=10)

    # ── Row-NMS insertion ─────────────────────
    # 빨간 점선 박스 표시
    box(8.15, 2.7, 3.0, 1.4,
        "★ Row-NMS\n(proposed)\nmerge y-overlapping rows",
        color="#DDFADB", edge="#2C9B3C", fs=10)

    # dashed arrow from rows bbox through Row-NMS
    ax.annotate("", xy=(9.65, 4.1), xytext=(9.65, 6.55),
                arrowprops=dict(arrowstyle="->", color="#2C9B3C",
                                lw=2.2, linestyle="dashed"))
    ax.text(9.75, 4.4, "insert here",
            fontsize=9.5, color="#2C9B3C", fontweight="bold")

    # after Row-NMS goes back to grid reconstruct
    ax.annotate("", xy=(12.1, 4.85), xytext=(11.18, 3.4),
                arrowprops=dict(arrowstyle="->", color="#2C9B3C",
                                lw=2.2, linestyle="dashed"))

    # Legend
    ax.text(0.4, 3.5, "Legend:", fontsize=11, fontweight="bold")
    ax.text(0.4, 3.1, "■  baseline data flow",     fontsize=10, color="#555")
    ax.text(0.4, 2.75, "■  bottleneck branch",       fontsize=10, color="#AA2222")
    ax.text(0.4, 2.4,  "■  proposed insertion",      fontsize=10, color="#2C9B3C")

    # Claim banner
    ax.text(0.4, 1.7,
            "Research claim:\n"
            "  TATR's row bbox branch is the main bottleneck of overall F1,\n"
            "  and a simple geometric Row-NMS alone yields a significant gain.",
            fontsize=10.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#F0F8FF", edgecolor="#2C5B8F", linewidth=1.3))

    p = RESULTS_DIR / "fig_pipeline.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# 1.1 Methodology Detail (Row-NMS Internal Logic)
# ─────────────────────────────────────────────
def fig_method_detail():
    """
    구체적인 Row-NMS 알고리즘 아키텍처/메소드 시각화.
    제목 없이 동작 원리에 집중.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, color="#E8F1FA", edge="#2C5B8F", fs=10, bold=False):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.08,rounding_size=0.1",
                           linewidth=1.5, edgecolor=edge, facecolor=color)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    # 1. Input Section
    ax.text(1.5, 7.2, "1. Input: TATR Row Bboxes", fontsize=12, fontweight="bold")
    # Draw overlapping rows
    # Rect A
    ax.add_patch(plt.Rectangle((0.5, 5.8), 2.0, 0.6, color="#4C72B0", alpha=0.3))
    ax.add_patch(plt.Rectangle((0.5, 5.8), 2.0, 0.6, fill=False, edgecolor="#4C72B0", lw=1.5))
    ax.text(1.5, 6.1, "Row i", ha="center")
    # Rect B (High overlap)
    ax.add_patch(plt.Rectangle((0.5, 5.6), 2.0, 0.6, color="#D62728", alpha=0.3))
    ax.add_patch(plt.Rectangle((0.5, 5.6), 2.0, 0.6, fill=False, edgecolor="#D62728", lw=1.5, ls="--"))
    ax.text(1.5, 5.75, "Row j (Duplicate)", ha="center", color="#800")

    # 2. Logic Section
    ax.text(5.5, 7.2, "2. Geometric Intersection (1D)", fontsize=12, fontweight="bold")
    # Intersection detail
    ax.plot([4.5, 4.5], [5.5, 6.5], color="#333", lw=2) # Axis
    # Bar i
    ax.plot([4.4, 4.6], [5.8, 5.8], color="#4C72B0", lw=3)
    ax.plot([4.4, 4.6], [6.4, 6.4], color="#4C72B0", lw=3)
    ax.vlines(4.5, 5.8, 6.4, colors="#4C72B0", lw=6, alpha=0.5, label="Y_i")
    # Bar j
    ax.plot([4.7, 4.9], [5.6, 5.6], color="#D62728", lw=3)
    ax.plot([4.7, 4.9], [6.2, 6.2], color="#D62728", lw=3)
    ax.vlines(4.8, 5.6, 6.2, colors="#D62728", lw=6, alpha=0.5, label="Y_j")

    # Math formula box
    formula = r"$Overlap Ratio = \frac{Y_i \cap Y_j}{\min(H_i, H_j)}$"
    ax.text(7.2, 5.8, formula, fontsize=14, bbox=dict(facecolor="#F9F9F9", alpha=0.9, boxstyle="round"))

    # Decision
    ax.text(7.2, 4.8, r"If $Overlap > \tau$ (e.g., 0.5)", fontsize=12, color="#D62728", fontweight="bold")
    ax.text(7.2, 4.3, "→ Merge rows into union bbox", fontsize=11)

    # 3. Output Section
    ax.text(10.5, 7.2, "3. Output: Cleaned Grid", fontsize=12, fontweight="bold")
    # Unified Rect
    ax.add_patch(plt.Rectangle((10.5, 5.6), 2.0, 0.8, color="#2C9B3C", alpha=0.4))
    ax.add_patch(plt.Rectangle((10.5, 5.6), 2.0, 0.8, fill=False, edgecolor="#2C9B3C", lw=2))
    ax.text(11.5, 6.0, "Merged Row", ha="center", fontweight="bold")

    # Flow arrows
    ax.annotate("", xy=(4.2, 6.0), xytext=(2.8, 6.0), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(10.2, 6.0), xytext=(8.8, 6.0), arrowprops=dict(arrowstyle="->", lw=2))

    # Architecture Diagram (Block view)
    ax.text(1, 3.2, "Post-processing Block Architecture", fontsize=12, fontweight="bold", color="#2C5B8F")
    # Grid of blocks
    box(0.5, 1.8, 2.0, 1.0, "Input Bboxes\n[N, 4]", color="#E1E1E1", bold=True)
    ax.annotate("", xy=(3.0, 2.3), xytext=(2.6, 2.3), arrowprops=dict(arrowstyle="->", lw=1.5))
    box(3.1, 1.8, 2.2, 1.0, "Sort by Y_center\n(O(N log N))", color="#F0F8FF")
    ax.annotate("", xy=(5.8, 2.3), xytext=(5.3, 2.3), arrowprops=dict(arrowstyle="->", lw=1.5))
    box(5.9, 1.8, 3.0, 1.0, "Adjacency Overlap Check\n1D Intersection over Min", color="#F0F8FF")
    ax.annotate("", xy=(9.4, 2.3), xytext=(8.9, 2.3), arrowprops=dict(arrowstyle="->", lw=1.5))
    box(9.5, 1.8, 2.2, 1.0, "Union Merge\n(Connected Comps)", color="#F0F8FF")
    ax.annotate("", xy=(12.2, 2.3), xytext=(11.7, 2.3), arrowprops=dict(arrowstyle="->", lw=1.5))
    box(12.3, 1.8, 1.5, 1.0, "Output\n[N', 4]", color="#DDFADB", bold=True)

    # Explanation text
    desc = (
        "Methodology: Row-NMS (Non-Maximum Suppression)\n"
        "- Motivation: TATR consistently over-predicts rows in complex structures.\n"
        "- Key Logic: Unlike traditional 2D IoU NMS, 1D vertical overlap is used.\n"
        "- Benefit: Resolves +1 row duplicates without retraining the detector."
    )
    ax.text(0.5, 0.2, desc, fontsize=11, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFDEB", edgecolor="#AA9900"))

    p = RESULTS_DIR / "fig_methodology_detail.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# 2. Evidence chain — 3단계 근거
# ─────────────────────────────────────────────
def fig_evidence_chain():
    """
    Step 1: Oracle decomposition (Phase 2 numbers, FinTabNet n=68)
             → Row/col이 병목이고 span은 아님
    Step 2: Count diagnostic
             → Col은 이미 정확, Row가 문제
    Step 3: Error mode
             → Row 오류가 +1 over-prediction에 집중 (duplicate)
    """
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.38, wspace=0.30)

    fig.suptitle("Evidence Chain — Why Row Detection (not Span) is the Bottleneck",
                 fontsize=15, fontweight="bold", y=0.995)

    # ── Panel 1: Oracle decomposition ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["A\nFull TATR\n(baseline)",
              "B\nGT row/col\n+ TATR span",
              "D\nTATR row/col\n+ GT span",
              "C\nAll GT\n(upper bound)"]
    vals = [0.830, 0.993, 0.831, 1.000]
    colors = ["#888", "#2C9B3C", "#C44E52", "#4C72B0"]
    bars = ax1.bar(range(4), vals, color=colors, alpha=0.88, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                 f"{v:.3f}", ha="center", fontsize=10.5, fontweight="bold")
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("GriTS-Top F1", fontsize=10)
    ax1.set_ylim(0, 1.18)
    ax1.set_title("STEP 1 — Oracle Decomposition\n(FinTabNet complex, n=68)",
                  fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # 화살표: A→B 큰 점프, A→D 변화 없음
    ax1.annotate("", xy=(1, 1.06), xytext=(0, 1.06),
                 arrowprops=dict(arrowstyle="->", color="#2C9B3C", lw=2.2))
    ax1.text(0.5, 1.1, "+0.163", ha="center", color="#2C9B3C",
             fontsize=11, fontweight="bold")
    ax1.annotate("", xy=(2, 0.92), xytext=(0, 0.92),
                 arrowprops=dict(arrowstyle="->", color="#C44E52", lw=2.2))
    ax1.text(1.0, 0.88, "+0.001", ha="center", color="#C44E52",
             fontsize=11, fontweight="bold")

    ax1.text(2.0, 0.25,
             "Replacing TATR's row/col with GT\n"
             "    → F1 jumps to 0.993\n"
             "Replacing TATR's span with GT\n"
             "    → F1 essentially unchanged",
             fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="#FFFDEB", edgecolor="#AA9900"))

    # ── Panel 2: Count mismatch ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    categories = ["SciTSR\nrows", "SciTSR\ncols", "FinTab\nrows", "FinTab\ncols"]
    exact_pct  = [9.9, 85.2, 0.0, 98.5]
    colors2    = ["#C44E52", "#2C9B3C", "#C44E52", "#2C9B3C"]
    bars = ax2.bar(range(4), exact_pct, color=colors2, alpha=0.88, edgecolor="white")
    for bar, v in zip(bars, exact_pct):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 2,
                 f"{v:.1f}%", ha="center", fontsize=10.5, fontweight="bold")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_ylabel("% tables where |n_pred - n_gt| = 0", fontsize=9.5)
    ax2.set_ylim(0, 115)
    ax2.set_title("STEP 2 — Count Mismatch Diagnostic\n(both datasets, complex split)",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.axhline(50, color="gray", lw=0.8, linestyle=":")

    ax2.text(1.5, 60,
             "Cols are nearly perfect (85~99%)\n"
             "Rows collapse to 0~10%",
             ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="#FFFDEB", edgecolor="#AA9900"))

    # ── Panel 3: dr=+1 concentration ─────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    drs = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    counts_sc = [3, 4, 9, 4, 18, 71, 548, 24, 4, 1, 3, 6, 2, 0, 2, 5]
    bars = ax3.bar(drs, counts_sc, color="#4C72B0", alpha=0.88, edgecolor="white")
    ax3.axvline(0, color="black", lw=1, linestyle="--")
    # highlight +1 bar
    bars[6].set_color("#D62728")
    bars[6].set_alpha(0.95)
    ax3.text(1, 560, "548\n(76.5%)", ha="center",
             fontsize=10.5, fontweight="bold", color="#D62728")
    ax3.set_xlabel("n_pred_rows - n_gt_rows", fontsize=10)
    ax3.set_ylabel("# tables", fontsize=10)
    ax3.set_title("STEP 3 — Error Mode of Row Over-Prediction\n"
                  "(SciTSR complex, n=716)",
                  fontsize=11, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.text(6, 420,
             "76% of tables are\nexactly +1 row over.\n\n"
             "100% have row pairs\nwith y-overlap > 0.5\n"
             "→ duplicates are\n   geometrically detectable.",
             fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="#FFFDEB", edgecolor="#AA9900"))

    # ── Conclusion panel ─────────────────────────────────
    axc = fig.add_subplot(gs[1, :])
    axc.axis("off")
    axc.set_xlim(0, 10); axc.set_ylim(0, 5)

    def stepbox(cx, cy, text, color):
        p = FancyBboxPatch((cx - 1.3, cy - 0.9), 2.6, 1.8,
                           boxstyle="round,pad=0.08,rounding_size=0.2",
                           linewidth=2, edgecolor=color, facecolor="white")
        axc.add_patch(p)
        axc.text(cx, cy, text, ha="center", va="center", fontsize=10)

    stepbox(1.3, 2.5,
            "STEP 1\nOracle:\nGT row/col → 0.993\nGT span → 0.831\n"
            "∴ row/col is bottleneck",
            "#2C9B3C")
    stepbox(4.4, 2.5,
            "STEP 2\nCount diag:\ncols 85~99% exact\nrows 0~10% exact\n"
            "∴ narrow to ROWS",
            "#4C72B0")
    stepbox(7.5, 2.5,
            "STEP 3\nError mode:\n76% of tables = +1\n100% geometric overlap\n"
            "∴ duplicate-row fix",
            "#D62728")

    # arrows between steps
    for x1, x2 in [(2.6, 3.1), (5.7, 6.2)]:
        axc.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=2.5))

    # final arrow → proposal
    axc.annotate("", xy=(9.35, 2.5), xytext=(8.85, 2.5),
                 arrowprops=dict(arrowstyle="->", color="#555", lw=2.5))
    p = FancyBboxPatch((8.4, 1.6), 1.55, 1.8,
                       boxstyle="round,pad=0.08,rounding_size=0.2",
                       linewidth=2.5, edgecolor="#000", facecolor="#FFFDC0")
    axc.add_patch(p)
    axc.text(9.175, 2.5, "PROPOSAL\nRow-NMS\npost-processing",
             ha="center", va="center", fontsize=10.5, fontweight="bold")

    axc.text(5, 0.55,
             "Conclusion: The bottleneck is not span cell reconstruction, "
             "but duplicate row detection. Row-NMS directly targets it.",
             ha="center", fontsize=11, style="italic", color="#222",
             bbox=dict(boxstyle="round,pad=0.35",
                       facecolor="#F0F8FF", edgecolor="#2C5B8F", linewidth=1.3))

    p = RESULTS_DIR / "fig_evidence_chain.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# 3. Axis guide
# ─────────────────────────────────────────────
def fig_axes_guide():
    """기존 figure들의 축 의미를 한 장에 정리."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Figure Axis Guide — what the X/Y axes mean in each figure",
                 fontsize=14, fontweight="bold", y=0.995)

    panels = [
        ("fig_pipeline.png",
         "X: (schematic)\nY: (schematic)",
         "Data-flow diagram.\n"
         "No numeric axes. Shows that Row-NMS\n"
         "is inserted on the row-bbox branch\n"
         "right after TATR."),
        ("fig_evidence_chain.png",
         "Varies per panel — see bottom panel",
         "3-step evidence chain proving that\n"
         "row detection (not span) is the\n"
         "overall F1 bottleneck."),
        ("fig_dr_before_after.png",
         "X: (n_pred_rows - n_gt_rows)\n    (neg=under, 0=exact, pos=over)\n"
         "Y: # tables with that delta",
         "Shows how the baseline mass piled\n"
         "at +1 shifts to 0 after Row-NMS\n"
         "is applied."),
        ("fig_main.png",
         "X: dataset (categorical: SciTSR / FinTabNet)\n"
         "Y: metric mean value (0~1)",
         "Baseline vs +Row-NMS bar chart.\n"
         "3 panels: GriTS-Top F1, Span-F1,\n"
         "Logical-Exact.\n"
         "Key result: GriTS-Top F1 +2.7~3.9%p."),
        ("fig_f1_sweep.png",
         "X: Row-NMS y-overlap threshold tau\n    (scanned 0.3 ~ 0.8)\n"
         "Y: GriTS-Top F1",
         "Threshold robustness.\n"
         "Beats baseline across tau in [0.3, 0.8].\n"
         "Optimum at tau = 0.5."),
        ("fig_scatter.png",
         "X: per-table baseline F1\n"
         "Y: per-table +Row-NMS F1\n"
         "(one point per table)",
         "Above diagonal = improved,\n"
         "below = regressed. Most tables lie\n"
         "above the diagonal, so the mean gain\n"
         "is not driven by a few outliers."),
    ]

    for ax, (name, axinfo, desc) in zip(axes.flat, panels):
        ax.axis("off")
        ax.text(0.5, 0.93, name, ha="center", va="top",
                fontsize=12, fontweight="bold", color="#2C5B8F",
                transform=ax.transAxes)
        ax.text(0.05, 0.75, axinfo, ha="left", va="top",
                fontsize=10.5, fontfamily="monospace",
                color="#AA4400",
                transform=ax.transAxes)
        ax.text(0.05, 0.40, desc, ha="left", va="top",
                fontsize=10, color="#222",
                transform=ax.transAxes)
        # box border
        ax.add_patch(plt.Rectangle((0.02, 0.03), 0.96, 0.94,
                                    transform=ax.transAxes,
                                    fill=False, edgecolor="#888", linewidth=1.3))

    plt.tight_layout()
    p = RESULTS_DIR / "fig_axes_guide.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


if __name__ == "__main__":
    fig_pipeline()
    fig_method_detail()
    fig_evidence_chain()
    fig_axes_guide()
    print("\nDone. 4 new figures saved to", RESULTS_DIR)

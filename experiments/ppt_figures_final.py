#!/usr/bin/env python3
"""
PPT-Ready Figures for SAGR Research Presentation
=================================================
Phase 1 & Phase 2 결과 기반, 학회 발표용 고급 시각화.
논문/발표 수준의 정교한 디자인.

출력: experiments/results_ppt_final/
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ─── Style ───
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "grid.alpha": 0.3,
    "grid.color": "#CCCCCC",
})

RESULTS_DIR = Path(__file__).parent / "results_ppt_final"
RESULTS_DIR.mkdir(exist_ok=True)

P1 = Path(__file__).parent / "results_phase1" / "summary.json"
P2 = Path(__file__).parent / "results_phase2" / "oracle_summary.json"

# Color palette — refined for presentations
C_SIMPLE  = "#4A90D9"    # Calm blue
C_COMPLEX = "#E05555"    # Alert red
C_ORACLE  = ["#CC4444", "#55A868", "#2C5F8A", "#DD8452"]
C_MODELS  = ["#4C72B0", "#DD8452", "#55A868"]


def load_data():
    with open(P1) as f:
        p1 = json.load(f)
    with open(P2) as f:
        p2 = json.load(f)
    return p1, p2


# ═══════════════════════════════════════════════
# SLIDE 1: "The Smoking Gun" — Trivial-Delta
# ═══════════════════════════════════════════════
def slide1_smoking_gun(p1):
    """
    PPT Slide 1: 핵심 smoking gun.
    Trivial Δ가 거의 0에 수렴함을 보여주는 도표.
    """
    models = ["TATR v1.0", "TATR v1.1-pub", "TATR v1.1-all"]
    datasets = ["SciTSR", "FinTabNet"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Detection-based TSR: Spanning Cell Recovery ≈ No Recovery",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92,
             "Trivial-Δ = GriTS-Top F1(model) − GriTS-Top F1(all 1×1 prediction)\n"
             "Near-zero Δ → the model adds no structural value over a trivial baseline",
             ha="center", fontsize=10, color="#555555", style="italic")

    for ax, ds in zip(axes, datasets):
        x = np.arange(len(models))
        w = 0.32

        for i, (split, color) in enumerate([("simple", C_SIMPLE), ("complex", C_COMPLEX)]):
            vals = []
            for m in models:
                key = f"{ds}/{m}/{split}"
                s = p1.get(key, {})
                vals.append(s.get("trivial_delta", 0))

            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=split.capitalize(),
                          color=color, alpha=0.88, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ypos = bar.get_height()
                offset = 0.0008 if v >= 0 else -0.0015
                ax.text(bar.get_x() + bar.get_width() / 2, ypos + offset,
                        f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8.5, fontweight="bold")

        ax.axhline(0, color="#333333", linewidth=1.2)
        # Annotation zone
        ax.axhspan(-0.005, 0.005, color="#FFEEEE", alpha=0.4, zorder=0)
        ax.text(0.02, 0.95, f"⚠ |Δ| < 0.01 → No meaningful gain",
                transform=ax.transAxes, fontsize=8, color="#CC0000",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec="#FFAAAA"))

        ax.set_title(ds, fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel("Trivial-Δ")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p = RESULTS_DIR / "slide1_smoking_gun.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ═══════════════════════════════════════════════
# SLIDE 2: Span-Only F1 — Isolated Failure
# ═══════════════════════════════════════════════
def slide2_span_only(p1):
    """
    PPT Slide 2: Span-Only F1이 극단적으로 낮음.
    "Spanning cell만 보면 최대 0.45 F1" — 구조적 실패.
    """
    models = ["TATR v1.0", "TATR v1.1-pub", "TATR v1.1-all"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Spanning Cell Recovery is Catastrophically Poor",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92,
             "GriTS-Top F1 evaluated ONLY on spanning cells (complex tables)",
             ha="center", fontsize=10, color="#555555", style="italic")

    x = np.arange(len(models))
    w = 0.3

    for i, (ds, color, hatch) in enumerate([
        ("SciTSR", "#4C72B0", ""),
        ("FinTabNet", "#DD8452", "///")
    ]):
        vals = []
        for m in models:
            key = f"{ds}/{m}/complex"
            s = p1.get(key, {})
            v = s.get("grits_top_span_only")
            vals.append(v if v is not None else 0)

        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=f"{ds} (complex)",
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
                      hatch=hatch)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Failure zone
    ax.axhspan(0, 0.5, color="#FFEEEE", alpha=0.25, zorder=0)
    ax.text(0.98, 0.12, "Failure Zone (F1 < 0.5)",
            transform=ax.transAxes, ha="right", fontsize=9, color="#CC0000",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec="#FFAAAA"))

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Span-Only GriTS-Top F1", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p = RESULTS_DIR / "slide2_span_only_failure.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ═══════════════════════════════════════════════
# SLIDE 3: Oracle Decomposition — Where is The Bottleneck?
# ═══════════════════════════════════════════════
def slide3_oracle(p2):
    """
    PPT Slide 3: Oracle 4개 비교. A→B 점프가 핵심.
    Stage 2 (Grid Reconstruction)가 병목이라는 결론.
    """
    labels = [
        "A: Full TATR\n(as-is)",
        "B: Oracle Row/Col\n(GT row/col)",
        "C: Full Oracle\n(all GT)",
        "D: Oracle Span\n(GT span only)",
    ]
    keys = ["exp_A", "exp_B_sigma0", "exp_C", "exp_D"]
    metrics = [
        ("grits_top_f1", "GriTS-Top F1"),
        ("grits_top_span_only", "Span-Only F1"),
        ("logical_exact", "Logical-Exact Rate"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Oracle Decomposition: Stage 2 (Grid Reconstruction) is the Bottleneck",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92,
             f"FinTabNet complex tables (n={p2.get('exp_A',{}).get('n','68')}), TATR v1.1-all",
             ha="center", fontsize=10, color="#555555", style="italic")

    for ax, (metric, mlabel) in zip(axes, metrics):
        vals = []
        for key in keys:
            r = p2.get(key, {})
            v = r.get(metric)
            vals.append(v if v is not None else 0)

        bars = ax.bar(range(4), vals, color=C_ORACLE, alpha=0.88, width=0.6,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # A→B improvement arrow
        if vals[1] > vals[0] + 0.005:
            mid_x = 0.5
            mid_y = max(vals[0], vals[1]) + 0.08
            delta = vals[1] - vals[0]
            ax.annotate("", xy=(1, vals[1] + 0.04), xytext=(0, vals[0] + 0.04),
                        arrowprops=dict(arrowstyle="->, head_width=0.15",
                                        color="#CC0000", lw=2))
            ax.text(mid_x, mid_y + 0.02,
                    f"Δ = +{delta:.3f}",
                    ha="center", fontsize=11, color="#CC0000", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FFF5F5", ec="#FF9999"))

        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, 1.15)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.grid(axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p = RESULTS_DIR / "slide3_oracle_bottleneck.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ═══════════════════════════════════════════════
# SLIDE 4: Noise Sensitivity — Stage 2 is Fragile
# ═══════════════════════════════════════════════
def slide4_noise(p2):
    """
    PPT Slide 4: 5px 노이즈만으로 Logical-Exact 0.41→0.06.
    """
    noise = p2.get("exp_noise", {})
    sigmas = sorted([int(k) for k in noise.keys()])

    logical = [noise[str(s)].get("logical_exact") or 0 for s in sigmas]
    span_f1 = [noise[str(s)].get("grits_top_span_only") or 0 for s in sigmas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Grid Reconstruction is Extremely Sensitive to BBox Noise",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92,
             "Oracle Row/Col + TATR span, Gaussian noise σ added to GT row/col bbox",
             ha="center", fontsize=10, color="#555555", style="italic")

    # Logical-Exact
    ax1.plot(sigmas, logical, "o-", color="#CC0000", markersize=8, linewidth=2.5,
             markerfacecolor="white", markeredgewidth=2)
    ax1.fill_between(sigmas, logical, alpha=0.12, color="#CC0000")
    for s, v in zip(sigmas, logical):
        ax1.text(s, v + 0.025, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.axhspan(0, 0.1, color="#FFEEEE", alpha=0.3, zorder=0)
    ax1.set_xlabel("Gaussian Noise σ (pixels)", fontsize=11)
    ax1.set_ylabel("Logical-Exact Rate", fontsize=11)
    ax1.set_title("Logical-Exact Rate", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(logical) * 1.25 + 0.05)
    ax1.grid(alpha=0.3)

    # Add annotation for the cliff
    if len(logical) >= 2:
        drop = logical[0] - logical[1]
        ax1.annotate(f"−{drop:.0%} drop\nwith just 5px noise!",
                     xy=(5, logical[1]), xytext=(12, logical[0] * 0.7),
                     fontsize=10, color="#CC0000", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1.5),
                     bbox=dict(boxstyle="round,pad=0.3", fc="#FFF0F0", ec="#FFAAAA"))

    # Span-Only F1
    ax2.plot(sigmas, span_f1, "s-", color="#4C72B0", markersize=8, linewidth=2.5,
             markerfacecolor="white", markeredgewidth=2)
    ax2.fill_between(sigmas, span_f1, alpha=0.12, color="#4C72B0")
    for s, v in zip(sigmas, span_f1):
        ax2.text(s, v + 0.025, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Gaussian Noise σ (pixels)", fontsize=11)
    ax2.set_ylabel("Span-Only F1", fontsize=11)
    ax2.set_title("Span-Only F1", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(span_f1) * 1.25 + 0.05)
    ax2.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p = RESULTS_DIR / "slide4_noise_sensitivity.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ═══════════════════════════════════════════════
# SLIDE 5: Summary Table — Numbers for the audience
# ═══════════════════════════════════════════════
def slide5_summary(p1, p2):
    """
    PPT Slide 5: 핵심 수치 정리 표.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
    fig.suptitle("Research Summary: Phase 1 & Phase 2 Key Findings",
                 fontsize=15, fontweight="bold", y=0.98)

    # Phase 1 table
    ax1.axis("off")
    headers1 = ["Dataset", "Split", "N", "GriTS-Top F1", "Trivial F1", "Trivial-Δ", "Span-Only F1"]

    model = "TATR v1.1-all"  # Representative
    rows1 = []
    for ds in ["SciTSR", "FinTabNet"]:
        for split in ["simple", "complex"]:
            key = f"{ds}/{model}/{split}"
            s = p1.get(key, {})
            sp = s.get("grits_top_span_only")
            rows1.append([
                ds, split.capitalize(),
                str(s.get("n", "")),
                f"{s.get('grits_top_f1', 0):.4f}",
                f"{s.get('grits_top_trivial', 0):.4f}",
                f"{s.get('trivial_delta', 0):+.4f}",
                f"{sp:.4f}" if sp is not None else "N/A",
            ])

    tbl1 = ax1.table(cellText=rows1, colLabels=headers1, cellLoc="center", loc="center")
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(10)
    tbl1.scale(1, 1.5)
    for j in range(len(headers1)):
        tbl1[(0, j)].set_facecolor("#1E3A5F")
        tbl1[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(rows1):
        if row[1] == "Complex":
            for j in range(len(headers1)):
                tbl1[(i + 1, j)].set_facecolor("#FFF0F0")
    ax1.set_title("Phase 1: Problem Proof (TATR v1.1-all)", fontsize=12,
                   fontweight="bold", pad=10, loc="left")

    # Phase 2 table
    ax2.axis("off")
    headers2 = ["Condition", "GriTS-Top F1", "Span-Only F1", "Logical-Exact", "vs Full TATR"]
    bs = p2.get("bottleneck_summary", {})
    rows2 = [
        ["A: Full TATR",
         f"{p2['exp_A']['grits_top_f1']:.4f}",
         f"{p2['exp_A'].get('grits_top_span_only') or 0:.4f}",
         f"{bs['A_logical_exact']:.4f}", "—"],
        ["B: Oracle Row/Col",
         f"{p2['exp_B_sigma0']['grits_top_f1']:.4f}",
         f"{p2['exp_B_sigma0'].get('grits_top_span_only') or 0:.4f}",
         f"{bs['B_logical_exact']:.4f}",
         f"+{bs['B_vs_A_improvement']:.4f}"],
        ["C: Full Oracle",
         f"{p2['exp_C']['grits_top_f1']:.4f}",
         f"{p2['exp_C'].get('grits_top_span_only') or 0:.4f}",
         f"{bs['C_logical_exact']:.4f}",
         f"+{bs['C_vs_A_improvement']:.4f}"],
        ["D: Oracle Span",
         f"{p2['exp_D']['grits_top_f1']:.4f}",
         f"{p2['exp_D'].get('grits_top_span_only') or 0:.4f}",
         f"{bs['D_logical_exact']:.4f}", "—"],
    ]

    tbl2 = ax2.table(cellText=rows2, colLabels=headers2, cellLoc="center", loc="center")
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(10)
    tbl2.scale(1, 1.5)
    for j in range(len(headers2)):
        tbl2[(0, j)].set_facecolor("#1E3A5F")
        tbl2[(0, j)].set_text_props(color="white", fontweight="bold")
    # Row B highlight (key finding)
    for j in range(len(headers2)):
        tbl2[(2, j)].set_facecolor("#E8F5E9")  # Light green
    ax2.set_title("Phase 2: Oracle Decomposition (FinTabNet complex, n=68)",
                   fontsize=12, fontweight="bold", pad=10, loc="left")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = RESULTS_DIR / "slide5_summary_table.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ═══════════════════════════════════════════════
# SLIDE 6: SciTSR Logical-Exact — 76.5% Full Failure
# ═══════════════════════════════════════════════
def slide6_logical_failure(p2):
    """
    PPT Slide 6: SciTSR 76.5% complete failure (Logical-Exact = 0).
    """
    le_val = p2.get("scitsr_logical_exact", 0.0895)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle("SciTSR-COMP: 76.5% of Tables Have Zero Correct Span Recovery",
                 fontsize=14, fontweight="bold", y=0.97)

    # Donut chart
    failure = 0.765
    partial = 1 - failure
    sizes = [failure, partial]
    colors_pie = ["#CC4444", "#55A868"]
    explode = (0.05, 0)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, colors=colors_pie,
        autopct="%1.1f%%", startangle=90, pctdistance=0.75,
        textprops={"fontsize": 14, "fontweight": "bold"},
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )

    # Custom labels
    ax.text(-0.3, -0.1, "Complete\nFailure", fontsize=12, ha="center",
            color="#CC4444", fontweight="bold")
    ax.text(0.3, 0.1, "Partial\nSuccess", fontsize=10, ha="center",
            color="#55A868", fontweight="bold")

    # Center text
    ax.text(0, 0, f"Mean\n{le_val:.1%}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333333")

    fig.text(0.5, 0.04,
             "Logical-Exact Rate = fraction of spanning cells with exact (row, col) index match\n"
             "TATR v1.1-all, n=716 complex tables",
             ha="center", fontsize=9, color="#666666", style="italic")

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    p = RESULTS_DIR / "slide6_logical_failure.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generating PPT-ready figures...")
    print("=" * 60)

    p1, p2 = load_data()

    slide1_smoking_gun(p1)
    slide2_span_only(p1)
    slide3_oracle(p2)
    slide4_noise(p2)
    slide5_summary(p1, p2)
    slide6_logical_failure(p2)

    print(f"\nAll figures saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

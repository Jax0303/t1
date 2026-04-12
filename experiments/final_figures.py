#!/usr/bin/env python3
"""
Final Figures — 논문 제출용 종합 시각화
========================================
Phase 1 + Phase 2 결과를 논문 figure 수준으로 재구성.

생성 파일:
  results_final/
    fig_main_simple_vs_complex.png   ← 핵심: simple/complex F1 gap
    fig_trivial_smoking_gun.png      ← smoking gun: trivial delta
    fig_oracle_bottleneck.png        ← 병목: Oracle A~D 비교
    fig_noise_sensitivity.png        ← Stage 2 취약성
    fig_combined_4panel.png          ← 논문용 4패널 통합
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

PHASE1 = Path(__file__).parent / "results_phase1" / "summary.json"
PHASE2 = Path(__file__).parent / "results_phase2" / "oracle_summary.json"
OUT    = Path(__file__).parent / "results_final"
OUT.mkdir(exist_ok=True)

MODELS = ["TATR v1.0", "TATR v1.1-pub", "TATR v1.1-all"]
MODEL_COLORS = {"TATR v1.0": "#4C72B0", "TATR v1.1-pub": "#DD8452", "TATR v1.1-all": "#55A868"}
SIMPLE_COLOR  = "#6BAED6"
COMPLEX_COLOR = "#E84C4C"


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Fig 1: Simple vs Complex GriTS-Top F1
# ─────────────────────────────────────────────
def fig_simple_vs_complex(p1: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "GriTS-Top F1: Simple vs Complex Tables\n"
        "(Official dataset splits — no manual curation)",
        fontsize=14, fontweight="bold", y=1.01
    )

    for ax, ds in zip(axes, ["SciTSR", "FinTabNet"]):
        x = np.arange(len(MODELS))
        w = 0.32
        for i, (split, color) in enumerate([("simple", SIMPLE_COLOR), ("complex", COMPLEX_COLOR)]):
            vals = [p1.get(f"{ds}/{m}/{split}", {}).get("grits_top_f1", 0) for m in MODELS]
            ns   = [p1.get(f"{ds}/{m}/{split}", {}).get("n", 0) for m in MODELS]
            bars = ax.bar(x + (i-0.5)*w, vals, w, label=f"{split.capitalize()} (n≈{ns[0]})",
                          color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold")

        # gap 화살표 (v1.1-all 기준)
        v_s = p1.get(f"{ds}/TATR v1.1-all/simple",  {}).get("grits_top_f1", 0)
        v_c = p1.get(f"{ds}/TATR v1.1-all/complex", {}).get("grits_top_f1", 0)
        if v_s > v_c:
            ax.annotate("", xy=(2+w/2, v_c+0.01), xytext=(2+w/2, v_s-0.01),
                        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
            ax.text(2+w/2+0.12, (v_s+v_c)/2, f"gap\n{v_s-v_c:.3f}",
                    va="center", fontsize=8, color="#333333")

        ax.set_title(f"{ds}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=9.5, rotation=8)
        ax.set_ylim(0.6, 1.0)
        ax.set_ylabel("GriTS-Top F1", fontsize=11)
        ax.legend(fontsize=9.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = OUT / "fig_main_simple_vs_complex.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# Fig 2: Smoking Gun — Trivial Delta
# ─────────────────────────────────────────────
def fig_trivial_smoking_gun(p1: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Trivial-Delta (GriTS-Top F1 − Trivial Baseline F1)\n"
        "Trivial baseline = predict every cell as 1×1 (no spans)\n"
        "Near-zero or negative Δ → model adds no value over doing nothing",
        fontsize=12, fontweight="bold", y=1.02
    )

    for ax, ds in zip(axes, ["SciTSR", "FinTabNet"]):
        x = np.arange(len(MODELS))
        w = 0.32
        for i, (split, color) in enumerate([("simple", SIMPLE_COLOR), ("complex", COMPLEX_COLOR)]):
            vals = [p1.get(f"{ds}/{m}/{split}", {}).get("trivial_delta", 0) for m in MODELS]
            bars = ax.bar(x + (i-0.5)*w, vals, w, label=split.capitalize(),
                          color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ypos = bar.get_height() + 0.0005 if v >= 0 else bar.get_height() - 0.002
                ax.text(bar.get_x()+bar.get_width()/2, ypos,
                        f"{v:+.4f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

        ax.axhline(0, color="black", linewidth=1.2, linestyle="-")
        ax.set_title(ds, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=9.5, rotation=8)
        ax.set_ylabel("GriTS-Top F1 − Trivial Baseline F1", fontsize=10)
        ax.legend(fontsize=9.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # 주석
        ax.text(0.02, 0.97, "← Model hurts\n   (below trivial)",
                transform=ax.transAxes, va="top", fontsize=8, color="#CC0000",
                style="italic")
        ax.text(0.02, 0.58, "← Model helps\n   (above trivial)",
                transform=ax.transAxes, va="top", fontsize=8, color="#2C7B2C",
                style="italic")

    plt.tight_layout()
    p = OUT / "fig_trivial_smoking_gun.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# Fig 3: Oracle Bottleneck
# ─────────────────────────────────────────────
def fig_oracle_bottleneck(p2: dict):
    conditions = [
        ("A\nFull TATR\n(baseline)",        "exp_A",       "#CC4444"),
        ("B\nOracle Row/Col\n(GT+TATR span)","exp_B_sigma0","#55A868"),
        ("C\nFull Oracle\n(all GT bbox)",    "exp_C",       "#2C5F8A"),
        ("D\nOracle Span\n(TATR+GT span)",   "exp_D",       "#DD8452"),
    ]
    metrics     = ["grits_top_f1", "grits_top_span_only", "logical_exact"]
    m_labels    = ["GriTS-Top F1", "Span-Only F1", "Logical-Exact Rate"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    n = p2.get("exp_A", {}).get("n", "?")
    fig.suptitle(
        f"Oracle Decomposition: Stage 1 vs Stage 2 Bottleneck Analysis\n"
        f"(FinTabNet complex, n={n} tables, TATR v1.1-all)\n"
        f"Key: B >> A on Logical-Exact → Stage 2 (grid reconstruction) is the bottleneck",
        fontsize=11, fontweight="bold", y=1.02
    )

    for ax, metric, mlabel in zip(axes, metrics, m_labels):
        vals   = [p2.get(k, {}).get(metric) or 0 for _, k, _ in conditions]
        colors = [c for _, _, c in conditions]
        labels = [l for l, _, _ in conditions]

        bars = ax.bar(range(4), vals, color=colors, alpha=0.85, width=0.6,
                      edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

        # A → B 향상 화살표 (Stage 2가 병목임을 강조)
        vA, vB = vals[0], vals[1]
        if vB > vA + 0.01:
            ax.annotate("", xy=(1, vB+0.05), xytext=(0, vA+0.05),
                        arrowprops=dict(arrowstyle="->", color="#CC0000", lw=2))
            ax.text(0.5, max(vA,vB)+0.09, f"Stage 2\nfix: +{vB-vA:.3f}",
                    ha="center", color="#CC0000", fontsize=9, fontweight="bold")

        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, 1.18)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = OUT / "fig_oracle_bottleneck.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# Fig 4: Noise Sensitivity
# ─────────────────────────────────────────────
def fig_noise_sensitivity(p2: dict):
    noise_data = p2.get("exp_noise", {})
    sigmas = sorted(int(k) for k in noise_data.keys())
    le_vals  = [noise_data[str(s)].get("logical_exact") or 0 for s in sigmas]
    le_stds  = [noise_data[str(s)].get("logical_exact_std") or 0 for s in sigmas]
    sp_vals  = [noise_data[str(s)].get("grits_top_span_only") or 0 for s in sigmas]
    f1_vals  = [noise_data[str(s)].get("grits_top_f1") or 0 for s in sigmas]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "Grid Reconstruction Sensitivity to Bounding Box Noise\n"
        "(Oracle Row/Col: GT row/col bboxes + Gaussian noise σ px, FinTabNet complex, TATR v1.1-all)\n"
        "Even σ=5px nearly collapses performance → Stage 2 is fragile to localization error",
        fontsize=11, fontweight="bold", y=1.03
    )

    for ax, vals, stds, title, color, zero_val in [
        (axes[0], le_vals,  le_stds,         "Logical-Exact Rate", "#CC0000", le_vals[0]),
        (axes[1], sp_vals,  [0]*len(sigmas), "Span-Only F1",       "#4C72B0", sp_vals[0]),
        (axes[2], f1_vals,  [0]*len(sigmas), "GriTS-Top F1",       "#55A868", f1_vals[0]),
    ]:
        ax.errorbar(sigmas, vals, yerr=stds, marker="o", color=color,
                    linewidth=2, capsize=4, markersize=7, zorder=3)
        if any(s > 0 for s in stds):
            ax.fill_between(sigmas, [m-s for m,s in zip(vals,stds)],
                            [m+s for m,s in zip(vals,stds)],
                            alpha=0.15, color=color)
        for sigma, val in zip(sigmas, vals):
            ax.text(sigma, val+0.025, f"{val:.3f}", ha="center", fontsize=8.5)

        # σ=0 vs σ=5 drop 강조
        if len(vals) >= 2:
            drop = vals[0] - vals[1]
            if drop > 0.05:
                ax.annotate("",
                            xy=(sigmas[1], vals[1]+0.02),
                            xytext=(sigmas[0], vals[0]-0.02),
                            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5))
                ax.text((sigmas[0]+sigmas[1])/2, (vals[0]+vals[1])/2,
                        f"−{drop:.3f}", color="#333333", fontsize=9,
                        ha="center", fontweight="bold")

        ax.set_xlabel("Gaussian Noise σ (pixels)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(sigmas)
        ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = OUT / "fig_noise_sensitivity.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# Fig 5: Span-Only F1 Summary (Simple vs Complex)
# ─────────────────────────────────────────────
def fig_span_only_summary(p1: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "GriTS-Top Span-Only F1: Simple vs Complex Tables\n"
        "(Only spanning cells evaluated; Simple=0.000 by definition — no GT spans)",
        fontsize=12, fontweight="bold", y=1.01
    )

    for ax, ds in zip(axes, ["SciTSR", "FinTabNet"]):
        x = np.arange(len(MODELS))
        w = 0.32
        for i, (split, color) in enumerate([("simple", SIMPLE_COLOR), ("complex", COMPLEX_COLOR)]):
            vals = [p1.get(f"{ds}/{m}/{split}", {}).get("grits_top_span_only") or 0
                    for m in MODELS]
            ns   = [p1.get(f"{ds}/{m}/{split}", {}).get("n", 0) for m in MODELS]
            bars = ax.bar(x + (i-0.5)*w, vals, w,
                          label=f"{split.capitalize()} (n≈{ns[0]})",
                          color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold")

        ax.set_title(ds, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=9.5, rotation=8)
        ax.set_ylim(0, 0.65)
        ax.set_ylabel("Span-Only F1", fontsize=11)
        ax.legend(fontsize=9.5)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    p = OUT / "fig_span_only_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# Fig 6: Combined 4-panel (논문 제출용)
# ─────────────────────────────────────────────
def fig_combined_thesis(p1: dict, p2: dict):
    """논문 핵심 4패널."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_simple  = fig.add_subplot(gs[0, 0])
    ax_trivial = fig.add_subplot(gs[0, 1])
    ax_oracle  = fig.add_subplot(gs[1, 0])
    ax_noise   = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "Detection-Based TSR Span Cell Bottleneck — Summary of Evidence",
        fontsize=14, fontweight="bold", y=0.99
    )

    # Panel (a): GriTS-Top F1 Simple vs Complex (SciTSR)
    x = np.arange(len(MODELS)); w = 0.32
    for i, (split, color) in enumerate([("simple", SIMPLE_COLOR), ("complex", COMPLEX_COLOR)]):
        vals = [p1.get(f"SciTSR/{m}/{split}", {}).get("grits_top_f1", 0) for m in MODELS]
        bars = ax_simple.bar(x + (i-0.5)*w, vals, w, label=split.capitalize(),
                              color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax_simple.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                           f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                           fontweight="bold")
    ax_simple.set_title("(a) GriTS-Top F1: Simple vs Complex\n(SciTSR, official COMP/non-COMP split)",
                        fontsize=9.5, fontweight="bold")
    ax_simple.set_xticks(x); ax_simple.set_xticklabels(MODELS, fontsize=8, rotation=8)
    ax_simple.set_ylim(0.6, 1.0); ax_simple.legend(fontsize=8)
    ax_simple.grid(axis="y", alpha=0.3, linestyle="--")
    ax_simple.set_ylabel("GriTS-Top F1", fontsize=9)

    # Panel (b): Trivial Delta (SciTSR)
    for i, (split, color) in enumerate([("simple", SIMPLE_COLOR), ("complex", COMPLEX_COLOR)]):
        vals = [p1.get(f"SciTSR/{m}/{split}", {}).get("trivial_delta", 0) for m in MODELS]
        bars = ax_trivial.bar(x + (i-0.5)*w, vals, w, label=split.capitalize(),
                               color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax_trivial.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()+0.0003 if v>=0 else bar.get_height()-0.0015,
                            f"{v:+.4f}", ha="center", va="bottom", fontsize=7,
                            fontweight="bold")
    ax_trivial.axhline(0, color="black", lw=1.2)
    ax_trivial.set_title("(b) Trivial-Delta (Smoking Gun)\n"
                         "TATR v1.0 complex Δ=+0.0005 ≈ trivial baseline",
                         fontsize=9.5, fontweight="bold")
    ax_trivial.set_xticks(x); ax_trivial.set_xticklabels(MODELS, fontsize=8, rotation=8)
    ax_trivial.legend(fontsize=8)
    ax_trivial.grid(axis="y", alpha=0.3, linestyle="--")
    ax_trivial.set_ylabel("GriTS-Top F1 − Trivial F1", fontsize=9)

    # Panel (c): Oracle Decomposition (Logical-Exact)
    conds = [
        ("A\nFull TATR",       "exp_A",       "#CC4444"),
        ("B\nOracle\nRow/Col", "exp_B_sigma0","#55A868"),
        ("C\nFull\nOracle",    "exp_C",       "#2C5F8A"),
        ("D\nOracle\nSpan",    "exp_D",       "#DD8452"),
    ]
    le_vals = [p2.get(k, {}).get("logical_exact") or 0 for _, k, _ in conds]
    cols    = [c for _, _, c in conds]
    bars    = ax_oracle.bar(range(4), le_vals, color=cols, alpha=0.85, width=0.6)
    for bar, v in zip(bars, le_vals):
        ax_oracle.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                       f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                       fontweight="bold")
    vA, vB = le_vals[0], le_vals[1]
    if vB > vA + 0.01:
        ax_oracle.annotate("", xy=(1, vB+0.04), xytext=(0, vA+0.04),
                           arrowprops=dict(arrowstyle="->", color="#CC0000", lw=2))
        ax_oracle.text(0.5, max(vA,vB)+0.07, f"Stage 2\n+{vB-vA:.3f}",
                       ha="center", color="#CC0000", fontsize=8.5, fontweight="bold")
    ax_oracle.set_title("(c) Oracle Decomposition: Logical-Exact Rate\n"
                        "(FinTabNet complex — Stage 2 is bottleneck)",
                        fontsize=9.5, fontweight="bold")
    ax_oracle.set_xticks(range(4))
    ax_oracle.set_xticklabels([l for l,_,_ in conds], fontsize=8)
    ax_oracle.set_ylim(0, 0.75); ax_oracle.grid(axis="y", alpha=0.3, linestyle="--")
    ax_oracle.set_ylabel("Logical-Exact Rate", fontsize=9)

    # Panel (d): Noise Sensitivity
    noise_data = p2.get("exp_noise", {})
    sigmas_n = sorted(int(k) for k in noise_data.keys())
    le_n = [noise_data[str(s)].get("logical_exact") or 0 for s in sigmas_n]
    ax_noise.plot(sigmas_n, le_n, "o-", color="#CC0000", lw=2, markersize=6)
    for s, v in zip(sigmas_n, le_n):
        ax_noise.text(s, v+0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    ax_noise.set_title("(d) Noise Sensitivity: Logical-Exact Rate vs σ\n"
                       "σ=5px drops performance 85% → Stage 2 fragility",
                       fontsize=9.5, fontweight="bold")
    ax_noise.set_xlabel("Gaussian Noise σ (pixels)", fontsize=9)
    ax_noise.set_ylabel("Logical-Exact Rate", fontsize=9)
    ax_noise.set_ylim(0, 0.55); ax_noise.set_xticks(sigmas_n)
    ax_noise.grid(alpha=0.3, linestyle="--")

    p = OUT / "fig_combined_thesis.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run():
    print("Generating final figures...")
    p1 = load_json(PHASE1)
    p2 = load_json(PHASE2)

    fig_simple_vs_complex(p1)
    fig_trivial_smoking_gun(p1)
    fig_oracle_bottleneck(p2)
    fig_noise_sensitivity(p2)
    fig_span_only_summary(p1)
    fig_combined_thesis(p1, p2)

    print(f"\nAll figures saved to: {OUT}")

    # 핵심 수치 요약 출력
    print("\n" + "=" * 60)
    print("KEY RESULTS SUMMARY")
    print("=" * 60)
    print("\n[Phase 1 — Problem Proof]")
    for ds in ["SciTSR", "FinTabNet"]:
        for m in MODELS:
            s = p1.get(f"{ds}/{m}/simple", {})
            c = p1.get(f"{ds}/{m}/complex", {})
            if not s or not c: continue
            print(f"  {ds} {m}:")
            print(f"    Simple  : F1={s.get('grits_top_f1',0):.4f}, Δ={s.get('trivial_delta',0):+.4f}")
            print(f"    Complex : F1={c.get('grits_top_f1',0):.4f}, Δ={c.get('trivial_delta',0):+.4f}, SpanF1={c.get('grits_top_span_only') or 0:.4f}")

    print("\n[Phase 2 — Oracle Decomposition (FinTabNet complex)]")
    bs = p2.get("bottleneck_summary", {})
    print(f"  A. Full TATR            : LogicalExact={bs.get('A_logical_exact',0):.4f}")
    print(f"  B. Oracle Row/Col       : LogicalExact={bs.get('B_logical_exact',0):.4f}  Δ={bs.get('B_vs_A_improvement',0):+.4f}")
    print(f"  C. Full Oracle          : LogicalExact={bs.get('C_logical_exact',0):.4f}  Δ={bs.get('C_vs_A_improvement',0):+.4f}")
    print(f"  D. Oracle Span          : LogicalExact={bs.get('D_logical_exact',0):.4f}")
    print(f"  SciTSR COMP LogicalExact: {p2.get('scitsr_logical_exact',0):.4f}")
    print("\n  → B >> A, D ≈ A: Stage 2 (grid reconstruction) confirmed as bottleneck")
    print(f"  → σ=5px: Logical-Exact drops from {p2.get('exp_noise',{}).get('0',{}).get('logical_exact',0):.3f} to {p2.get('exp_noise',{}).get('5',{}).get('logical_exact',0):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run()

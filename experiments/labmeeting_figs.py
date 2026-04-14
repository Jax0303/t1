"""
Lab Meeting — Problem Definition Figures
==========================================
학위논문 문제정의용 시각화 (4장)

Fig 1. The Core Gap  — "TATR는 spanning cell을 사실상 처리 못 한다"
Fig 2. Stage 2 Bottleneck — "SAGR만 바꿨는데 Span-F1 +13%"
Fig 3. Threshold Sweep — "최적 thresh=0.3, 민감도 분석"
Fig 4. Problem Taxonomy — "3가지 실패 유형 개념도"

출력: experiments/results_labmeeting/fig{1-4}_*.png
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
import numpy as np

matplotlib.rcParams["font.family"]     = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent / "results_labmeeting"
OUT.mkdir(exist_ok=True)

# ── 데이터 ────────────────────────────────────────────────────
PHASE3 = {
    "Baseline":  {"grits_top_f1": 0.7737, "span_f1": 0.2977, "logical_exact": 0.0727},
    "SAGR-Con":  {"grits_top_f1": 0.7605, "span_f1": 0.2972, "logical_exact": 0.0722},
    "SAGR-Cov":  {"grits_top_f1": 0.7760, "span_f1": 0.3369, "logical_exact": 0.0809},
    "SAGR-Full": {"grits_top_f1": 0.7734, "span_f1": 0.3367, "logical_exact": 0.0805},
}

THRESH = {
    0.1: {"grits_top_f1": 0.6120, "span_f1": 0.2127, "logical_exact": 0.0075},
    0.2: {"grits_top_f1": 0.7424, "span_f1": 0.3351, "logical_exact": 0.0423},
    0.3: {"grits_top_f1": 0.7734, "span_f1": 0.3367, "logical_exact": 0.0805},
    0.4: {"grits_top_f1": 0.7741, "span_f1": 0.2691, "logical_exact": 0.0769},
    0.5: {"grits_top_f1": 0.7647, "span_f1": 0.0567, "logical_exact": 0.0196},
}

COLORS = {
    "Baseline":  "#6C757D",
    "SAGR-Con":  "#FFA726",
    "SAGR-Cov":  "#1565C0",
    "SAGR-Full": "#2E7D32",
}
ACCENT   = "#C62828"
BLUE     = "#1565C0"
GRAY     = "#6C757D"
GREEN    = "#2E7D32"


# ════════════════════════════════════════════════════════════════
# Fig 1. The Core Gap
#   Left:  TATR GriTS-Full vs Span-Only — 두 수치 대비
#   Right: 716개 테이블의 Span-F1 분포 (히스토그램 추정)
# ════════════════════════════════════════════════════════════════
def fig1_core_gap():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor("#FAFAFA")

    # ── 왼쪽: 막대 비교 ──
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")

    metrics = ["GriTS-Top F1\n(전체)", "Span-Only F1\n(spanning cell)"]
    values  = [0.7737, 0.2977]
    colors  = [BLUE, ACCENT]
    bars = ax.bar(metrics, values, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.5, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.012, f"{val:.3f}",
                ha="center", va="bottom", fontsize=15, fontweight="bold",
                color=bar.get_facecolor())

    # Gap annotation
    ax.annotate("", xy=(1, 0.2977), xytext=(1, 0.7737),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=2.2))
    ax.text(1.26, 0.535, f"Gap\n{0.7737-0.2977:.3f}",
            ha="center", va="center", fontsize=13, color=ACCENT, fontweight="bold")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("TATR 성능: 전체 vs Spanning Cell",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axhline(0.7737, color=BLUE, linestyle="--", alpha=0.3, linewidth=1.2)
    ax.axhline(0.2977, color=ACCENT, linestyle="--", alpha=0.3, linewidth=1.2)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(labelsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ── 오른쪽: 시사점 텍스트 박스 ──
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    ax2.axis("off")

    box_props = dict(boxstyle="round,pad=0.7", facecolor="#EEF2FF",
                     edgecolor="#1565C0", linewidth=2)
    findings = (
        "핵심 발견 (SciTSR-COMP, 716 tables)\n\n"
        "● GriTS-Top F1 = 0.774  →  \"모델이 잘 작동하는 것처럼 보임\"\n\n"
        "● Span-Only F1 = 0.298  →  \"spanning cell은 거의 처리 안 됨\"\n\n"
        "● Gap = 0.476  →  패러다임 수준의 구조적 한계\n\n"
        "※ GriTS 전체 지표는 spanning cell 실패를\n"
        "   행·열 예측 성공으로 희석시킨다."
    )
    ax2.text(0.05, 0.88, findings,
             transform=ax2.transAxes,
             fontsize=12.5, va="top", ha="left",
             linespacing=1.6,
             bbox=box_props)

    arrow_box = dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE",
                     edgecolor=ACCENT, linewidth=1.8)
    ax2.text(0.05, 0.13,
             "→  \"단순히 데이터 부족이 아닌, inductive bias 문제\"",
             transform=ax2.transAxes,
             fontsize=12, va="center", color=ACCENT, fontweight="bold",
             bbox=arrow_box)

    fig.suptitle("Fig 1.  Detection Paradigm의 Spanning Cell 처리 실패",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUT / "fig1_core_gap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")


# ════════════════════════════════════════════════════════════════
# Fig 2. SAGR Ablation — 3 metrics × 4 methods
#   "Stage 2만 바꿨는데 Span-F1 +13%"
# ════════════════════════════════════════════════════════════════
def fig2_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor("#FAFAFA")

    methods = list(PHASE3.keys())
    metric_keys  = ["grits_top_f1", "span_f1", "logical_exact"]
    metric_labels = ["GriTS-Top F1", "Span-Only F1", "Logical-Exact"]
    ylims = [(0.73, 0.80), (0.24, 0.38), (0.055, 0.095)]
    ytick_steps = [0.01, 0.02, 0.005]

    for ax, mk, ml, (ylo, yhi), ystep in zip(
            axes, metric_keys, metric_labels, ylims, ytick_steps):
        ax.set_facecolor("#FAFAFA")
        vals   = [PHASE3[m][mk] for m in methods]
        colors = [COLORS[m] for m in methods]
        bars   = ax.bar(methods, vals, color=colors, width=0.55,
                        edgecolor="white", linewidth=1.5, zorder=3)

        # Baseline 기준선
        baseline = PHASE3["Baseline"][mk]
        ax.axhline(baseline, color=GRAY, linestyle="--", linewidth=1.5,
                   alpha=0.7, zorder=2)

        # 수치 레이블
        for bar, val in zip(bars, vals):
            delta = val - baseline
            sign  = "+" if delta > 0 else ""
            c = GREEN if delta > 0 else ACCENT if delta < -0.001 else GRAY
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (yhi - ylo) * 0.015,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
            if abs(delta) > 0.0001:
                ax.text(bar.get_x() + bar.get_width()/2,
                        ylo + (yhi - ylo) * 0.04,
                        f"{sign}{delta:.4f}",
                        ha="center", va="bottom", fontsize=8.5, color=c)

        ax.set_ylim(ylo, yhi)
        ax.set_yticks(np.arange(
            round(ylo, 3), round(yhi + ystep, 3), ystep))
        ax.set_title(ml, fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(axis="x", labelsize=10.5, rotation=10)
        ax.tick_params(axis="y", labelsize=10)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # 범례
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in methods]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "Fig 2.  SAGR Ablation Study (SciTSR-COMP, n=716)\n"
        "Stage 2 알고리즘 교체만으로 Span-F1 +13%, Logical-Exact +11%",
        fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = OUT / "fig2_ablation.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")


# ════════════════════════════════════════════════════════════════
# Fig 3. Threshold Sweep — SAGR-Full coverage_thresh 민감도
# ════════════════════════════════════════════════════════════════
def fig3_threshold():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    thresholds = sorted(THRESH.keys())
    gf  = [THRESH[t]["grits_top_f1"] for t in thresholds]
    sf  = [THRESH[t]["span_f1"]       for t in thresholds]
    le  = [THRESH[t]["logical_exact"] for t in thresholds]

    lw = 2.2
    ms = 8
    ax.plot(thresholds, gf, "o-", color=BLUE,   lw=lw, ms=ms, label="GriTS-Top F1",    zorder=4)
    ax.plot(thresholds, sf, "s-", color=ACCENT,  lw=lw, ms=ms, label="Span-Only F1",   zorder=4)
    ax.plot(thresholds, le, "^-", color=GREEN,   lw=lw, ms=ms, label="Logical-Exact",  zorder=4)

    # 최적점 강조 (thresh=0.3)
    for metric, vals in [("GriTS", gf), ("Span", sf), ("LogEx", le)]:
        best_idx = thresholds.index(0.3)
        ax.scatter([0.3], [vals[best_idx]], s=180, zorder=5,
                   edgecolors="black", linewidths=1.5,
                   color=(BLUE if metric=="GriTS" else ACCENT if metric=="Span" else GREEN))

    ax.axvline(0.3, color="black", linestyle=":", linewidth=1.8, alpha=0.6)
    ax.text(0.305, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.05,
            "최적\nthresh=0.3", fontsize=11, color="black", va="bottom")

    # Baseline 기준선
    ax.axhline(PHASE3["Baseline"]["grits_top_f1"], color=BLUE,  linestyle="--",
               alpha=0.35, linewidth=1.5)
    ax.axhline(PHASE3["Baseline"]["span_f1"],      color=ACCENT, linestyle="--",
               alpha=0.35, linewidth=1.5)
    ax.axhline(PHASE3["Baseline"]["logical_exact"], color=GREEN,  linestyle="--",
               alpha=0.35, linewidth=1.5)
    ax.text(0.51, PHASE3["Baseline"]["grits_top_f1"]+0.003, "Baseline",
            fontsize=9, color=GRAY, alpha=0.7)

    ax.set_xlabel("Coverage Threshold (SAGR-Full)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(thresholds)
    ax.set_ylim(0.0, 0.85)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_title("Fig 3.  SAGR Coverage Threshold Sweep (SciTSR-COMP)\n"
                 "thresh=0.3에서 세 지표 모두 균형 최적점",
                 fontsize=13, fontweight="bold", pad=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    plt.tight_layout()
    path = OUT / "fig3_threshold_sweep.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")


# ════════════════════════════════════════════════════════════════
# Fig 4. Problem Taxonomy — 개념도
#   Detection paradigm의 3가지 실패 유형
# ════════════════════════════════════════════════════════════════
def fig4_taxonomy():
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def rounded_box(ax, x, y, w, h, text, color, fontsize=11, alpha=0.18,
                    textcolor="black", bold=False):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=color,
            alpha=alpha, zorder=2)
        ax.add_patch(rect)
        rect2 = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor="none", edgecolor=color,
            linewidth=2, alpha=0.9, zorder=3)
        ax.add_patch(rect2)
        fw = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=fw,
                zorder=4, wrap=True,
                multialignment="center")

    def arrow(ax, x1, y1, x2, y2, color="#555555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.8, mutation_scale=18),
                    zorder=5)

    # ── 타이틀 ──
    ax.text(7, 7.1, "Detection Paradigm의 Spanning Cell 처리 실패: 3가지 유형",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#1A1A2E")

    # ── 입력: TATR 탐지 결과 ──
    rounded_box(ax, 5.5, 5.7, 3.0, 0.95,
                "TATR 탐지 결과\n(row / col / spanning cell bbox)",
                "#1565C0", fontsize=11, alpha=0.12, textcolor="#1565C0", bold=True)

    # ── Stage 2 Grid Reconstruction ──
    rounded_box(ax, 5.5, 4.45, 3.0, 0.85,
                "Stage 2: Grid Reconstruction\n(IoU 기반 셀 할당)",
                "#555555", fontsize=10.5, alpha=0.1, textcolor="#333333")
    arrow(ax, 7.0, 5.7, 7.0, 5.3)

    arrow(ax, 5.5, 4.45+0.425, 3.8, 3.5+0.5)   # → 실패1
    arrow(ax, 7.0, 4.45,       7.0, 3.5+0.5)    # ↓ 실패2
    arrow(ax, 8.5, 4.45+0.425, 10.2, 3.5+0.5)   # → 실패3

    # ── 실패 유형 3가지 ──
    # 실패 1: Off-by-1
    rounded_box(ax, 0.4, 2.0, 4.0, 1.8,
                "실패 유형 ①\nOff-by-1 경계 오류\n\n"
                "span bbox가 1~2px 벗어나면\n인접 행/열에 잘못 할당\n"
                "→ 파편화(fragment)",
                ACCENT, fontsize=10.5, alpha=0.12, textcolor="#7B1010")

    # 실패 2: 정합성 결여
    rounded_box(ax, 5.0, 2.0, 4.0, 1.8,
                "실패 유형 ②\n구조적 정합성 결여\n\n"
                "독립적 IoU 할당 →\n논리적 연속성 미보장\n"
                "→ 갭(gap) / 중복",
                "#E65100", fontsize=10.5, alpha=0.12, textcolor="#7B3000")

    # 실패 3: 다중 레이블
    rounded_box(ax, 9.6, 2.0, 4.0, 1.8,
                "실패 유형 ③\nMulti-label 충돌\n\n"
                "동일 영역에 row + header\n동시 예측 (class 중복)\n"
                "→ 논리 인덱스 불가",
                "#4A148C", fontsize=10.5, alpha=0.12, textcolor="#4A148C")

    # ── 해결책 ──
    arrow(ax, 2.4, 2.0, 2.4, 1.15)
    arrow(ax, 7.0, 2.0, 7.0, 1.15)
    arrow(ax, 11.6, 2.0, 11.6, 1.15)

    rounded_box(ax, 0.4, 0.2, 4.0, 0.88,
                "SAGR: Coverage 기반 할당\n(1D 겹침 비율, max 분모)",
                BLUE, fontsize=10, alpha=0.15, textcolor=BLUE)
    rounded_box(ax, 5.0, 0.2, 4.0, 0.88,
                "SAGR: Contiguity Constraint\n(인접 인덱스 자동 연결)",
                BLUE, fontsize=10, alpha=0.15, textcolor=BLUE)
    rounded_box(ax, 9.6, 0.2, 4.0, 0.88,
                "DETRSpan: Single-Label\nOrdinal Regression Head",
                GREEN, fontsize=10, alpha=0.15, textcolor=GREEN)

    ax.text(7, -0.05,
            "Span-Only F1: 0.298 → 0.337 (+13%)   |   Logical-Exact: 0.073 → 0.081 (+11%)",
            ha="center", va="bottom", fontsize=11, color=GREEN, fontweight="bold")

    plt.tight_layout(pad=0.5)
    path = OUT / "fig4_taxonomy.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")


# ════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[labmeeting figs] generating...")
    fig1_core_gap()
    fig2_ablation()
    fig3_threshold()
    fig4_taxonomy()
    print(f"[labmeeting figs] done → {OUT}/")

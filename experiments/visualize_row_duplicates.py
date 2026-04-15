#!/usr/bin/env python3
"""
Row Duplicate Visualization
=============================
TATR이 row를 중복 검출하는 실제 사례를 시각화.
tatr_inference_cache.pkl에서 dr=+1인 worst-case를 뽑아
GT row 개수 vs TATR predicted row bbox를 비교 표시.

출력:
  experiments/results_rownms/fig_qualitative_rows.png
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent))

CACHE_PATH  = Path(__file__).parent / "results_phase3" / "tatr_inference_cache.pkl"
RESULTS_DIR = Path(__file__).parent / "results_rownms"
RESULTS_DIR.mkdir(exist_ok=True)


def _1d_overlap_over_min(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    min_len = min(a1 - a0, b1 - b0)
    return inter / min_len if min_len > 0 else 0.0


def find_overlapping_pairs(rows_bbox, thresh=0.5):
    """y-overlap > thresh인 row 쌍을 찾는다."""
    pairs = []
    sorted_rows = sorted(range(len(rows_bbox)), key=lambda i: (rows_bbox[i][1] + rows_bbox[i][3]) / 2)
    for idx in range(len(sorted_rows) - 1):
        i, j = sorted_rows[idx], sorted_rows[idx + 1]
        ov = _1d_overlap_over_min(rows_bbox[i][1], rows_bbox[i][3],
                                   rows_bbox[j][1], rows_bbox[j][3])
        if ov > thresh:
            pairs.append((i, j, ov))
    return pairs


def visualize_cases(cache, n_show=6):
    """dr=+1인 테이블 중 대표 사례를 시각화."""

    # dr=+1이면서 이미지가 있는 케이스 수집
    candidates = []
    for raw, item in cache:
        gt = item["gt"]
        n_gt_rows = gt["n_rows"]
        n_pred_rows = len(raw["rows_bbox"])
        dr = n_pred_rows - n_gt_rows

        if dr == 1:
            pairs = find_overlapping_pairs(raw["rows_bbox"])
            if pairs:
                f1_approx = 1.0 / (1.0 + abs(dr))  # rough proxy
                candidates.append({
                    "raw": raw,
                    "item": item,
                    "dr": dr,
                    "pairs": pairs,
                    "n_gt_rows": n_gt_rows,
                    "n_pred_rows": n_pred_rows,
                })

    if not candidates:
        print("[warn] No dr=+1 candidates found")
        return

    # 다양한 테이블 크기를 보여주기 위해 GT row 수 기준으로 분산 샘플링
    candidates.sort(key=lambda c: c["n_gt_rows"])
    step = max(1, len(candidates) // n_show)
    selected = candidates[::step][:n_show]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Row Duplicate Detection — TATR predicts exactly +1 extra row\n"
        "Blue = predicted row bbox | Red dashed = overlapping pair (y-overlap > 0.5)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, case in zip(axes.flat, selected):
        raw = case["raw"]
        rows = raw["rows_bbox"]

        # 테이블 영역 계산
        all_boxes = rows + raw["cols_bbox"]
        if not all_boxes:
            ax.axis("off")
            continue

        x_min = min(b[0] for b in all_boxes)
        y_min = min(b[1] for b in all_boxes)
        x_max = max(b[2] for b in all_boxes)
        y_max = max(b[3] for b in all_boxes)

        # 빈 캔버스에 row bbox들을 그림
        ax.set_xlim(x_min - 10, x_max + 10)
        ax.set_ylim(y_max + 10, y_min - 10)  # 이미지 좌표계 (y 반전)
        ax.set_aspect("equal")

        # 모든 predicted row 그리기 (파란색)
        overlap_indices = set()
        for i, j, ov in case["pairs"]:
            overlap_indices.add(i)
            overlap_indices.add(j)

        for ri, rb in enumerate(rows):
            if ri in overlap_indices:
                color = "#D62728"
                lw = 2.5
                ls = "--"
                alpha = 0.35
            else:
                color = "#4C72B0"
                lw = 1.5
                ls = "-"
                alpha = 0.2

            rect = patches.Rectangle(
                (rb[0], rb[1]), rb[2] - rb[0], rb[3] - rb[1],
                linewidth=lw, edgecolor=color, facecolor=color,
                alpha=alpha, linestyle=ls
            )
            ax.add_patch(rect)
            # y 중심 표시
            y_center = (rb[1] + rb[3]) / 2
            ax.plot([rb[0], rb[2]], [y_center, y_center],
                    color=color, lw=0.8, alpha=0.6)

        # overlap pair 간 연결선
        for i, j, ov in case["pairs"]:
            yi = (rows[i][1] + rows[i][3]) / 2
            yj = (rows[j][1] + rows[j][3]) / 2
            xr = x_max + 5
            ax.annotate(
                f"ov={ov:.2f}",
                xy=(xr, (yi + yj) / 2),
                fontsize=8, color="#D62728", fontweight="bold",
                ha="left", va="center"
            )

        # Column boxes (가벼운 회색)
        for cb in raw["cols_bbox"]:
            rect = patches.Rectangle(
                (cb[0], cb[1]), cb[2] - cb[0], cb[3] - cb[1],
                linewidth=0.8, edgecolor="#999", facecolor="#999",
                alpha=0.08
            )
            ax.add_patch(rect)

        ax.set_title(
            f"GT rows={case['n_gt_rows']}  |  Pred rows={case['n_pred_rows']}  (Δ=+1)\n"
            f"{len(case['pairs'])} overlapping pair(s)",
            fontsize=10, fontweight="bold"
        )
        ax.grid(alpha=0.15)

    # 빈 subplot 처리
    for ax in axes.flat[len(selected):]:
        ax.axis("off")

    plt.tight_layout()
    p = RESULTS_DIR / "fig_qualitative_rows.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


def main():
    print("=" * 60)
    print("Row Duplicate Visualization")
    print("=" * 60)

    if not CACHE_PATH.exists():
        print(f"[error] Cache not found: {CACHE_PATH}")
        print("  Run row_nms_eval.py (or phase3_sagr_eval.py) first to generate cache.")
        return

    with open(CACHE_PATH, "rb") as f:
        scitsr_cache, fintab_cache = pickle.load(f)
    print(f"[cache] SciTSR={len(scitsr_cache)}  FinTab={len(fintab_cache)}")

    print("\n[SciTSR] Visualizing dr=+1 cases...")
    visualize_cases(scitsr_cache, n_show=6)

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Spanning Cell Boundary Precision Analysis — SciTSR-COMP
========================================================
boundary_precision_analysis.py의 SciTSR-COMP 버전.
원본은 PubTables-1M XML 기반이나 데이터가 없으므로,
동일한 연구 질문을 SciTSR-COMP + chunk 기반 GT bbox로 분석.

연구 질문:
  "TATR은 spanning cell을 탐지하지만, bbox 경계가 row/col separator에
   정밀하게 맞지 않아 grid 복원 단계에서 colspan/rowspan 오류 발생"

측정:
  - Recall@0.5 / @0.75 / @0.9 : 매칭된 span / 전체 GT span
  - Grid index error rate : 매칭된 span 중 span boundary가
    separator를 넘어 colspan/rowspan이 틀린 비율
  - IoU bin vs grid accuracy

출력: experiments/results/boundary_scitsr_summary.json
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from transformers import AutoModelForObjectDetection, AutoImageProcessor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from _eval_utils import (
    load_valid_ids, load_gt, load_image,
    infer_gt_cell_bboxes, bbox_iou,
)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CONF       = 0.5
N_SAMPLES  = 300   # 원본과 동일
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─── 모델 ────────────────────────────────────────────────────────────────────
def load_model():
    name = "microsoft/table-transformer-structure-recognition-v1.1-all"
    proc = AutoImageProcessor.from_pretrained(name)
    if isinstance(proc.size, dict) and list(proc.size.keys()) == ["longest_edge"]:
        v = proc.size["longest_edge"]
        proc.size = {"shortest_edge": v, "longest_edge": v}
    model = AutoModelForObjectDetection.from_pretrained(name).to(DEVICE).eval()
    return proc, model


@torch.no_grad()
def infer(image, proc, model):
    inputs = proc(images=image, return_tensors="pt").to(DEVICE)
    out    = model(**inputs)
    tgt    = torch.tensor([image.size[::-1]])
    res    = proc.post_process_object_detection(out, threshold=CONF,
                                                target_sizes=tgt)[0]
    return {k: v.cpu() for k, v in res.items()}


def get_pred_spans(result, model_obj) -> list:
    """TATR 예측 중 spanning cell bbox만 반환."""
    id2label = model_obj.config.id2label
    spans = []
    for box, label, score in zip(
        result["boxes"], result["labels"], result["scores"]
    ):
        lname = id2label[label.item()].lower()
        if "spanning" in lname or "projected" in lname:
            spans.append(box.numpy().tolist())
    return spans


def get_pred_rows_cols(result, model_obj):
    """TATR 예측 row/col bbox 반환."""
    id2label = model_obj.config.id2label
    rows, cols = [], []
    for box, label in zip(result["boxes"], result["labels"]):
        lname = id2label[label.item()].lower()
        b = box.numpy().tolist()
        if "column" in lname:
            cols.append(b)
        elif "spanning" not in lname and "projected" not in lname:
            rows.append(b)
    rows.sort(key=lambda b: (b[1] + b[3]) / 2)
    cols.sort(key=lambda b: (b[0] + b[2]) / 2)
    return rows, cols


def grid_index_error(pred_span: list, gt_cell: dict,
                     pred_rows: list, pred_cols: list) -> bool:
    """
    매칭된 (pred_span, gt_cell) 쌍에서 grid index 오류 여부.
    pred_span bbox와 pred_rows/cols 기반으로 colspan/rowspan을 추정한 뒤
    GT rowspan/colspan과 비교.
    """
    if not pred_rows or not pred_cols:
        return False

    def _overlap_1d(a1, a2, b1, b2):
        inter = max(0.0, min(a2, b2) - max(a1, b1))
        length = max(a2, b2) - min(a1, b1)
        return inter / length if length > 0 else 0.0

    x0, y0, x1, y1 = pred_span
    row_hits = [i for i, rb in enumerate(pred_rows)
                if _overlap_1d(y0, y1, rb[1], rb[3]) >= 0.5]
    col_hits = [j for j, cb in enumerate(pred_cols)
                if _overlap_1d(x0, x1, cb[0], cb[2]) >= 0.5]

    if not row_hits or not col_hits:
        return False

    pred_rs = max(row_hits) - min(row_hits) + 1
    pred_cs = max(col_hits) - min(col_hits) + 1
    gt_rs   = gt_cell.get("row_span", 1)
    gt_cs   = gt_cell.get("col_span", 1)
    return (pred_rs != gt_rs) or (pred_cs != gt_cs)


# ─── 메인 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Boundary Precision Analysis — SciTSR-COMP")
    print(f"Device: {DEVICE}, n_samples={N_SAMPLES}, conf={CONF}")
    print("=" * 65)

    proc, model_obj = load_model()

    valid_ids = load_valid_ids()
    rng = np.random.default_rng(42)
    ids = valid_ids[:] if len(valid_ids) <= N_SAMPLES else \
          [valid_ids[i] for i in rng.choice(len(valid_ids), N_SAMPLES, replace=False)]
    print(f"  Using {len(ids)} tables from SciTSR-COMP")

    iou_thresholds = [0.5, 0.75, 0.9]
    n_gt_total     = 0
    n_detected     = {t: 0 for t in iou_thresholds}
    grid_errors_detected = 0     # IoU>=0.5 매칭 중 grid 오류
    grid_errors_90       = 0     # IoU>=0.9 매칭 중 grid 오류
    n_matched_50         = 0
    n_matched_90         = 0
    iou_bins = np.zeros(10)      # 0.0–1.0 을 0.1 단위로
    iou_bins_gerr = np.zeros(10)

    skip = 0
    for tid in tqdm(ids, desc="boundary analysis"):
        try:
            gt  = load_gt(tid)
            img = load_image(tid)
            gt_bboxes = infer_gt_cell_bboxes(gt, tid, img_size=img.size)
            gt_spans  = [c for c in gt_bboxes
                         if c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1]
            if not gt_spans:
                continue

            result = infer(img, proc, model_obj)
            pred_spans = get_pred_spans(result, model_obj)
            pred_rows, pred_cols = get_pred_rows_cols(result, model_obj)

            if not pred_spans:
                n_gt_total += len(gt_spans)
                continue

            n_gt_total += len(gt_spans)

            # Hungarian matching (GT spans × pred spans)
            iou_mat = np.zeros((len(gt_spans), len(pred_spans)))
            for i, gs in enumerate(gt_spans):
                gb = gs["bbox"]
                for j, pb in enumerate(pred_spans):
                    iou_mat[i, j] = bbox_iou(gb, pb)

            gi, pi = linear_sum_assignment(1.0 - iou_mat)

            for g_idx, p_idx in zip(gi, pi):
                iou = iou_mat[g_idx, p_idx]
                bin_idx = min(int(iou * 10), 9)
                iou_bins[bin_idx] += 1

                gerr = grid_index_error(
                    pred_spans[p_idx], gt_spans[g_idx], pred_rows, pred_cols)
                if gerr:
                    iou_bins_gerr[bin_idx] += 1

                for thr in iou_thresholds:
                    if iou >= thr:
                        n_detected[thr] += 1

                if iou >= 0.5:
                    n_matched_50 += 1
                    if gerr:
                        grid_errors_detected += 1
                if iou >= 0.9:
                    n_matched_90 += 1
                    if gerr:
                        grid_errors_90 += 1

        except Exception:
            skip += 1
            continue

    print(f"\n  Skipped: {skip}")
    print(f"\n  GT spans (total):      {n_gt_total}")
    for thr in iou_thresholds:
        r = n_detected[thr] / n_gt_total if n_gt_total else 0
        print(f"  Recall @ IoU>={thr:.2f}: {r:.3f}  ({n_detected[thr]}/{n_gt_total})")

    gerr_all = grid_errors_detected / n_matched_50 if n_matched_50 else 0.0
    gerr_90  = grid_errors_90       / n_matched_90 if n_matched_90 else 0.0
    print(f"\n  Grid err (IoU>=0.5):  {gerr_all:.4f}  ({grid_errors_detected}/{n_matched_50})")
    print(f"  Grid err (IoU>=0.9):  {gerr_90:.4f}   ({grid_errors_90}/{n_matched_90})")

    summary = {
        "dataset":                  "SciTSR-COMP",
        "n_tables":                 len(ids),
        "n_gt_spans":               n_gt_total,
        "n_detected_50":            n_detected[0.5],
        "n_detected_75":            n_detected[0.75],
        "n_detected_90":            n_detected[0.9],
        "recall_50":                n_detected[0.5] / n_gt_total if n_gt_total else 0,
        "recall_75":                n_detected[0.75] / n_gt_total if n_gt_total else 0,
        "recall_90":                n_detected[0.9] / n_gt_total if n_gt_total else 0,
        "n_matched_iou50":          n_matched_50,
        "n_matched_iou90":          n_matched_90,
        "grid_err_rate_all_detected": gerr_all,
        "grid_err_rate_iou_90plus":   gerr_90,
    }

    out_path = RESULTS_DIR / "boundary_scitsr_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {out_path}")

    # 그림
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Spanning Cell Boundary Precision — TATR v1.1-all on SciTSR-COMP",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    lbls = ["@IoU≥0.5", "@IoU≥0.75", "@IoU≥0.9"]
    vals = [n_detected[t] / n_gt_total if n_gt_total else 0 for t in iou_thresholds]
    bars = ax.bar(lbls, vals, color=["#5B8DD9", "#F5A623", "#E05C4B"], alpha=0.9)
    for b, v, n in zip(bars, vals, [n_detected[t] for t in iou_thresholds]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                f"{v:.1%}\n({n}/{n_gt_total})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Recall"); ax.set_title("Detection Recall by IoU threshold")

    ax = axes[1]
    bins_center = np.arange(0.05, 1.05, 0.1)
    ax.bar(bins_center, iou_bins, width=0.08, alpha=0.6, label="matched pairs")
    ax.bar(bins_center, iou_bins_gerr, width=0.08, alpha=0.9, color="red", label="grid error")
    ax.set_xlabel("IoU"); ax.set_ylabel("Count")
    ax.set_title("Grid Index Error by IoU bin"); ax.legend()
    ax.text(0.02, 0.92,
            f"Grid err rate (all): {gerr_all:.2%}\nGrid err rate (IoU≥0.9): {gerr_90:.2%}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))

    plt.tight_layout()
    fig_path = RESULTS_DIR / "boundary_scitsr.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fig_path}")

    return summary


if __name__ == "__main__":
    main()

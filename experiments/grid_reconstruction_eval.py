#!/usr/bin/env python3
"""
=======================================================================
TATR Cell Grid Reconstruction Evaluation (v3)
=======================================================================
v2의 설계 오류를 수정한 올바른 실험.

TATR은 cell을 직접 예측하지 않고 row/column/spanning-cell 단위로 예측한다.
따라서 올바른 평가는:
  1. GT/Pred 각각에서 row×col 교차 → span merge → cell grid 구성
  2. GT cell grid vs Pred cell grid를 비교

3-Layer 평가:
  Layer 1: Row/Col 경계 품질 (GT rows/cols vs pred rows/cols)
  Layer 2: Spanning cell 탐지 품질 (GT spans vs pred label=5)
  Layer 3: Cell grid 복원 정확도 (핵심)

가설: "Spanning cell의 bbox 부정확성과 미탐지가 row×col 후처리에서
       셀 분할 오류로 증폭되며, spanning cell이 없는 table 대비
       cell-level 정확도를 유의미하게 낮춘다."
=======================================================================
"""

import os
import json
import argparse
import warnings
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_ind, kruskal

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE = 0.5
NUM_SAMPLES = 500
SEED = 42
MODEL_NAME = "microsoft/table-transformer-structure-recognition-v1.1-all"

RESULTS_DIR = Path(__file__).parent / "results_v3"


# ═════════════════════════════════════════════
# 1. GT Loading
# ═════════════════════════════════════════════
def load_gt_xml(xml_path):
    """PubTables-1M GT XML → category별 bbox dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cats = defaultdict(list)
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        bbox = [float(bb.findtext(d, 0)) for d in ("xmin", "ymin", "xmax", "ymax")]
        cats[name].append(bbox)
    return dict(cats)


# ═════════════════════════════════════════════
# 2. TATR Inference
# ═════════════════════════════════════════════
def load_model():
    print(f"[Model] Loading {MODEL_NAME} → {DEVICE}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    if isinstance(processor.size, dict) and list(processor.size.keys()) == ["longest_edge"]:
        le = processor.size["longest_edge"]
        processor.size = {"shortest_edge": le, "longest_edge": le}
    model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def run_tatr(image, processor, model):
    """TATR inference → rows, cols, spans 분리."""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=CONFIDENCE, target_sizes=target_sizes
    )[0]

    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    id2label = model.config.id2label

    rows, cols, spans = [], [], []
    for box, lid, sc in zip(boxes, labels, scores):
        lname = id2label[lid].lower()
        entry = {"bbox": box.tolist(), "score": float(sc), "label": lname}
        if "column" in lname and "header" not in lname:
            cols.append(entry)
        elif "spanning" in lname:
            spans.append(entry)
        elif "row" in lname:
            rows.append(entry)

    rows.sort(key=lambda r: (r["bbox"][1] + r["bbox"][3]) / 2)
    cols.sort(key=lambda c: (c["bbox"][0] + c["bbox"][2]) / 2)
    return {"rows": rows, "cols": cols, "spans": spans}


# ═════════════════════════════════════════════
# 3. Cell Grid Construction
# ═════════════════════════════════════════════
def bbox_iou(b1, b2):
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def bbox_overlap_ratio(small, big):
    """small bbox가 big bbox에 얼마나 포함되는지 (0~1)."""
    x0 = max(small[0], big[0]); y0 = max(small[1], big[1])
    x1 = min(small[2], big[2]); y1 = min(small[3], big[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_small = (small[2] - small[0]) * (small[3] - small[1])
    return inter / area_small if area_small > 0 else 0.0


def build_cell_grid(rows, cols, spans, overlap_thresh=0.5):
    """
    Row×Col 교차점으로 1×1 cell grid 생성 후, spanning cell로 merge.

    Args:
        rows: list of bbox [x1,y1,x2,y2]
        cols: list of bbox [x1,y1,x2,y2]
        spans: list of bbox [x1,y1,x2,y2]
        overlap_thresh: cell center가 span bbox에 포함되는 기준

    Returns:
        cells: list of dict with 'bbox', 'start_row', 'end_row', 'start_col', 'end_col'
    """
    n_rows, n_cols = len(rows), len(cols)
    if n_rows == 0 or n_cols == 0:
        return []

    # Step 1: row×col 교차로 grid cell 생성
    grid = {}  # (r, c) → bbox
    for ri, rb in enumerate(rows):
        for ci, cb in enumerate(cols):
            x0 = max(rb[0], cb[0]); y0 = max(rb[1], cb[1])
            x1 = min(rb[2], cb[2]); y1 = min(rb[3], cb[3])
            if x1 > x0 and y1 > y0:
                grid[(ri, ci)] = [x0, y0, x1, y1]

    if not grid:
        return []

    # Step 2: spanning cell로 merge
    # 각 grid cell의 중심점이 span bbox 안에 있으면 해당 span에 속함
    cell_to_span = {}  # (r, c) → span_idx
    for si, sp in enumerate(spans):
        for (r, c), bbox in grid.items():
            if (r, c) in cell_to_span:
                continue
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            if sp[0] <= cx <= sp[2] and sp[1] <= cy <= sp[3]:
                cell_to_span[(r, c)] = si

    # Step 3: span 그룹별로 merge, 나머지는 1×1
    span_groups = defaultdict(list)
    for (r, c), si in cell_to_span.items():
        span_groups[si].append((r, c))

    merged_cells = []
    used = set()

    # Spanning cells
    for si, positions in span_groups.items():
        rs = [p[0] for p in positions]
        cs = [p[1] for p in positions]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)

        # Merged bbox = union of all grid cells in this span
        all_bboxes = [grid[p] for p in positions if p in grid]
        if not all_bboxes:
            continue
        merged_bbox = [
            min(b[0] for b in all_bboxes),
            min(b[1] for b in all_bboxes),
            max(b[2] for b in all_bboxes),
            max(b[3] for b in all_bboxes),
        ]
        merged_cells.append({
            "bbox": merged_bbox,
            "start_row": r0, "end_row": r1,
            "start_col": c0, "end_col": c1,
            "is_span": True,
        })
        for p in positions:
            used.add(p)

    # Non-spanning cells
    for (r, c), bbox in grid.items():
        if (r, c) not in used:
            merged_cells.append({
                "bbox": bbox,
                "start_row": r, "end_row": r,
                "start_col": c, "end_col": c,
                "is_span": False,
            })

    return merged_cells


# ═════════════════════════════════════════════
# 4. Cell Grid Matching & Metrics
# ═════════════════════════════════════════════
def match_cells(gt_cells, pred_cells):
    """Hungarian matching of cell bboxes → mean IoU, precision, recall."""
    n_gt = len(gt_cells)
    n_pred = len(pred_cells)
    if n_gt == 0 and n_pred == 0:
        return {"mean_iou": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "over_seg": 1.0, "matched_ious": []}
    if n_gt == 0 or n_pred == 0:
        return {"mean_iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "over_seg": n_pred / max(n_gt, 1), "matched_ious": []}

    gt_boxes = np.array([c["bbox"] for c in gt_cells])
    pred_boxes = np.array([c["bbox"] for c in pred_cells])

    # IoU matrix
    iou_mat = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            iou_mat[i, j] = bbox_iou(gt_boxes[i], pred_boxes[j])

    cost = 1.0 - iou_mat
    gi, pi = linear_sum_assignment(cost)

    matched_ious = []
    matched_at_50 = 0
    for g, p in zip(gi, pi):
        iou = iou_mat[g, p]
        matched_ious.append(iou)
        if iou >= 0.5:
            matched_at_50 += 1

    mean_iou = np.mean(matched_ious) if matched_ious else 0.0
    precision = matched_at_50 / n_pred if n_pred > 0 else 0.0
    recall = matched_at_50 / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "mean_iou": float(mean_iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "over_seg": n_pred / n_gt,
        "matched_ious": [float(x) for x in matched_ious],
    }


def compute_adjacency_f1(cells):
    """Extract horizontal and vertical adjacency pairs from cell list."""
    adj = set()
    for i, c in enumerate(cells):
        for j, d in enumerate(cells):
            if i >= j:
                continue
            # Horizontal adjacency: same row range, col adjacent
            if (c["start_row"] == d["start_row"] and c["end_row"] == d["end_row"]
                    and c["end_col"] + 1 == d["start_col"]):
                adj.add((i, j))
            elif (d["start_row"] == c["start_row"] and d["end_row"] == c["end_row"]
                  and d["end_col"] + 1 == c["start_col"]):
                adj.add((j, i))
            # Vertical adjacency: same col range, row adjacent
            if (c["start_col"] == d["start_col"] and c["end_col"] == d["end_col"]
                    and c["end_row"] + 1 == d["start_row"]):
                adj.add((i, j))
            elif (d["start_col"] == c["start_col"] and d["end_col"] == c["end_col"]
                  and d["end_row"] + 1 == c["start_row"]):
                adj.add((j, i))
    return adj


def adjacency_f1_between(gt_cells, pred_cells):
    """
    Adjacency-F1: Compare adjacency structure.
    Since GT and pred may have different logical coords, we use
    bbox-based adjacency instead.
    """
    def bbox_adj_pairs(cells):
        """Find adjacent cell pairs based on bbox proximity."""
        pairs = set()
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                bi, bj = cells[i]["bbox"], cells[j]["bbox"]
                # Horizontal adjacency: right edge of i ≈ left edge of j (or vice versa)
                h_gap = min(abs(bi[2] - bj[0]), abs(bj[2] - bi[0]))
                v_overlap = min(bi[3], bj[3]) - max(bi[1], bj[1])
                if h_gap < 5 and v_overlap > 0:
                    pairs.add((min(i, j), max(i, j)))
                # Vertical adjacency
                v_gap = min(abs(bi[3] - bj[1]), abs(bj[3] - bi[1]))
                h_overlap = min(bi[2], bj[2]) - max(bi[0], bj[0])
                if v_gap < 5 and h_overlap > 0:
                    pairs.add((min(i, j), max(i, j)))
        return pairs

    gt_adj = bbox_adj_pairs(gt_cells)
    pred_adj = bbox_adj_pairs(pred_cells)
    if not gt_adj and not pred_adj:
        return 1.0
    if not gt_adj or not pred_adj:
        return 0.0

    # Match adj pairs via bbox overlap
    tp = 0
    for ga, gb in gt_adj:
        gb1, gb2 = gt_cells[ga]["bbox"], gt_cells[gb]["bbox"]
        for pa, pb in pred_adj:
            pb1, pb2 = pred_cells[pa]["bbox"], pred_cells[pb]["bbox"]
            if bbox_iou(gb1, pb1) > 0.3 and bbox_iou(gb2, pb2) > 0.3:
                tp += 1
                break
            if bbox_iou(gb1, pb2) > 0.3 and bbox_iou(gb2, pb1) > 0.3:
                tp += 1
                break

    prec = tp / len(pred_adj) if pred_adj else 0
    rec = tp / len(gt_adj) if gt_adj else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return f1


# ═════════════════════════════════════════════
# 5. Layer 1: Row/Col Evaluation
# ═════════════════════════════════════════════
def eval_row_col(gt_rows, gt_cols, pred_rows, pred_cols):
    """Row/Col 경계 품질 평가."""
    def match_1d(gt_list, pred_list):
        n_g, n_p = len(gt_list), len(pred_list)
        if n_g == 0 or n_p == 0:
            return {"mean_iou": 0.0, "count_error": abs(n_g - n_p)}
        iou_mat = np.zeros((n_g, n_p))
        for i in range(n_g):
            for j in range(n_p):
                iou_mat[i, j] = bbox_iou(gt_list[i], pred_list[j])
        gi, pi = linear_sum_assignment(1.0 - iou_mat)
        ious = [iou_mat[g, p] for g, p in zip(gi, pi)]
        return {"mean_iou": float(np.mean(ious)), "count_error": abs(n_g - n_p)}

    return {
        "row": match_1d(gt_rows, pred_rows),
        "col": match_1d(gt_cols, pred_cols),
    }


# ═════════════════════════════════════════════
# 6. Layer 2: Span Detection Evaluation
# ═════════════════════════════════════════════
def eval_span_detection(gt_spans, pred_spans):
    """Spanning cell 탐지 품질."""
    n_gt, n_pred = len(gt_spans), len(pred_spans)
    if n_gt == 0:
        return {"n_gt": 0, "n_pred": n_pred, "recall": None, "ap50": None, "ap75": None, "ap90": None}
    if n_pred == 0:
        return {"n_gt": n_gt, "n_pred": 0, "recall": 0.0, "ap50": 0.0, "ap75": 0.0, "ap90": 0.0}

    iou_mat = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            iou_mat[i, j] = bbox_iou(gt_spans[i], pred_spans[j])

    gi, pi = linear_sum_assignment(1.0 - iou_mat)
    matched_ious = [iou_mat[g, p] for g, p in zip(gi, pi)]

    result = {"n_gt": n_gt, "n_pred": n_pred}
    for tname, thresh in [("ap50", 0.5), ("ap75", 0.75), ("ap90", 0.9)]:
        hits = sum(1 for iou in matched_ious if iou >= thresh)
        result[tname] = hits / n_gt
    result["recall"] = sum(1 for iou in matched_ious if iou >= 0.5) / n_gt
    result["matched_ious"] = matched_ious

    # Classify each GT span: B-1 (accurate), B-2 (imprecise), B-3 (missed)
    classifications = []
    for g, p in zip(gi, pi):
        iou = iou_mat[g, p]
        if iou >= 0.9:
            classifications.append("B-1")
        elif iou >= 0.5:
            classifications.append("B-2")
        else:
            classifications.append("B-3")
    # Unmatched GT spans are B-3
    for i in range(n_gt):
        if i not in gi:
            classifications.append("B-3")
    result["classifications"] = classifications
    return result


# ═════════════════════════════════════════════
# 7. Visualization
# ═════════════════════════════════════════════
def draw_cell_grid(ax, cells, title, color='#2196F3'):
    """Draw cell bboxes on an axis."""
    ax.set_title(title, fontsize=9, fontweight='bold')
    for cell in cells:
        b = cell["bbox"]
        w, h = b[2] - b[0], b[3] - b[1]
        is_sp = cell.get("is_span", False)
        fc = color if is_sp else '#E0E0E0'
        alpha = 0.4 if is_sp else 0.15
        lw = 1.5 if is_sp else 0.5
        rect = plt.Rectangle((b[0], b[1]), w, h, linewidth=lw,
                              edgecolor=color, facecolor=fc, alpha=alpha)
        ax.add_patch(rect)
        if is_sp:
            ax.text(b[0] + w / 2, b[1] + h / 2, "SPAN",
                    ha='center', va='center', fontsize=6, color='white', fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.tick_params(labelsize=5)


def visualize_example(image, gt_cells, pred_cells, stats, save_path):
    """GT grid vs Pred grid 비교 시각화."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=9)
    axes[0].axis('off')

    if gt_cells:
        all_b = [c["bbox"] for c in gt_cells]
        xmin = min(b[0] for b in all_b) - 5
        xmax = max(b[2] for b in all_b) + 5
        ymin = min(b[1] for b in all_b) - 5
        ymax = max(b[3] for b in all_b) + 5
        for ax in axes[1:]:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)

    n_gt_span = sum(1 for c in gt_cells if c.get("is_span"))
    draw_cell_grid(axes[1], gt_cells,
                   f"GT Grid ({len(gt_cells)} cells, {n_gt_span} spanning)", '#1565C0')
    n_pred_span = sum(1 for c in pred_cells if c.get("is_span"))
    draw_cell_grid(axes[2], pred_cells,
                   f"Pred Grid ({len(pred_cells)} cells, {n_pred_span} spanning)\n"
                   f"Cell IoU={stats['cell_iou']:.3f}, F1={stats['cell_f1']:.3f}",
                   '#E65100')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_summary_figures(results_df, save_dir):
    """Summary 시각화 5개."""
    os.makedirs(save_dir, exist_ok=True)

    grp_a = results_df[results_df["has_spanning"] == False]
    grp_b = results_df[results_df["has_spanning"] == True]

    # Fig 1: Row/Col IoU
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in [(axes[0], "row_iou", "Row IoU"), (axes[1], "col_iou", "Col IoU")]:
        data = [grp_a[metric].dropna().values, grp_b[metric].dropna().values]
        bp = ax.boxplot(data, labels=["No Spanning (A)", "Has Spanning (B)"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#FF9800")
        for b in bp["boxes"]:
            b.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')
        means = [np.mean(d) for d in data]
        for i, m in enumerate(means):
            ax.text(i + 1, m, f'μ={m:.3f}', ha='center', va='bottom', fontsize=8, color='red')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig1_row_col_iou.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig 2: Span detection (already known, but include for completeness)
    span_tables = results_df[results_df["has_spanning"] == True]
    if len(span_tables) > 0 and "span_ap50" in span_tables.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col, label in [("span_ap50", "AP@0.5"), ("span_ap75", "AP@0.75"), ("span_ap90", "AP@0.9")]:
            vals = span_tables[col].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=20, alpha=0.5, label=f'{label} (μ={vals.mean():.3f})')
        ax.set_xlabel("AP")
        ax.set_title("Spanning Cell Detection AP Distribution", fontweight='bold')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "fig2_span_ap.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Fig 3: Cell IoU — Group A vs B (+ B-1/B-2/B-3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3a: A vs B
    data_ab = [grp_a["cell_iou"].dropna().values, grp_b["cell_iou"].dropna().values]
    bp = axes[0].boxplot(data_ab, labels=["No Spanning (A)", "Has Spanning (B)"],
                         patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")
    for b in bp["boxes"]:
        b.set_alpha(0.6)
    means = [np.mean(d) for d in data_ab if len(d) > 0]
    for i, m in enumerate(means):
        axes[0].text(i + 1, m, f'μ={m:.3f}', ha='center', va='bottom', fontsize=9, color='red')
    axes[0].set_ylabel("Cell Mean IoU")
    axes[0].set_title("Cell IoU: No Spanning vs Has Spanning", fontweight='bold')

    # 3b: B-1 vs B-2 vs B-3
    sub_labels = ["B-1\n(accurate)", "B-2\n(imprecise)", "B-3\n(missed)"]
    sub_data = []
    for cat in ["B-1", "B-2", "B-3"]:
        vals = results_df[results_df["span_category"] == cat]["cell_iou"].dropna().values
        sub_data.append(vals if len(vals) > 0 else np.array([]))
    bp2 = axes[1].boxplot([d for d in sub_data], labels=sub_labels,
                          patch_artist=True, widths=0.5)
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for i, d in enumerate(sub_data):
        if len(d) > 0:
            axes[1].text(i + 1, np.mean(d), f'μ={np.mean(d):.3f}',
                         ha='center', va='bottom', fontsize=9, color='red')
    axes[1].set_ylabel("Cell Mean IoU")
    axes[1].set_title("Error Cascade: Span Detection Quality → Grid Accuracy", fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig3_cell_iou_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig 4: Error cascade scatter
    if len(span_tables) > 0 and "span_mean_iou" in span_tables.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = span_tables["span_mean_iou"].dropna()
        y = span_tables["cell_iou"].dropna()
        common = x.index.intersection(y.index)
        if len(common) > 0:
            ax.scatter(x[common], y[common], alpha=0.5, s=30, c='#E65100')
            z = np.polyfit(x[common], y[common], 1)
            p = np.poly1d(z)
            xs = np.linspace(x[common].min(), x[common].max(), 100)
            ax.plot(xs, p(xs), 'r--', linewidth=1.5,
                    label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            ax.set_xlabel("Span Detection Mean IoU")
            ax.set_ylabel("Cell Grid Mean IoU")
            ax.set_title("Span Detection Quality → Grid Reconstruction Quality", fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "fig4_error_cascade.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Fig 5: Cell F1 comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    data_f1 = [grp_a["cell_f1"].dropna().values, grp_b["cell_f1"].dropna().values]
    bp = ax.boxplot(data_f1, labels=["No Spanning (A)", "Has Spanning (B)"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")
    for b in bp["boxes"]:
        b.set_alpha(0.6)
    for i, d in enumerate(data_f1):
        if len(d) > 0:
            ax.text(i + 1, np.mean(d), f'μ={np.mean(d):.3f}',
                    ha='center', va='bottom', fontsize=9, color='red')
    ax.set_ylabel("Cell F1 @ IoU 0.5")
    ax.set_title("Cell-level F1: No Spanning vs Has Spanning", fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig5_cell_f1.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═════════════════════════════════════════════
# 8. Main
# ═════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="TATR Grid Reconstruction Eval v3")
    parser.add_argument("--pubtables_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--n_visualize", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("  TATR Cell Grid Reconstruction Evaluation (v3)")
    print("=" * 70)
    print(f"  Data:     {args.pubtables_root}")
    print(f"  Samples:  {args.num_samples}")
    print(f"  Device:   {DEVICE}")
    print()

    # Find data dirs
    img_dir = os.path.join(args.pubtables_root, "test", "images")
    ann_dir = os.path.join(args.pubtables_root, "test", "annotations")
    if not os.path.exists(img_dir):
        # Try alternative structure
        for root, dirs, _ in os.walk(args.pubtables_root):
            if "images" in dirs and "annotations" in dirs:
                img_dir = os.path.join(root, "images")
                ann_dir = os.path.join(root, "annotations")
                break

    print(f"  Images:   {img_dir}")
    print(f"  Annots:   {ann_dir}")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"  Total:    {len(img_files)}")

    # Sample
    np.random.seed(SEED)
    indices = np.random.choice(len(img_files), min(args.num_samples, len(img_files)), replace=False)
    img_files = [img_files[i] for i in sorted(indices)]
    print(f"  Sampled:  {len(img_files)}")

    # Load model
    processor, model = load_model()

    # Run evaluation
    all_results = []
    vis_count = 0

    for img_file in tqdm(img_files, desc="Evaluating"):
        stem = os.path.splitext(img_file)[0]
        xml_path = os.path.join(ann_dir, stem + ".xml")
        if not os.path.exists(xml_path):
            continue

        try:
            image = Image.open(os.path.join(img_dir, img_file)).convert("RGB")
        except Exception:
            continue

        # Load GT
        gt_data = load_gt_xml(xml_path)
        gt_rows = gt_data.get("table row", [])
        gt_cols = gt_data.get("table column", [])
        gt_spans = gt_data.get("table spanning cell", [])

        if not gt_rows or not gt_cols:
            continue

        has_spanning = len(gt_spans) > 0

        # Build GT cell grid
        gt_cells = build_cell_grid(gt_rows, gt_cols, gt_spans)

        # TATR inference
        pred = run_tatr(image, processor, model)
        pred_rows = [r["bbox"] for r in pred["rows"]]
        pred_cols = [c["bbox"] for c in pred["cols"]]
        pred_spans = [s["bbox"] for s in pred["spans"]]

        if not pred_rows or not pred_cols:
            continue

        # Build Pred cell grid
        pred_cells = build_cell_grid(pred_rows, pred_cols, pred_spans)

        # Layer 1: Row/Col eval
        rc_eval = eval_row_col(gt_rows, gt_cols, pred_rows, pred_cols)

        # Layer 2: Span detection eval
        span_eval = eval_span_detection(gt_spans, pred_spans)

        # Layer 3: Cell grid comparison
        cell_match = match_cells(gt_cells, pred_cells)
        adj_f1 = adjacency_f1_between(gt_cells, pred_cells)

        # Determine span category for error cascade
        span_cat = "A"  # no spanning
        if has_spanning:
            clss = span_eval.get("classifications", [])
            if all(c == "B-1" for c in clss):
                span_cat = "B-1"
            elif any(c == "B-3" for c in clss):
                span_cat = "B-3"
            else:
                span_cat = "B-2"

        result = {
            "image": stem,
            "has_spanning": has_spanning,
            "n_gt_rows": len(gt_rows),
            "n_gt_cols": len(gt_cols),
            "n_gt_spans": len(gt_spans),
            "n_pred_rows": len(pred_rows),
            "n_pred_cols": len(pred_cols),
            "n_pred_spans": len(pred_spans),
            "n_gt_cells": len(gt_cells),
            "n_pred_cells": len(pred_cells),
            "row_iou": rc_eval["row"]["mean_iou"],
            "col_iou": rc_eval["col"]["mean_iou"],
            "row_count_err": rc_eval["row"]["count_error"],
            "col_count_err": rc_eval["col"]["count_error"],
            "span_ap50": span_eval.get("ap50"),
            "span_ap75": span_eval.get("ap75"),
            "span_ap90": span_eval.get("ap90"),
            "span_mean_iou": float(np.mean(span_eval.get("matched_ious", [0]))) if span_eval.get("matched_ious") else None,
            "cell_iou": cell_match["mean_iou"],
            "cell_f1": cell_match["f1"],
            "cell_precision": cell_match["precision"],
            "cell_recall": cell_match["recall"],
            "over_seg": cell_match["over_seg"],
            "adj_f1": adj_f1,
            "span_category": span_cat,
        }
        all_results.append(result)

        # Visualize
        if vis_count < args.n_visualize:
            vis_path = out_dir / f"grid_example_{vis_count + 1}_{stem}.png"
            visualize_example(image, gt_cells, pred_cells,
                              {"cell_iou": cell_match["mean_iou"],
                               "cell_f1": cell_match["f1"]},
                              str(vis_path))
            vis_count += 1

    # ── Analysis ──
    df = pd.DataFrame(all_results)
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Tables evaluated: {len(df)}")

    grp_a = df[df["has_spanning"] == False]
    grp_b = df[df["has_spanning"] == True]
    print(f"  Group A (no spanning):  {len(grp_a)}")
    print(f"  Group B (has spanning): {len(grp_b)}")

    print(f"\n{'─' * 40}")
    print("  Layer 1: Row/Col Boundary Quality")
    print(f"{'─' * 40}")
    print(f"  {'Group':<20} {'Row IoU':>10} {'Col IoU':>10}")
    for label, grp in [("A (no span)", grp_a), ("B (has span)", grp_b)]:
        ri = grp["row_iou"].mean() if len(grp) > 0 else 0
        ci = grp["col_iou"].mean() if len(grp) > 0 else 0
        print(f"  {label:<20} {ri:>10.4f} {ci:>10.4f}")

    print(f"\n{'─' * 40}")
    print("  Layer 2: Span Detection Quality")
    print(f"{'─' * 40}")
    if len(grp_b) > 0:
        for col, label in [("span_ap50", "AP@0.50"), ("span_ap75", "AP@0.75"), ("span_ap90", "AP@0.90")]:
            vals = grp_b[col].dropna()
            print(f"  {label}: {vals.mean():.4f} (n={len(vals)})")

    print(f"\n{'─' * 40}")
    print("  Layer 3: Cell Grid Reconstruction (KEY)")
    print(f"{'─' * 40}")
    print(f"  {'Group':<20} {'Cell IoU':>10} {'Cell F1':>10} {'Adj F1':>10} {'OverSeg':>10}")
    for label, grp in [("A (no span)", grp_a), ("B (has span)", grp_b)]:
        ci = grp["cell_iou"].mean() if len(grp) > 0 else 0
        cf = grp["cell_f1"].mean() if len(grp) > 0 else 0
        af = grp["adj_f1"].mean() if len(grp) > 0 else 0
        os_ = grp["over_seg"].mean() if len(grp) > 0 else 0
        print(f"  {label:<20} {ci:>10.4f} {cf:>10.4f} {af:>10.4f} {os_:>10.2f}x")

    gap_iou = grp_a["cell_iou"].mean() - grp_b["cell_iou"].mean() if len(grp_a) > 0 and len(grp_b) > 0 else 0
    gap_f1 = grp_a["cell_f1"].mean() - grp_b["cell_f1"].mean() if len(grp_a) > 0 and len(grp_b) > 0 else 0
    print(f"\n  ★ Cell IoU Gap (A - B): {gap_iou:+.4f}")
    print(f"  ★ Cell F1 Gap (A - B):  {gap_f1:+.4f}")

    # Error Cascade
    print(f"\n{'─' * 40}")
    print("  Error Cascade (Spanning tables only)")
    print(f"{'─' * 40}")
    for cat in ["B-1", "B-2", "B-3"]:
        sub = df[df["span_category"] == cat]
        if len(sub) > 0:
            print(f"  {cat}: n={len(sub)}, Cell IoU={sub['cell_iou'].mean():.4f}, "
                  f"Cell F1={sub['cell_f1'].mean():.4f}")

    # Statistical tests
    print(f"\n{'─' * 40}")
    print("  Statistical Tests")
    print(f"{'─' * 40}")
    stats = {}
    if len(grp_a) > 1 and len(grp_b) > 1:
        t_stat, t_pval = ttest_ind(grp_a["cell_iou"].values, grp_b["cell_iou"].values, equal_var=False)
        pooled = np.sqrt((grp_a["cell_iou"].std()**2 + grp_b["cell_iou"].std()**2) / 2)
        d = (grp_a["cell_iou"].mean() - grp_b["cell_iou"].mean()) / pooled if pooled > 0 else 0
        print(f"  Welch's t: t={t_stat:.4f}, p={t_pval:.2e}")
        print(f"  Cohen's d: {d:.4f} ({'large' if abs(d) >= 0.8 else 'medium' if abs(d) >= 0.5 else 'small'})")
        stats = {"welch_t": float(t_stat), "p_value": float(t_pval), "cohens_d": float(d)}

    # Kruskal-Wallis for B-1 vs B-2 vs B-3
    b_groups = [df[df["span_category"] == c]["cell_iou"].values for c in ["B-1", "B-2", "B-3"]]
    b_groups = [g for g in b_groups if len(g) > 0]
    if len(b_groups) >= 2:
        h_stat, h_pval = kruskal(*b_groups)
        print(f"  Kruskal-Wallis (B-1/B-2/B-3): H={h_stat:.4f}, p={h_pval:.2e}")
        stats["kruskal_h"] = float(h_stat)
        stats["kruskal_p"] = float(h_pval)

    print(f"\n{'=' * 70}")

    # Save results
    csv_path = out_dir / "grid_eval_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  📄 {csv_path}")

    summary = {
        "config": {"num_samples": args.num_samples, "seed": SEED, "model": MODEL_NAME,
                    "confidence": CONFIDENCE, "device": DEVICE},
        "counts": {"total": len(df), "group_a": len(grp_a), "group_b": len(grp_b)},
        "layer1_row_col": {
            "a_row_iou": float(grp_a["row_iou"].mean()) if len(grp_a) else None,
            "a_col_iou": float(grp_a["col_iou"].mean()) if len(grp_a) else None,
            "b_row_iou": float(grp_b["row_iou"].mean()) if len(grp_b) else None,
            "b_col_iou": float(grp_b["col_iou"].mean()) if len(grp_b) else None,
        },
        "layer2_span": {
            "ap50": float(grp_b["span_ap50"].dropna().mean()) if len(grp_b) else None,
            "ap75": float(grp_b["span_ap75"].dropna().mean()) if len(grp_b) else None,
            "ap90": float(grp_b["span_ap90"].dropna().mean()) if len(grp_b) else None,
        },
        "layer3_cell_grid": {
            "a_cell_iou": float(grp_a["cell_iou"].mean()) if len(grp_a) else None,
            "b_cell_iou": float(grp_b["cell_iou"].mean()) if len(grp_b) else None,
            "gap_iou": float(gap_iou),
            "a_cell_f1": float(grp_a["cell_f1"].mean()) if len(grp_a) else None,
            "b_cell_f1": float(grp_b["cell_f1"].mean()) if len(grp_b) else None,
            "gap_f1": float(gap_f1),
        },
        "error_cascade": {
            cat: {"n": int(len(df[df["span_category"] == cat])),
                  "cell_iou": float(df[df["span_category"] == cat]["cell_iou"].mean())
                  if len(df[df["span_category"] == cat]) > 0 else None}
            for cat in ["B-1", "B-2", "B-3"]
        },
        "statistical_tests": stats,
    }
    json_path = out_dir / "summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  📄 {json_path}")

    # Figures
    make_summary_figures(df, str(out_dir))
    print(f"  📊 Figures saved to {out_dir}")

    # Conclusion
    print(f"\n{'=' * 70}")
    if gap_iou > 0:
        print(f"  ✅ 가설 지지: Spanning cell이 있는 테이블의 Cell IoU가")
        print(f"     {gap_iou:.4f} 낮음 (A={grp_a['cell_iou'].mean():.4f} vs B={grp_b['cell_iou'].mean():.4f})")
    else:
        print(f"  ❌ 가설 기각: Gap 방향이 예상과 다름 ({gap_iou:.4f})")
    if stats.get("p_value", 1) < 0.05:
        print(f"     통계적으로 유의미 (p={stats['p_value']:.2e})")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

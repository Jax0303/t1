"""
=======================================================================
TATR Experiment v3: 3-Layer Structural Evaluation
=======================================================================
Layer 1: Component-level evaluation
  - GT rows vs TATR rows (Hungarian IoU matching)
  - GT cols vs TATR cols (Hungarian IoU matching)
  - GT spans vs TATR spans (Hungarian IoU matching)
  -> per-image: mean_iou, recall@0.5, precision@0.5, F1@0.5

Layer 2: Cell grid reconstruction evaluation
  - GT cell grid: GT rows × GT cols + GT spanning cells
  - Pred cell grid: TATR rows × TATR cols + TATR spanning cells
  - Hungarian matching -> cell-level IoU, recall, F1
  - span cell / non-span cell separately

Layer 3: Span presence comparison
  - has_span images (GT spanning cell >= 1) vs no_span images
  - All Layer 1, 2 metrics compared between groups
  - Welch's t-test for statistical significance

Usage:
    python experiment_v3.py \
        --pubtables_root /home/ugh/t1/data/pubtables-1m \
        --output_dir ./results_v3 \
        --num_samples 500 \
        --confidence 0.5
=======================================================================
"""

import os
import json
import argparse
import warnings
from collections import defaultdict
import xml.etree.ElementTree as ET

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_ind

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# TATR v1.1 label IDs
LABEL_ROW = 2
LABEL_COL = 1
LABEL_SPAN = 5


# ─────────────────────────────────────────────
# 1. Model Loading
# ─────────────────────────────────────────────
def load_model():
    print(f"[Model] Loading... (device: {DEVICE})")
    model_name = "microsoft/table-transformer-structure-recognition-v1.1-all"
    processor = AutoImageProcessor.from_pretrained(model_name)
    # Patch for some transformers versions that only have 'longest_edge'
    if isinstance(processor.size, dict) and list(processor.size.keys()) == ["longest_edge"]:
        longest = processor.size["longest_edge"]
        processor.size = {"shortest_edge": longest, "longest_edge": longest}
    model = AutoModelForObjectDetection.from_pretrained(model_name).to(DEVICE)
    model.eval()
    print(f"  -> {model_name}")
    return processor, model


# ─────────────────────────────────────────────
# 2. GT Parsing
# ─────────────────────────────────────────────
def load_gt_annotation(xml_path):
    """PubTables-1M GT XML (Pascal VOC) -> category bbox dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {
        "rows": [],
        "cols": [],
        "spans": [],
        "categories": defaultdict(list),
    }

    for obj in root.findall("object"):
        cat = obj.findtext("name", "").strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", 0))
        ymin = float(bndbox.findtext("ymin", 0))
        xmax = float(bndbox.findtext("xmax", 0))
        ymax = float(bndbox.findtext("ymax", 0))
        bbox = [xmin, ymin, xmax, ymax]
        result["categories"][cat].append(bbox)
        if cat == "table row":
            result["rows"].append(bbox)
        elif cat == "table column":
            result["cols"].append(bbox)
        elif cat == "table spanning cell":
            result["spans"].append(bbox)

    return result


# ─────────────────────────────────────────────
# 3. Cell Grid Construction
# ─────────────────────────────────────────────
def build_cell_grid(rows, cols, spans):
    """
    Build cell grid from rows, cols, and spans.
    1. Add spanning cells directly.
    2. Add row x col intersections, excluding those whose center falls in a span.
    """
    cells = []

    # Spanning cells added directly
    for sp in spans:
        cells.append({"bbox": sp, "is_span": True})

    # row x col intersections, excluding span-covered cells
    for row in rows:
        for col in cols:
            x1 = max(row[0], col[0])
            y1 = max(row[1], col[1])
            x2 = min(row[2], col[2])
            y2 = min(row[3], col[3])
            if x1 >= x2 or y1 >= y2:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if any(sp[0] <= cx <= sp[2] and sp[1] <= cy <= sp[3] for sp in spans):
                continue
            cells.append({"bbox": [x1, y1, x2, y2], "is_span": False})

    return cells


# ─────────────────────────────────────────────
# 4. Inference
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(image, processor, model, threshold):
    """TATR inference -> boxes, scores, labels."""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    boxes = results["boxes"].cpu()
    scores = results["scores"].cpu()
    labels = results["labels"].cpu()

    pred_rows = boxes[labels == LABEL_ROW].tolist()
    pred_cols = boxes[labels == LABEL_COL].tolist()
    pred_spans = boxes[labels == LABEL_SPAN].tolist()

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "rows": pred_rows,
        "cols": pred_cols,
        "spans": pred_spans,
    }


# ─────────────────────────────────────────────
# 5. Hungarian Matching Utilities
# ─────────────────────────────────────────────
def match_components(gt_list, pred_list, iou_threshold=0.5):
    """
    Hungarian IoU matching between GT and pred bbox lists.
    Returns dict with mean_iou, recall, precision, F1 and per-pair info.
    """
    n_gt = len(gt_list)
    n_pred = len(pred_list)

    if n_gt == 0 and n_pred == 0:
        return {"mean_iou": 0.0, "recall": 1.0, "precision": 1.0, "f1": 1.0,
                "n_gt": 0, "n_pred": 0, "tp": 0, "iou_values": []}
    if n_gt == 0:
        return {"mean_iou": 0.0, "recall": 1.0, "precision": 0.0, "f1": 0.0,
                "n_gt": 0, "n_pred": n_pred, "tp": 0, "iou_values": []}
    if n_pred == 0:
        return {"mean_iou": 0.0, "recall": 0.0, "precision": 1.0, "f1": 0.0,
                "n_gt": n_gt, "n_pred": 0, "tp": 0, "iou_values": [0.0] * n_gt}

    gt_t = torch.tensor(gt_list, dtype=torch.float32)
    pred_t = torch.tensor(pred_list, dtype=torch.float32)
    iou_mat = box_iou(gt_t, pred_t).numpy()

    cost_mat = 1.0 - iou_mat
    gt_idx, pred_idx = linear_sum_assignment(cost_mat)

    matched_ious = np.zeros(n_gt)
    for gi, pi in zip(gt_idx, pred_idx):
        matched_ious[gi] = iou_mat[gi, pi]

    tp = int(np.sum(matched_ious >= iou_threshold))
    mean_iou = float(np.mean(matched_ious))
    recall = tp / n_gt if n_gt > 0 else 0.0
    precision = tp / n_pred if n_pred > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "mean_iou": mean_iou,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "iou_values": matched_ious.tolist(),
    }


def match_cells(gt_cells, pred_cells, iou_threshold=0.5):
    """
    Hungarian matching of cell grids.
    Returns overall and split-by-span metrics + per-cell IoU list.
    """
    n_gt = len(gt_cells)
    n_pred = len(pred_cells)

    if n_gt == 0:
        return {
            "mean_iou": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "span_mean_iou": 0.0, "span_recall": 0.0, "span_f1": 0.0,
            "nonspan_mean_iou": 0.0, "nonspan_recall": 0.0, "nonspan_f1": 0.0,
            "n_gt": 0, "n_pred": n_pred, "span_iou_values": [], "nonspan_iou_values": [],
        }
    if n_pred == 0:
        n_span = sum(1 for c in gt_cells if c["is_span"])
        n_nonspan = n_gt - n_span
        return {
            "mean_iou": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "span_mean_iou": 0.0, "span_recall": 0.0, "span_f1": 0.0,
            "nonspan_mean_iou": 0.0, "nonspan_recall": 0.0, "nonspan_f1": 0.0,
            "n_gt": n_gt, "n_pred": 0,
            "span_iou_values": [0.0] * n_span,
            "nonspan_iou_values": [0.0] * n_nonspan,
        }

    gt_boxes = torch.tensor([c["bbox"] for c in gt_cells], dtype=torch.float32)
    pred_boxes = torch.tensor([c["bbox"] for c in pred_cells], dtype=torch.float32)
    iou_mat = box_iou(gt_boxes, pred_boxes).numpy()

    cost_mat = 1.0 - iou_mat
    gt_idx, pred_idx = linear_sum_assignment(cost_mat)

    matched_ious = np.zeros(n_gt)
    for gi, pi in zip(gt_idx, pred_idx):
        matched_ious[gi] = iou_mat[gi, pi]

    # Overall
    tp = int(np.sum(matched_ious >= iou_threshold))
    mean_iou = float(np.mean(matched_ious))
    recall = tp / n_gt
    precision = tp / n_pred if n_pred > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # Split by span type
    span_mask = np.array([c["is_span"] for c in gt_cells])
    nonspan_mask = ~span_mask

    def sub_metrics(mask):
        if not np.any(mask):
            return {"mean_iou": 0.0, "recall": 0.0, "f1": 0.0, "iou_values": []}
        sub_ious = matched_ious[mask]
        sub_n_gt = int(np.sum(mask))
        sub_tp = int(np.sum(sub_ious >= iou_threshold))
        sub_mean_iou = float(np.mean(sub_ious))
        sub_recall = sub_tp / sub_n_gt
        # For sub-group f1, use sub-group n_pred as same ratio
        sub_n_pred = max(1, int(n_pred * sub_n_gt / n_gt))
        sub_prec = sub_tp / sub_n_pred
        sub_f1 = (2 * sub_prec * sub_recall / (sub_prec + sub_recall)
                  if (sub_prec + sub_recall) > 0 else 0.0)
        return {"mean_iou": sub_mean_iou, "recall": sub_recall,
                "f1": sub_f1, "iou_values": sub_ious.tolist()}

    span_metrics = sub_metrics(span_mask)
    nonspan_metrics = sub_metrics(nonspan_mask)

    return {
        "mean_iou": mean_iou,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "span_mean_iou": span_metrics["mean_iou"],
        "span_recall": span_metrics["recall"],
        "span_f1": span_metrics["f1"],
        "nonspan_mean_iou": nonspan_metrics["mean_iou"],
        "nonspan_recall": nonspan_metrics["recall"],
        "nonspan_f1": nonspan_metrics["f1"],
        "n_gt": n_gt,
        "n_pred": n_pred,
        "span_iou_values": span_metrics["iou_values"],
        "nonspan_iou_values": nonspan_metrics["iou_values"],
    }


def span_recall_at_thresholds(gt_spans, pred_spans, thresholds):
    """Compute span recall at multiple IoU thresholds for recall curve."""
    if not gt_spans or not pred_spans:
        return {t: 0.0 for t in thresholds}

    gt_t = torch.tensor(gt_spans, dtype=torch.float32)
    pred_t = torch.tensor(pred_spans, dtype=torch.float32)
    iou_mat = box_iou(gt_t, pred_t).numpy()

    cost_mat = 1.0 - iou_mat
    gt_idx, pred_idx = linear_sum_assignment(cost_mat)

    matched_ious = np.zeros(len(gt_spans))
    for gi, pi in zip(gt_idx, pred_idx):
        matched_ious[gi] = iou_mat[gi, pi]

    recalls = {}
    for t in thresholds:
        tp = int(np.sum(matched_ious >= t))
        recalls[t] = tp / len(gt_spans)
    return recalls


# ─────────────────────────────────────────────
# 6. Data Path Discovery
# ─────────────────────────────────────────────
def find_data_dirs(root):
    candidates = [
        (os.path.join(root, "test", "images"),
         os.path.join(root, "test", "annotations")),
        (os.path.join(root, "images", "test"),
         os.path.join(root, "annotations", "test")),
        (os.path.join(root, "PubTables-1M-Structure", "test", "images"),
         os.path.join(root, "PubTables-1M-Structure", "test", "annotations")),
    ]
    for img_dir, ann_dir in candidates:
        if os.path.exists(img_dir) and os.path.exists(ann_dir):
            return img_dir, ann_dir

    for dirpath, dirnames, filenames in os.walk(root):
        if "images" in dirnames:
            img_candidate = os.path.join(dirpath, "images")
            ann_candidate = os.path.join(dirpath, "annotations")
            if os.path.exists(ann_candidate):
                return img_candidate, ann_candidate

    raise FileNotFoundError(
        f"Data not found: {root}\n"
        f"Expected structure: root/test/images/ + root/test/annotations/"
    )


# ─────────────────────────────────────────────
# 7. Main Evaluation Loop
# ─────────────────────────────────────────────
SPAN_RECALL_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def evaluate(data_dir, processor, model, num_samples=500, confidence=0.5):
    img_dir, ann_dir = find_data_dirs(data_dir)
    print(f"  Images:      {img_dir}")
    print(f"  Annotations: {ann_dir}")

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"  Total images: {len(img_files)}")

    if num_samples and num_samples < len(img_files):
        np.random.seed(SEED)
        indices = np.random.choice(len(img_files), num_samples, replace=False)
        img_files = [img_files[i] for i in sorted(indices)]
        print(f"  Sampling: {num_samples}")

    per_image_rows = []
    skipped = 0

    # Accumulators for span recall curve (aggregated across all images)
    span_recall_accum = defaultdict(list)

    for img_file in tqdm(img_files, desc="Inference"):
        stem = os.path.splitext(img_file)[0]
        ann_file = os.path.join(ann_dir, stem + ".xml")

        if not os.path.exists(ann_file):
            skipped += 1
            continue

        gt_data = load_gt_annotation(ann_file)
        gt_rows = gt_data["rows"]
        gt_cols = gt_data["cols"]
        gt_spans = gt_data["spans"]

        if not gt_rows or not gt_cols:
            skipped += 1
            continue

        try:
            image = Image.open(os.path.join(img_dir, img_file)).convert("RGB")
        except Exception:
            skipped += 1
            continue

        pred = run_inference(image, processor, model, confidence)

        # ── Layer 1: Component matching ──────────────────────────────
        row_metrics = match_components(gt_rows, pred["rows"])
        col_metrics = match_components(gt_cols, pred["cols"])
        span_metrics = match_components(gt_spans, pred["spans"])

        # ── Layer 2: Cell grid reconstruction ────────────────────────
        gt_cells = build_cell_grid(gt_rows, gt_cols, gt_spans)
        pred_cells = build_cell_grid(pred["rows"], pred["cols"], pred["spans"])
        cell_metrics = match_cells(gt_cells, pred_cells)

        # ── Span recall at multiple thresholds ───────────────────────
        if gt_spans:
            thresh_recalls = span_recall_at_thresholds(
                gt_spans, pred["spans"], SPAN_RECALL_THRESHOLDS
            )
            for t, r in thresh_recalls.items():
                span_recall_accum[t].append(r)

        has_span = len(gt_spans) >= 1

        row_data = {
            "image": stem,
            "has_span": has_span,
            "n_gt_rows": len(gt_rows),
            "n_gt_cols": len(gt_cols),
            "n_gt_spans": len(gt_spans),
            # Layer 1
            "row_mean_iou": row_metrics["mean_iou"],
            "row_recall": row_metrics["recall"],
            "row_precision": row_metrics["precision"],
            "row_f1": row_metrics["f1"],
            "col_mean_iou": col_metrics["mean_iou"],
            "col_recall": col_metrics["recall"],
            "col_precision": col_metrics["precision"],
            "col_f1": col_metrics["f1"],
            "span_mean_iou": span_metrics["mean_iou"],
            "span_recall": span_metrics["recall"],
            "span_precision": span_metrics["precision"],
            "span_f1": span_metrics["f1"],
            # Layer 2
            "cell_mean_iou": cell_metrics["mean_iou"],
            "cell_recall": cell_metrics["recall"],
            "cell_precision": cell_metrics["precision"],
            "cell_f1": cell_metrics["f1"],
            "span_cell_mean_iou": cell_metrics["span_mean_iou"],
            "span_cell_recall": cell_metrics["span_recall"],
            "span_cell_f1": cell_metrics["span_f1"],
            "nonspan_cell_mean_iou": cell_metrics["nonspan_mean_iou"],
            "nonspan_cell_recall": cell_metrics["nonspan_recall"],
            "nonspan_cell_f1": cell_metrics["nonspan_f1"],
            # Cell counts
            "n_gt_cells": cell_metrics["n_gt"],
            "n_pred_cells": cell_metrics["n_pred"],
        }
        per_image_rows.append(row_data)

        # Store individual cell IoU values keyed by image for histogram
        row_data["_span_iou_values"] = cell_metrics["span_iou_values"]
        row_data["_nonspan_iou_values"] = cell_metrics["nonspan_iou_values"]

    print(f"\n  Processed: {len(per_image_rows)} / Skipped: {skipped}")

    df = pd.DataFrame(per_image_rows)
    return df, span_recall_accum


# ─────────────────────────────────────────────
# 8. Statistical Tests
# ─────────────────────────────────────────────
def welch_t_test(a, b):
    """Welch's t-test between two groups. Returns dict."""
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": None, "p_value": None, "significant": None}
    t_stat, p_val = ttest_ind(a, b, equal_var=False)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
    }


def layer3_analysis(df):
    """Layer 3: Compare has_span vs no_span groups across all L1/L2 metrics."""
    has = df[df["has_span"] == True]
    nope = df[df["has_span"] == False]

    metrics = [
        "row_mean_iou", "row_recall", "row_f1",
        "col_mean_iou", "col_recall", "col_f1",
        "span_mean_iou", "span_recall", "span_f1",
        "cell_mean_iou", "cell_recall", "cell_f1",
        "span_cell_mean_iou", "span_cell_recall", "span_cell_f1",
        "nonspan_cell_mean_iou", "nonspan_cell_recall", "nonspan_cell_f1",
    ]

    results = {}
    for m in metrics:
        if m not in df.columns:
            continue
        a_vals = has[m].dropna().tolist()
        b_vals = nope[m].dropna().tolist()
        test = welch_t_test(a_vals, b_vals)
        results[m] = {
            "has_span_mean": float(np.mean(a_vals)) if a_vals else None,
            "has_span_std": float(np.std(a_vals)) if a_vals else None,
            "no_span_mean": float(np.mean(b_vals)) if b_vals else None,
            "no_span_std": float(np.std(b_vals)) if b_vals else None,
            "n_has_span": len(a_vals),
            "n_no_span": len(b_vals),
            "welch_t_test": test,
        }
    return results


# ─────────────────────────────────────────────
# 9. Visualizations
# ─────────────────────────────────────────────
def plot_component_iou_boxplot(df, output_dir):
    """Viz 1: Component IoU comparison (has_span vs no_span) for Row/Col/Span."""
    has = df[df["has_span"] == True]
    nope = df[df["has_span"] == False]

    components = [
        ("Row IoU", "row_mean_iou"),
        ("Col IoU", "col_mean_iou"),
        ("Span IoU", "span_mean_iou"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Component-Level IoU: has_span vs no_span", fontsize=13, fontweight="bold")

    for ax, (title, col) in zip(axes, components):
        data_has = has[col].dropna().tolist()
        data_nope = nope[col].dropna().tolist()
        bp = ax.boxplot(
            [data_has, data_nope],
            labels=["has_span", "no_span"],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Mean IoU")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        # Annotate medians
        for i, data in enumerate([data_has, data_nope], 1):
            if data:
                med = np.median(data)
                ax.text(i, med + 0.02, f"{med:.3f}", ha="center", va="bottom",
                        fontsize=9, color="black")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "viz1_component_iou_boxplot.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cell_reconstruction_f1(df, output_dir):
    """Viz 2: Cell reconstruction F1 (has_span vs no_span) bar chart with error bars."""
    has = df[df["has_span"] == True]
    nope = df[df["has_span"] == False]

    metrics = [
        ("Cell F1\n(all)", "cell_f1"),
        ("Cell F1\n(span)", "span_cell_f1"),
        ("Cell F1\n(non-span)", "nonspan_cell_f1"),
    ]

    labels = [m[0] for m in metrics]
    cols = [m[1] for m in metrics]

    has_means = [has[c].dropna().mean() for c in cols]
    has_stds = [has[c].dropna().std() for c in cols]
    nope_means = [nope[c].dropna().mean() for c in cols]
    nope_stds = [nope[c].dropna().std() for c in cols]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, has_means, width, yerr=has_stds,
                   label="has_span", color="#4C72B0", capsize=5, alpha=0.85)
    bars2 = ax.bar(x + width / 2, nope_means, width, yerr=nope_stds,
                   label="no_span", color="#DD8452", capsize=5, alpha=0.85)

    ax.set_title("Cell Reconstruction F1@0.5: has_span vs no_span", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "viz2_cell_reconstruction_f1.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_span_recall_curve(span_recall_accum, output_dir):
    """Viz 3: Span detection recall at IoU threshold 0.5~0.9."""
    thresholds = SPAN_RECALL_THRESHOLDS
    mean_recalls = [np.mean(span_recall_accum[t]) if span_recall_accum[t] else 0.0
                    for t in thresholds]
    std_recalls = [np.std(span_recall_accum[t]) if span_recall_accum[t] else 0.0
                   for t in thresholds]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, mean_recalls, marker="o", color="#4C72B0",
            linewidth=2, markersize=7, label="Mean Recall")
    ax.fill_between(
        thresholds,
        [m - s for m, s in zip(mean_recalls, std_recalls)],
        [m + s for m, s in zip(mean_recalls, std_recalls)],
        alpha=0.2, color="#4C72B0", label="±1 std"
    )

    for t, m in zip(thresholds, mean_recalls):
        ax.annotate(f"{m:.3f}", (t, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_title("Span Detection Recall Curve (IoU threshold 0.5 ~ 0.9)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("Recall")
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, "viz3_span_recall_curve.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cell_iou_histogram(df, output_dir):
    """Viz 4: Cell IoU distribution — span cell vs non-span cell histogram."""
    all_span_ious = []
    all_nonspan_ious = []
    for _, row in df.iterrows():
        all_span_ious.extend(row.get("_span_iou_values", []))
        all_nonspan_ious.extend(row.get("_nonspan_iou_values", []))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle("Cell IoU Distribution: Span vs Non-Span", fontsize=13, fontweight="bold")

    bins = np.linspace(0, 1, 21)

    for ax, (title, data, color) in zip(axes, [
        ("Span Cells", all_span_ious, "#4C72B0"),
        ("Non-Span Cells", all_nonspan_ious, "#DD8452"),
    ]):
        if data:
            ax.hist(data, bins=bins, color=color, alpha=0.75, edgecolor="white")
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
                       label=f"Mean={mean_val:.3f}")
            ax.axvline(median_val, color="green", linestyle=":", linewidth=1.5,
                       label=f"Median={median_val:.3f}")
            ax.legend(fontsize=9)
        ax.set_title(f"{title} (n={len(data):,})", fontsize=11)
        ax.set_xlabel("IoU")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "viz4_cell_iou_histogram.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────
# 10. Summary Statistics
# ─────────────────────────────────────────────
def compute_summary(df, layer3, span_recall_accum):
    """Aggregate all metrics into a summary dict."""
    def group_stats(grp, col):
        vals = grp[col].dropna()
        if len(vals) == 0:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(vals.mean()), "std": float(vals.std()), "n": len(vals)}

    has = df[df["has_span"] == True]
    nope = df[df["has_span"] == False]

    layer1_metrics = [
        "row_mean_iou", "row_recall", "row_precision", "row_f1",
        "col_mean_iou", "col_recall", "col_precision", "col_f1",
        "span_mean_iou", "span_recall", "span_precision", "span_f1",
    ]
    layer2_metrics = [
        "cell_mean_iou", "cell_recall", "cell_precision", "cell_f1",
        "span_cell_mean_iou", "span_cell_recall", "span_cell_f1",
        "nonspan_cell_mean_iou", "nonspan_cell_recall", "nonspan_cell_f1",
    ]

    summary = {
        "total_images": len(df),
        "has_span_images": int(df["has_span"].sum()),
        "no_span_images": int((~df["has_span"]).sum()),
        "layer1": {},
        "layer2": {},
        "layer3_group_comparison": layer3,
        "span_recall_curve": {
            str(t): float(np.mean(span_recall_accum[t])) if span_recall_accum[t] else 0.0
            for t in SPAN_RECALL_THRESHOLDS
        },
    }

    for col in layer1_metrics:
        if col not in df.columns:
            continue
        summary["layer1"][col] = {
            "overall": group_stats(df, col),
            "has_span": group_stats(has, col),
            "no_span": group_stats(nope, col),
        }

    for col in layer2_metrics:
        if col not in df.columns:
            continue
        summary["layer2"][col] = {
            "overall": group_stats(df, col),
            "has_span": group_stats(has, col),
            "no_span": group_stats(nope, col),
        }

    return summary


# ─────────────────────────────────────────────
# 11. Entry Point
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="TATR v3: 3-Layer Structural Evaluation"
    )
    parser.add_argument("--pubtables_root", type=str,
                        default="/home/ugh/t1/data/pubtables-1m",
                        help="Root dir for PubTables-1M dataset")
    parser.add_argument("--output_dir", type=str, default="./results_v3",
                        help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of images to evaluate")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for TATR predictions")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("TATR Experiment v3: 3-Layer Structural Evaluation")
    print("=" * 60)
    print(f"  pubtables_root : {args.pubtables_root}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  num_samples    : {args.num_samples}")
    print(f"  confidence     : {args.confidence}")
    print(f"  device         : {DEVICE}")
    print()

    # Step 1: Load model
    processor, model = load_model()

    # Step 2: Evaluate
    print("\n[Evaluate]")
    df_raw, span_recall_accum = evaluate(
        args.pubtables_root, processor, model,
        num_samples=args.num_samples, confidence=args.confidence
    )

    if df_raw.empty:
        print("No results collected. Exiting.")
        return

    # Step 3: Layer 3 analysis
    print("\n[Layer 3: has_span vs no_span group comparison]")
    layer3 = layer3_analysis(df_raw)

    # Step 4: Summary
    summary = compute_summary(df_raw, layer3, span_recall_accum)

    # Step 5: Save results
    print("\n[Saving results]")

    # Drop internal list columns before saving CSV
    csv_cols = [c for c in df_raw.columns if not c.startswith("_")]
    csv_path = os.path.join(args.output_dir, "per_image_stats.csv")
    df_raw[csv_cols].to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")

    # Step 6: Visualizations
    print("\n[Visualizations]")
    plot_component_iou_boxplot(df_raw, args.output_dir)
    plot_cell_reconstruction_f1(df_raw, args.output_dir)
    plot_span_recall_curve(span_recall_accum, args.output_dir)
    plot_cell_iou_histogram(df_raw, args.output_dir)

    # Step 7: Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"  Total images evaluated : {summary['total_images']}")
    print(f"  has_span images        : {summary['has_span_images']}")
    print(f"  no_span  images        : {summary['no_span_images']}")
    print()

    # Layer 1 highlights
    print("[Layer 1 — Component Matching]")
    for comp in ["row", "col", "span"]:
        col = f"{comp}_f1"
        if col in summary["layer1"]:
            ov = summary["layer1"][col]["overall"]
            print(f"  {comp:5s} F1@0.5 (overall): {ov['mean']:.4f} ± {ov['std']:.4f}")

    # Layer 2 highlights
    print("\n[Layer 2 — Cell Grid Reconstruction]")
    for col in ["cell_f1", "span_cell_f1", "nonspan_cell_f1"]:
        if col in summary["layer2"]:
            ov = summary["layer2"][col]["overall"]
            print(f"  {col:<22}: {ov['mean']:.4f} ± {ov['std']:.4f}")

    # Layer 3 highlights
    print("\n[Layer 3 — Group Comparison (Welch's t-test)]")
    for col in ["cell_f1", "span_cell_f1", "nonspan_cell_f1"]:
        if col in layer3:
            info = layer3[col]
            wt = info["welch_t_test"]
            print(f"  {col:<22}: "
                  f"has_span={info['has_span_mean']:.4f}, "
                  f"no_span={info['no_span_mean']:.4f}, "
                  f"p={wt['p_value']:.4e} "
                  f"({'*' if wt['significant'] else 'ns'})")

    # Span recall curve
    print("\n[Span Detection Recall Curve]")
    for t in SPAN_RECALL_THRESHOLDS:
        r = summary["span_recall_curve"][str(t)]
        print(f"  IoU@{t:.1f}: {r:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

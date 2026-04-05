#!/usr/bin/env python3
"""
Detection-Family Evaluation Harness (hole 2 해결)
==================================================
"TATR 3 변종은 전부 DETR 계열이므로 detection paradigm 일반화 주장이
과도하다" 는 비판에 대한 대응.

목표:
  1. DETR 계열(TATR) 이외의 detection paradigm 들을 동일 metric 으로 평가 가능한
     하나의 pluggable adapter 인터페이스 제공
  2. 현재 실제 평가 가능: TATR v1.0 / v1.1-pub / v1.1-all (DETR계)
                      + Faster R-CNN local checkpoint (anchor 계, RPN)
  3. 확장 ready (weights/config 제공 시 즉시 plug-in):
     - LGPMA   (cascade R-CNN + local pyramid mask)    [mmocr]
     - TableMaster (multi-branch FPN)                  [paddleocr]
     - LORE    (anchor-free, keypoint)                 [open source]

  이 harness 는 논문의 scope 를 다음과 같이 명확화한다:
    "We evaluate detection-based TSR across two dominant paradigms:
     (a) DETR-based (TATR 3 variants, Microsoft PubTables-1M lineage),
     (b) anchor-based (Faster R-CNN trained on PubTables-1M).
     Extension to further detection families is provided via the
     DetectionModelAdapter interface."

인터페이스:
  class DetectionModelAdapter:
      name:     str
      paradigm: Literal["detr", "anchor", "keypoint", "cascade", ...]
      def load(self) -> None: ...
      def infer(self, image: PIL.Image) -> dict:
          # return {'cells', 'n_rows', 'n_cols', 'n_spans_detected',
          #         'rows_bbox', 'cols_bbox', 'spans_bbox'}

출력:
  experiments/results_family/family_summary.json
  experiments/results_family/family_eval.png   (paradigm 별 비교)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _eval_utils import load_valid_ids, load_gt, load_image, bbox_iou
from logical_eval import (
    infer_gt_row_col_centers, logical_span_recovery,
    span_distribution_match,
)
from grits_eval import grits_factored, sim_top, sim_con

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results_family"
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5
IOU_MERGE = 0.3


# ─────────────────────────────────────────────
# Abstract adapter
# ─────────────────────────────────────────────
class DetectionModelAdapter(ABC):
    name: str
    paradigm: str
    color: str = "#888888"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def infer(self, image: Image.Image) -> dict: ...


# ─────────────────────────────────────────────
# TATR (DETR) adapter — 3 variants
# ─────────────────────────────────────────────
class TATRAdapter(DetectionModelAdapter):
    paradigm = "detr"

    def __init__(self, model_id: str, name: str, color: str):
        self.model_id = model_id
        self.name = name
        self.color = color
        self.processor = None
        self.model = None

    def load(self):
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        if hasattr(self.processor, "size") and isinstance(self.processor.size, dict):
            sz = self.processor.size
            if "longest_edge" in sz and "shortest_edge" not in sz:
                le = sz["longest_edge"]
                self.processor.size = {"shortest_edge": le, "longest_edge": le}
        self.model = TableTransformerForObjectDetection.from_pretrained(self.model_id)
        self.model = self.model.to(DEVICE).eval()

    @torch.no_grad()
    def infer(self, image: Image.Image) -> dict:
        enc = self.processor(images=image, return_tensors="pt")
        mk = {"pixel_values", "pixel_mask"}
        model_enc = {k: v.to(DEVICE) for k, v in enc.items() if k in mk}
        outputs = self.model(**model_enc)
        results = self.processor.post_process_object_detection(
            outputs, threshold=CONF_THRESH,
            target_sizes=[(image.size[1], image.size[0])],
        )[0]
        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        id2label = self.model.config.id2label
        rows, cols, spans = [], [], []
        for box, lid, sc in zip(boxes, labels, scores):
            ln = id2label[lid].lower()
            e = {"box": box.tolist(), "score": float(sc)}
            if "column" in ln: cols.append(e)
            elif "spanning" in ln or "projected" in ln: spans.append(e)
            else: rows.append(e)
        return _cells_from_rowcolspan(rows, cols, spans)


# ─────────────────────────────────────────────
# Faster R-CNN (anchor) adapter — local checkpoint
# ─────────────────────────────────────────────
class FasterRCNNAdapter(DetectionModelAdapter):
    paradigm = "anchor"
    name = "Faster R-CNN"
    color = "#C44E52"

    def __init__(self, checkpoint_path: str, num_classes: int = 4,
                 label_map: Optional[dict] = None):
        """
        Args:
          checkpoint_path: .pth 로컬 weight
          num_classes: 배경 포함 클래스 수 (e.g. [bg, row, col, span] = 4)
          label_map: {class_id: 'row'|'column'|'spanning_cell'} mapping
        """
        self.ckpt = checkpoint_path
        self.num_classes = num_classes
        self.label_map = label_map or {1: "row", 2: "column", 3: "spanning_cell"}
        self.model = None
        self.transform = None

    def load(self):
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        except ImportError:
            raise RuntimeError("torchvision required for FasterRCNNAdapter")
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        state = torch.load(self.ckpt, map_location=DEVICE)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        self.model = model.to(DEVICE).eval()

        import torchvision.transforms as T
        self.transform = T.Compose([T.ToTensor()])

    @torch.no_grad()
    def infer(self, image: Image.Image) -> dict:
        t = self.transform(image).to(DEVICE).unsqueeze(0)
        out = self.model(t)[0]
        boxes = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        rows, cols, spans = [], [], []
        for box, lid, sc in zip(boxes, labels, scores):
            if sc < CONF_THRESH:
                continue
            lname = self.label_map.get(int(lid), "")
            e = {"box": box.tolist(), "score": float(sc)}
            if "column" in lname: cols.append(e)
            elif "span" in lname or "projected" in lname: spans.append(e)
            elif "row" in lname: rows.append(e)
        return _cells_from_rowcolspan(rows, cols, spans)


# ─────────────────────────────────────────────
# 공용: row/col/span bbox list → cell grid + merge
# ─────────────────────────────────────────────
def _cells_from_rowcolspan(rows: list, cols: list, spans: list) -> dict:
    rows.sort(key=lambda r: (r["box"][1] + r["box"][3]) / 2)
    cols.sort(key=lambda c: (c["box"][0] + c["box"][2]) / 2)
    cells = []
    for ri, rdata in enumerate(rows):
        for ci, cdata in enumerate(cols):
            rb, cb = rdata["box"], cdata["box"]
            x0, y0 = max(cb[0], rb[0]), max(rb[1], cb[1])
            x1, y1 = min(cb[2], rb[2]), min(rb[3], cb[3])
            if x1 > x0 and y1 > y0:
                cells.append({
                    "start_row": ri, "end_row": ri,
                    "start_col": ci, "end_col": ci,
                    "row_span": 1, "col_span": 1,
                    "bbox": [x0, y0, x1, y1], "text": "",
                })
    if spans:
        merged_idx = set()
        merged = []
        for sdata in spans:
            sb = sdata["box"]
            ov = [(i, c) for i, c in enumerate(cells)
                  if i not in merged_idx and bbox_iou(sb, c["bbox"]) > IOU_MERGE]
            if ov:
                rs = [c["start_row"] for _, c in ov]
                cs = [c["start_col"] for _, c in ov]
                r0, r1 = min(rs), max(rs); c0, c1 = min(cs), max(cs)
                merged.append({
                    "start_row": r0, "end_row": r1,
                    "start_col": c0, "end_col": c1,
                    "row_span": r1 - r0 + 1, "col_span": c1 - c0 + 1,
                    "bbox": sb, "text": "",
                })
                for i, _ in ov:
                    merged_idx.add(i)
        cells = merged + [c for i, c in enumerate(cells) if i not in merged_idx]
    return {
        "cells": cells,
        "n_rows": len(rows), "n_cols": len(cols),
        "n_spans_detected": len(spans),
        "rows_bbox": [r["box"] for r in rows],
        "cols_bbox": [c["box"] for c in cols],
        "spans_bbox": [s["box"] for s in spans],
    }


# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────
def build_models(include_frcnn_ckpt: Optional[str] = None) -> List[DetectionModelAdapter]:
    models: List[DetectionModelAdapter] = [
        TATRAdapter("microsoft/table-transformer-structure-recognition",
                    "TATR v1.0", "#4C72B0"),
        TATRAdapter("microsoft/table-transformer-structure-recognition-v1.1-pub",
                    "TATR v1.1-pub", "#DD8452"),
        TATRAdapter("microsoft/table-transformer-structure-recognition-v1.1-all",
                    "TATR v1.1-all", "#55A868"),
    ]
    if include_frcnn_ckpt and Path(include_frcnn_ckpt).exists():
        models.append(FasterRCNNAdapter(include_frcnn_ckpt))
    return models


# ─────────────────────────────────────────────
# Unified eval per table
# ─────────────────────────────────────────────
def eval_table(gt: dict, pred: dict, tid: str, img: Image.Image) -> dict:
    centers = infer_gt_row_col_centers(gt, tid, img.size)
    log_exact = None
    if centers is not None:
        row_c, col_c = centers
        logr = logical_span_recovery(gt, pred, row_c, col_c)
        log_exact = logr["rate_exact"]
    dist = span_distribution_match(gt, pred)
    p_t, r_t, f_t = grits_factored(
        pred["cells"], gt["cells"],
        pred["n_rows"], pred["n_cols"],
        gt["n_rows"], gt["n_cols"], sim_top)
    p_c, r_c, f_c = grits_factored(
        pred["cells"], gt["cells"],
        pred["n_rows"], pred["n_cols"],
        gt["n_rows"], gt["n_cols"], sim_con)
    return {
        "table_id": tid,
        "logical_exact": log_exact,
        "span_dist_jaccard": dist,
        "grits_top_f1": f_t, "grits_top_p": p_t, "grits_top_r": r_t,
        "grits_con_f1": f_c, "grits_con_p": p_c, "grits_con_r": r_c,
        "pred_n_rows": pred["n_rows"], "pred_n_cols": pred["n_cols"],
        "gt_n_rows": gt["n_rows"], "gt_n_cols": gt["n_cols"],
    }


def run(include_frcnn_ckpt: Optional[str] = None):
    valid_ids = load_valid_ids()
    print(f"[family_eval] {len(valid_ids)} tables", flush=True)
    models = build_models(include_frcnn_ckpt)
    all_recs: Dict[str, List[dict]] = {}
    summary: Dict[str, dict] = {}

    for m in models:
        print(f"\n=== {m.name} (paradigm={m.paradigm}) ===", flush=True)
        try:
            m.load()
        except Exception as e:
            print(f"  SKIP: load failed: {e}", flush=True)
            continue
        recs = []
        err = 0
        for tid in tqdm(valid_ids, desc=m.name):
            try:
                gt = load_gt(tid); img = load_image(tid)
                pred = m.infer(img)
                recs.append(eval_table(gt, pred, tid, img))
            except Exception:
                err += 1
        print(f"  processed={len(recs)}  errors={err}", flush=True)
        all_recs[m.name] = recs
        if recs:
            summary[m.name] = {
                "paradigm": m.paradigm,
                "n_tables": len(recs),
                "logical_exact_macro":
                    float(np.mean([r["logical_exact"] for r in recs
                                   if r["logical_exact"] is not None]))
                    if any(r["logical_exact"] is not None for r in recs) else None,
                "span_dist_jaccard": float(np.mean([r["span_dist_jaccard"] for r in recs])),
                "grits_top_f1": float(np.mean([r["grits_top_f1"] for r in recs])),
                "grits_con_f1": float(np.mean([r["grits_con_f1"] for r in recs])),
                "grits_top_p":  float(np.mean([r["grits_top_p"]  for r in recs])),
                "grits_top_r":  float(np.mean([r["grits_top_r"]  for r in recs])),
            }

    (RESULTS_DIR / "family_per_table.json").write_text(
        json.dumps(all_recs, indent=2), encoding="utf-8")
    (RESULTS_DIR / "family_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print("DETECTION-FAMILY SUMMARY")
    print("=" * 68)
    for name, s in summary.items():
        print(f"\n{name}  [{s['paradigm']}]  n={s['n_tables']}")
        print(f"  logical-exact(macro) : {s['logical_exact_macro']}")
        print(f"  span-dist Jaccard    : {s['span_dist_jaccard']:.3f}")
        print(f"  GriTS-Top F1         : {s['grits_top_f1']:.3f}")
        print(f"  GriTS-Con F1         : {s['grits_con_f1']:.3f}")

    make_figure(summary)


def make_figure(summary: dict):
    if not summary:
        return
    names = list(summary.keys())
    paradigms = [summary[n]["paradigm"] for n in names]
    gtop = [summary[n]["grits_top_f1"] for n in names]
    gcon = [summary[n]["grits_con_f1"] for n in names]
    log_ex = [summary[n]["logical_exact_macro"] or 0.0 for n in names]
    dist = [summary[n]["span_dist_jaccard"] for n in names]
    colors = ["#4C72B0" if p == "detr" else "#C44E52" if p == "anchor" else "#888888"
              for p in paradigms]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(len(names))
    for ax, vals, title, ylim in [
        (axes[0], gtop, "GriTS-Top F1",       1.0),
        (axes[1], gcon, "GriTS-Con F1",       1.0),
        (axes[2], log_ex, "Logical-Exact (macro)", max(0.05, max(log_ex)*1.3) if log_ex else 0.1),
        (axes[3], dist, "Span-Dist Jaccard",   1.0),
    ]:
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=7)
        ax.set_ylim(0, ylim); ax.set_title(title, fontsize=10)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.suptitle("Detection-Family Comparison (DETR vs anchor)", fontsize=12)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "family_eval.png", dpi=150)
    plt.close()
    print(f"\n[saved] {RESULTS_DIR / 'family_eval.png'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frcnn_ckpt", type=str, default=None,
                    help="optional Faster R-CNN checkpoint path (.pth)")
    args = ap.parse_args()
    run(include_frcnn_ckpt=args.frcnn_ckpt)

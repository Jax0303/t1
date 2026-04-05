#!/usr/bin/env python3
"""
Multi-Paradigm TSR Benchmark — Type-Stratified
===============================================
연구 목적: "detection paradigm은 spanning cell을 구조적으로 처리 못 한다"
           의 근거를 paradigm 비교 + 표 유형 층화 분석으로 확립.

표 유형 (span complexity 기준 — 연구 테마 직접 연결):
  Type-A (Simple)   : spanning cell 0개 — 순수 grid
  Type-B (Header)   : spanning cell 1~3개 — 단순 column header span
  Type-C (Multi)    : spanning cell 4~10개 — 복합 구조
  Type-D (Dense)    : spanning cell 11개 이상 — 고밀도 계층 구조

모델:
  TATR v1.1-all     : DETR 기반 detection paradigm  [현재 SOTA detection]
  Docling-TableFormer: 생성/하이브리드 paradigm       [IBM, VLM계열]
  CascadeTabNet     : Cascade R-CNN 기반 (harness 제공; weights 별도 필요)

데이터셋:
  SciTSR-COMP (716) : 논리적 GT → 완전한 정량 평가
  PubTables-1M (25) : bbox GT XML → span 감지율 평가
  FinTabNet    (25) : bbox GT XML → span 감지율 평가

Metric (연구 테마 aligned):
  GriTS-Top F1 (full)      : 전체 grid 구조 (Smock CVPR'23 표준)
  GriTS-Top F1 (span-only) : spanning cell 만 격리 평가
  Logical-Exact rate       : (sr,er,sc,ec) exact match — GT bbox 의존 없음
  Span-class recall @iou0.1: detection이 span 영역을 최소한 찾는가 (loose)

출력:
  experiments/results_benchmark/
    benchmark_summary.json
    benchmark_per_table.json
    benchmark_stratified.png   ← 유형별 모델 비교 메인 그림
    benchmark_visual.png       ← 유형별 대표 table 시각화
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

os.environ["DOCLING_LOG_LEVEL"] = "ERROR"
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _eval_utils import (
    load_valid_ids, load_gt, load_image, bbox_iou,
    STRUCT_DIR, IMG_DIR, CHUNK_DIR,
)
from grits_eval import grits_factored, sim_top, _span_only_grits_top
from logical_eval import infer_gt_row_col_centers, logical_span_recovery

RESULTS_DIR = Path(__file__).parent / "results_benchmark"
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5
IOU_MERGE   = 0.3

DATA_ROOT = Path(__file__).parent.parent / "data"
PUBTABLES_IMG  = DATA_ROOT / "pubtables-1m" / "images"
PUBTABLES_ANN  = DATA_ROOT / "pubtables-1m" / "structure_annotations_test"
FINTABNET_IMG  = DATA_ROOT / "fintabnet" / "images"
FINTABNET_ANN  = DATA_ROOT / "fintabnet" / "annotations"

# ─────────────────────────────────────────────
# 표 유형 정의
# ─────────────────────────────────────────────
TABLE_TYPES = {
    "A-Simple":  (0,  0),   # spans: 0
    "B-Header":  (1,  3),   # spans: 1-3
    "C-Multi":   (4, 10),   # spans: 4-10
    "D-Dense":   (11, 999), # spans: 11+
}
TYPE_COLORS = {
    "A-Simple": "#4C72B0",
    "B-Header": "#55A868",
    "C-Multi":  "#DD8452",
    "D-Dense":  "#C44E52",
}
SAMPLES_PER_TYPE = 30  # 유형별 최대 샘플 수


def classify_type(n_spans: int) -> str:
    for tname, (lo, hi) in TABLE_TYPES.items():
        if lo <= n_spans <= hi:
            return tname
    return "D-Dense"


# ─────────────────────────────────────────────
# GT 구조 파싱
# ─────────────────────────────────────────────
def parse_xml_gt(xml_path: Path, img_path: Path) -> Optional[dict]:
    """
    PubTables-1M / FinTabNet XML → logical cell grid 재구성.
    XML에는 row/col/spanning cell bbox만 있으므로
    get_spanning_cell_rows_and_columns 방식으로 논리 구조 복원.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objs = root.findall("object")
        rows, cols, spans = [], [], []
        for o in objs:
            name = o.find("name").text.lower()
            bb = o.find("bndbox")
            box = [float(bb.find(t).text)
                   for t in ["xmin", "ymin", "xmax", "ymax"]]
            if name == "table row":
                rows.append(box)
            elif name == "table column":
                cols.append(box)
            elif "spanning" in name or "projected" in name:
                spans.append(box)
        if not rows or not cols:
            return None
        rows.sort(key=lambda b: (b[1]+b[3])/2)
        cols.sort(key=lambda b: (b[0]+b[2])/2)
        n_r, n_c = len(rows), len(cols)

        # 각 spanning cell을 grid 위치로 변환
        cells = []
        for sr, sc in [(r, c) for r in range(n_r) for c in range(n_c)]:
            cells.append({
                "start_row": sr, "end_row": sr,
                "start_col": sc, "end_col": sc,
                "row_span": 1, "col_span": 1, "text": "",
            })

        gt_spans_cells = []
        for sp in spans:
            row_hits = [i for i, rb in enumerate(rows)
                        if _overlap_1d(sp[1], sp[3], rb[1], rb[3]) >= 0.5]
            col_hits = [j for j, cb in enumerate(cols)
                        if _overlap_1d(sp[0], sp[2], cb[0], cb[2]) >= 0.5]
            if row_hits and col_hits:
                gt_spans_cells.append({
                    "start_row": min(row_hits), "end_row": max(row_hits),
                    "start_col": min(col_hits), "end_col": max(col_hits),
                    "row_span": max(row_hits)-min(row_hits)+1,
                    "col_span": max(col_hits)-min(col_hits)+1,
                    "text": "", "bbox": sp,
                })

        img = Image.open(img_path).convert("RGB")
        return {
            "cells": cells + gt_spans_cells,
            "n_rows": n_r, "n_cols": n_c,
            "rows_bbox": rows, "cols_bbox": cols,
            "spans_bbox": spans,
            "image": img,
        }
    except Exception:
        return None


def _overlap_1d(a1, a2, b1, b2) -> float:
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    union = max(a2, b2) - min(a1, b1)
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# 데이터셋 로딩 + 유형 분류
# ─────────────────────────────────────────────
def load_scitsr_typed() -> Dict[str, List[dict]]:
    """SciTSR-COMP 716 tables → type별 분류."""
    typed: Dict[str, List] = {t: [] for t in TABLE_TYPES}
    for tid in load_valid_ids():
        try:
            gt = load_gt(tid)
            img = load_image(tid)
            n_spans = sum(1 for c in gt["cells"]
                          if c["row_span"] > 1 or c["col_span"] > 1)
            ttype = classify_type(n_spans)
            typed[ttype].append({
                "source": "SciTSR", "table_id": tid,
                "gt": gt, "image": img, "n_spans": n_spans,
            })
        except Exception:
            continue
    return typed


def load_xml_dataset_typed(img_dir: Path, ann_dir: Path,
                            source_name: str) -> Dict[str, List[dict]]:
    """PubTables / FinTabNet XML → type별 분류."""
    typed: Dict[str, List] = {t: [] for t in TABLE_TYPES}
    for xml_file in ann_dir.glob("*.xml"):
        try:
            fname = ET.parse(xml_file).getroot().find("filename").text
            img_path = img_dir / fname
            if not img_path.exists():
                continue
            gt = parse_xml_gt(xml_file, img_path)
            if gt is None:
                continue
            n_spans = len([c for c in gt["cells"]
                           if c["row_span"] > 1 or c["col_span"] > 1])
            ttype = classify_type(n_spans)
            typed[ttype].append({
                "source": source_name, "table_id": xml_file.stem,
                "gt": gt, "image": gt.pop("image"), "n_spans": n_spans,
            })
        except Exception:
            continue
    return typed


def merge_typed(*dicts: Dict[str, List]) -> Dict[str, List]:
    out: Dict[str, List] = {t: [] for t in TABLE_TYPES}
    for d in dicts:
        for t, lst in d.items():
            out[t].extend(lst)
    return out


def sample_typed(typed: Dict[str, List], n: int, seed: int = 42) -> Dict[str, List]:
    rng = np.random.default_rng(seed)
    return {t: lst[:n] if len(lst) <= n
            else [lst[i] for i in rng.choice(len(lst), n, replace=False)]
            for t, lst in typed.items()}


# ─────────────────────────────────────────────
# Model Adapters
# ─────────────────────────────────────────────
class ModelAdapter(ABC):
    name: str
    paradigm: str
    color: str

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def infer(self, image: Image.Image) -> dict:
        """Returns dict with 'cells', 'n_rows', 'n_cols', 'spans_bbox'."""
        ...


# --- TATR ---
class TATRAdapter(ModelAdapter):
    paradigm = "detection-DETR"

    def __init__(self, model_id: str, name: str, color: str):
        self.model_id = model_id
        self.name = name
        self.color = color
        self._proc = None
        self._model = None

    def load(self):
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        proc = AutoImageProcessor.from_pretrained(self.model_id)
        if hasattr(proc, "size") and isinstance(proc.size, dict):
            if "longest_edge" in proc.size and "shortest_edge" not in proc.size:
                le = proc.size["longest_edge"]
                proc.size = {"shortest_edge": le, "longest_edge": le}
        self._proc = proc
        self._model = TableTransformerForObjectDetection.from_pretrained(
            self.model_id).to(DEVICE).eval()

    @torch.no_grad()
    def infer(self, image: Image.Image) -> dict:
        enc = self._proc(images=image, return_tensors="pt")
        out = self._model(**{k: v.to(DEVICE) for k, v in enc.items()
                             if k in {"pixel_values", "pixel_mask"}})
        res = self._proc.post_process_object_detection(
            out, threshold=CONF_THRESH,
            target_sizes=[(image.size[1], image.size[0])])[0]
        boxes  = res["boxes"].cpu().numpy()
        labels = res["labels"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        id2label = self._model.config.id2label
        rows, cols, spans = [], [], []
        for box, lid, sc in zip(boxes, labels, scores):
            ln = id2label[lid].lower()
            e  = {"box": box.tolist(), "score": float(sc)}
            if "column" in ln:             cols.append(e)
            elif "spanning" in ln or "projected" in ln: spans.append(e)
            else:                          rows.append(e)
        return _build_cells(rows, cols, spans)


# --- Docling-TableFormer ---
class DoclingAdapter(ModelAdapter):
    name     = "Docling-TableFormer"
    paradigm = "generation-VLM"
    color    = "#9467BD"

    def load(self):
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.IMAGE])

    def infer(self, image: Image.Image) -> dict:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            image.save(tmp)
            result = self._converter.convert(tmp)
            doc    = result.document
            tables = list(doc.tables)
            if not tables:
                return {"cells": [], "n_rows": 0, "n_cols": 0,
                        "spans_bbox": [], "rows_bbox": [], "cols_bbox": []}
            t  = tables[0]
            nr = t.data.num_rows
            nc = t.data.num_cols
            seen, cells = set(), []
            for row_cells in t.data.grid:
                for cell in row_cells:
                    sr = cell.start_row_offset_idx
                    sc = cell.start_col_offset_idx
                    rs = cell.row_span
                    cs = cell.col_span
                    key = (sr, sc)
                    if key in seen:
                        continue
                    seen.add(key)
                    cells.append({
                        "start_row": sr, "end_row": sr + rs - 1,
                        "start_col": sc, "end_col": sc + cs - 1,
                        "row_span": rs, "col_span": cs, "text": "",
                    })
            return {"cells": cells, "n_rows": nr, "n_cols": nc,
                    "spans_bbox": [], "rows_bbox": [], "cols_bbox": []}
        finally:
            os.unlink(tmp)


# --- CascadeTabNet (harness stub) ---
class CascadeTabNetAdapter(ModelAdapter):
    """
    CascadeTabNet (Prasad et al. ICDAR'20) adapter stub.
    Paradigm: Cascade Mask R-CNN (anchor-based detection, NOT DETR).

    To activate:
      1. pip install mmdet==2.x
      2. Download pretrained weights from:
         https://github.com/DevashishPrasad/CascadeTabNet
      3. Set checkpoint_path below and uncomment load() body.

    Why this matters: CascadeTabNet uses anchor-based RPN + cascade heads
    (different from TATR's DETR) → if it also fails at spanning cells,
    the failure is paradigm-level, not DETR-specific.
    """
    name     = "CascadeTabNet"
    paradigm = "detection-anchor"
    color    = "#8C564B"
    SKIP_MSG = ("CascadeTabNet requires mmdetection + pretrained weights. "
                "See adapter docstring. Skipping inference.")

    def load(self):
        print(f"  [SKIP] {self.SKIP_MSG}", flush=True)

    def infer(self, image: Image.Image) -> dict:
        return {"cells": [], "n_rows": 0, "n_cols": 0,
                "spans_bbox": [], "rows_bbox": [], "cols_bbox": [],
                "skipped": True}


def _build_cells(rows, cols, spans) -> dict:
    rows.sort(key=lambda r: (r["box"][1]+r["box"][3])/2)
    cols.sort(key=lambda c: (c["box"][0]+c["box"][2])/2)
    cells = []
    for ri, rd in enumerate(rows):
        for ci, cd in enumerate(cols):
            rb, cb = rd["box"], cd["box"]
            x0, y0 = max(cb[0], rb[0]), max(rb[1], cb[1])
            x1, y1 = min(cb[2], rb[2]), min(rb[3], cb[3])
            if x1 > x0 and y1 > y0:
                cells.append({"start_row": ri, "end_row": ri,
                               "start_col": ci, "end_col": ci,
                               "row_span": 1, "col_span": 1,
                               "bbox": [x0, y0, x1, y1], "text": ""})
    if spans:
        merged_idx, merged = set(), []
        for sd in spans:
            sb = sd["box"]
            ov = [(i, c) for i, c in enumerate(cells)
                  if i not in merged_idx and bbox_iou(sb, c["bbox"]) > IOU_MERGE]
            if ov:
                rs = [c["start_row"] for _, c in ov]
                cs = [c["start_col"] for _, c in ov]
                r0, r1 = min(rs), max(rs); c0, c1 = min(cs), max(cs)
                merged.append({"start_row": r0, "end_row": r1,
                                "start_col": c0, "end_col": c1,
                                "row_span": r1-r0+1, "col_span": c1-c0+1,
                                "bbox": sb, "text": ""})
                for i, _ in ov: merged_idx.add(i)
        cells = merged + [c for i, c in enumerate(cells) if i not in merged_idx]
    return {"cells": cells,
            "n_rows": len(rows), "n_cols": len(cols),
            "spans_bbox": [s["box"] for s in spans],
            "rows_bbox": [r["box"] for r in rows],
            "cols_bbox": [c["box"] for c in cols]}


# ─────────────────────────────────────────────
# Per-table evaluation
# ─────────────────────────────────────────────
def eval_one(item: dict, pred: dict) -> dict:
    gt = item["gt"]
    img = item["image"]
    nr_p, nc_p = pred["n_rows"], pred["n_cols"]
    nr_g, nc_g = gt["n_rows"], gt["n_cols"]

    rec: dict = {
        "table_id": item["table_id"],
        "source":   item["source"],
        "n_spans_gt": item["n_spans"],
        "table_type": classify_type(item["n_spans"]),
    }

    if nr_p == 0 or nc_p == 0 or nr_g == 0 or nc_g == 0:
        rec.update({"grits_top_f1": None, "grits_top_span_only": None,
                    "logical_exact": None, "span_class_recall": None})
        return rec

    _, _, f1 = grits_factored(pred["cells"], gt["cells"],
                               nr_p, nc_p, nr_g, nc_g, sim_top)
    f1_span  = _span_only_grits_top(pred["cells"], gt["cells"])

    # Logical-Exact (SciTSR only: has chunk-based centers)
    logical = None
    if item["source"] == "SciTSR":
        try:
            centers = infer_gt_row_col_centers(gt, item["table_id"], img.size)
            if centers is not None:
                res = logical_span_recovery(gt, pred, centers[0], centers[1])
                logical = res["rate_exact"]
        except Exception:
            pass

    # Span-class recall (loose @iou0.1) — only if GT bboxes available
    span_recall = None
    if "spans_bbox" in gt and gt["spans_bbox"] and pred["spans_bbox"]:
        hits = sum(1 for gb in gt["spans_bbox"]
                   if any(bbox_iou(gb, pb) >= 0.1
                          for pb in pred["spans_bbox"]))
        span_recall = hits / len(gt["spans_bbox"])
    elif "spans_bbox" in gt and gt["spans_bbox"] and not pred["spans_bbox"]:
        span_recall = 0.0

    rec.update({"grits_top_f1": f1, "grits_top_span_only": f1_span,
                "logical_exact": logical, "span_class_recall": span_recall})
    return rec


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
MODELS: List[ModelAdapter] = [
    TATRAdapter("microsoft/table-transformer-structure-recognition-v1.1-all",
                "TATR v1.1-all", "#DD8452"),
    DoclingAdapter(),
    # CascadeTabNetAdapter(),  # uncomment after obtaining weights
]


def run():
    print("=== Loading datasets ===", flush=True)
    scitsr  = load_scitsr_typed()
    pubtab  = load_xml_dataset_typed(PUBTABLES_IMG, PUBTABLES_ANN, "PubTables")
    fintab  = load_xml_dataset_typed(FINTABNET_IMG, FINTABNET_ANN, "FinTabNet")
    all_typed = merge_typed(scitsr, pubtab, fintab)

    print("\nTable distribution (before sampling):")
    for t, lst in all_typed.items():
        src = {}
        for item in lst:
            src[item["source"]] = src.get(item["source"], 0) + 1
        print(f"  {t}: {len(lst)} tables  {src}")

    samples = sample_typed(all_typed, SAMPLES_PER_TYPE)
    print("\nAfter sampling:")
    for t, lst in samples.items():
        print(f"  {t}: {len(lst)} tables")

    # pick representative table per type (for visualization)
    vis_items = {}
    for t, lst in samples.items():
        scitsr_items = [x for x in lst if x["source"] == "SciTSR"]
        vis_items[t] = (scitsr_items + lst)[0] if lst else None

    all_records: Dict[str, Dict[str, List[dict]]] = {}  # model → type → records

    for model in MODELS:
        print(f"\n=== {model.name} ({model.paradigm}) ===", flush=True)
        try:
            model.load()
        except Exception as e:
            print(f"  SKIP (load failed): {e}", flush=True)
            continue

        model_records: Dict[str, List[dict]] = {t: [] for t in TABLE_TYPES}
        for ttype, items in samples.items():
            if not items:
                continue
            for item in tqdm(items, desc=f"  {ttype}", leave=False):
                try:
                    pred = model.infer(item["image"])
                    if pred.get("skipped"):
                        continue
                    rec = eval_one(item, pred)
                    model_records[ttype].append(rec)
                except Exception as e:
                    continue
            n = len(model_records[ttype])
            print(f"    {ttype}: {n} tables evaluated", flush=True)
        all_records[model.name] = model_records

    # ── aggregate ──────────────────────────────────────────────
    summary: Dict[str, dict] = {}
    for mname, type_recs in all_records.items():
        summary[mname] = {}
        for ttype, recs in type_recs.items():
            if not recs:
                summary[mname][ttype] = None
                continue
            def _mean(key):
                vals = [r[key] for r in recs if r.get(key) is not None]
                return float(np.mean(vals)) if vals else None
            summary[mname][ttype] = {
                "n": len(recs),
                "grits_top_f1":       _mean("grits_top_f1"),
                "grits_top_span_only":_mean("grits_top_span_only"),
                "logical_exact":      _mean("logical_exact"),
                "span_class_recall":  _mean("span_class_recall"),
            }

    # ── save ───────────────────────────────────────────────────
    (RESULTS_DIR / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    flat = {m: [r for recs in tr.values() for r in recs]
            for m, tr in all_records.items()}
    (RESULTS_DIR / "benchmark_per_table.json").write_text(
        json.dumps(flat, indent=2), encoding="utf-8")

    _print_summary(summary)
    _make_stratified_figure(summary)
    _make_visual_figure(vis_items, all_records)


def _print_summary(summary: dict):
    print("\n" + "="*72)
    print("PARADIGM BENCHMARK — TYPE-STRATIFIED")
    print("="*72)
    ttypes = list(TABLE_TYPES.keys())
    for mname, ts in summary.items():
        print(f"\n[{mname}]")
        print(f"  {'Type':<14} {'n':>4}  {'GriTS-Top':>10} {'Span-only':>10} {'Logical-Ex':>10} {'Cls-Rec':>8}")
        for t in ttypes:
            s = ts.get(t)
            if s is None:
                print(f"  {t:<14} {'—':>4}  {'—':>10} {'—':>10} {'—':>10} {'—':>8}")
                continue
            def fmt(v): return f"{v:.3f}" if v is not None else "N/A"
            print(f"  {t:<14} {s['n']:>4}  "
                  f"{fmt(s['grits_top_f1']):>10} "
                  f"{fmt(s['grits_top_span_only']):>10} "
                  f"{fmt(s['logical_exact']):>10} "
                  f"{fmt(s['span_class_recall']):>8}")


def _make_stratified_figure(summary: dict):
    """메인 비교 그림: 유형별 × 모델별 4-metric 바 차트."""
    ttypes  = list(TABLE_TYPES.keys())
    mnames  = list(summary.keys())
    metrics = [("grits_top_f1",        "GriTS-Top F1 (full)"),
               ("grits_top_span_only", "GriTS-Top F1 (span-only)"),
               ("logical_exact",       "Logical-Exact Rate"),
               ("span_class_recall",   "Span-Class Recall @IoU≥0.1")]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    x     = np.arange(len(ttypes))
    width = 0.8 / max(len(mnames), 1)
    model_colors = [m.color for m in MODELS if m.name in mnames]

    for ax, (key, title) in zip(axes, metrics):
        for mi, (mname, col) in enumerate(zip(mnames, model_colors)):
            vals = []
            for t in ttypes:
                s = summary[mname].get(t)
                vals.append(s[key] if s and s.get(key) is not None else 0.0)
            offset = (mi - len(mnames)/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width=width, label=mname, color=col, alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, v,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ttypes, rotation=12, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.axhline(0, color="black", linewidth=0.5)
        if mi == 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(
        "Paradigm Benchmark — Detection vs Generation by Table Span Complexity\n"
        "(SciTSR-COMP + PubTables-1M + FinTabNet, stratified sample)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "benchmark_stratified.png", dpi=150)
    plt.close()
    print(f"\n[saved] {RESULTS_DIR / 'benchmark_stratified.png'}")


def _make_visual_figure(vis_items: dict, all_records: dict):
    """유형별 대표 표 이미지에 GT bbox + 모델 예측 overlay."""
    ttypes = [t for t in TABLE_TYPES if vis_items.get(t) is not None]
    n_types = len(ttypes)
    mnames  = list(all_records.keys())
    n_cols  = 1 + len(mnames)   # GT + each model

    fig, axes = plt.subplots(n_types, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_types))
    if n_types == 1:
        axes = [axes]
    col_labels = ["GT (SciTSR)"] + mnames

    for ri, ttype in enumerate(ttypes):
        item = vis_items[ttype]
        img  = item["image"]
        gt   = item["gt"]
        W, H = img.size

        for ci in range(n_cols):
            ax = axes[ri][ci] if n_types > 1 else axes[ci]
            ax.imshow(np.array(img), aspect="auto")
            ax.axis("off")
            if ci == 0:
                # GT cells
                for cell in gt["cells"]:
                    if "bbox" in cell:
                        x0,y0,x1,y1 = cell["bbox"]
                        is_span = cell["row_span"]>1 or cell["col_span"]>1
                        ec = "#FF4444" if is_span else "#4444FF"
                        lw = 2.0 if is_span else 0.8
                        ax.add_patch(mpatches.FancyBboxPatch(
                            (x0, y0), x1-x0, y1-y0,
                            boxstyle="square,pad=0", fill=False,
                            edgecolor=ec, linewidth=lw, alpha=0.85))
                title = f"{ttype}\nGT ({gt['n_rows']}r×{gt['n_cols']}c, {item['n_spans']}sp)"
            else:
                mname = mnames[ci - 1]
                # find this table's pred record
                recs = [r for recs in all_records[mname].values() for r in recs
                        if r["table_id"] == item["table_id"]]
                # rerun inference for drawing
                title = f"{mname}"
                if recs:
                    r = recs[0]
                    g_top = r.get("grits_top_f1")
                    g_sp  = r.get("grits_top_span_only")
                    title += (f"\nGriTS:{g_top:.2f}" if g_top else "\nGriTS:N/A")
                    title += (f" sp:{g_sp:.2f}" if g_sp else " sp:N/A")
            if ri == 0:
                ax.set_title(col_labels[ci], fontsize=9, fontweight="bold")
            ax.text(0.01, 0.99, title, transform=ax.transAxes,
                    fontsize=7, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.suptitle("Representative Tables by Type — GT overlay",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "benchmark_visual.png", dpi=120)
    plt.close()
    print(f"[saved] {RESULTS_DIR / 'benchmark_visual.png'}")


if __name__ == "__main__":
    run()

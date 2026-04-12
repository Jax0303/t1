"""
experiments/_eval_utils.py
==========================
공유 유틸리티 — 데이터 로딩, GriTS 계산, bbox_iou.

데이터셋:
  SciTSR  : test/structure/*.json + test/img/*.png
            공식 split → SciTSR-COMP.list = complex, 나머지 = simple
  FinTabNet: annotations/*.xml + images/*.jpg
            공식 split → 'table spanning cell' 존재 여부 = complex
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_ROOT  = Path(__file__).parent.parent / "data"

SCITSR_DIR   = DATA_ROOT / "SciTSR"
STRUCT_DIR   = SCITSR_DIR / "test" / "structure"
IMG_DIR      = SCITSR_DIR / "test" / "img"
CHUNK_DIR    = SCITSR_DIR / "test" / "chunk"
COMP_LIST    = SCITSR_DIR / "SciTSR-COMP.list"

FINTAB_ANN   = DATA_ROOT / "fintabnet" / "annotations"
FINTAB_IMG   = DATA_ROOT / "fintabnet" / "images"

SPAN_RECOVERY_IOU = 0.5


# ─────────────────────────────────────────────
# SciTSR — ID 로딩
# ─────────────────────────────────────────────
def load_comp_ids() -> List[str]:
    """SciTSR-COMP 공식 복잡 테이블 ID 목록 (716개)."""
    return [l.strip() for l in COMP_LIST.read_text().splitlines() if l.strip()]


def load_all_scitsr_ids() -> List[str]:
    """SciTSR test 전체 ID (구조 파일 기준, 최대 3000개)."""
    return [f.stem for f in sorted(STRUCT_DIR.glob("*.json"))]


def load_valid_ids() -> List[str]:
    """하위 호환: COMP IDs 반환."""
    return load_comp_ids()


# ─────────────────────────────────────────────
# SciTSR — GT 로딩
# ─────────────────────────────────────────────
def load_gt(table_id: str) -> dict:
    """
    SciTSR GT 로딩.
    cells 각각에 start_row/end_row/start_col/end_col/row_span/col_span 보장.
    """
    path = STRUCT_DIR / f"{table_id}.json"
    with open(path) as f:
        raw = json.load(f)

    cells_raw = raw["cells"]
    cells = []
    for c in cells_raw:
        sr = c.get("start_row", 0)
        er = c.get("end_row", sr)
        sc = c.get("start_col", 0)
        ec = c.get("end_col", sc)
        cells.append({
            "start_row": sr, "end_row": er,
            "start_col": sc, "end_col": ec,
            "row_span": er - sr + 1,
            "col_span": ec - sc + 1,
            "text": " ".join(c.get("content", [])),
        })

    n_rows = max(c["end_row"] for c in cells) + 1 if cells else 0
    n_cols = max(c["end_col"] for c in cells) + 1 if cells else 0
    return {"cells": cells, "n_rows": n_rows, "n_cols": n_cols, "table_id": table_id}


def load_image(table_id: str) -> Image.Image:
    path = IMG_DIR / f"{table_id}.png"
    return Image.open(path).convert("RGB")


# ─────────────────────────────────────────────
# SciTSR — split 태깅
# ─────────────────────────────────────────────
def scitsr_split(table_id: str, comp_set: set) -> str:
    """'complex' or 'simple'."""
    return "complex" if table_id in comp_set else "simple"


# ─────────────────────────────────────────────
# FinTabNet — GT 로딩
# ─────────────────────────────────────────────
def load_fintabnet_gt(xml_path: Path) -> Optional[dict]:
    """
    FinTabNet XML → cells (bbox 기반 row/col 인덱스 재구성).
    반환: {cells, n_rows, n_cols, is_complex}
    """
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None

    objects = root.findall("object")
    has_span = any(
        o.findtext("name", "").strip() == "table spanning cell"
        for o in objects
    )

    # row/col/spanning cell bbox 수집
    rows_b, cols_b, spans_b = [], [], []
    for o in objects:
        name = o.findtext("name", "").strip().lower()
        bb = o.find("bndbox")
        if bb is None:
            continue
        box = [
            float(bb.findtext("xmin", 0)),
            float(bb.findtext("ymin", 0)),
            float(bb.findtext("xmax", 0)),
            float(bb.findtext("ymax", 0)),
        ]
        if name == "table row":
            rows_b.append(box)
        elif name == "table column":
            cols_b.append(box)
        elif name == "table spanning cell":
            spans_b.append(box)

    if not rows_b or not cols_b:
        return None

    rows_b.sort(key=lambda b: (b[1] + b[3]) / 2)
    cols_b.sort(key=lambda b: (b[0] + b[2]) / 2)

    # 교차로 기본 셀 생성
    cells = []
    for ri, rb in enumerate(rows_b):
        for ci, cb in enumerate(cols_b):
            x0 = max(cb[0], rb[0]); y0 = max(rb[1], cb[1])
            x1 = min(cb[2], rb[2]); y1 = min(rb[3], cb[3])
            if x1 > x0 and y1 > y0:
                cells.append({
                    "start_row": ri, "end_row": ri,
                    "start_col": ci, "end_col": ci,
                    "row_span": 1, "col_span": 1,
                    "bbox": [x0, y0, x1, y1], "text": "",
                })

    # span merge
    if spans_b:
        merged_idx = set()
        merged = []
        for sb in spans_b:
            ov = [(i, c) for i, c in enumerate(cells)
                  if i not in merged_idx and bbox_iou(sb, c["bbox"]) > 0.3]
            if ov:
                rs = [c["start_row"] for _, c in ov]
                cs = [c["start_col"] for _, c in ov]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                merged.append({
                    "start_row": r0, "end_row": r1,
                    "start_col": c0, "end_col": c1,
                    "row_span": r1 - r0 + 1, "col_span": c1 - c0 + 1,
                    "bbox": sb, "text": "",
                })
                for i, _ in ov:
                    merged_idx.add(i)
        cells = merged + [c for i, c in enumerate(cells) if i not in merged_idx]

    n_rows = len(rows_b)
    n_cols = len(cols_b)
    return {
        "cells": cells, "n_rows": n_rows, "n_cols": n_cols,
        "is_complex": has_span,
        "table_id": xml_path.stem,
    }


def load_fintabnet_image(xml_path: Path) -> Optional[Image.Image]:
    stem = xml_path.stem
    img_path = FINTAB_IMG / f"{stem}.jpg"
    if not img_path.exists():
        return None
    return Image.open(img_path).convert("RGB")


# ─────────────────────────────────────────────
# bbox IoU
# ─────────────────────────────────────────────
def bbox_iou(a: list, b: list) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)

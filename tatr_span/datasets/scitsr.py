"""
SciTSR / SciTSR-COMP Dataset Loader
======================================
TATR 학습용 — bbox 어노테이션을 DETR target 형식으로 변환.

디렉토리 구조:
  data_root/
    images/         *.png
    structure/      *.json   (SciTSR 논리 인덱스 형식)
    SciTSR-COMP.list  (complex table id 목록, 옵션)

SciTSR structure JSON 형식:
  {
    "cells": [
      {
        "id": 0,
        "content": "...",
        "start_row": 0, "end_row": 0,   # inclusive, 0-indexed
        "start_col": 0, "end_col": 0,
        "bbox": [x0, y0, x1, y1]        # absolute pixel
      }, ...
    ],
    "cols": 5,
    "rows": 3
  }

TATR 6-class 매핑 (구조 기반 추론):
  - bbox가 여러 행을 침범하는 셀 → table spanning cell (4)
  - 첫 번째 행 셀                 → table column header (2)
  - 나머지 셀                     → table row로 bbox 역추론

참고: SciTSR는 행/열 bbox가 명시되어 있지 않습니다.
      TATR 학습에는 cell bbox를 행/열 bbox로 집계하거나
      직접 logical-index supervision 방식을 사용합니다.
      여기서는 cell bbox → row/col bbox를 역추론합니다.

Cell → Row/Col Aggregation:
  row r: x0 = min(cell.x0 for cells in row r)
          x1 = max(cell.x1 ...)
          y0 = min(cell.y0 ...), y1 = max(cell.y1 ...)
  col c: analogous
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

STRUCTURE_CLASS_MAP: Dict[str, int] = {
    "table row":                  0,
    "table column":               1,
    "table column header":        2,
    "table projected row header": 3,
    "table spanning cell":        4,
}
SPAN_CLASS_IDX = 4
MAX_SPAN = 10


def _load_structure(json_path: Path) -> Optional[Dict]:
    """SciTSR structure JSON 파싱."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None

    cells = data.get("cells", [])
    n_rows = data.get("rows", 0)
    n_cols = data.get("cols", 0)

    if not cells:
        return None

    # --- Row/Col bbox 역추론 ---
    row_bboxes: Dict[int, List[float]] = {}
    col_bboxes: Dict[int, List[float]] = {}

    for cell in cells:
        bbox = cell.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = map(float, bbox[:4])
        if x1 <= x0 or y1 <= y0:
            continue

        sr, er = cell.get("start_row", 0), cell.get("end_row", 0)
        sc, ec = cell.get("start_col", 0), cell.get("end_col", 0)

        # 행 bbox에 기여 (셀이 span하는 모든 행에)
        for r in range(sr, er + 1):
            if r not in row_bboxes:
                row_bboxes[r] = [x0, y0, x1, y1]
            else:
                b = row_bboxes[r]
                b[0] = min(b[0], x0); b[1] = min(b[1], y0)
                b[2] = max(b[2], x1); b[3] = max(b[3], y1)

        # 열 bbox에 기여
        for c in range(sc, ec + 1):
            if c not in col_bboxes:
                col_bboxes[c] = [x0, y0, x1, y1]
            else:
                b = col_bboxes[c]
                b[0] = min(b[0], x0); b[1] = min(b[1], y0)
                b[2] = max(b[2], x1); b[3] = max(b[3], y1)

    boxes, labels, rowspans, colspans = [], [], [], []

    # Row objects
    for r_idx in sorted(row_bboxes):
        b = row_bboxes[r_idx]
        lbl = STRUCTURE_CLASS_MAP["table column header"] if r_idx == 0 \
              else STRUCTURE_CLASS_MAP["table row"]
        boxes.append(b)
        labels.append(lbl)
        rowspans.append(1)
        colspans.append(1)

    # Column objects
    for c_idx in sorted(col_bboxes):
        b = col_bboxes[c_idx]
        boxes.append(b)
        labels.append(STRUCTURE_CLASS_MAP["table column"])
        rowspans.append(1)
        colspans.append(1)

    # Spanning cell objects
    for cell in cells:
        sr, er = cell.get("start_row", 0), cell.get("end_row", 0)
        sc, ec = cell.get("start_col", 0), cell.get("end_col", 0)
        rs = er - sr + 1
        cs = ec - sc + 1
        if rs == 1 and cs == 1:
            continue  # not spanning

        bbox = cell.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = map(float, bbox[:4])
        if x1 <= x0 or y1 <= y0:
            continue

        boxes.append([x0, y0, x1, y1])
        labels.append(SPAN_CLASS_IDX)
        rowspans.append(max(1, min(MAX_SPAN, rs)))
        colspans.append(max(1, min(MAX_SPAN, cs)))

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "boxes":    boxes,
        "labels":   labels,
        "rowspans": rowspans,
        "colspans": colspans,
    }


class SciTSRDataset(Dataset):
    """SciTSR / SciTSR-COMP 데이터셋.

    Args:
        data_root:    데이터 루트 (images/, structure/ 포함)
        split:        'train' | 'test' (SciTSR 공식 분할 따름)
        transforms:   torchvision 호환 transform
        use_span_attr: True면 target에 rowspans/colspans 포함
        complex_only: True면 spanning cell 포함 테이블만 로드
                      (SciTSR-COMP.list 파일 우선, 없으면 추론)
    """

    IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]

    def __init__(self,
                 data_root: str,
                 split: str = "train",
                 transforms=None,
                 use_span_attr: bool = True,
                 complex_only: bool = False):
        self.data_root    = Path(data_root)
        self.split        = split
        self.transforms   = transforms
        self.use_span_attr = use_span_attr
        self.complex_only  = complex_only

        self.img_dir  = self.data_root / "images"
        self.json_dir = self.data_root / "structure"

        # SciTSR-COMP.list 파일 (complex table ids)
        comp_list_file = self.data_root / "SciTSR-COMP.list"
        comp_ids: Optional[set] = None
        if complex_only and comp_list_file.exists():
            comp_ids = {
                l.strip() for l in comp_list_file.read_text().splitlines()
                if l.strip()
            }

        # 샘플 수집
        json_files = sorted(self.json_dir.glob("*.json"))
        self.samples: List[Dict] = []

        for jf in json_files:
            sid = jf.stem

            if complex_only:
                if comp_ids is not None:
                    if sid not in comp_ids:
                        continue
                # comp_ids 없으면 파싱 후 span 유무로 필터

            ann = _load_structure(jf)
            if ann is None:
                continue

            if complex_only and comp_ids is None:
                if SPAN_CLASS_IDX not in ann["labels"]:
                    continue

            ann["id"] = sid
            self.samples.append(ann)

        print(f"[SciTSR] split={split}, n={len(self.samples)}"
              + (" (complex only)" if complex_only else ""))

    def __len__(self):
        return len(self.samples)

    def _find_image(self, sid: str) -> Optional[Path]:
        for ext in self.IMG_EXTENSIONS:
            p = self.img_dir / f"{sid}{ext}"
            if p.exists():
                return p
        return None

    def __getitem__(self, idx: int):
        ann = self.samples[idx]
        sid = ann["id"]

        img_path = self._find_image(sid)
        if img_path is None:
            # 이미지 없으면 fallback (학습 시 거의 없어야 함)
            img = Image.new("RGB", (800, 600), 128)
            h, w = 600, 800
        else:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

        boxes  = torch.as_tensor(ann["boxes"],    dtype=torch.float32) \
                 if ann["boxes"] else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(ann["labels"],   dtype=torch.long) \
                 if ann["labels"] else torch.zeros((0,), dtype=torch.long)

        target: dict = {
            "image_id":  torch.tensor([idx]),
            "boxes":     boxes,
            "labels":    labels,
            "orig_size": torch.as_tensor([h, w]),
            "size":      torch.as_tensor([h, w]),
        }

        if self.use_span_attr:
            rs = torch.as_tensor(ann["rowspans"], dtype=torch.long) \
                 if ann["rowspans"] else torch.zeros((0,), dtype=torch.long)
            cs = torch.as_tensor(ann["colspans"], dtype=torch.long) \
                 if ann["colspans"] else torch.zeros((0,), dtype=torch.long)
            target["rowspans"] = rs
            target["colspans"] = cs

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):
        imgs, targets = zip(*batch)
        return list(imgs), list(targets)

    def get_height_and_width(self, idx: int) -> Tuple[int, int]:
        ann = self.samples[idx]
        img_path = self._find_image(ann["id"])
        if img_path is None:
            return 600, 800
        with Image.open(img_path) as img:
            return img.height, img.width

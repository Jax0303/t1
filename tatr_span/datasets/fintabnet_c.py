"""
FinTabNet.c Dataset Loader
===========================
HuggingFace `bsmock/FinTabNet.c` — canonical corrected version.

데이터 형식: Parquet (HuggingFace datasets)
  split: 'train' | 'validation' | 'test'

컬럼:
  - image:        PIL image (bytes)
  - bbox:         [[x0,y0,x1,y1], ...] absolute pixel coords
  - label:        [str, ...]  e.g. "table row", "table column header", ...
  - rowspan:      [int, ...]  (spanning cells only, else 1)
  - colspan:      [int, ...]

TATR 6-class 레이블 (PubTables-1M 동일):
  0: table row
  1: table column
  2: table column header
  3: table projected row header
  4: table spanning cell
  5: no object (DETR eos, 학습 타깃 아님)
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional

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

# HuggingFace dataset id
HF_DATASET_ID = "bsmock/FinTabNet.c"


def _load_hf_dataset(split: str, cache_dir: Optional[str] = None):
    """HuggingFace datasets 로드 (lazy import)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets 패키지 필요: pip install datasets"
        ) from e
    return load_dataset(HF_DATASET_ID, split=split, cache_dir=cache_dir)


class FinTabNetCDataset(Dataset):
    """FinTabNet.c HuggingFace 데이터셋 래퍼.

    Args:
        split:        'train' | 'validation' | 'test'
        cache_dir:    HF 캐시 경로 (None → 기본값 ~/.cache/huggingface)
        transforms:   torchvision 호환 (image, target) → (image, target)
        use_span_attr: True면 target에 rowspans/colspans 포함
        complex_only:  True면 spanning cell 포함 샘플만 유지
        label_col:    레이블 컬럼명 (기본 'label')
        bbox_col:     bbox 컬럼명 (기본 'bbox')
        image_col:    이미지 컬럼명 (기본 'image')
        rowspan_col:  rowspan 컬럼명 (기본 'rowspan', 없으면 1)
        colspan_col:  colspan 컬럼명 (기본 'colspan', 없으면 1)
    """

    def __init__(self,
                 split: str = "train",
                 cache_dir: Optional[str] = None,
                 transforms=None,
                 use_span_attr: bool = True,
                 complex_only: bool = False,
                 label_col: str = "label",
                 bbox_col: str = "bbox",
                 image_col: str = "image",
                 rowspan_col: str = "rowspan",
                 colspan_col: str = "colspan"):
        self.split        = split
        self.transforms   = transforms
        self.use_span_attr = use_span_attr
        self.complex_only  = complex_only
        self.label_col    = label_col
        self.bbox_col     = bbox_col
        self.image_col    = image_col
        self.rowspan_col  = rowspan_col
        self.colspan_col  = colspan_col

        raw = _load_hf_dataset(split, cache_dir)
        self._columns = set(raw.column_names)

        # 필터링
        if complex_only:
            def _has_span(ex):
                labels = ex[label_col]
                return any(l == "table spanning cell" for l in labels)
            raw = raw.filter(_has_span)

        self._ds = raw
        print(f"[FinTabNetC] split={split}, n={len(self._ds)}"
              + (" (complex only)" if complex_only else ""))

    def __len__(self):
        return len(self._ds)

    def _parse_example(self, ex: dict, idx: int):
        """HF example → (PIL.Image, target dict)."""
        # 이미지
        img_raw = ex[self.image_col]
        if isinstance(img_raw, dict) and "bytes" in img_raw:
            img = Image.open(io.BytesIO(img_raw["bytes"])).convert("RGB")
        elif isinstance(img_raw, bytes):
            img = Image.open(io.BytesIO(img_raw)).convert("RGB")
        elif isinstance(img_raw, Image.Image):
            img = img_raw.convert("RGB")
        else:
            img = Image.fromarray(img_raw).convert("RGB")

        w, h = img.size

        raw_labels: List[str] = ex[self.label_col]
        raw_boxes:  List     = ex[self.bbox_col]

        boxes, labels, rowspans, colspans = [], [], [], []
        has_rowspan = self.rowspan_col in self._columns
        has_colspan = self.colspan_col in self._columns
        raw_rs = ex.get(self.rowspan_col, None) if has_rowspan else None
        raw_cs = ex.get(self.colspan_col, None) if has_colspan else None

        for i, (lbl, bb) in enumerate(zip(raw_labels, raw_boxes)):
            class_idx = STRUCTURE_CLASS_MAP.get(lbl)
            if class_idx is None:
                continue

            # bbox: list [x0,y0,x1,y1] or dict
            if isinstance(bb, dict):
                x0, y0 = bb.get("xmin", bb.get("x0", 0)), bb.get("ymin", bb.get("y0", 0))
                x1, y1 = bb.get("xmax", bb.get("x1", 0)), bb.get("ymax", bb.get("y1", 0))
            else:
                x0, y0, x1, y1 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])

            if x1 <= x0 or y1 <= y0:
                continue

            rs = 1
            cs = 1
            if class_idx == SPAN_CLASS_IDX:
                if raw_rs is not None and i < len(raw_rs):
                    try:
                        rs = max(1, min(MAX_SPAN, int(raw_rs[i])))
                    except (TypeError, ValueError):
                        rs = 1
                if raw_cs is not None and i < len(raw_cs):
                    try:
                        cs = max(1, min(MAX_SPAN, int(raw_cs[i])))
                    except (TypeError, ValueError):
                        cs = 1

            boxes.append([x0, y0, x1, y1])
            labels.append(class_idx)
            rowspans.append(rs)
            colspans.append(cs)

        if not boxes:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.long)
            rs_t     = torch.zeros((0,),   dtype=torch.long)
            cs_t     = torch.zeros((0,),   dtype=torch.long)
        else:
            boxes_t  = torch.as_tensor(boxes,    dtype=torch.float32)
            labels_t = torch.as_tensor(labels,   dtype=torch.long)
            rs_t     = torch.as_tensor(rowspans, dtype=torch.long)
            cs_t     = torch.as_tensor(colspans, dtype=torch.long)

        target: dict = {
            "image_id":  torch.tensor([idx]),
            "boxes":     boxes_t,
            "labels":    labels_t,
            "orig_size": torch.as_tensor([h, w]),
            "size":      torch.as_tensor([h, w]),
        }
        if self.use_span_attr:
            target["rowspans"] = rs_t
            target["colspans"] = cs_t

        return img, target

    def __getitem__(self, idx: int):
        ex = self._ds[idx]
        img, target = self._parse_example(ex, idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):
        imgs, targets = zip(*batch)
        return list(imgs), list(targets)

    def get_height_and_width(self, idx: int):
        ex = self._ds[idx]
        img_raw = ex[self.image_col]
        if isinstance(img_raw, dict) and "bytes" in img_raw:
            img = Image.open(io.BytesIO(img_raw["bytes"]))
        elif isinstance(img_raw, bytes):
            img = Image.open(io.BytesIO(img_raw))
        elif isinstance(img_raw, Image.Image):
            img = img_raw
        else:
            img = Image.fromarray(img_raw)
        return img.height, img.width

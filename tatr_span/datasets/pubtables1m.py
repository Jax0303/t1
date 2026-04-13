"""
PubTables-1M Dataset Loader
==============================
공식 PASCAL VOC XML 어노테이션 형식 파싱.

디렉토리 구조:
  data_root/
    images/         *.png  (또는 *.jpg)
    xml/            *.xml  (PASCAL VOC)
    splits/
      train.txt
      val.txt
      test.txt

TATR 공식 6-class 레이블:
  0: table row
  1: table column
  2: table column header
  3: table projected row header
  4: table spanning cell   ← rowspan/colspan 속성 포함
  5: no object (배경, DETR eos)

spanning cell XML 속성 예시:
  <object>
    <name>table spanning cell</name>
    <attributes>
      <attribute><name>rowspan</name><value>2</value></attribute>
      <attribute><name>colspan</name><value>1</value></attribute>
    </attributes>
    <bndbox>...</bndbox>
  </object>
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

# TATR 공식 구조 인식 클래스 매핑
STRUCTURE_CLASS_MAP: Dict[str, int] = {
    "table row":                0,
    "table column":             1,
    "table column header":      2,
    "table projected row header": 3,
    "table spanning cell":      4,
}
SPAN_CLASS_IDX = 4
MAX_SPAN = 10


def parse_xml(xml_path: Path) -> Optional[Dict]:
    """PASCAL VOC XML → 어노테이션 dict.

    Returns:
        {
          'filename': str,
          'width': int, 'height': int,
          'boxes':    [[x0,y0,x1,y1], ...],   # absolute pixel
          'labels':   [int, ...],
          'rowspans': [int, ...],   # 1 if not spanning cell
          'colspans': [int, ...],
        }
        None on parse error.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return None

    size = root.find("size")
    w = int(size.findtext("width",  "0"))
    h = int(size.findtext("height", "0"))
    filename = root.findtext("filename", xml_path.stem)

    boxes, labels, rowspans, colspans = [], [], [], []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in STRUCTURE_CLASS_MAP:
            continue
        label = STRUCTURE_CLASS_MAP[name]

        bb = obj.find("bndbox")
        if bb is None:
            continue
        x0 = float(bb.findtext("xmin", "0"))
        y0 = float(bb.findtext("ymin", "0"))
        x1 = float(bb.findtext("xmax", "0"))
        y1 = float(bb.findtext("ymax", "0"))
        if x1 <= x0 or y1 <= y0:
            continue

        rs, cs = 1, 1
        if label == SPAN_CLASS_IDX:
            # <attributes> → rowspan/colspan 파싱
            attrs = obj.find("attributes")
            if attrs is not None:
                for attr in attrs.findall("attribute"):
                    aname = attr.findtext("name", "").strip()
                    aval  = attr.findtext("value", "1").strip()
                    try:
                        val = max(1, min(MAX_SPAN, int(aval)))
                    except ValueError:
                        val = 1
                    if aname == "rowspan":
                        rs = val
                    elif aname == "colspan":
                        cs = val

        boxes.append([x0, y0, x1, y1])
        labels.append(label)
        rowspans.append(rs)
        colspans.append(cs)

    return {
        "filename": filename,
        "width": w, "height": h,
        "boxes":    boxes,
        "labels":   labels,
        "rowspans": rowspans,
        "colspans": colspans,
    }


class PubTables1MDataset(Dataset):
    """PubTables-1M 구조 인식 데이터셋.

    Args:
        data_root:   데이터 루트 경로 (images/, xml/, splits/ 포함)
        split:       'train' | 'val' | 'test'
        transforms:  torchvision 호환 transform (image, target) → (image, target)
        max_size:    이미지 장축 최대 크기
        use_span_attr: True면 target에 rowspans/colspans 포함
        complex_only: True면 spanning cell 포함 테이블만 로드
    """

    IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]

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

        self.img_dir = self.data_root / "images"
        self.xml_dir = self.data_root / "xml"

        # split 파일 로드
        split_file = self.data_root / "splits" / f"{split}.txt"
        if split_file.exists():
            ids = [l.strip() for l in split_file.read_text().splitlines()
                   if l.strip()]
        else:
            # splits/ 없으면 xml dir 전체 사용
            ids = [f.stem for f in sorted(self.xml_dir.glob("*.xml"))]

        # 어노테이션 파싱 + 필터
        self.samples: List[Dict] = []
        for sid in ids:
            xml_path = self.xml_dir / f"{sid}.xml"
            if not xml_path.exists():
                continue
            ann = parse_xml(xml_path)
            if ann is None:
                continue
            if complex_only and SPAN_CLASS_IDX not in ann["labels"]:
                continue
            ann["id"] = sid
            self.samples.append(ann)

        print(f"[PubTables1M] split={split}, n={len(self.samples)}"
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
        ann  = self.samples[idx]
        sid  = ann["id"]
        img_path = self._find_image(sid)

        if img_path is None:
            # 이미지가 없으면 빈 RGB 반환 (드문 경우)
            img = Image.new("RGB", (ann["width"], ann["height"]), 128)
        else:
            img = Image.open(img_path).convert("RGB")

        w, h = img.size
        boxes  = torch.as_tensor(ann["boxes"],    dtype=torch.float32)
        labels = torch.as_tensor(ann["labels"],   dtype=torch.long)

        target: Dict = {
            "image_id": torch.tensor([idx]),
            "boxes":    boxes,          # [N, 4] xyxy absolute
            "labels":   labels,         # [N]
            "orig_size": torch.as_tensor([h, w]),
            "size":      torch.as_tensor([h, w]),
        }

        if self.use_span_attr:
            target["rowspans"] = torch.as_tensor(ann["rowspans"], dtype=torch.long)
            target["colspans"] = torch.as_tensor(ann["colspans"], dtype=torch.long)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):
        """DataLoader collate: 이미지 리스트 + target 리스트 반환."""
        imgs, targets = zip(*batch)
        return list(imgs), list(targets)

    # TATR 공식 eval 호환 메서드
    def get_height_and_width(self, idx: int) -> Tuple[int, int]:
        ann = self.samples[idx]
        return ann["height"], ann["width"]

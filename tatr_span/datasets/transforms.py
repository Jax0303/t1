"""
Data Augmentation Transforms
==============================
DETR/TATR 호환 transform — (PIL.Image, target) → (PIL.Image | Tensor, target).

target dict keys:
  boxes    [N, 4] xyxy absolute → xyxy absolute (resize 후 스케일 조정)
  labels   [N]
  rowspans [N]  (있으면 통과)
  colspans [N]  (있으면 통과)
  orig_size [H, W]
  size      [H, W]  transform 이후 실제 크기
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _clip_boxes(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """bbox를 이미지 영역 내로 클리핑."""
    boxes = boxes.clone()
    boxes[:, 0].clamp_(0, w)
    boxes[:, 1].clamp_(0, h)
    boxes[:, 2].clamp_(0, w)
    boxes[:, 3].clamp_(0, h)
    return boxes


def _filter_boxes(
    target: dict, h: int, w: int, min_area: float = 1.0
) -> dict:
    """크기 0 박스 제거."""
    boxes = target.get("boxes")
    if boxes is None or len(boxes) == 0:
        return target

    boxes = _clip_boxes(boxes, h, w)
    keep = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) >= min_area

    new_target = {k: v for k, v in target.items()}
    for key in ["boxes", "labels", "rowspans", "colspans"]:
        if key in new_target and len(new_target[key]) > 0:
            new_target[key] = new_target[key][keep]
    new_target["boxes"] = boxes[keep]
    return new_target


# ──────────────────────────────────────────────────────────────
# Transform Classes
# ──────────────────────────────────────────────────────────────

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image, target: dict):
        if random.random() < self.p:
            img = F.hflip(img)
            w, h = img.size
            boxes = target.get("boxes")
            if boxes is not None and len(boxes):
                flipped = boxes.clone()
                flipped[:, 0] = w - boxes[:, 2]
                flipped[:, 2] = w - boxes[:, 0]
                target = {**target, "boxes": flipped}
        return img, target


class RandomResize:
    """단축(shortest edge)을 sizes 중 하나로 랜덤 리사이즈.
    max_size: 장축 최대 크기 (None이면 무제한).
    """

    def __init__(self, sizes: List[int], max_size: Optional[int] = None):
        self.sizes    = sizes
        self.max_size = max_size

    def __call__(self, img: Image.Image, target: dict):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class Resize:
    def __init__(self, size: int, max_size: Optional[int] = None):
        self.size     = size
        self.max_size = max_size

    def __call__(self, img: Image.Image, target: dict):
        return resize(img, target, self.size, self.max_size)


class ToTensor:
    def __call__(self, img: Image.Image, target: dict):
        return F.to_tensor(img), target


class Normalize:
    def __init__(self,
                 mean: List[float] = (0.485, 0.456, 0.406),
                 std:  List[float] = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std  = std

    def __call__(self, img: torch.Tensor, target: dict):
        img = F.normalize(img, mean=self.mean, std=self.std)

        h, w = img.shape[-2], img.shape[-1]
        target = {**target}

        # boxes를 [0,1] 정규화 cxcywh 로 변환 (DETR 학습 형식)
        boxes = target.get("boxes")
        if boxes is not None and len(boxes):
            boxes = xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        return img, target


class RandomSelect:
    """두 transform 중 p 확률로 첫 번째 선택."""

    def __init__(self, t1, t2, p: float = 0.5):
        self.t1, self.t2, self.p = t1, t2, p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.t1(img, target)
        return self.t2(img, target)


class RandomSizeCrop:
    """DETR 스타일 랜덤 크롭."""

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: Image.Image, target: dict):
        w, h = img.size
        crop_w = random.randint(self.min_size, min(w, self.max_size))
        crop_h = random.randint(self.min_size, min(h, self.max_size))
        region = T.RandomCrop.get_params(img, (crop_h, crop_w))
        return crop(img, target, region)


# ──────────────────────────────────────────────────────────────
# Functional helpers
# ──────────────────────────────────────────────────────────────

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    bw = x1 - x0
    bh = y1 - y0
    return torch.stack([cx, cy, bw, bh], dim=-1)


def _get_size(orig_w: int, orig_h: int,
              target_size: int,
              max_size: Optional[int]) -> Tuple[int, int]:
    """단축=target_size, 장축≤max_size 가 되는 (new_w, new_h) 계산."""
    min_orig = min(orig_w, orig_h)
    max_orig = max(orig_w, orig_h)
    scale = target_size / min_orig
    if max_size is not None and scale * max_orig > max_size:
        scale = max_size / max_orig
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    return new_w, new_h


def resize(img: Image.Image, target: dict,
           size: int, max_size: Optional[int] = None):
    orig_w, orig_h = img.size
    new_w, new_h = _get_size(orig_w, orig_h, size, max_size)

    img = F.resize(img, (new_h, new_w))

    # boxes 스케일
    target = {**target}
    boxes = target.get("boxes")
    if boxes is not None and len(boxes):
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        target["boxes"] = boxes

    target["size"] = torch.as_tensor([new_h, new_w])
    return img, target


def crop(img: Image.Image, target: dict,
         region: Tuple[int, int, int, int]):
    """(top, left, height, width) 크롭."""
    top, left, ht, wt = region
    img = F.crop(img, top, left, ht, wt)

    target = {**target}
    target["size"] = torch.as_tensor([ht, wt])

    boxes = target.get("boxes")
    if boxes is not None and len(boxes):
        boxes = boxes - torch.tensor([left, top, left, top], dtype=torch.float32)
        target["boxes"] = boxes
        target = _filter_boxes(target, ht, wt)

    return img, target


# ──────────────────────────────────────────────────────────────
# 표준 transform 팩토리
# ──────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def make_train_transforms(min_size: int = 800, max_size: int = 1333):
    """TATR 학습용 표준 augmentation."""
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return Compose([
        RandomHorizontalFlip(),
        RandomSelect(
            RandomResize(scales, max_size=max_size),
            Compose([
                RandomResize([400, 500, 600]),
                RandomSizeCrop(384, 600),
                RandomResize(scales, max_size=max_size),
            ]),
        ),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_val_transforms(min_size: int = 800, max_size: int = 1333):
    """평가/추론용 transform (augmentation 없음)."""
    return Compose([
        Resize(min_size, max_size=max_size),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

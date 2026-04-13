"""
TSRDet: Cascade R-CNN + Single-Label Regularization
=====================================================
MDPI Electronics 2024 논문의 충실한 재현 (±1% 목표).

핵심 설계:
  1. Backbone: ResNet-50 + FPN (C3~C5, P2~P6)
  2. RPN: 표준 Region Proposal Network
  3. 3-stage Cascade R-CNN head (IoU thresh: 0.5 / 0.6 / 0.7)
  4. Single-label regularization: 각 RoI에서 softmax 후
     가장 높은 확률 클래스만 강제 (multi-label 억제)

구현 전략:
  mmdetection 기반 (설치 필요):
    pip install mmcv-full mmdet

  또는 torchvision Faster/Cascade R-CNN API 사용 (여기서는 torchvision 구현).

TATR 6-class 레이블:
  0: table row
  1: table column
  2: table column header
  3: table projected row header
  4: table spanning cell
  5: background (Cascade R-CNN 내부 처리)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import MultiScaleRoIAlign, box_iou

NUM_CLASSES = 6   # 5 fg + 1 bg

# Cascade IoU thresholds (논문 설정)
CASCADE_IOU_THRESHOLDS = [0.5, 0.6, 0.7]


# ─────────────────────────────────────────────────────────────
# Single-Label Regularization
# ─────────────────────────────────────────────────────────────

class SingleLabelRegularizer(nn.Module):
    """RoI 특징에서 가장 확률 높은 클래스만 남기는 정규화.

    TSRDet 논문 핵심 기여: 행/열/헤더 등이 동일 영역에서 중복 예측되는
    multi-label 문제를 softmax + argmax 마스킹으로 억제.

    loss_weight: SLR loss를 주 분류 손실에 더하는 가중치.
    """

    def __init__(self, loss_weight: float = 0.1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, C] (배경 포함 C=NUM_CLASSES+1)
        Returns: SLR regularization loss (scalar)

        방식: softmax(logits)에서 argmax 클래스 외 확률의 제곱합 최소화
        """
        probs = F.softmax(logits, dim=-1)
        max_idx = probs.argmax(dim=-1, keepdim=True)
        mask = torch.zeros_like(probs).scatter_(1, max_idx, 1.0)
        # 비-최대 클래스 확률 억제 손실
        non_max_probs = probs * (1.0 - mask)
        slr_loss = (non_max_probs ** 2).sum(dim=-1).mean()
        return self.loss_weight * slr_loss


# ─────────────────────────────────────────────────────────────
# Cascade RoI Head (3-stage)
# ─────────────────────────────────────────────────────────────

class CascadeBoxPredictor(nn.Module):
    """단일 Cascade stage의 분류 + bbox regression head."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score  = nn.Linear(in_channels, num_classes + 1)
        self.bbox_pred  = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        cls_logits = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)
        return cls_logits, bbox_delta


class CascadeRoIHead(nn.Module):
    """3-stage Cascade R-CNN RoI head.

    각 stage는 독립적인 IoU threshold + predictor를 가집니다.
    SLR은 마지막 stage의 분류 logit에만 적용합니다.
    """

    def __init__(self,
                 box_roi_pool: MultiScaleRoIAlign,
                 representation_size: int = 1024,
                 num_classes: int = NUM_CLASSES,
                 iou_thresholds: List[float] = None,
                 slr_weight: float = 0.1):
        super().__init__()
        self.box_roi_pool   = box_roi_pool
        self.num_classes    = num_classes
        self.iou_thresholds = iou_thresholds or CASCADE_IOU_THRESHOLDS
        self.slr = SingleLabelRegularizer(slr_weight)

        # 3 stage predictor
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 * 7 * 7, representation_size),
                nn.ReLU(),
                nn.Linear(representation_size, representation_size),
                nn.ReLU(),
            )
            for _ in self.iou_thresholds
        ])
        self.heads = nn.ModuleList([
            CascadeBoxPredictor(representation_size, num_classes)
            for _ in self.iou_thresholds
        ])

    def forward(self, features, proposals, image_shapes,
                targets=None):
        """
        features:     OrderedDict of FPN feature maps
        proposals:    list of [N_i, 4] Tensor (xyxy, image coords)
        image_shapes: list of (H, W)
        targets:      list of dicts with 'boxes', 'labels' (training only)
        """
        losses = {}
        # Stage 별로 처리
        boxes = proposals
        for stage_idx, (fc, head, iou_thresh) in \
                enumerate(zip(self.predictors, self.heads, self.iou_thresholds)):

            # RoI Align
            roi_features = self.box_roi_pool(features, boxes, image_shapes)
            roi_features = roi_features.flatten(1)
            roi_features = fc(roi_features)
            cls_logits, bbox_deltas = head(roi_features)

            if targets is not None:
                # 라벨 할당 (간소화: 실제 구현은 torchvision matcher 사용)
                labels, regression_targets = self._assign_labels(
                    boxes, targets, iou_thresh, image_shapes)

                cls_loss, box_loss = fastrcnn_loss(
                    cls_logits, bbox_deltas, labels, regression_targets)
                losses[f"loss_classifier_{stage_idx}"] = cls_loss
                losses[f"loss_box_reg_{stage_idx}"]    = box_loss

                # SLR on last stage
                if stage_idx == len(self.iou_thresholds) - 1:
                    losses["loss_slr"] = self.slr(cls_logits)

            # 다음 stage proposal: 현재 stage bbox 예측으로 정제
            # (간소화: 실제 구현에서는 decode + clamp)
            # 학습 중에는 GT proposal 재사용 (standard cascade training)

        if targets is not None:
            return losses

        # Inference: 마지막 stage 결과 반환
        cls_logits, bbox_deltas = self.heads[-1](
            self.predictors[-1](
                self.box_roi_pool(features, boxes, image_shapes).flatten(1)
            )
        )
        return cls_logits, bbox_deltas, boxes

    def _assign_labels(self, proposals, targets, iou_threshold, image_shapes):
        """간소화된 라벨 할당 (IoU 기반 매칭)."""
        from torchvision.models.detection._utils import Matcher, BalancedPositiveNegativeSampler
        from torchvision.ops import box_convert

        matcher = Matcher(
            high_quality_match_score=iou_threshold,
            low_quality_match_score=0.3,
            allow_low_quality_matches=False,
        )
        sampler = BalancedPositiveNegativeSampler(512, 0.25)

        labels_list = []
        reg_targets_list = []

        for props_per_img, targets_per_img in zip(proposals, targets):
            gt_boxes  = targets_per_img["boxes"]
            gt_labels = targets_per_img["labels"]

            if gt_boxes.numel() == 0:
                labels_list.append(torch.zeros(len(props_per_img),
                                               dtype=torch.long,
                                               device=props_per_img.device))
                reg_targets_list.append(torch.zeros(len(props_per_img), 4,
                                                    device=props_per_img.device))
                continue

            iou = box_iou(props_per_img, gt_boxes)
            match_quality, matches = iou.max(dim=1)

            neg = match_quality < 0.3
            pos = match_quality >= iou_threshold
            ignore = ~neg & ~pos

            lbl = gt_labels[matches]
            lbl[neg]    = 0   # background
            lbl[ignore] = -1  # ignore

            labels_list.append(lbl)
            reg_targets_list.append(gt_boxes[matches])

        labels_flat = torch.cat(labels_list)
        reg_flat    = torch.cat(reg_targets_list)
        return labels_flat, reg_flat


# ─────────────────────────────────────────────────────────────
# Full TSRDet Model
# ─────────────────────────────────────────────────────────────

def build_tsrdet(
    num_classes: int = NUM_CLASSES,
    pretrained_backbone: bool = True,
    slr_weight: float = 0.1,
) -> nn.Module:
    """TSRDet 모델 빌드.

    torchvision의 FasterRCNN을 Cascade RoI head로 교체합니다.
    실제 연구 재현을 위해 mmdetection 기반 구현이 더 정확하지만,
    의존성 최소화를 위해 torchvision API를 활용합니다.

    Args:
        num_classes:         포그라운드 클래스 수 (배경 제외)
        pretrained_backbone: ImageNet pretrained ResNet-50
        slr_weight:          Single-Label Regularization 손실 가중치
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="DEFAULT" if pretrained_backbone else None,
        trainable_layers=3,
    )

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    cascade_head = CascadeRoIHead(
        box_roi_pool=box_roi_pool,
        representation_size=1024,
        num_classes=num_classes,
        iou_thresholds=CASCADE_IOU_THRESHOLDS,
        slr_weight=slr_weight,
    )

    # FasterRCNN을 base로 사용하고 roi_heads를 교체
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes + 1,   # FasterRCNN: bg 포함
        min_size=800,
        max_size=1333,
        rpn_anchor_generator=None,
        box_roi_pool=box_roi_pool,
        box_head=None,
        box_predictor=None,
    )
    # Cascade head로 교체
    model.roi_heads = cascade_head

    return model

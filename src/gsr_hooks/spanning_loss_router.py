"""
SpanningCellLossRouter
======================
BBox regression loss를 spanning cell / non-spanning cell 두 그룹으로
분기하는 라우터 모듈.

설계 원칙
---------
- 이 클래스만 교체하면 loss 전략이 바뀜 (head 구조는 건드리지 않음).
- baseline config  : both_loss = SmoothL1Loss  →  vanilla Cascade R-CNN 재현
- GSR config       : spanning_loss = GIoU/DIoU →  spanning cell 전용 loss 주입

PubTables-1M structure class 인덱스 (0-based)
---------------------------------------------
  0 : table column
  1 : table row
  2 : table column header
  3 : table projected row header
  4 : table spanning cell          ← spanning_class_ids 기본값
"""

from __future__ import annotations
from typing import Optional, Set

import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor


@MODELS.register_module()
class SpanningCellLossRouter(nn.Module):
    """Regression loss를 class label 기준으로 두 그룹에 분기한다.

    Args:
        spanning_class_ids (list[int]): spanning cell 취급할 class index 목록.
        span_loss (dict): spanning cell용 loss config.
            decoded bbox를 입력으로 받는 IoU 계열 권장
            (GIoULoss / DIoULoss / CIoULoss).
        nonspan_loss (dict): non-spanning cell용 loss config.
            기본 SmoothL1Loss.
        span_loss_weight (float): spanning loss 최종 스케일. 두 그룹 간
            샘플 수 불균형을 보정할 때 사용.
    """

    def __init__(
        self,
        spanning_class_ids: list[int],
        span_loss: dict = dict(type='GIoULoss', loss_weight=1.0),
        nonspan_loss: dict = dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        span_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.spanning_class_ids: Set[int] = set(spanning_class_ids)
        self.span_loss_weight = span_loss_weight

        self.span_loss_fn = MODELS.build(span_loss)
        self.nonspan_loss_fn = MODELS.build(nonspan_loss)

        # IoU 계열 loss는 decoded bbox를 요구함 → 상위 head에 알림
        self.requires_decoded = span_loss.get('type', '') not in (
            'SmoothL1Loss', 'L1Loss', 'MSELoss'
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        bbox_pred: torch.Tensor,      # (N, 4)  delta 또는 decoded box
        bbox_targets: torch.Tensor,   # (N, 4)
        bbox_weights: torch.Tensor,   # (N, 4)
        labels: torch.Tensor,         # (N,)  positive sample의 GT class
        avg_factor: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
              'loss_bbox'          – non-spanning regression loss
              'loss_bbox_spanning' – spanning cell regression loss
        """
        device = bbox_pred.device
        span_mask = torch.tensor(
            [l.item() in self.spanning_class_ids for l in labels],
            dtype=torch.bool, device=device,
        )
        nonspan_mask = ~span_mask

        losses: dict[str, torch.Tensor] = {}

        # ── non-spanning ────────────────────────────────────────────────
        if nonspan_mask.any():
            losses['loss_bbox'] = self.nonspan_loss_fn(
                bbox_pred[nonspan_mask],
                bbox_targets[nonspan_mask],
                bbox_weights[nonspan_mask],
                avg_factor=avg_factor,
            )
        else:
            losses['loss_bbox'] = bbox_pred.sum() * 0.0  # 그래디언트 유지

        # ── spanning ────────────────────────────────────────────────────
        if span_mask.any():
            span_weight = bbox_weights[span_mask].max(dim=-1).values  # (M,)
            losses['loss_bbox_spanning'] = (
                self.span_loss_fn(
                    bbox_pred[span_mask],
                    bbox_targets[span_mask],
                    weight=span_weight,
                    avg_factor=span_mask.float().sum().item(),
                )
                * self.span_loss_weight
            )
        else:
            losses['loss_bbox_spanning'] = bbox_pred.sum() * 0.0

        return losses

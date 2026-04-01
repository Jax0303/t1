"""
SpanningAwareCascadeBBoxHead
============================
Cascade R-CNN의 Shared2FCBBoxHead를 최소한으로 override하여
SpanningCellLossRouter를 끼워 넣는다.

변경점 요약
-----------
  1. __init__ : loss_bbox 대신 loss_router(SpanningCellLossRouter)를 주입.
  2. loss()   : positive sample을 router에 전달 → 두 개의 loss dict 반환.
  3. 나머지   : 완전히 부모(Shared2FCBBoxHead) 위임.

이 head를 config에서 교체하는 것만으로 vanilla ↔ GSR 전환이 가능하다.
"""

from __future__ import annotations

import torch
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.registry import MODELS
from mmdet.utils import ConfigType


@MODELS.register_module()
class SpanningAwareCascadeBBoxHead(Shared2FCBBoxHead):
    """Shared2FCBBoxHead + SpanningCellLossRouter.

    Args:
        loss_router (dict): SpanningCellLossRouter config.
            ``type`` 키는 반드시 'SpanningCellLossRouter' 여야 한다.
        **kwargs: Shared2FCBBoxHead 에 그대로 전달.

    Note:
        부모의 ``loss_bbox`` 인자는 무시된다. router가 loss를 완전히 대체함.
    """

    def __init__(self, loss_router: ConfigType, **kwargs):
        # 부모가 loss_bbox를 build하지 못하도록 더미로 대체
        kwargs.setdefault(
            'loss_bbox', dict(type='SmoothL1Loss', loss_weight=0.0)
        )
        super().__init__(**kwargs)

        # 실제 loss 담당자
        self.loss_router: 'SpanningCellLossRouter' = MODELS.build(loss_router)

        # router가 decoded box를 요구하면 head도 decoded 모드로 전환
        if self.loss_router.requires_decoded:
            self.reg_decoded_bbox = True

    # ------------------------------------------------------------------
    def loss(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        rois: torch.Tensor,
        labels: torch.Tensor,
        label_weights: torch.Tensor,
        bbox_targets: torch.Tensor,
        bbox_weights: torch.Tensor,
        reduction_override=None,
    ) -> dict[str, torch.Tensor]:
        """부모 loss()의 bbox regression 부분만 router로 교체한다."""

        losses: dict[str, torch.Tensor] = {}

        # ── 1. Classification loss (부모 로직 재사용) ──────────────────
        if cls_score is not None:
            avg_factor = max(label_weights.sum().item(), 1.0)
            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override,
            )
            losses['loss_cls'] = loss_cls
            losses['acc'] = self._accuracy(cls_score, labels)

        # ── 2. Regression loss (router 분기) ───────────────────────────
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_mask = (labels >= 0) & (labels < bg_class_ind)

            if pos_mask.any():
                pos_bbox_pred = bbox_pred[pos_mask]
                pos_targets = bbox_targets[pos_mask]
                pos_weights = bbox_weights[pos_mask]
                pos_labels = labels[pos_mask]

                # decoded bbox가 필요하면 변환
                if self.reg_decoded_bbox:
                    pos_bbox_pred = self.bbox_coder.decode(
                        rois[pos_mask, 1:],
                        pos_bbox_pred.reshape(-1, 4),
                    )
                    pos_targets = self.bbox_coder.decode(
                        rois[pos_mask, 1:],
                        pos_targets,
                    )

                router_losses = self.loss_router(
                    pos_bbox_pred,
                    pos_targets,
                    pos_weights,
                    labels=pos_labels,
                    avg_factor=pos_mask.float().sum().item(),
                )
                losses.update(router_losses)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0.0
                losses['loss_bbox_spanning'] = bbox_pred.sum() * 0.0

        return losses

    # ------------------------------------------------------------------
    @staticmethod
    def _accuracy(cls_score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """top-1 accuracy (부모의 accuracy 계산을 로컬에서 재현)."""
        with torch.no_grad():
            pred = cls_score.argmax(dim=1)
            acc = pred.eq(labels).float().mean()
        return acc

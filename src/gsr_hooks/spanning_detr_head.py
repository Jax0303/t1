"""
[OPTIONAL] SpanningAwareDeformableDETRHead / SpanningAwareTATRHead
==================================================================
향후 GSR loss를 DETR 계열로 일반화하는 실험용 파일.
현재 실험(Deformable-DETR / TATR vanilla baseline)에서는 사용하지 않음.

현재 실험 구조
--------------
  Deformable-DETR : configs/deformable_detr_pubtables_baseline.py
                    → 기본 DeformableDETRHead, GSR hook 없음
  TATR            : configs/tatr_pubtables_baseline.py
                    → 기본 DETRHead, GSR hook 없음
  GSR 적용 대상   : Cascade R-CNN 전용
                    (spanning_bbox_head.py + cascade_rcnn_r50_pubtables_gsr.py)

이 파일을 활성화하는 시점
--------------------------
  "GSR loss가 DETR 계열에서도 유효한가?"를 추가 ablation으로 검증할 때.
  그 전까지는 import하지 말 것 (__init__.py에서 제외됨).

DETR 계열과 Cascade R-CNN의 차이
---------------------------------
- Cascade R-CNN : RoI proposal → positive sample이 GT class를 직접 보유
- DETR 계열     : Hungarian matching 이후에야 query ↔ GT 대응이 결정됨
                  → matching index를 통해 spanning cell 여부를 역추적

MMDetection v3 에서 DeformableDETRHead.loss_by_feat() 의 흐름
-------------------------------------------------------------
  1. batch_gt_instances 로부터 GT cls / bbox 수집
  2. Hungarian matching (HungarianAssigner) → (query_idx, gt_idx) 쌍
  3. matched query에 대해 loss_cls / loss_bbox(L1) / loss_iou(GIoU) 계산

이 head는 step 3에서 spanning cell로 매칭된 query의 regression loss만
별도 weight로 교체한다.
"""

from __future__ import annotations
from typing import Union

import torch
from torch import Tensor

from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList


@MODELS.register_module()
class SpanningAwareDeformableDETRHead(DeformableDETRHead):
    """DeformableDETRHead + spanning cell 전용 loss 가중치.

    Hungarian matching 이후 spanning cell에 매칭된 query에 대해
    bbox / iou loss의 weight를 별도로 지정한다.

    Args:
        spanning_class_ids (list[int]): spanning cell 취급할 GT class index.
        span_bbox_loss_weight (float):  spanning query의 L1 loss weight 배율.
        span_iou_loss_weight  (float):  spanning query의 IoU loss weight 배율.
        span_iou_loss (dict | None):    None이면 부모의 loss_iou 그대로 사용.
            GIoU → DIoU / CIoU 로 교체할 때 사용.
        **kwargs: DeformableDETRHead에 전달.
    """

    def __init__(
        self,
        spanning_class_ids: list[int],
        span_bbox_loss_weight: float = 1.0,
        span_iou_loss_weight: float = 2.0,
        span_iou_loss: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spanning_class_ids = set(spanning_class_ids)
        self.span_bbox_loss_weight = span_bbox_loss_weight
        self.span_iou_loss_weight = span_iou_loss_weight

        # spanning 전용 IoU loss (None이면 부모 loss_iou 재사용)
        self._span_iou_loss_fn = (
            MODELS.build(span_iou_loss) if span_iou_loss is not None
            else None
        )

    # ------------------------------------------------------------------
    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
    ) -> tuple:
        """부모 loss_by_feat_single을 호출한 뒤 spanning loss를 패치."""

        # ── 1. 부모 loss 계산 (cls / bbox / iou 모두 포함) ──────────────
        loss_cls, loss_bbox, loss_iou = super().loss_by_feat_single(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas
        )

        # ── 2. spanning cell에 대한 추가 loss 항 계산 ───────────────────
        # 부모와 동일한 assigner / sampler 로 matching을 재실행하여
        # spanning cell query 마스크를 얻는다.
        span_iou_extra = self._spanning_iou_loss(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas
        )

        # ── 3. 합산 (loss_iou 에 spanning 기여분 추가) ─────────────────
        loss_iou = loss_iou + span_iou_extra

        return loss_cls, loss_bbox, loss_iou

    # ------------------------------------------------------------------
    def _spanning_iou_loss(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: list[dict],
    ) -> Tensor:
        """spanning cell로 매칭된 query에만 추가 IoU loss를 계산한다."""

        device = cls_scores.device
        num_imgs = cls_scores.size(0)
        extra_losses: list[Tensor] = []

        for img_id in range(num_imgs):
            gt_instances = batch_gt_instances[img_id]
            img_meta = batch_img_metas[img_id]
            h, w = img_meta['img_shape'][:2]

            gt_labels = gt_instances.labels          # (G,)
            gt_bboxes = gt_instances.bboxes          # (G, 4) xyxy

            # spanning cell GT index
            span_gt_mask = torch.tensor(
                [l.item() in self.spanning_class_ids for l in gt_labels],
                dtype=torch.bool, device=device,
            )
            if not span_gt_mask.any():
                extra_losses.append(cls_scores.sum() * 0.0)
                continue

            # ── assigner로 matching 재실행 ───────────────────────────
            pred_scores = cls_scores[img_id]         # (Q, C)
            pred_boxes = bbox_preds[img_id]          # (Q, 4) cxcywh norm

            assign_result = self.assigner.assign(
                pred_instances=dict(
                    scores=pred_scores,
                    bboxes=pred_boxes,
                ),
                gt_instances=dict(
                    labels=gt_labels,
                    bboxes=gt_bboxes,
                ),
                img_meta=img_meta,
            )

            assigned_gt_inds = assign_result.gt_inds  # (Q,)
            pos_mask = assigned_gt_inds > 0           # positive query

            if not pos_mask.any():
                extra_losses.append(cls_scores.sum() * 0.0)
                continue

            # positive query의 matched GT index (0-based)
            pos_gt_inds = assigned_gt_inds[pos_mask] - 1  # (P,)
            # spanning cell로 매칭된 positive query 마스크
            span_pos_mask = span_gt_mask[pos_gt_inds]     # (P,)

            if not span_pos_mask.any():
                extra_losses.append(cls_scores.sum() * 0.0)
                continue

            # ── decoded box 계산 ─────────────────────────────────────
            pos_pred = pred_boxes[pos_mask][span_pos_mask]  # (S, 4) cxcywh
            pos_gt = gt_bboxes[pos_gt_inds[span_pos_mask]]  # (S, 4) xyxy

            # cxcywh norm → xyxy abs
            scale = pred_boxes.new_tensor([w, h, w, h])
            cx, cy, bw, bh = pos_pred.unbind(-1)
            pred_xyxy = torch.stack(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1
            ) * scale

            iou_fn = self._span_iou_loss_fn or self.loss_iou
            span_loss = iou_fn(
                pred_xyxy,
                pos_gt,
                avg_factor=max(span_pos_mask.float().sum().item(), 1.0),
            ) * self.span_iou_loss_weight

            extra_losses.append(span_loss)

        return torch.stack(extra_losses).mean()


# TATR 는 구조적으로 DETR 과 동일 → 별칭 등록
@MODELS.register_module()
class SpanningAwareTATRHead(SpanningAwareDeformableDETRHead):
    """Table Transformer(TATR) 용 별칭.

    TATR = DETR encoder-decoder + PubTables-1M fine-tune.
    HuggingFace checkpoint 평가는 eval_tatr_hf.py 참조.
    MMDetection 재현 학습 시 이 head를 사용.
    """
    pass

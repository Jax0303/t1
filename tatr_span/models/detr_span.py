"""
TATR + Span Attribute Regression Branch
=========================================
DETR의 기존 class head + bbox head에 병렬로
rowspan/colspan 예측을 위한 ordinal regression head를 추가합니다.

Ordinal Regression 방식:
  - MAX_SPAN = 10 (rowspan/colspan 최대값)
  - K = MAX_SPAN - 1 = 9 ordinal threshold
  - head output: sigmoid([P(rs>1), P(rs>2), ..., P(rs>9),
                           P(cs>1), P(cs>2), ..., P(cs>9)])
  - 디코딩: span = 1 + sum(P > 0.5)

Loss 적용 범위:
  - "table spanning cell" (class idx = SPAN_CLASS_IDX) 매칭 쿼리에만 적용
  - 비매칭 / 비-spanning 클래스는 span loss 제외

PubTables-1M 6-class 구조 인식 레이블:
  0: table row
  1: table column
  2: table column header
  3: table projected row header
  4: table spanning cell  ← span 속성 대상
  5: no object (background/eos)
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# TATR base detr 경로 추가
TATR_BASE = Path(__file__).parents[2] / "tatr_base" / "detr"
sys.path.insert(0, str(TATR_BASE))

from util import box_ops
from util.misc import (
    NestedTensor, nested_tensor_from_tensor_list,
    accuracy, get_world_size, is_dist_avail_and_initialized,
)
from models.backbone import build_backbone
from models.matcher import build_matcher
from models.transformer import build_transformer
from models.detr import MLP, SetCriterion, PostProcess

# TATR 구조 인식의 spanning cell 클래스 인덱스
SPAN_CLASS_IDX = 4
MAX_SPAN = 10          # rowspan/colspan 최대값 (포함)
ORDINAL_K = MAX_SPAN - 1  # 9개의 binary threshold


# ──────────────────────────────────────────────────────────
# 1. 모델: DETRSpan
# ──────────────────────────────────────────────────────────
class DETRSpan(nn.Module):
    """DETR + Ordinal Regression Head for spanning cell attributes.

    기존 DETR과 동일하되, transformer decoder 출력에서
    span_embed를 통해 rowspan/colspan ordinal 예측을 추가합니다.
    """

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, ordinal_k=ORDINAL_K):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.ordinal_k = ordinal_k
        hidden_dim = transformer.d_model

        # 기존 DETR heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # Span Ordinal Regression Head (병렬 추가)
        # 출력: 2 * ordinal_k (rowspan K개 + colspan K개)
        self.span_embed = MLP(hidden_dim, hidden_dim, 2 * ordinal_k, 3)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask,
                              self.query_embed.weight, pos[-1])[0]
        # hs: [num_decoder_layers, B, num_queries, hidden_dim]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # ordinal: sigmoid → values in (0,1), shape [layers, B, Q, 2K]
        outputs_span  = self.span_embed(hs).sigmoid()

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes':  outputs_coord[-1],
            'pred_spans':  outputs_span[-1],   # [B, Q, 2*ordinal_k]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_span)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_span):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_spans': c}
                for a, b, c in zip(outputs_class[:-1],
                                   outputs_coord[:-1],
                                   outputs_span[:-1])]


# ──────────────────────────────────────────────────────────
# 2. Span Loss Utilities
# ──────────────────────────────────────────────────────────
def span_to_ordinal_label(span_val: torch.Tensor, k: int = ORDINAL_K) -> torch.Tensor:
    """
    정수 span 값 → ordinal binary label 변환.

    span_val: [N] int tensor, values in {1, ..., MAX_SPAN}
    Returns:  [N, k] float tensor

    Example (k=9, span=3):
      P(span>1)=1, P(span>2)=1, P(span>3)=0, ..., P(span>9)=0
      → [1, 1, 0, 0, 0, 0, 0, 0, 0]
    """
    thresholds = torch.arange(1, k + 1, device=span_val.device).float()
    # span_val[:,None] > thresholds[None,:] → [N, k] bool
    return (span_val.float().unsqueeze(1) > thresholds.unsqueeze(0)).float()


def ordinal_label_to_span(ordinal_pred: torch.Tensor) -> torch.Tensor:
    """
    Ordinal 예측 sigmoid → 정수 span 값 디코딩.

    ordinal_pred: [..., k] sigmoid values
    Returns: [...] int tensor, values in {1, ..., MAX_SPAN}
    """
    # 1 + number of thresholds exceeded (P > 0.5)
    return 1 + (ordinal_pred > 0.5).sum(dim=-1)


# ──────────────────────────────────────────────────────────
# 3. Span Criterion
# ──────────────────────────────────────────────────────────
class SpanCriterion(nn.Module):
    """Ordinal regression loss for rowspan/colspan prediction.

    Hungarian matching 결과를 재활용하여
    SPAN_CLASS_IDX로 매칭된 쿼리에 대해서만 손실 계산.
    """

    def __init__(self, ordinal_k: int = ORDINAL_K,
                 span_class_idx: int = SPAN_CLASS_IDX,
                 row_weight: float = 0.5,
                 col_weight: float = 0.5):
        super().__init__()
        self.ordinal_k      = ordinal_k
        self.span_class_idx = span_class_idx
        self.row_weight     = row_weight
        self.col_weight     = col_weight

    def forward(self, outputs, targets, indices):
        """
        outputs: dict with 'pred_spans' [B, Q, 2K]
        targets: list of dicts, each may contain 'rowspans', 'colspans' [M]
        indices: Hungarian matching indices [(src_idx, tgt_idx), ...]

        Returns: dict with 'loss_rowspan', 'loss_colspan'
        """
        pred_spans = outputs['pred_spans']   # [B, Q, 2K]
        B = pred_spans.shape[0]
        k = self.ordinal_k

        row_loss_list, col_loss_list = [], []

        for b in range(B):
            src_idx, tgt_idx = indices[b]
            if len(src_idx) == 0:
                continue

            tgt = targets[b]
            tgt_labels = tgt['labels'][tgt_idx]   # [M_matched]

            # spanning cell 매칭 쿼리만 선택
            is_span = (tgt_labels == self.span_class_idx)
            if not is_span.any():
                continue

            src_span_idx = src_idx[is_span]
            tgt_span_idx = tgt_idx[is_span]

            # span 속성이 없으면 모두 1로 처리 (fallback)
            rs_gt = tgt.get('rowspans', torch.ones(len(tgt['labels']),
                            dtype=torch.long, device=pred_spans.device))
            cs_gt = tgt.get('colspans', torch.ones(len(tgt['labels']),
                            dtype=torch.long, device=pred_spans.device))

            rs_gt = rs_gt[tgt_span_idx]   # [S]
            cs_gt = cs_gt[tgt_span_idx]   # [S]

            # Ordinal GT labels
            rs_ord = span_to_ordinal_label(rs_gt, k)   # [S, k]
            cs_ord = span_to_ordinal_label(cs_gt, k)   # [S, k]

            # Predicted ordinal (already sigmoid-activated in forward)
            pred = pred_spans[b][src_span_idx]          # [S, 2k]
            rs_pred = pred[:, :k]                        # [S, k]
            cs_pred = pred[:, k:]                        # [S, k]

            row_loss_list.append(
                F.binary_cross_entropy(rs_pred, rs_ord, reduction='mean'))
            col_loss_list.append(
                F.binary_cross_entropy(cs_pred, cs_ord, reduction='mean'))

        loss_rs = torch.stack(row_loss_list).mean() if row_loss_list \
                  else pred_spans.new_zeros(1).squeeze()
        loss_cs = torch.stack(col_loss_list).mean() if col_loss_list \
                  else pred_spans.new_zeros(1).squeeze()

        return {
            'loss_rowspan': self.row_weight * loss_rs,
            'loss_colspan': self.col_weight * loss_cs,
        }


# ──────────────────────────────────────────────────────────
# 4. Combined Criterion
# ──────────────────────────────────────────────────────────
class SetCriterionWithSpan(SetCriterion):
    """기존 SetCriterion에 SpanCriterion을 통합."""

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 emphasized_weights=None, ordinal_k=ORDINAL_K,
                 span_loss_coef=1.0):
        super().__init__(num_classes, matcher, weight_dict, eos_coef,
                         losses, emphasized_weights or {})
        self.span_criterion = SpanCriterion(ordinal_k=ordinal_k)
        self.span_loss_coef = span_loss_coef

    def forward(self, outputs, targets):
        # 기존 DETR 손실 계산
        outputs_without_aux = {k: v for k, v in outputs.items()
                                if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Span ordinal regression 손실 추가
        span_losses = self.span_criterion(outputs, targets, indices)
        losses.update({k: v * self.span_loss_coef for k, v in span_losses.items()})

        # Auxiliary layers
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets,
                                           aux_indices, num_boxes, **kwargs)
                    losses.update({k + f'_{i}': v for k, v in l_dict.items()})

                # Aux span loss
                aux_span = self.span_criterion(aux_outputs, targets, aux_indices)
                losses.update({
                    k + f'_{i}': v * self.span_loss_coef
                    for k, v in aux_span.items()
                })

        return losses


# ──────────────────────────────────────────────────────────
# 5. PostProcess (spans 포함)
# ──────────────────────────────────────────────────────────
class PostProcessWithSpan(PostProcess):
    """기존 PostProcess에 span 디코딩 추가."""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        results = super().forward(outputs, target_sizes)

        if 'pred_spans' in outputs:
            pred_spans = outputs['pred_spans']   # [B, Q, 2K]
            k = pred_spans.shape[-1] // 2
            rs_pred = ordinal_label_to_span(pred_spans[..., :k])  # [B, Q]
            cs_pred = ordinal_label_to_span(pred_spans[..., k:])  # [B, Q]

            # 각 이미지의 결과에 span 추가
            # results는 [{'scores':, 'labels':, 'boxes':}, ...]
            for i, r in enumerate(results):
                # scores로 필터된 idx 없이 전체 쿼리 span 저장
                r['rowspans'] = rs_pred[i]
                r['colspans'] = cs_pred[i]

        return results


# ──────────────────────────────────────────────────────────
# 6. Build
# ──────────────────────────────────────────────────────────
def build_detr_span(args):
    """DETRSpan + SetCriterionWithSpan + PostProcessWithSpan 빌드."""
    device = torch.device(args.device)

    backbone    = build_backbone(args)
    transformer = build_transformer(args)

    model = DETRSpan(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        ordinal_k=getattr(args, 'ordinal_k', ORDINAL_K),
    )

    matcher     = build_matcher(args)
    weight_dict = {
        'loss_ce':   args.ce_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
    }

    span_loss_coef = getattr(args, 'span_loss_coef', 1.0)
    weight_dict['loss_rowspan'] = span_loss_coef
    weight_dict['loss_colspan'] = span_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterionWithSpan(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        emphasized_weights=getattr(args, 'emphasized_weights', {}),
        ordinal_k=getattr(args, 'ordinal_k', ORDINAL_K),
        span_loss_coef=span_loss_coef,
    )
    criterion.to(device)

    postprocessors = {'bbox': PostProcessWithSpan()}
    return model, criterion, postprocessors


def load_pretrained_weights(model: DETRSpan, checkpoint_path: str,
                             strict: bool = False) -> None:
    """bsmock/tatr-pubtables1m-v1.0 가중치 로드 (span_embed 제외).

    strict=False: span_embed 등 새 레이어는 랜덤 초기화 유지.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    new_keys = [k for k in missing if 'span_embed' in k]
    print(f"[load_pretrained] Loaded {checkpoint_path}")
    print(f"  New span keys (random init): {new_keys}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}...")

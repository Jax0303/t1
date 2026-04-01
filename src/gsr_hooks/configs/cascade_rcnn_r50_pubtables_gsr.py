"""
GSR variant: spanning cell에만 GIoU loss 주입
=============================================
"같은 detection 패러다임 내에서 loss만 교체해도 spanning cell 성능이 오른다"
는 GSR 핵심 주장을 검증하는 config.

baseline config 대비 변경사항:
  - loss_router.span_loss : SmoothL1 → GIoULoss
  - reg_decoded_bbox=True  (GIoU는 decoded box 입력 필요)
  - span_loss_weight       : 샘플 불균형 보정 (spanning cell이 소수)

필요 시 DIoU / CIoU로 교체하려면 span_loss.type만 바꾸면 된다.
"""

_base_ = ['cascade_rcnn_r50_pubtables_baseline.py']

NUM_CLASSES = 5

# ── router: spanning → GIoU, non-spanning → SmoothL1 ─────────────────
_loss_router_gsr = dict(
    type='SpanningCellLossRouter',
    spanning_class_ids=[4],
    span_loss=dict(
        type='GIoULoss',
        loss_weight=2.0,            # IoU loss는 보통 SmoothL1보다 크게 세팅
    ),
    nonspan_loss=dict(
        type='SmoothL1Loss',
        beta=1.0 / 9.0,
        loss_weight=1.0,
    ),
    span_loss_weight=1.5,           # spanning cell 샘플 수가 적으므로 업-웨이트
)

_bbox_head_gsr = dict(
    type='SpanningAwareCascadeBBoxHead',
    num_classes=NUM_CLASSES,
    loss_router=_loss_router_gsr,
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    reg_class_agnostic=True,
    reg_decoded_bbox=True,          # GIoU는 반드시 decoded bbox 필요
    loss_cls=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0,
    ),
)

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(**_bbox_head_gsr),  # stage 1
            dict(**_bbox_head_gsr),  # stage 2
            dict(**_bbox_head_gsr),  # stage 3
        ],
    ),
)

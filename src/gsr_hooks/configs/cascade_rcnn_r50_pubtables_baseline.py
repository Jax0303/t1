"""
Baseline: vanilla Cascade R-CNN
================================
SmoothL1Loss를 전 class에 균등 적용.
Xiao et al. (2023) Table 2의 "Cascade R-CNN" 수치와 비교 기준.

loss_router.span_loss     = SmoothL1Loss  (spanning도 동일)
loss_router.nonspan_loss  = SmoothL1Loss
"""

_base_ = [
    'mmdetection/configs/_base_/models/cascade-rcnn_r50_fpn.py',
    'mmdetection/configs/_base_/datasets/coco_detection.py',
    'mmdetection/configs/_base_/schedules/schedule_1x.py',
    'mmdetection/configs/_base_/default_runtime.py',
]

# ── PubTables-1M structure class 수 ──────────────────────────────────
NUM_CLASSES = 5  # col / row / col-header / proj-row-header / spanning-cell

# ── router: spanning도 동일하게 SmoothL1 ─────────────────────────────
_loss_router = dict(
    type='SpanningCellLossRouter',
    spanning_class_ids=[4],           # 'table spanning cell' index
    span_loss=dict(
        type='SmoothL1Loss',
        beta=1.0 / 9.0,
        loss_weight=1.0,
    ),
    nonspan_loss=dict(
        type='SmoothL1Loss',
        beta=1.0 / 9.0,
        loss_weight=1.0,
    ),
)

# ── Cascade R-CNN 각 stage head를 SpanningAwareCascadeBBoxHead로 교체 ─
_bbox_head = dict(
    type='SpanningAwareCascadeBBoxHead',
    num_classes=NUM_CLASSES,
    loss_router=_loss_router,
    # 부모(Shared2FCBBoxHead) 파라미터
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    reg_class_agnostic=True,
    loss_cls=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0,
    ),
)

model = dict(
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=[
            dict(**_bbox_head, reg_decoded_bbox=False),  # stage 1
            dict(**_bbox_head, reg_decoded_bbox=False),  # stage 2
            dict(**_bbox_head, reg_decoded_bbox=False),  # stage 3
        ],
    ),
)

# ── 데이터셋 경로 (필요 시 수정) ──────────────────────────────────────
data_root = 'data/pubtables1m/'
data = dict(
    train=dict(ann_file=data_root + 'train.json',
               img_prefix=data_root + 'train/'),
    val=dict(ann_file=data_root + 'val.json',
             img_prefix=data_root + 'val/'),
    test=dict(ann_file=data_root + 'test.json',
              img_prefix=data_root + 'test/'),
)

"""
Table Transformer (TATR) vanilla baseline
==========================================
GSR hook 없음. 모든 class에 L1 + GIoU loss를 균등 적용.

역할
----
- Xiao et al. (2023) Table 2의 "TATR" 수치와 비교하는 순수 재현 config.
- HuggingFace 공식 체크포인트(microsoft/table-transformer-structure-recognition)
  평가는 eval_tatr_hf.py 사용 → 수치 직접 인용 가능.
  MMDetection 재현 학습이 필요한 경우에만 이 config를 사용할 것.

구조
----
  TATR = DETR (encoder 6층, decoder 6층) + ResNet-18 backbone
  FPN 없이 마지막 stage feature(stride 32)만 사용.
  커스텀 head 불필요 — 기본 DETRHead 그대로 사용.
"""

_base_ = [
    'mmdetection/configs/detr/detr_r50_8xb2-150e_coco.py'
]

NUM_CLASSES = 5  # PubTables-1M structure classes

model = dict(
    backbone=dict(
        type='ResNet',
        depth=18,
        out_indices=(3,),
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
    ),
    neck=None,   # TATR는 FPN 미사용
    bbox_head=dict(
        type='DETRHead',             # 기본 head, 커스텀 없음
        num_classes=NUM_CLASSES,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0] * NUM_CLASSES + [0.1],  # bg down-weight
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='ClassificationCost', weight=1.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0),
                ],
            )
        ),
    ),
)

data_root = 'data/pubtables1m/'
data = dict(
    train=dict(ann_file=data_root + 'train.json',
               img_prefix=data_root + 'train/'),
    val=dict(ann_file=data_root + 'val.json',
             img_prefix=data_root + 'val/'),
    test=dict(ann_file=data_root + 'test.json',
              img_prefix=data_root + 'test/'),
)

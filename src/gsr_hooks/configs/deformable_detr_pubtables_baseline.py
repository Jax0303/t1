"""
Deformable-DETR vanilla baseline
==================================
GSR hook 없음. 모든 class(row / column / spanning cell 포함)에
L1 + GIoU loss를 균등 적용.

역할
----
- Xiao et al. (2023) Table 2의 "Deformable-DETR" 수치와 비교하는
  순수 재현 config.
- 수치 직접 인용이 가능하므로 재학습 없이 결과표에 채울 수 있음.
  재현 학습이 필요한 경우에만 이 config를 사용할 것.

커스텀 head 불필요 — MMDetection 기본 DeformableDETRHead 그대로 사용.
"""

_base_ = [
    'mmdetection/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'
]

NUM_CLASSES = 5  # PubTables-1M structure classes

model = dict(
    bbox_head=dict(
        type='DeformableDETRHead',   # 기본 head, 커스텀 없음
        num_classes=NUM_CLASSES,
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
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

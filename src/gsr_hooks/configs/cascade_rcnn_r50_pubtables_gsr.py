"""
GSR variant: SnapGTBoxesToSeparators + SmoothL1 (Option B)
===========================================================
"같은 detection 패러다임 내에서 regression target만 바꿔도
spanning cell 성능이 오른다"는 GSR 핵심 주장 검증 config.

baseline config 대비 변경사항:
  - train pipeline에 SnapGTBoxesToSeparators 추가
    → spanning cell GT bbox가 nearest separator로 교체되어 저장됨
  - loss는 기존 SmoothL1 그대로 (model 구조 변경 없음)

SpanningAwareCascadeBBoxHead / SpanningCellLossRouter는
향후 loss weighting ablation 시 추가 가능 (현재 미사용).
"""

_base_ = ['cascade_rcnn_r50_pubtables_baseline.py']

# ── 데이터 파이프라인: GSR transform 주입 ─────────────────────────────
# baseline pipeline 끝에 SnapGTBoxesToSeparators를 삽입.
# PackDetInputs 직전에 위치해야 gt_bboxes가 올바르게 전달됨.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction')),
]

# SnapGTBoxesToSeparators를 PackDetInputs 바로 앞에 삽입
_snap_transform = dict(
    type='SnapGTBoxesToSeparators',
    spanning_class_ids=[4],    # table spanning cell
    row_class_ids=[1, 3],      # table row, table projected row header
    col_class_ids=[0, 2],      # table column, table column header
    snap_non_spanning=False,   # spanning cell만 snap
)
train_pipeline.insert(-1, _snap_transform)  # PackDetInputs 직전

data_root = 'data/pubtables1m/'
train_dataloader = dict(
    dataset=dict(
        ann_file=data_root + 'train.json',
        data_prefix=dict(img=data_root + 'train/'),
        pipeline=train_pipeline,
    )
)

from .snap_gt_transform import SnapGTBoxesToSeparators   # GSR 핵심
from .spanning_loss_router import SpanningCellLossRouter
from .spanning_bbox_head import SpanningAwareCascadeBBoxHead

# gsr_loss: 보조 regularizer (선택적 사용)
# from .gsr_loss import HardSnappingLoss, SoftSnappingLoss

# spanning_detr_head: 향후 DETR 계열 GSR 일반화 실험용 (현재 미사용)
# from .spanning_detr_head import SpanningAwareDeformableDETRHead, SpanningAwareTATRHead

__all__ = [
    'SnapGTBoxesToSeparators',
    'SpanningCellLossRouter',
    'SpanningAwareCascadeBBoxHead',
]

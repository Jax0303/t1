from .snap_gt_transform import SnapGTBoxesToSeparators   # GSR 핵심
try:
    from .spanning_loss_router import SpanningCellLossRouter
    from .spanning_bbox_head import SpanningAwareCascadeBBoxHead
    _MMDET_OK = True
except ImportError:
    _MMDET_OK = False

__all__ = [
    'SnapGTBoxesToSeparators',
]
if _MMDET_OK:
    __all__.extend(['SpanningCellLossRouter', 'SpanningAwareCascadeBBoxHead'])

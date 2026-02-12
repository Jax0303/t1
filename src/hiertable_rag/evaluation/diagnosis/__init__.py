"""
SciTSR Diagnosis Module

Comprehensive diagnostic harness for table structure recognition evaluation.
"""

from .utils import setup_logging
from .scitsr_dataset import SciTSRDataset
from .stratify import TableTypeClassifier, FeatureExtractor, StratifiedSampler
from .structures import Cell, TableStruct, OcrResult, convert_polygon_to_xyxy, bbox_iou
from .ocr_sources import BaseOCRSource, NoisyOCRSource, PerfectOCRSource, GTOCRSource
from .tsr_engines import BaseTSREngine, BaselineTSR, GNNCSPTSREngine, GTStructureEngine
from .metrics import DiagnosisMetrics
from .attribution import AttributionCalculator
from .overlay import OverlayGenerator
from .cache import ResultCache

__all__ = [
    'setup_logging',
    'SciTSRDataset',
    'TableTypeClassifier',
    'FeatureExtractor',
    'StratifiedSampler',
    'Cell',
    'TableStruct',
    'OcrResult',
    'convert_polygon_to_xyxy',
    'bbox_iou',
    'BaseOCRSource',
    'NoisyOCRSource',
    'PerfectOCRSource',
    'GTOCRSource',
    'BaseTSREngine',
    'BaselineTSR',
    'GNNCSPTSREngine',
    'GTStructureEngine',
    'DiagnosisMetrics',
    'AttributionCalculator',
    'OverlayGenerator',
    'ResultCache'
]

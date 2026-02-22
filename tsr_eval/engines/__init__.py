"""TSR Eval Engines package."""
from .tatr_engine import TATREngine
from .paddle_engine import PaddleTSREngine
from .tesseract_engine import TesseractEngine
from .paddle_ocr_engine import PaddleOCREngine

__all__ = ["TATREngine", "PaddleTSREngine", "TesseractEngine", "PaddleOCREngine"]

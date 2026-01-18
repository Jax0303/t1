import logging
from typing import Dict, Any
from src.hiertable_rag.core.uncertainty import TTAUncertaintyEstimator
from src.hiertable_rag.extraction.baselines import OCRExtractor, TSRExtractor, VLMExtractor

logger = logging.getLogger(__name__)

class AdaptiveExtractor:
    """
    Adaptive extractor that chooses between different OCR/Parsing strategies
    based on TTA-based confidence scores.
    
    1. High Confidence (> 0.8): Basic OCR
    2. Mid Confidence (0.4 - 0.8): OCR + Structural TSR
    3. Low Confidence (<= 0.4): Vision-LLM / Visual Verification
    """
    
    def __init__(self, high_threshold: float = 0.8, low_threshold: float = 0.4):
        self.confidence_estimator = TTAUncertaintyEstimator()
        self.basic_ocr = OCRExtractor()
        self.structural_ocr = TSRExtractor()
        self.vision_vlm = VLMExtractor()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def extract(self, table_raw: Dict[str, Any], force_strategy: str = None) -> Dict[str, Any]:
        """
        Routes the extraction request based on confidence or a forced strategy.
        """
        confidence = self.confidence_estimator.estimate_confidence(table_raw)
        
        if force_strategy == "OCR_ONLY":
            result = self.basic_ocr.extract(table_raw)
            result["strategy_tier"] = "FORCED (BASIC)"
        elif force_strategy == "OCR_TSR":
            result = self.structural_ocr.extract(table_raw)
            result["strategy_tier"] = "FORCED (STRUCTURAL)"
        elif force_strategy == "OCR_VLM":
            result = self.vision_vlm.extract(table_raw)
            result["strategy_tier"] = "FORCED (VISION_LLM)"
        elif confidence > self.high_threshold:
            logger.info(f"Confidence ({confidence:.2f}) > {self.high_threshold}: Routing to BASIC OCR")
            result = self.basic_ocr.extract(table_raw)
            result["strategy_tier"] = "HIGH_CONFIDENCE (BASIC)"
        elif confidence > self.low_threshold:
            logger.info(f"Confidence ({confidence:.2f}) in ({self.low_threshold}, {self.high_threshold}]: Routing to STRUCTURAL TSR")
            result = self.structural_ocr.extract(table_raw)
            result["strategy_tier"] = "MID_CONFIDENCE (STRUCTURAL)"
        else:
            logger.info(f"Confidence ({confidence:.2f}) <= {self.low_threshold}: Routing to VISION LLM")
            result = self.vision_vlm.extract(table_raw)
            result["strategy_tier"] = "LOW_CONFIDENCE (VISION_LLM)"
            
        result["tta_confidence"] = confidence
        return result

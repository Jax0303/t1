"""
Table Detection using Hugging Face models
Hugging Face의 사전 훈련된 모델을 사용한 표 검출
"""
from typing import List, Dict, Any
from PIL import Image
import torch
import numpy as np

try:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed")

from .base_model import TableDetector, BoundingBox


class HuggingFaceTableDetector(TableDetector):
    """
    Hugging Face의 Table Transformer (DETR 기반) 사용
    microsoft/table-transformer-detection 모델
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is not installed")
        
        self.model_name = config.get('model_name', 'microsoft/table-transformer-detection')
        self.threshold = config.get('threshold', 0.7)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = None
        
    def load_model(self):
        """모델과 이미지 프로세서 로드"""
        print(f"Loading table detection model: {self.model_name}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict(self, image: Image.Image) -> List[BoundingBox]:
        """표 검출 수행"""
        return self.detect_tables(image)
    
    def detect_tables(self, image: Image.Image) -> List[BoundingBox]:
        """
        이미지에서 표 영역 검출
        
        Args:
            image: PIL Image
            
        Returns:
            List of BoundingBox objects for detected tables
        """
        if self.model is None:
            self.load_model()
        
        # 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 결과 후처리
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=self.threshold, 
            target_sizes=target_sizes
        )[0]
        
        # BoundingBox 객체로 변환
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_val = score.item()
            label_val = label.item()
            box_coords = box.cpu().numpy()
            
            # 라벨 이름 가져오기
            label_name = self.model.config.id2label[label_val]
            
            # 표(table)만 필터링
            if 'table' in label_name.lower():
                bbox = BoundingBox(
                    x1=float(box_coords[0]),
                    y1=float(box_coords[1]),
                    x2=float(box_coords[2]),
                    y2=float(box_coords[3]),
                    confidence=score_val,
                    label=label_name
                )
                tables.append(bbox)
        
        return tables


class DETRTableDetector(TableDetector):
    """
    DETR (DEtection TRansformer) 기반 표 검출
    facebook/detr-resnet-50 등의 범용 객체 검출 모델도 사용 가능
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is not installed")
        
        self.model_name = config.get('model_name', 'facebook/detr-resnet-50')
        self.threshold = config.get('threshold', 0.7)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """모델과 프로세서 로드"""
        print(f"Loading DETR model: {self.model_name}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict(self, image: Image.Image) -> List[BoundingBox]:
        """표 검출 수행"""
        return self.detect_tables(image)
    
    def detect_tables(self, image: Image.Image) -> List[BoundingBox]:
        """표 영역 검출 (DETR 모델 사용)"""
        if self.model is None:
            self.load_model()
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=self.threshold, 
            target_sizes=target_sizes
        )[0]
        
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_val = score.item()
            label_val = label.item()
            box_coords = box.cpu().numpy()
            label_name = self.model.config.id2label[label_val]
            
            bbox = BoundingBox(
                x1=float(box_coords[0]),
                y1=float(box_coords[1]),
                x2=float(box_coords[2]),
                y2=float(box_coords[3]),
                confidence=score_val,
                label=label_name
            )
            tables.append(bbox)
        
        return tables


def get_table_detector(detector_type: str, config: Dict[str, Any]) -> TableDetector:
    """
    표 검출 모델 팩토리 함수
    
    Args:
        detector_type: 'huggingface' or 'detr'
        config: Model configuration
        
    Returns:
        TableDetector instance
    """
    if detector_type.lower() in ['huggingface', 'table-transformer']:
        return HuggingFaceTableDetector(config)
    elif detector_type.lower() == 'detr':
        return DETRTableDetector(config)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

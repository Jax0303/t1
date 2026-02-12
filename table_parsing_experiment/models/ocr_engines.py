"""
OCR Engines Integration
Tesseract, EasyOCR, DocTR 통합
"""
from typing import List, Dict, Any
from PIL import Image
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not installed")

try:
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
    print("Warning: doctr not installed")

from .base_model import OCREngine, BoundingBox


class TesseractEngine(OCREngine):
    """Tesseract OCR 엔진"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract is not installed")
        self.lang = config.get('lang', 'eng')
        self.psm_config = config.get('config', '--psm 6')
        
    def load_model(self):
        """Tesseract는 별도 로드 불필요"""
        pass
    
    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        """텍스트 인식 및 위치 정보 반환"""
        return self.recognize_text(image)
    
    def recognize_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        이미지에서 텍스트 및 바운딩 박스 추출
        
        Returns:
            List of dicts with 'text', 'bbox', 'confidence'
        """
        # Tesseract로 텍스트와 위치 정보 추출
        data = pytesseract.image_to_data(
            image, 
            lang=self.lang,
            config=self.psm_config,
            output_type=pytesseract.Output.DICT
        )
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:  # 빈 텍스트 제외
                continue
            
            conf = float(data['conf'][i])
            if conf < 0:  # 신뢰도가 음수인 경우 제외
                continue
            
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            bbox = BoundingBox(
                x1=x,
                y1=y,
                x2=x + w,
                y2=y + h,
                confidence=conf / 100.0  # 0-1 범위로 정규화
            )
            
            results.append({
                'text': text,
                'bbox': bbox,
                'confidence': conf / 100.0
            })
        
        return results


class EasyOCREngine(OCREngine):
    """EasyOCR 엔진"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not EASYOCR_AVAILABLE:
            raise ImportError("easyocr is not installed")
        self.languages = config.get('languages', ['en'])
        self.gpu = config.get('gpu', True)
        self.reader = None
        
    def load_model(self):
        """EasyOCR 리더 초기화"""
        if self.reader is None:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
    
    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        """텍스트 인식 및 위치 정보 반환"""
        return self.recognize_text(image)
    
    def recognize_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        이미지에서 텍스트 및 바운딩 박스 추출
        
        Returns:
            List of dicts with 'text', 'bbox', 'confidence'
        """
        if self.reader is None:
            self.load_model()
        
        # PIL Image를 numpy array로 변환
        image_np = np.array(image)
        
        # EasyOCR 실행
        detections = self.reader.readtext(image_np)
        
        results = []
        for detection in detections:
            bbox_coords, text, confidence = detection
            
            # 바운딩 박스 좌표 (4개 점) -> (x1, y1, x2, y2)
            x_coords = [point[0] for point in bbox_coords]
            y_coords = [point[1] for point in bbox_coords]
            
            bbox = BoundingBox(
                x1=min(x_coords),
                y1=min(y_coords),
                x2=max(x_coords),
                y2=max(y_coords),
                confidence=confidence
            )
            
            results.append({
                'text': text,
                'bbox': bbox,
                'confidence': confidence
            })
        
        return results


class DocTROCREngine(OCREngine):
    """DocTR OCR 엔진 (SOTA OCR from Mindee)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not DOCTR_AVAILABLE:
            raise ImportError("doctr is not installed. Install with: pip install python-doctr[torch]")
        
        self.pretrained = config.get('pretrained', True)
        self.det_arch = config.get('det_arch', 'db_resnet50')
        self.reco_arch = config.get('reco_arch', 'crnn_vgg16_bn')
        self.model = None
        
    def load_model(self):
        """DocTR 모델 로드"""
        if self.model is None:
            print(f"Loading DocTR model (det: {self.det_arch}, reco: {self.reco_arch})...")
            self.model = ocr_predictor(
                det_arch=self.det_arch,
                reco_arch=self.reco_arch,
                pretrained=self.pretrained
            )
            print("DocTR model loaded successfully")
    
    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        """텍스트 인식 및 위치 정보 반환"""
        return self.recognize_text(image)
    
    def recognize_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        이미지에서 텍스트 및 바운딩 박스 추출
        
       Returns:
            List of dicts with 'text', 'bbox', 'confidence'
        """
        if self.model is None:
            self.load_model()
        
        # PIL Image를 numpy array로 변환
        image_np = np.array(image)
        
        # DocTR 실행
        result = self.model([image_np])
        
        results = []
        
        # DocTR 결과 파싱
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text = word.value
                        confidence = word.confidence
                        
                        # 바운딩 박스 좌표 (normalized 0-1)
                        # DocTR는 (x_min, y_min, x_max, y_max) 형식으로 반환
                        geometry = word.geometry
                        
                        # 이미지 크기 기준으로 변환
                        img_width, img_height = image.size
                        bbox = BoundingBox(
                            x1=int(geometry[0][0] * img_width),
                            y1=int(geometry[0][1] * img_height),
                            x2=int(geometry[1][0] * img_width),
                            y2=int(geometry[1][1] * img_height),
                            confidence=confidence
                        )
                        
                        results.append({
                            'text': text,
                            'bbox': bbox,
                            'confidence': confidence
                        })
        
        return results


def get_ocr_engine(engine_type: str, config: Dict[str, Any]) -> OCREngine:
    """
    OCR 엔진 팩토리 함수
    
    Args:
        engine_type: 'tesseract', 'easyocr', or 'doctr'
        config: Engine configuration
    
    Returns:
        OCREngine instance
    """
    if engine_type.lower() == 'tesseract':
        return TesseractEngine(config)
    elif engine_type.lower() == 'easyocr':
        return EasyOCREngine(config)
    elif engine_type.lower() == 'doctr':
        return DocTROCREngine(config)
    else:
        raise ValueError(f"Unknown OCR engine type: {engine_type}. Supported: tesseract, easyocr, doctr")

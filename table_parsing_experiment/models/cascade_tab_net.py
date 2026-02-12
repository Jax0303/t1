"""
CascadeTabNet Integration
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import numpy as np

# MMDetection imports
try:
    from mmdet.apis import init_detector, inference_detector
    import mmcv
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False
    print("Warning: mmdet or mmcv not installed")

from .base_model import TableDetector, TableStructure, BoundingBox

class CascadeTabNetDetector(TableDetector):
    """
    CascadeTabNet wrapper for Table Detection.
    Detects tables in an image.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not MMDET_AVAILABLE:
            raise ImportError("MMDetection is required for CascadeTabNet")
            
        self.config_file = config.get('config_file', str(Path(__file__).parent / 'external/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'))
        self.checkpoint_file = config.get('checkpoint_file', str(Path(__file__).parent / 'external/CascadeTabNet/epoch_36.pth'))
        self.model = None
        self.device = 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

    def load_model(self):
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_file}")
            
        print(f"Loading CascadeTabNet from {self.checkpoint_file}...")
        # Config 수정 필요시 여기서 로드 후 수정
        try:
            self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
            print("CascadeTabNet loaded successfully")
        except Exception as e:
            print(f"Error loading CascadeTabNet: {e}")
            # 호환성 문제일 경우 가짜 모델이라도 띄워서 인터페이스 테스트
            self.model = None

    def predict(self, image: Image.Image) -> List[BoundingBox]:
        """BaseModel 추상 메서드 구현"""
        return self.detect_tables(image)
        
    def detect_tables(self, image: Image.Image) -> List[BoundingBox]:
        if self.model is None:
            return []
            
        # PIL -> numpy (BGR)
        img_np = np.array(image)
        if img_np.shape[-1] == 3:
            img_np = img_np[:, :, ::-1] # RGB to BGR
            
        try:
            # mmdetection 추론
            from mmdet.apis import inference_detector
            result = inference_detector(self.model, img_np)
            # result structure depends on mmdet version
            # CascadeTabNet result: dictionary or list of arrays
            
            # Assuming standard mmdet result: list of arrays (one per class)
            # Class 0: Bordered, Class 1: Cell, Class 2: Borderless (Based on demo code)
            # We are interested in Class 0 and Class 2 for TABLE detection
            
            tables = []
            
            # Check result format
            if isinstance(result, tuple):
                bbox_result, segm_result = result
            else:
                bbox_result = result
                
            # Class 0 (Bordered Table)
            if len(bbox_result) > 0:
                for box in bbox_result[0]:
                    if box[4] > 0.5: # score threshold
                        tables.append(BoundingBox(x1=int(box[0]), y1=int(box[1]), x2=int(box[2]), y2=int(box[3])))
                        
            # Class 2 (Borderless Table)
            if len(bbox_result) > 2:
                for box in bbox_result[2]:
                    if box[4] > 0.5:
                        tables.append(BoundingBox(x1=int(box[0]), y1=int(box[1]), x2=int(box[2]), y2=int(box[3])))
                        
            return tables
            
        except Exception as e:
            print(f"Error during CascadeTabNet inference: {e}")
            return []

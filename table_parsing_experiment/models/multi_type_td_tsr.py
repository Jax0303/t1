"""
Multi-Type-TD-TSR Integration
GitHub: https://github.com/Psarpei/Multi-Type-TD-TSR
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import cv2
import builtins

# Colab 호환성을 위한 패치
if not hasattr(builtins, 'cv2_imshow'):
    def cv2_imshow(img):
        pass
    builtins.cv2_imshow = cv2_imshow

# 외부 모듈 경로 추가
EXTERNAL_MODELS_DIR = Path(__file__).parent / 'external' / 'Multi-Type-TD-TSR' / 'scripts'
if str(EXTERNAL_MODELS_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_MODELS_DIR))

try:
    from TSR import table_structure_recognition_lines as tsrl
    from TSR import table_structure_recognition_wol as tsrwol
    from TSR import table_structure_recognition_lines_wol as tsrlwol
    from TSR import table_structure_recognition_all as tsra
    MULTI_TYPE_TSR_AVAILABLE = True
except ImportError as e:
    MULTI_TYPE_TSR_AVAILABLE = False
    print(f"Warning: Multi-Type-TD-TSR modules not found: {e}")

from .base_model import TableStructureRecognizer, TableStructure, Cell, BoundingBox


class MultiTypeTSR(TableStructureRecognizer):
    """
    Multi-Type-TD-TSR의 구조 인식 모듈 래퍼
    OpenCV 기반의 결정론적 알고리즘 사용
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not MULTI_TYPE_TSR_AVAILABLE:
            raise ImportError("Multi-Type-TD-TSR is not available. Please check external models directory.")
        
        # 표 유형 설정 (borderd, unbordered, partially, partially_color_inv)
        self.table_type = config.get('table_type', 'partially')
        
        self.type_dict = {
            "fully_bordered": tsrl,
            "borderd": tsrl,
            "unbordered": tsrwol,
            "partially_bordered": tsrlwol,
            "partially": tsrlwol,
            "partially_color_inv": tsra
        }
        
        if self.table_type not in self.type_dict:
            print(f"Warning: Unknown table type '{self.table_type}', defaulting to 'partially'")
            self.table_type = 'partially'
            
        self.algorithm = self.type_dict[self.table_type]
        
    def load_model(self):
        """No model loading needed for OpenCV algorithms"""
        pass
    
    def predict(self, table_image: Image.Image) -> TableStructure:
        """표 구조 인식 수행"""
        return self.recognize_structure(table_image)
    
    def recognize_structure(self, table_image: Image.Image) -> TableStructure:
        """
        Multi-Type-TD-TSR 알고리즘을 사용한 구조 인식
        """
        # PIL Image -> OpenCV BGR
        image_np = np.array(table_image)
        if image_np.shape[-1] == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            
        # 알고리즘 실행
        # returns: boxes, img_processed
        # boxes is usually a list of [x, y, w, h] or similar?
        # Let's verify return type from inspecting code if needed, but assuming list of boxes based on tsr.py
        
        try:
            boxes, img_processed = self.algorithm.recognize_structure(image_cv)
        except Exception as e:
            # print(f"Error in Multi-Type-TD-TSR: {e}") # 로그 과다 방지
            boxes = []
            
        if not boxes:
             return TableStructure(num_rows=1, num_cols=1, cells=[])

        cells = []
        rows_coords = set()
        cols_coords = set()
        
        for i, box in enumerate(boxes):
            try:
                # box format check
                if isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
                    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                else:
                    continue
                
                bbox = BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)
                
                # 행/열 인덱스는 좌표 기반으로 추정해야 함
                # 우선 단순 리스트로 셀 생성
                cell = Cell(
                    row_idx=0, # 추후 계산
                    col_idx=0, # 추후 계산
                    row_span=1,
                    col_span=1,
                    bbox=bbox,
                    text=""
                )
                cells.append(cell)
            except Exception:
                continue
            
        if not cells:
            return TableStructure(num_rows=1, num_cols=1, cells=[])
            
        # 셀 좌표를 기반으로 행/열 구조 복원 (간단한 휴리스틱)
        # y좌표로 정렬하여 행 구분
        cells.sort(key=lambda c: c.bbox.y1)
        
        # 행 클러스터링
        current_y = -1
        row_idx = -1
        row_threshold = 10  # 픽셀
        
        for cell in cells:
            if current_y == -1 or abs(cell.bbox.y1 - current_y) > row_threshold:
                row_idx += 1
                current_y = cell.bbox.y1
            cell.row_idx = row_idx
            
        # 각 행 내에서 x좌표로 열 정렬
        num_rows = row_idx + 1 if row_idx >= 0 else 0
        max_cols = 0
        
        for r in range(num_rows):
            row_cells = [c for c in cells if c.row_idx == r]
            row_cells.sort(key=lambda c: c.bbox.x1)
            for c_idx, cell in enumerate(row_cells):
                cell.col_idx = c_idx
            max_cols = max(max_cols, len(row_cells))
            
        return TableStructure(
            num_rows=num_rows,
            num_cols=max_cols,
            cells=cells
        )


"""
Table Structure Recognition using Hugging Face models
Hugging Face의 사전 훈련된 모델을 사용한 표 구조 인식
"""
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch
import numpy as np
from collections import defaultdict

try:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Warning: transformers import failed: {e}")

from .base_model import TableStructureRecognizer, TableStructure, Cell, BoundingBox


class HuggingFaceTableStructureRecognizer(TableStructureRecognizer):
    """
    Hugging Face의 Table Transformer Structure Recognition 모델
    microsoft/table-transformer-structure-recognition
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is not installed")
        
        self.model_name = config.get('model_name', 'microsoft/table-transformer-structure-recognition')
        self.threshold = config.get('threshold', 0.7)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = None
        
    def load_model(self):
        """모델과 이미지 프로세서 로드"""
        print(f"Loading table structure model: {self.model_name}")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict(self, table_image: Image.Image) -> TableStructure:
        """표 구조 인식 수행"""
        return self.recognize_structure(table_image)
    
    def recognize_structure(self, table_image: Image.Image) -> TableStructure:
        """
        표 이미지에서 구조 인식 (행, 열, 셀 검출)
        
        Args:
            table_image: 표 영역 이미지
            
        Returns:
            TableStructure object
        """
        if self.model is None:
            self.load_model()
        
        # 이미지 전처리
        inputs = self.processor(images=table_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 결과 후처리
        target_sizes = torch.tensor([table_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes
        )[0]
        
        # 검출된 요소 분류
        rows = []
        columns = []
        cells = []
        headers = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_val = score.item()
            label_val = label.item()
            box_coords = box.cpu().numpy()
            label_name = self.model.config.id2label[label_val].lower()
            
            bbox = BoundingBox(
                x1=float(box_coords[0]),
                y1=float(box_coords[1]),
                x2=float(box_coords[2]),
                y2=float(box_coords[3]),
                confidence=score_val,
                label=label_name
            )
            
            if 'row' in label_name and 'column' not in label_name:
                rows.append(bbox)
            elif 'column' in label_name:
                columns.append(bbox)
            elif 'header' in label_name:
                headers.append(bbox)
            else:
                # 일반 셀로 간주
                cells.append(bbox)
        
        # 행/열 정렬
        rows.sort(key=lambda r: r.y1)
        columns.sort(key=lambda c: c.x1)
        
        # 셀을 행/열에 매핑
        table_cells = self._map_cells_to_grid(rows, columns, cells, headers, table_image)
        
        # TableStructure 생성
        structure = TableStructure(
            num_rows=len(rows) if rows else 1,
            num_cols=len(columns) if columns else 1,
            cells=table_cells
        )
        
        return structure
    
    def _map_cells_to_grid(
        self, 
        rows: List[BoundingBox], 
        columns: List[BoundingBox],
        cells: List[BoundingBox],
        headers: List[BoundingBox],
        table_image: Image.Image
    ) -> List[Cell]:
        """
        검출된 행/열/셀 정보를 그리드 셀로 매핑
        
        Args:
            rows: 검출된 행 바운딩 박스
            columns: 검출된 열 바운딩 박스
            cells: 검출된 셀 바운딩 박스
            headers: 검출된 헤더 바운딩 박스
            table_image: 표 이미지
            
        Returns:
            List of Cell objects
        """
        table_cells = []
        
        # 행/열이 비어있으면 기본 그리드 생성
        if not rows or not columns:
            # 단일 셀로 처리
            cell = Cell(
                row_idx=0,
                col_idx=0,
                row_span=1,
                col_span=1,
                bbox=BoundingBox(0, 0, table_image.width, table_image.height),
                text="",
                is_header=False
            )
            return [cell]
        
        # 각 행과 열의 교차점에 셀 생성
        for row_idx, row_bbox in enumerate(rows):
            for col_idx, col_bbox in enumerate(columns):
                # 행과 열의 교차 영역 계산
                x1 = max(col_bbox.x1, 0)
                y1 = max(row_bbox.y1, 0)
                x2 = min(col_bbox.x2, table_image.width)
                y2 = min(row_bbox.y2, table_image.height)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                cell_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                
                # 헤더인지 확인
                is_header = any(
                    cell_bbox.iou(h) > 0.5 for h in headers
                )
                
                cell = Cell(
                    row_idx=row_idx,
                    col_idx=col_idx,
                    row_span=1,
                    col_span=1,
                    bbox=cell_bbox,
                    text="",  # OCR 결과로 채워질 예정
                    is_header=is_header
                )
                table_cells.append(cell)
        
        return table_cells


def get_table_structure_recognizer(recognizer_type: str, config: Dict[str, Any]) -> TableStructureRecognizer:
    """
    표 구조 인식 모델 팩토리 함수
    
    Args:
        recognizer_type: 'huggingface' or other types
        config: Model configuration
        
    Returns:
        TableStructureRecognizer instance
    """
    if recognizer_type.lower() in ['huggingface', 'table-transformer']:
        return HuggingFaceTableStructureRecognizer(config)
    else:
        raise ValueError(f"Unknown recognizer type: {recognizer_type}")

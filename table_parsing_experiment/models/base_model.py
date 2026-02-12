"""
Base Model Interface
모든 표 파싱 모델의 추상 베이스 클래스
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from PIL import Image
import numpy as np


@dataclass
class BoundingBox:
    """바운딩 박스 데이터 클래스"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    label: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "label": self.label
        }
    
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Intersection over Union 계산"""
        x_left = max(self.x1, other.x1)
        y_top = max(self.y1, other.y1)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class Cell:
    """표 셀 데이터 클래스"""
    row_idx: int
    col_idx: int
    row_span: int = 1
    col_span: int = 1
    bbox: Optional[BoundingBox] = None
    text: str = ""
    is_header: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "row": self.row_idx,
            "col": self.col_idx,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "text": self.text,
            "is_header": self.is_header
        }


@dataclass
class TableStructure:
    """표 구조 데이터 클래스"""
    num_rows: int
    num_cols: int
    cells: List[Cell]
    bbox: Optional[BoundingBox] = None
    
    def to_dict(self) -> Dict:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "cells": [cell.to_dict() for cell in self.cells],
            "bbox": self.bbox.to_dict() if self.bbox else None
        }
    
    def to_html(self) -> str:
        """HTML 표 형식으로 변환 (TEDS 메트릭용)"""
        html = "<table>\n"
        
        # 셀을 행/열 순서로 정렬
        grid = {}
        for cell in self.cells:
            if cell.row_idx not in grid:
                grid[cell.row_idx] = {}
            grid[cell.row_idx][cell.col_idx] = cell
        
        # HTML 생성
        for row_idx in sorted(grid.keys()):
            tag = "th" if any(c.is_header for c in grid[row_idx].values()) else "td"
            html += "  <tr>\n"
            for col_idx in sorted(grid[row_idx].keys()):
                cell = grid[row_idx][col_idx]
                attrs = []
                if cell.row_span > 1:
                    attrs.append(f'rowspan="{cell.row_span}"')
                if cell.col_span > 1:
                    attrs.append(f'colspan="{cell.col_span}"')
                attr_str = " " + " ".join(attrs) if attrs else ""
                html += f"    <{tag}{attr_str}>{cell.text}</{tag}>\n"
            html += "  </tr>\n"
        
        html += "</table>"
        return html


class BaseModel(ABC):
    """모든 표 파싱 모델의 베이스 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """모델 로드"""
        pass
    
    @abstractmethod
    def predict(self, image: Image.Image) -> Any:
        """예측 수행"""
        pass
    
    def preprocess_image(self, image: Image.Image) -> Any:
        """이미지 전처리 (서브클래스에서 오버라이드 가능)"""
        return image
    
    def postprocess_output(self, output: Any) -> Any:
        """출력 후처리 (서브클래스에서 오버라이드 가능)"""
        return output


class OCREngine(BaseModel):
    """OCR 엔진 베이스 클래스"""
    
    @abstractmethod
    def recognize_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        이미지에서 텍스트 인식
        
        Returns:
            List of dicts with keys: 'text', 'bbox', 'confidence'
        """
        pass


class TableDetector(BaseModel):
    """표 검출 모델 베이스 클래스"""
    
    @abstractmethod
    def detect_tables(self, image: Image.Image) -> List[BoundingBox]:
        """
        이미지에서 표 영역 검출
        
        Returns:
            List of BoundingBox objects
        """
        pass


class TableStructureRecognizer(BaseModel):
    """표 구조 인식 모델 베이스 클래스"""
    
    @abstractmethod
    def recognize_structure(self, table_image: Image.Image) -> TableStructure:
        """
        표 이미지에서 구조 인식
        
        Returns:
            TableStructure object
        """
        pass

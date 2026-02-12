"""
간단한 표 분류기 - 표 유형을 자동으로 분류
"""
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import cv2


class TableClassifier:
    """표 유형 자동 분류기"""
    
    TABLE_TYPES = [
        'fully_bordered',
        'unbordered',
        'partially_bordered',
        'spanning_cells',
        'empty_cells',
        'complex'
    ]
    
    def __init__(self):
        pass
    
    def classify_table(self, image: Image.Image) -> Dict[str, any]:
        """
        표 이미지를 분석하여 유형 분류
        
        Args:
            image: PIL Image (표 영역)
            
        Returns:
            Dict with classification results
        """
        # PIL to OpenCV
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # 경계선 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 수평/수직 선 검출
        horizontal_lines = self._detect_lines(edges, 'horizontal')
        vertical_lines = self._detect_lines(edges, 'vertical')
        
        # 빈 영역 비율 계산
        empty_ratio = self._calculate_empty_ratio(gray)
        
        # 분류 로직
        classification = {
            'type': [],
            'has_borders': False,
            'border_coverage': 0.0,
            'empty_cell_ratio': empty_ratio,
            'horizontal_lines': len(horizontal_lines),
            'vertical_lines': len(vertical_lines),
            'complexity_score': 0.0
        }
        
        # 테두리 분석
        total_pixels = edges.size
        border_pixels = np.sum(edges > 0)
        border_coverage = border_pixels / total_pixels
        
        classification['border_coverage'] = border_coverage
        
        if border_coverage > 0.15:
            classification['type'].append('fully_bordered')
            classification['has_borders'] = True
        elif border_coverage < 0.05:
            classification['type'].append('unbordered')
        else:
            classification['type'].append('partially_bordered')
            classification['has_borders'] = True
        
        # 빈 셀 많음
        if empty_ratio > 0.3:
            classification['type'].append('empty_cells')
        
        # 복잡도 점수 (주관적 지표)
        complexity_score = (
            len(horizontal_lines) * 0.3 +
            len(vertical_lines) * 0.3 +
            empty_ratio * 0.2 +
            border_coverage * 0.2
        )
        classification['complexity_score'] = complexity_score
        
        if complexity_score > 0.7:
            classification['type'].append('complex')
        
        return classification
    
    def _detect_lines(self, edges: np.ndarray, direction: str) -> List[Tuple[int, int, int, int]]:
        """
        수평 또는 수직 선 검출
        
        Args:
            edges: Canny edge 이미지
            direction: 'horizontal' or 'vertical'
            
        Returns:
            검출된 선 목록
        """
        if direction == 'horizontal':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        detected = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            detected,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=50,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        return [tuple(line[0]) for line in lines]
    
    def _calculate_empty_ratio(self, gray: np.ndarray) -> float:
        """빈 영역 비율 계산"""
        # 이진화
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # 흰색(빈) 픽셀 비율
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        
        return white_pixels / total_pixels if total_pixels > 0 else 0.0


def classify_table_image(image_path: str) -> Dict:
    """
    이미지 파일에서 표 유형 분류
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        분류 결과
    """
    image = Image.open(image_path)
    classifier = TableClassifier()
    return classifier.classify_table(image)

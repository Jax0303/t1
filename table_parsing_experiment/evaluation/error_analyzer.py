"""
Error Analyzer
RAPTOR 연구의 오류 분류 체계 기반 오류 분석기
"""
from typing import Dict, List, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

from models.base_model import BoundingBox, TableStructure, Cell


class ErrorCategory(Enum):
    """오류 카테고리"""
    TD = "Table Detection"
    TSR = "Table Structure Recognition"
    OCR = "Optical Character Recognition"


@dataclass
class Error:
    """오류 데이터 클래스"""
    category: ErrorCategory
    error_type: str
    severity: str  # 'low', 'medium', 'high'
    description: str
    sample_id: str
    
    def to_dict(self) -> Dict:
        return {
            'category': self.category.value,
            'error_type': self.error_type,
            'severity': self.severity,
            'description': self.description,
            'sample_id': self.sample_id
        }


class ErrorAnalyzer:
    """오류 분석 클래스"""
    
    # RAPTOR 연구 기반 오류 유형 정의
    TD_ERROR_TYPES = [
        'noise_detected_above',
        'noise_detected_below',
        'wrong_table_detected',
        'false_positive',
        'false_negative'
    ]
    
    TSR_ERROR_TYPES = [
        'missing_elements',
        'wrong_header_detected',
        'segmentation_error',
        'fused_lines',
        'splitted_line',
        'implicit_header',
        'spanning_cells_error',
        'sub_tables'
    ]
    
    OCR_ERROR_TYPES = [
        'text_recognition_error',
        'missing_text',
        'extra_text'
    ]
    
    def __init__(self):
        self.errors: List[Error] = []
        self.error_counts = defaultdict(lambda: defaultdict(int))
    
    def analyze_detection_errors(
        self,
        pred_tables: List[BoundingBox],
        gt_tables: List[BoundingBox],
        sample_id: str,
        iou_threshold: float = 0.5
    ) -> List[Error]:
        """
        표 검출 오류 분석
        
        Args:
            pred_tables: 예측된 표 바운딩 박스
            gt_tables: 정답 표 바운딩 박스
            sample_id: 샘플 ID
            iou_threshold: IoU 임계값
            
        Returns:
            List of Error objects
        """
        errors = []
        
        # False Negatives (놓친 표)
        matched_gt = set()
        for pred in pred_tables:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_tables):
                iou = pred.iou(gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold:
                matched_gt.add(best_idx)
        
        for idx in range(len(gt_tables)):
            if idx not in matched_gt:
                error = Error(
                    category=ErrorCategory.TD,
                    error_type='false_negative',
                    severity='high',
                    description=f'Failed to detect table at {gt_tables[idx].to_dict()}',
                    sample_id=sample_id
                )
                errors.append(error)
                self._add_error(error)
        
        # False Positives (잘못 검출한 표)
        matched_pred = set()
        for gt in gt_tables:
            best_iou = 0.0
            best_idx = -1
            for idx, pred in enumerate(pred_tables):
                iou = pred.iou(gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold:
                matched_pred.add(best_idx)
        
        for idx in range(len(pred_tables)):
            if idx not in matched_pred:
                error = Error(
                    category=ErrorCategory.TD,
                    error_type='false_positive',
                    severity='medium',
                    description=f'Incorrectly detected table at {pred_tables[idx].to_dict()}',
                    sample_id=sample_id
                )
                errors.append(error)
                self._add_error(error)
        
        return errors
    
    def analyze_structure_errors(
        self,
        pred_structure: TableStructure,
        gt_structure: TableStructure,
        sample_id: str
    ) -> List[Error]:
        """
        표 구조 인식 오류 분석
        
        Args:
            pred_structure: 예측된 표 구조
            gt_structure: 정답 표 구조
            sample_id: 샘플 ID
            
        Returns:
            List of Error objects
        """
        errors = []
        
        # 행/열 개수 불일치
        if pred_structure.num_rows != gt_structure.num_rows:
            error = Error(
                category=ErrorCategory.TSR,
                error_type='segmentation_error',
                severity='high',
                description=f'Row count mismatch: predicted {pred_structure.num_rows}, expected {gt_structure.num_rows}',
                sample_id=sample_id
            )
            errors.append(error)
            self._add_error(error)
        
        if pred_structure.num_cols != gt_structure.num_cols:
            error = Error(
                category=ErrorCategory.TSR,
                error_type='segmentation_error',
                severity='high',
                description=f'Column count mismatch: predicted {pred_structure.num_cols}, expected {gt_structure.num_cols}',
                sample_id=sample_id
            )
            errors.append(error)
            self._add_error(error)
        
        # 셀 개수 불일치
        if len(pred_structure.cells) != len(gt_structure.cells):
            error = Error(
                category=ErrorCategory.TSR,
                error_type='missing_elements',
                severity='medium',
                description=f'Cell count mismatch: predicted {len(pred_structure.cells)}, expected {len(gt_structure.cells)}',
                sample_id=sample_id
            )
            errors.append(error)
            self._add_error(error)
        
        # 헤더 인식 오류
        pred_headers = [c for c in pred_structure.cells if c.is_header]
        gt_headers = [c for c in gt_structure.cells if c.is_header]
        
        if len(pred_headers) != len(gt_headers):
            error = Error(
                category=ErrorCategory.TSR,
                error_type='wrong_header_detected',
                severity='medium',
                description=f'Header count mismatch: predicted {len(pred_headers)}, expected {len(gt_headers)}',
                sample_id=sample_id
            )
            errors.append(error)
            self._add_error(error)
        
        # 스팬 셀 오류 (간단한 체크)
        pred_spanning = [c for c in pred_structure.cells if c.row_span > 1 or c.col_span > 1]
        gt_spanning = [c for c in gt_structure.cells if c.row_span > 1 or c.col_span > 1]
        
        if len(pred_spanning) != len(gt_spanning):
            error = Error(
                category=ErrorCategory.TSR,
                error_type='spanning_cells_error',
                severity='medium',
                description=f'Spanning cells mismatch: predicted {len(pred_spanning)}, expected {len(gt_spanning)}',
                sample_id=sample_id
            )
            errors.append(error)
            self._add_error(error)
        
        return errors
    
    def analyze_ocr_errors(
        self,
        pred_structure: TableStructure,
        gt_structure: TableStructure,
        sample_id: str
    ) -> List[Error]:
        """
        OCR 오류 분석
        
        Args:
            pred_structure: 예측된 표 구조 (텍스트 포함)
            gt_structure: 정답 표 구조 (텍스트 포함)
            sample_id: 샘플 ID
            
        Returns:
            List of Error objects
        """
        errors = []
        
        # 셀 위치 매칭하여 텍스트 비교
        pred_cells_dict = {(c.row_idx, c.col_idx): c for c in pred_structure.cells}
        gt_cells_dict = {(c.row_idx, c.col_idx): c for c in gt_structure.cells}
        
        all_positions = set(pred_cells_dict.keys()) | set(gt_cells_dict.keys())
        
        text_errors = 0
        for pos in all_positions:
            pred_cell = pred_cells_dict.get(pos)
            gt_cell = gt_cells_dict.get(pos)
            
            if pred_cell and gt_cell:
                if pred_cell.text != gt_cell.text:
                    text_errors += 1
            elif not pred_cell:
                text_errors += 1
            elif not gt_cell:
                text_errors += 1
        
        if text_errors > 0:
            error = Error(
                category=ErrorCategory.OCR,
                error_type='text_recognition_error',
                severity='low',
                description=f'{text_errors} cells have text recognition errors',
                sample_id=sample_id
            )
            errors.append(error)
            self._add_error(error)
        
        return errors
    
    def _add_error(self, error: Error):
        """오류를 리스트와 카운트에 추가"""
        self.errors.append(error)
        self.error_counts[error.category.value][error.error_type] += 1
    
    def get_error_distribution(self) -> Dict[str, Dict[str, int]]:
        """오류 분포 반환"""
        return dict(self.error_counts)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """오류 요약 통계"""
        summary = {
            'total_errors': len(self.errors),
            'by_category': {},
            'by_type': {},
            'by_severity': defaultdict(int)
        }
        
        for error in self.errors:
            # 카테고리별
            cat = error.category.value
            if cat not in summary['by_category']:
                summary['by_category'][cat] = 0
            summary['by_category'][cat] += 1
            
            # 타입별
            error_type = error.error_type
            if error_type not in summary['by_type']:
                summary['by_type'][error_type] = 0
            summary['by_type'][error_type] += 1
            
            # 심각도별
            summary['by_severity'][error.severity] += 1
        
        summary['by_severity'] = dict(summary['by_severity'])
        
        return summary
    
    def reset(self):
        """오류 데이터 초기화"""
        self.errors = []
        self.error_counts = defaultdict(lambda: defaultdict(int))

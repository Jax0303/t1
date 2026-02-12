"""
Complete Table Parsing Pipeline
전체 표 파싱 파이프라인 통합
"""
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import time
from pathlib import Path

from models import (
    get_ocr_engine,
    get_table_detector,
    get_table_structure_recognizer,
    BoundingBox,
    TableStructure,
    Cell
)
from evaluation import MetricsCalculator, ErrorAnalyzer


class TableParsingPipeline:
    """표 파싱 전체 파이프라인"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 파이프라인 설정
        """
        self.config = config
        
        # 모델 초기화
        self.ocr_engine = None
        self.table_detector = None
        self.structure_recognizer = None
        
        # 평가 모듈
        self.metrics_calculator = None
        self.error_analyzer = ErrorAnalyzer()
        
        self._initialize_models()
    
    def _initialize_models(self):
        """모델 초기화"""
        print("Initializing models...")
        
        # OCR 엔진
        ocr_config = self.config.get('ocr', {})
        ocr_type = ocr_config.get('type', 'tesseract')
        self.ocr_engine = get_ocr_engine(ocr_type, ocr_config)
        if hasattr(self.ocr_engine, 'load_model'):
            self.ocr_engine.load_model()
        
        # 표 검출
        td_config = self.config.get('table_detection', {})
        td_type = td_config.get('type', 'huggingface')
        self.table_detector = get_table_detector(td_type, td_config)
        self.table_detector.load_model()
        
        # 표 구조 인식
        tsr_config = self.config.get('table_structure', {})
        tsr_type = tsr_config.get('type', 'huggingface')
        self.structure_recognizer = get_table_structure_recognizer(tsr_type, tsr_config)
        self.structure_recognizer.load_model()
        
        # 메트릭 계산기
        metrics_config = self.config.get('metrics', {})
        self.metrics_calculator = MetricsCalculator(metrics_config)
        
        print("All models initialized successfully!")
    
    def parse_image(
        self,
        image: Image.Image,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        이미지에서 표 파싱
        
        Args:
            image: PIL Image
            return_intermediates: 중간 결과 반환 여부
            
        Returns:
            Dict containing parsing results
        """
        results = {
            'success': False,
            'tables': [],
            'execution_time': {}
        }
        
        try:
            # 1. 표 검출
            start_time = time.time()
            detected_tables = self.table_detector.detect_tables(image)
            results['execution_time']['table_detection'] = time.time() - start_time
            
            if not detected_tables:
                print("No tables detected in the image")
                return results
            
            # 2. 각 표에 대해 구조 인식 및 OCR
            for table_idx, table_bbox in enumerate(detected_tables):
                # 표 영역 추출
                table_region = image.crop((
                    int(table_bbox.x1),
                    int(table_bbox.y1),
                    int(table_bbox.x2),
                    int(table_bbox.y2)
                ))
                
                # 구조 인식
                start_time = time.time()
                table_structure = self.structure_recognizer.recognize_structure(table_region)
                tsr_time = time.time() - start_time
                
                # OCR
                start_time = time.time()
                ocr_results = self.ocr_engine.recognize_text(table_region)
                ocr_time = time.time() - start_time
                
                # OCR 결과를 셀에 매핑
                self._assign_text_to_cells(table_structure, ocr_results)
                
                table_result = {
                    'table_idx': table_idx,
                    'bbox': table_bbox.to_dict(),
                    'structure': table_structure.to_dict(),
                    'html': table_structure.to_html(),
                    'execution_time': {
                        'structure_recognition': tsr_time,
                        'ocr': ocr_time
                    }
                }
                
                if return_intermediates:
                    table_result['ocr_raw'] = ocr_results
                    table_result['table_image'] = table_region
                
                results['tables'].append(table_result)
            
            results['success'] = True
            
        except Exception as e:
            print(f"Error during table parsing: {e}")
            results['error'] = str(e)
        
        return results
    
    def _assign_text_to_cells(
        self,
        table_structure: TableStructure,
        ocr_results: List[Dict[str, Any]]
    ):
        """
        OCR 결과를 표 셀에 할당
        
        Args:
            table_structure: TableStructure (수정됨)
            ocr_results: OCR 결과 리스트
        """
        for cell in table_structure.cells:
            if cell.bbox is None:
                continue
            
            # 이 셀과 겹치는 텍스트 찾기
            cell_texts = []
            for ocr_item in ocr_results:
                ocr_bbox = ocr_item['bbox']
                
                # IoU 계산
                iou = cell.bbox.iou(ocr_bbox)
                
                # IoU가 임계값 이상이면 텍스트 포함
                if iou > 0.3:  # 임계값 조정 가능
                    cell_texts.append((ocr_item['text'], ocr_bbox.y1, ocr_bbox.x1))
            
            # 위에서 아래, 왼쪽에서 오른쪽 순으로 정렬
            cell_texts.sort(key=lambda x: (x[1], x[2]))
            
            # 텍스트 결합
            cell.text = ' '.join([text for text, _, _ in cell_texts])
    
    def evaluate(
        self,
        pred_results: Dict[str, Any],
        ground_truth: Dict[str, Any],
        sample_id: str
    ) -> Dict[str, Any]:
        """
        파싱 결과 평가
        
        Args:
            pred_results: parse_image()의 결과
            ground_truth: 정답 데이터
            sample_id: 샘플 ID
            
        Returns:
            평가 메트릭 및 오류 분석 결과
        """
        evaluation_results = {
            'sample_id': sample_id,
            'metrics': {},
            'errors': []
        }
        
        # 표 검출 평가
        pred_tables = [
            BoundingBox(**t['bbox']) for t in pred_results.get('tables', [])
        ]
        gt_tables = ground_truth.get('tables', [])
        
        # 구조 평가 (첫 번째 표만)
        if pred_results.get('tables') and gt_tables:
            pred_structure_dict = pred_results['tables'][0]['structure']
            gt_structure_dict = gt_tables[0]
            
            # Dict를 TableStructure로 변환
            pred_structure = self._dict_to_structure(pred_structure_dict)
            gt_structure = self._dict_to_structure(gt_structure_dict)
            
            # 메트릭 계산
            metrics = self.metrics_calculator.calculate_all_metrics(
                pred_structure, gt_structure, pred_tables, 
                [BoundingBox(**gt['bbox']) for gt in gt_tables]
            )
            evaluation_results['metrics'] = metrics
            
            # 오류 분석
            td_errors = self.error_analyzer.analyze_detection_errors(
                pred_tables,
                [BoundingBox(**gt['bbox']) for gt in gt_tables],
                sample_id
            )
            
            tsr_errors = self.error_analyzer.analyze_structure_errors(
                pred_structure, gt_structure, sample_id
            )
            
            ocr_errors = self.error_analyzer.analyze_ocr_errors(
                pred_structure, gt_structure, sample_id
            )
            
            all_errors = td_errors + tsr_errors + ocr_errors
            evaluation_results['errors'] = [e.to_dict() for e in all_errors]
        
        return evaluation_results
    
    def _dict_to_structure(self, structure_dict: Dict) -> TableStructure:
        """Dict를 TableStructure 객체로 변환"""
        cells = []
        for cell_dict in structure_dict.get('cells', []):
            bbox = None
            if cell_dict.get('bbox'):
                bbox = BoundingBox(**cell_dict['bbox'])
            
            cell = Cell(
                row_idx=cell_dict['row'],
                col_idx=cell_dict['col'],
                row_span=cell_dict.get('row_span', 1),
                col_span=cell_dict.get('col_span', 1),
                bbox=bbox,
                text=cell_dict.get('text', ''),
                is_header=cell_dict.get('is_header', False)
            )
            cells.append(cell)
        
        return TableStructure(
            num_rows=structure_dict['num_rows'],
            num_cols=structure_dict['num_cols'],
            cells=cells,
            bbox=BoundingBox(**structure_dict['bbox']) if structure_dict.get('bbox') else None
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """누적된 오류 요약 반환"""
        return self.error_analyzer.get_error_summary()
    
    def reset_errors(self):
        """오류 카운터 초기화"""
        self.error_analyzer.reset()

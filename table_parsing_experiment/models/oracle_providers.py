"""
Oracle Providers for Oracle Swap Experiments.
GT 데이터를 기반으로 동작하는 OCR 및 TSR 프로바이더.
"""
from typing import List, Dict, Any, Optional
from PIL import Image
from .base_model import OCREngine, TableStructureRecognizer, TableStructure, Cell, BoundingBox

class OracleOCRProvider(OCREngine):
    """
    Ground Truth 데이터를 사용하는 Oracle OCR 프로바이더.
    """
    def __init__(self, mode: str = "TOKEN"):
        """
        Args:
            mode: "TOKEN" (GT 토큰 기반) 또는 "CELL" (GT 셀 텍스트 기반)
        """
        super().__init__({"mode": mode})
        self.mode = mode.upper()
        self.current_gt = None

    def load_model(self):
        pass

    def set_gt(self, gt_data: Dict[str, Any]):
        """현재 샘플의 GT 데이터를 설정"""
        self.current_gt = gt_data

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        return self.recognize_text(image)

    def recognize_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        if self.current_gt is None:
            return []

        results = []
        if self.mode == "TOKEN":
            # GT가 token bbox를 제공하는 경우 (SciTSR 등)
            # SciTSR의 경우 'chunk'가 token 역할을 함
            tokens = self.current_gt.get('tokens', [])
            if not tokens:
                # tokens이 없으면 cells에서 추출 시도
                for cell in self.current_gt.get('cells', []):
                    if cell.get('bbox'):
                        results.append({
                            'text': cell.get('text', ''),
                            'bbox': cell.get('bbox'),
                            'confidence': 1.0
                        })
            else:
                for token in tokens:
                    results.append({
                        'text': token.get('text', ''),
                        'bbox': token.get('bbox'),
                        'confidence': 1.0
                    })
        elif self.mode == "CELL":
            # GT cell_text를 직접 사용 (mapping 병목 제외)
            for cell in self.current_gt.get('cells', []):
                results.append({
                    'text': cell.get('text', ''),
                    'bbox': cell.get('bbox'),
                    'confidence': 1.0
                })
        
        return results

class OracleTSRProvider(TableStructureRecognizer):
    """
    Ground Truth 데이터를 사용하는 Oracle TSR 프로바이더.
    """
    def __init__(self):
        super().__init__({})
        self.current_gt = None

    def load_model(self):
        pass

    def set_gt(self, gt_data: Dict[str, Any]):
        """현재 샘플의 GT 데이터를 설정"""
        self.current_gt = gt_data

    def predict(self, table_image: Image.Image) -> TableStructure:
        return self.recognize_structure(table_image)

    def recognize_structure(self, table_image: Image.Image) -> TableStructure:
        if self.current_gt is None:
            return TableStructure(num_rows=1, num_cols=1, cells=[])

        gt_cells = self.current_gt.get('cells', [])
        cells = []
        max_row = 0
        max_col = 0
        
        for gt_cell in gt_cells:
            row = gt_cell.get('row', 0)
            col = gt_cell.get('col', 0)
            row_span = gt_cell.get('row_span', 1)
            col_span = gt_cell.get('col_span', 1)
            bbox_data = gt_cell.get('bbox')
            
            bbox = None
            if bbox_data:
                if isinstance(bbox_data, BoundingBox):
                    bbox = bbox_data
                elif isinstance(bbox_data, dict):
                    bbox = BoundingBox(**bbox_data)
                elif isinstance(bbox_data, (list, tuple)):
                    bbox = BoundingBox(x1=bbox_data[0], y1=bbox_data[1], x2=bbox_data[2], y2=bbox_data[3])
            
            cell = Cell(
                row_idx=row,
                col_idx=col,
                row_span=row_span,
                col_span=col_span,
                bbox=bbox,
                text=gt_cell.get('text', ''),
                is_header=gt_cell.get('is_header', False)
            )
            cells.append(cell)
            max_row = max(max_row, row + row_span)
            max_col = max(max_col, col + col_span)

        return TableStructure(
            num_rows=max_row,
            num_cols=max_col,
            cells=cells
        )

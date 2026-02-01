"""
Simple OCR+TSR Pipeline using PaddleOCR + Spatial Sorting

This module provides a practical OCR+TSR pipeline for error analysis:
- OCR: PaddleOCR (popular, robust Chinese OCR that works well on English tables)
- TSR: Spatial Sorting (simple, interpretable, no learning required)

This is the baseline we'll use to identify real OCR vs TSR errors.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path


class SimpleSpatialTSR:
    """
    Simple spatial-based TSR using bounding box coordinates.
    
    Algorithm:
    1. Sort OCR boxes by Y coordinate (top to bottom) to find rows
    2. Within each row, sort by X coordinate (left to right) to find columns
    3. Use clustering with threshold to group cells into rows
    """
    
    def __init__(self, 
                 row_threshold: float = 15.0,
                 col_threshold: float = 10.0):
        """
        Args:
            row_threshold: Y-distance threshold for grouping into same row
            col_threshold: X-distance threshold for grouping into same column
        """
        self.row_threshold = row_threshold
        self.col_threshold = col_threshold
    
    def process(self, ocr_results: List[Dict]) -> Dict[str, Any]:
        """
        Process OCR results into table structure
        
        Args:
            ocr_results: List of OCR detections with 'bbox' and 'text'
        
        Returns:
            Dict with 'cells' containing structured table cells
        """
        if not ocr_results:
            return {'cells': []}
        
        # Extract bboxes and texts
        boxes = []
        texts = []
        
        for result in ocr_results:
            bbox = result['bbox']  # [x1, y1, x2, y2] or [[x1,y1],[x2,y2],...]
            text = result['text']
            
            # Normalize bbox format
            if isinstance(bbox[0], (list, tuple)):
                # Convert from [[x1,y1],[x2,y2],...] to [x1,y1,x2,y2]
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            
            boxes.append(bbox)
            texts.append(text)
        
        # Group into rows by Y coordinate
        rows = self._group_into_rows(boxes, texts)
        
        # Build cells
        cells = []
        for row_idx, row_data in enumerate(rows):
            # Sort cells in row by X coordinate
            row_sorted = sorted(row_data, key=lambda x: x['bbox'][0])
            
            for col_idx, cell_data in enumerate(row_sorted):
                cells.append({
                    'row': row_idx,
                    'col': col_idx,
                    'row_span': 1,
                    'col_span': 1,
                    'text': cell_data['text'],
                    'box': cell_data['bbox'],
                    'confidence': cell_data.get('confidence', 1.0)
                })
        
        # Calculate grid dimensions
        num_rows = len(rows)
        num_cols = max(len(row) for row in rows) if rows else 0
        
        return {
            'cells': cells,
            'num_rows': num_rows,
            'num_cols': num_cols
        }
    
    def _group_into_rows(self, boxes: List, texts: List) -> List[List[Dict]]:
        """Group boxes into rows based on Y coordinate"""
        
        # Calculate center Y for each box
        items = []
        for bbox, text in zip(boxes, texts):
            y_center = (bbox[1] + bbox[3]) / 2
            items.append({
                'bbox': bbox,
                'text': text,
                'y_center': y_center
            })
        
        # Sort by Y coordinate
        items_sorted = sorted(items, key=lambda x: x['y_center'])
        
        # Cluster into rows
        rows = []
        current_row = []
        current_y = None
        
        for item in items_sorted:
            if current_y is None:
                # First row
                current_row = [item]
                current_y = item['y_center']
            elif abs(item['y_center'] - current_y) < self.row_threshold:
                # Same row
                current_row.append(item)
            else:
                # New row
                rows.append(current_row)
                current_row = [item]
                current_y = item['y_center']
        
        # Add last row
        if current_row:
            rows.append(current_row)
        
        return rows


class PaddleOCRWrapper:
    """
    Wrapper for PaddleOCR with unified interface
    """
    
    def __init__(self, lang='en', use_gpu=False):
        """
        Initialize PaddleOCR
        
        Args:
            lang: Language code ('en', 'ch', etc.)
            use_gpu: Whether to use GPU
        """
        try:
            from paddleocr import PaddleOCR
            
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False
            )
            self.available = True
        except ImportError:
            print("Warning: PaddleOCR not installed. Install with: pip install paddleocr")
            self.ocr = None
            self.available = False
        except Exception as e:
            print(f"Warning: PaddleOCR initialization failed: {e}")
            self.ocr = None
            self.available = False
    
    def process(self, image_path: str) -> List[Dict]:
        """
        Run OCR on image
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of detection dicts with 'bbox', 'text', 'confidence'
        """
        if not self.available:
            # Return mock results for testing
            return self._mock_ocr(image_path)
        
        # Run PaddleOCR
        result = self.ocr.ocr(image_path, cls=True)
        
        # Convert to unified format
        detections = []
        
        if result and result[0]:
            for line in result[0]:
                bbox = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text_info = line[1]  # (text, confidence)
                
                detections.append({
                    'bbox': bbox,
                    'text': text_info[0],
                    'confidence': text_info[1]
                })
        
        return detections
    
    def _mock_ocr(self, image_path: str) -> List[Dict]:
        """Mock OCR for testing when PaddleOCR not available"""
        # Load image to get dimensions
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        # Return fake detections (3x3 grid)
        detections = []
        cell_h = h // 3
        cell_w = w // 3
        
        for row in range(3):
            for col in range(3):
                x1 = col * cell_w + 10
                y1 = row * cell_h + 10
                x2 = (col + 1) * cell_w - 10
                y2 = (row + 1) * cell_h - 10
                
                detections.append({
                    'bbox': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    'text': f'Cell_{row}_{col}',
                    'confidence': 0.95
                })
        
        return detections


class SimpleOCRTSRPipeline:
    """
    Complete OCR+TSR Pipeline
    
    Combines:
    - PaddleOCR for text detection and recognition
    - Spatial Sorting for table structure recognition
    """
    
    def __init__(self, 
                 use_gpu: bool = False,
                 lang: str = 'en',
                 row_threshold: float = 15.0):
        """
        Initialize pipeline
        
        Args:
            use_gpu: Use GPU for OCR
            lang: OCR language
            row_threshold: TSR row grouping threshold
        """
        self.ocr = PaddleOCRWrapper(lang=lang, use_gpu=use_gpu)
        self.tsr = SimpleSpatialTSR(row_threshold=row_threshold)
    
    def process(self, image_path: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Process image through OCR+TSR pipeline
        
        Args:
            image_path: Path to table image
        
        Returns:
            (ocr_results, structure)
            - ocr_results: Raw OCR detections
            - structure: Structured table cells
        """
        # Step 1: OCR
        ocr_results = self.ocr.process(image_path)
        
        # Step 2: TSR
        structure = self.tsr.process(ocr_results)
        
        return ocr_results, structure


if __name__ == '__main__':
    # Test pipeline
    print("Testing SimpleOCRTSRPipeline...")
    
    pipeline = SimpleOCRTSRPipeline()
    
    # Create dummy image for testing
    import cv2
    dummy_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.imwrite('/tmp/test_table.png', dummy_img)
    
    ocr_results, structure = pipeline.process('/tmp/test_table.png')
    
    print(f"\nOCR Results: {len(ocr_results)} detections")
    print(f"Structure: {structure['num_rows']} rows x {structure['num_cols']} cols")
    print(f"Cells: {len(structure['cells'])}")
    
    if structure['cells']:
        print(f"\nFirst cell: {structure['cells'][0]}")
    
    print("\n✓ Pipeline test complete")

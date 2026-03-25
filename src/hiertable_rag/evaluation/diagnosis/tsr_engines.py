"""
TSR Engine Wrappers

Provides unified interface for baseline, GNN-CSP, and GT TSR engines.
"""

import sys
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import torch

from .utils import extract_text_content, normalize_bbox

logger = logging.getLogger(__name__)


class BaseTSREngine(ABC):
    """
    Abstract base class for TSR engines
    """
    
    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        ocr_boxes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict table structure from image and OCR results
        
        Args:
            image: Image array (H, W, 3)
            ocr_boxes: List of OCR results with 'bbox', 'text', 'confidence'
        
        Returns:
            Structure dict with format:
            {
                'cells': [
                    {
                        'row': int,
                        'col': int,
                        'row_span': int,
                        'col_span': int,
                        'text': str,
                        'box': [x1, y1, x2, y2],
                        'confidence': float
                    }
                ],
                'num_rows': int,
                'num_cols': int
            }
        """
        pass


class BaselineTSR(BaseTSREngine):
    """
    Baseline TSR using simple spatial clustering
    
    Reuses SimpleSpatialTSR from existing codebase if available.
    """
    
    def __init__(self, row_threshold: float = 15.0, col_threshold: float = 15.0):
        """
        Initialize baseline TSR
        
        Args:
            row_threshold: Y-coordinate threshold for row grouping
            col_threshold: X-coordinate threshold for column grouping
        """
        self.row_threshold = row_threshold
        self.col_threshold = col_threshold
        
        # Try to import SimpleSpatialTSR from existing code
        self.simple_tsr = None
        try:
            # Add project paths
            project_root = Path(__file__).parent.parent.parent.parent.parent
            sys.path.insert(0, str(project_root / 'src'))
            
            from ocr_tsr.simple_pipeline import SimpleSpatialTSR
            self.simple_tsr = SimpleSpatialTSR(row_threshold=row_threshold)
            logger.info("SimpleSpatialTSR loaded successfully")
        except ImportError:
            logger.warning("SimpleSpatialTSR not available, using fallback implementation")
    
    def predict(
        self,
        image: np.ndarray,
        ocr_boxes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict structure using spatial clustering
        
        Args:
            image: Image array
            ocr_boxes: OCR results
        
        Returns:
            Structure dict
        """
        if self.simple_tsr is not None:
            return self._use_existing_tsr(ocr_boxes)
        else:
            return self._fallback_tsr(ocr_boxes)
    
    def _use_existing_tsr(self, ocr_boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use existing SimpleSpatialTSR"""
        try:
            result = self.simple_tsr.process(ocr_boxes)
            return result
        except Exception as e:
            logger.error(f"SimpleSpatialTSR failed: {e}")
            return self._fallback_tsr(ocr_boxes)
    
    def _fallback_tsr(self, ocr_boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback implementation: simple spatial clustering
        
        Groups boxes by Y-coordinate for rows and X-coordinate for columns.
        """
        if not ocr_boxes:
            return {'cells': [], 'num_rows': 0, 'num_cols': 0}
        
        # Normalize bboxes
        boxes_with_coords = []
        for box in ocr_boxes:
            bbox = normalize_bbox(box['bbox'])
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            boxes_with_coords.append({
                'bbox': bbox,
                'text': box['text'],
                'cx': cx,
                'cy': cy,
                'x1': x1,
                'y1': y1
            })
        
        # Sort by Y then X
        boxes_with_coords.sort(key=lambda b: (b['cy'], b['cx']))
        
        # Assign rows
        row_assignments = []
        current_row = 0
        prev_cy = None
        
        for box in boxes_with_coords:
            if prev_cy is None or abs(box['cy'] - prev_cy) > self.row_threshold:
                current_row += 1
            row_assignments.append(current_row - 1)
            prev_cy = box['cy']
        
        # Assign columns (per row)
        cells = []
        for i, box in enumerate(boxes_with_coords):
            row = row_assignments[i]
            
            # Find boxes in same row
            row_boxes = [(j, b) for j, (b, r) in enumerate(zip(boxes_with_coords, row_assignments)) if r == row]
            row_boxes.sort(key=lambda x: x[1]['cx'])
            
            # Find column index
            col = next((j for j, (idx, _) in enumerate(row_boxes) if idx == i), 0)
            
            cells.append({
                'row': row,
                'col': col,
                'row_span': 1,
                'col_span': 1,
                'text': box['text'],
                'box': box['bbox'],
                'confidence': 0.8
            })
        
        num_rows = max(c['row'] for c in cells) + 1 if cells else 0
        num_cols = max(c['col'] for c in cells) + 1 if cells else 0
        
        return {
            'cells': cells,
            'num_rows': num_rows,
            'num_cols': num_cols
        }


class GNNCSPTSREngine(BaseTSREngine):
    """
    GNN-CSP TSR engine wrapper
    
    Wraps existing GNNCSPPipeline from src/hiertable_rag/gnn_csp/pipeline.py
    """
    
    def __init__(
        self,
        rca_checkpoint: Optional[str] = None,
        gnn_checkpoint: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize GNN-CSP engine
        
        Args:
            rca_checkpoint: Path to RCA model checkpoint
            gnn_checkpoint: Path to GNN model checkpoint
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Import GNN-CSP pipeline
        try:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            sys.path.insert(0, str(project_root / 'src'))
            
            from hiertable_rag.gnn_csp.pipeline import GNNCSPPipeline
            
            self.pipeline = GNNCSPPipeline(
                rca_checkpoint=rca_checkpoint,
                gnn_checkpoint=gnn_checkpoint,
                device=self.device
            )
            logger.info(f"GNN-CSP pipeline initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GNN-CSP pipeline: {e}")
            raise
    
    def predict(
        self,
        image: np.ndarray,
        ocr_boxes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict structure using GNN-CSP
        
        Args:
            image: Image array
            ocr_boxes: OCR results
        
        Returns:
            Structure dict
        """
        try:
            # Convert image to tensor
            if isinstance(image, np.ndarray):
                # Normalize to [0, 1] and transpose to (C, H, W)
                image_tensor = torch.from_numpy(image).float() / 255.0
                if image_tensor.ndim == 3 and image_tensor.shape[2] == 3:
                    image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                image_tensor = image_tensor.to(self.device)
            else:
                image_tensor = image
            
            # Convert OCR boxes to expected format
            formatted_boxes = []
            for box in ocr_boxes:
                bbox = normalize_bbox(box['bbox'])
                formatted_boxes.append({
                    'box': bbox,  # [x1, y1, x2, y2]
                    'content': box['text']
                })
            
            # Run pipeline
            result = self.pipeline.parse(
                image=image_tensor,
                ocr_boxes=formatted_boxes,
                use_csp=True,
                use_semantic=True
            )
            
            # Convert output format
            return self._convert_output(result)
        
        except Exception as e:
            logger.error(f"GNN-CSP prediction failed: {e}")
            # Return empty structure
            return {'cells': [], 'num_rows': 0, 'num_cols': 0}
    
    def _convert_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert GNN-CSP output to standard format
        
        Args:
            result: Raw GNN-CSP output
        
        Returns:
            Standardized structure dict
        """
        cells = result.get('cells', [])
        
        # Ensure cells have required fields
        standardized_cells = []
        for cell in cells:
            standardized_cells.append({
                'row': cell.get('row', 0),
                'col': cell.get('col', 0),
                'row_span': cell.get('row_span', 1),
                'col_span': cell.get('col_span', 1),
                'text': cell.get('text', cell.get('content', '')),
                'box': normalize_bbox(cell.get('box', cell.get('bbox', [0, 0, 10, 10]))),
                'confidence': cell.get('confidence', 0.9)
            })
        
        num_rows = result.get('num_rows', max([c['row'] + c['row_span'] for c in standardized_cells], default=0))
        num_cols = result.get('num_cols', max([c['col'] + c['col_span'] for c in standardized_cells], default=0))
        
        return {
            'cells': standardized_cells,
            'num_rows': num_rows,
            'num_cols': num_cols
        }


class GTStructureEngine(BaseTSREngine):
    """
    Ground truth structure engine
    
    Not a real TSR - simply returns the ground truth structure.
    Used for S3 and S4 scenarios.
    """
    
    def __init__(self, structure_map: Optional[Dict[str, Dict]] = None):
        """
        Initialize GT engine
        
        Args:
            structure_map: Mapping of table_id to GT structure
        """
        self.structure_map = structure_map or {}
        self.current_table_id = None
    
    def set_structure(self, table_id: str, structure: Dict[str, Any]):
        """
        Set GT structure for a table
        
        Args:
            table_id: Table identifier
            structure: GT structure dict
        """
        self.structure_map[table_id] = structure
    
    def set_current_table(self, table_id: str):
        """
        Set current table ID for prediction
        
        Args:
            table_id: Table identifier
        """
        self.current_table_id = table_id
    
    def predict(
        self,
        image: np.ndarray,
        ocr_boxes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Return GT structure
        
        Args:
            image: Image array (not used)
            ocr_boxes: OCR results (used for matching text to cells)
        
        Returns:
            GT structure
        """
        if self.current_table_id is None:
            logger.warning("No current table ID set for GT engine")
            return {'cells': [], 'num_rows': 0, 'num_cols': 0}
        
        structure = self.structure_map.get(self.current_table_id)
        
        if structure is None:
            logger.warning(f"No GT structure available for {self.current_table_id}")
            return {'cells': [], 'num_rows': 0, 'num_cols': 0}
        
        # Convert GT cells to standard format
        gt_cells = structure.get('cells', [])
        standardized_cells = []
        
        for cell in gt_cells:
            row = cell.get('start_row', cell.get('row', 0))
            col = cell.get('start_col', cell.get('col', 0))
            row_span = cell.get('end_row', row) - row + 1
            col_span = cell.get('end_col', col) - col + 1
            
            standardized_cells.append({
                'row': row,
                'col': col,
                'row_span': row_span,
                'col_span': col_span,
                'text': extract_text_content(cell),
                'box': normalize_bbox(cell.get('box', cell.get('bbox', [0, 0, 10, 10]))),
                'confidence': 1.0
            })
        
        num_rows = max([c['row'] + c['row_span'] for c in standardized_cells], default=0)
        num_cols = max([c['col'] + c['col_span'] for c in standardized_cells], default=0)
        
        return {
            'cells': standardized_cells,
            'num_rows': num_rows,
            'num_cols': num_cols
        }

# ============================================================================
# External Baseline TSR Engines  
# ============================================================================

class TATREngine(BaseTSREngine):
    """Table Transformer (TATR) - structure-only detector"""
    
    def __init__(self, model_name="microsoft/table-transformer-structure-recognition",
                 device="cuda", threshold=0.2, iou_threshold=0.3):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.uses_ocr = False
        
        try:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
            from PIL import Image
            
            logger.info(f"Loading TATR: {model_name}")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
            self.model = self.model.to(self.device).eval()
            self.Image = Image
            logger.info(f"TATR loaded on {self.device}")
        except Exception as e:
            logger.error(f"TATR load failed: {e}")
            raise
    
    def predict(self, image, ocr_boxes):
        """Predict using TATR"""
        try:
            if isinstance(image, np.ndarray):
                pil_image = self.Image.fromarray(image.astype(np.uint8))
            else:
                pil_image = image
            
            encoding = self.processor(images=pil_image, return_tensors="pt")
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoding)
            
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=self.threshold, target_sizes=target_sizes)[0]
            
            boxes = results["boxes"].cpu().numpy()
            labels = results["labels"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            
            id2label = self.model.config.id2label
            label_strings = [id2label[lid] for lid in labels]
            
            return self._canonicalize_grid(boxes, label_strings, scores)
        except Exception as e:
            logger.error(f"TATR predict failed: {e}")
            return {'cells': [], 'num_rows': 0, 'num_cols': 0}
    
    def _canonicalize_grid(self, boxes, labels, scores):
        """Grid canonicalization from row/col/cell boxes"""  
        from .structures import bbox_iou
        
        rows, cols, spanning = [], [], []
        for box, label, score in zip(boxes, labels, scores):
            data = {'box': box, 'score': score}
            if 'row' in label.lower() and 'header' not in label.lower():
                rows.append(data)
            elif 'column' in label.lower():
                cols.append(data)
            elif 'spanning' in label.lower():
                spanning.append(data)
        
        rows.sort(key=lambda b: (b['box'][1] + b['box'][3]) / 2)
        cols.sort(key=lambda b: (b['box'][0] + b['box'][2]) / 2)
        
        n_rows, n_cols = len(rows), len(cols)
        if n_rows == 0 or n_cols == 0:
            logger.warning("TATR: no rows/cols detected")
            return {'cells': [], 'num_rows': 0, 'num_cols': 0}
        
        # Base grid
        cells = []
        for i, rdata in enumerate(rows):
            for j, cdata in enumerate(cols):
                rb, cb = rdata['box'], cdata['box']
                x0, y0 = max(cb[0], rb[0]), max(rb[1], cb[1])
                x1, y1 = min(cb[2], rb[2]), min(rb[3], cb[3])
                
                if x1 > x0 and y1 > y0:
                    cells.append({
                        'row': i, 'col': j, 'row_span': 1, 'col_span': 1,
                        'box': [x0, y0, x1, y1], 'text': '',
                        'confidence': (rdata['score'] + cdata['score']) / 2
                    })
        
        # Merge spanning cells
        if spanning:
            merged_idx = set()
            final = []
            for sdata in spanning:
                sbox = tuple(sdata['box'])
                overlaps = []
                for idx, cell in enumerate(cells):
                    if idx in merged_idx:
                        continue
                    if bbox_iou(sbox, tuple(cell['box'])) > self.iou_threshold:
                        overlaps.append((idx, cell))
                
                if overlaps:
                    rs = [c['row'] for _, c in overlaps]
                    cs = [c['col'] for _, c in overlaps]
                    r0, r1 = min(rs), max(rs) + 1
                    c0, c1 = min(cs), max(cs) + 1
                    
                    final.append({
                        'row': r0, 'col': c0, 'row_span': r1-r0, 'col_span': c1-c0,
                        'box': list(sbox), 'text': '', 'confidence': sdata['score']
                    })
                    for idx, _ in overlaps:
                        merged_idx.add(idx)
            
            for idx, cell in enumerate(cells):
                if idx not in merged_idx:
                    final.append(cell)
            cells = final
        
        return {'cells': cells, 'num_rows': n_rows, 'num_cols': n_cols}


class PaddleXSLANetEngine(BaseTSREngine):
    """PaddleX SLANet_plus - HTML parsing based TSR"""
    
    def __init__(self, device="gpu:0"):
        self.device = device
        self.uses_ocr = False
        
        try:
            import paddlex
            from bs4 import BeautifulSoup
            
            logger.info(f"Loading PaddleX SLANet_plus on {device}")
            self.model = paddlex.create_model(model_name="SLANet_plus", device=device)
            self.BeautifulSoup = BeautifulSoup
            logger.info("PaddleX SLANet_plus loaded")
        except Exception as e:
            logger.error(f"PaddleX load failed: {e}")
            raise
    
    def predict(self, image, ocr_boxes):
        """Predict using SLANet_plus"""
        try:
            from PIL import Image as PILImage
            if isinstance(image, np.ndarray):
                image = PILImage.fromarray(image.astype(np.uint8))
            
            outputs = self.model.predict(input=image, batch_size=1)
            if not outputs:
                return {'cells': [], 'num_rows': 0, 'num_cols': 0}
            
            res = outputs[0].json.get("res", {})
            bboxes = res.get("bbox", [])
            tokens = res.get("structure", [])
            
            return self._parse_structure(bboxes, tokens)
        except Exception as e:
            logger.error(f"PaddleX predict failed: {e}")
            return {'cells': [], 'num_rows': 0, 'num_cols': 0}
    
    def _parse_structure(self, bboxes, tokens):
        """Parse HTML tokens and match bboxes"""
        from .structures import convert_polygon_to_xyxy
        
        html = ''.join(tokens)
        if not html.startswith('<table'):
            html = f'<table>{html}</table>'
        
        try:
            soup = self.BeautifulSoup(html, 'html.parser')
            html_cells = []
            for ridx, row in enumerate(soup.find_all('tr')):
                for cidx, td in enumerate(row.find_all(['td', 'th'])):
                    html_cells.append({
                        'row': ridx, 'col': cidx,
                        'rowspan': int(td.get('rowspan', 1)),
                        'colspan': int(td.get('colspan', 1))
                    })
            
            if len(html_cells) == len(bboxes):
                # Match
                cells = []
                for cell_info, bbox_poly in zip(html_cells, bboxes):
                    bbox = convert_polygon_to_xyxy(bbox_poly)
                    cells.append({
                        'row': cell_info['row'], 'col': cell_info['col'],
                        'row_span': cell_info['rowspan'], 'col_span': cell_info['colspan'],
                        'box': list(bbox), 'text': '', 'confidence': 0.9
                    })
                
                nr = max(c['row'] + c['row_span'] for c in cells) if cells else 0
                nc = max(c['col'] + c['col_span'] for c in cells) if cells else 0
                return {'cells': cells, 'num_rows': nr, 'num_cols': nc}
            else:
                logger.warning(f"Cell-bbox mismatch: {len(html_cells)} vs {len(bboxes)}")
                return self._fallback_grid(bboxes)
        except Exception as e:
            logger.error(f"HTML parse failed: {e}")
            return self._fallback_grid(bboxes)
    
    def _fallback_grid(self, bboxes):
        """Fallback: spatial clustering"""
        from .structures import convert_polygon_to_xyxy
        
        if not bboxes:
            return {'cells': [], 'num_rows': 0, 'num_cols': 0}
        
        boxes = []
        for bp in bboxes:
            bbox = convert_polygon_to_xyxy(bp)
            cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
            boxes.append({'bbox': bbox, 'cx': cx, 'cy': cy})
        
        boxes.sort(key=lambda b: (b['cy'], b['cx']))
        
        # Cluster rows
        rows, curr_row = [], []
        prev_cy = None
        for bd in boxes:
            if prev_cy is None or abs(bd['cy'] - prev_cy) > 20:
                if curr_row:
                    rows.append(curr_row)
                curr_row = [bd]
            else:
                curr_row.append(bd)
            prev_cy = bd['cy']
        if curr_row:
            rows.append(curr_row)
        
        cells = []
        for ridx, row_boxes in enumerate(rows):
            row_boxes.sort(key=lambda b: b['cx'])
            for cidx, bd in enumerate(row_boxes):
                cells.append({
                    'row': ridx, 'col': cidx, 'row_span': 1, 'col_span': 1,
                    'box': list(bd['bbox']), 'text': '', 'confidence': 0.7
                })
        
        nr = len(rows)
        nc = max(len(r) for r in rows) if rows else 0
        return {'cells': cells, 'num_rows': nr, 'num_cols': nc}


class DoclingEngine(BaseTSREngine):
    """Docling (IBM) - supports images and PDFs"""
    
    def __init__(self):
        self.uses_ocr = False
        
        try:
            from docling.document_converter import DocumentConverter
            logger.info("Initializing Docling")
            self.converter = DocumentConverter()
            logger.info("Docling loaded")
        except Exception as e:
            logger.error(f"Docling load failed: {e}")
            raise
    
    def predict(self, image, ocr_boxes):
        """Predict using Docling"""
        try:    
            import tempfile
            from PIL import Image as PILImage
            from pathlib import Path as PathType
            
            if isinstance(image, (str, PathType)):
                input_path = str(image)
            else:
                pil_image = PILImage.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    pil_image.save(f.name)
                    input_path = f.name
            
            conv = self.converter.convert(input_path)
            doc = conv.document
            
            if not doc.tables:
                logger.warning("Docling: no tables found")
                return {'cells': [], 'num_rows': 0, 'num_cols': 0}
            
            table = doc.tables[0]  # Select first table
            return self._parse_table(table)
        except Exception as e:
            logger.error(f"Docling predict failed: {e}")
            return {'cells': [], 'num_rows': 0, 'num_cols': 0}
    
    def _parse_table(self, table):
        """Parse Docling table"""
        try:
            grid = table.data.grid
            cells = []
            
            for row_cells in grid:
                for tcell in row_cells:
                    if tcell is None:
                        continue
                    
                    r0 = tcell.start_row_offset_idx
                    c0 = tcell.start_col_offset_idx
                    rs = tcell.row_span
                    cs = tcell.col_span
                    
                    bbox = tcell.bbox
                    if bbox is None:
                        bbox = [0, 0, 100, 100]
                    else:
                        if hasattr(bbox, 'as_tuple'):
                            bbox = list(bbox.as_tuple())
                        else:
                            bbox = list(bbox)
                    
                    bbox_xyxy = bbox[:4] if len(bbox) >= 4 else [0, 0, 100, 100]
                    
                    cells.append({
                        'row': r0, 'col': c0, 'row_span': rs, 'col_span': cs,
                        'box': bbox_xyxy, 'text': tcell.text or '', 'confidence': 0.95
                    })
            
            nr = max(c['row'] + c['row_span'] for c in cells) if cells else 0
            nc = max(c['col'] + c['col_span'] for c in cells) if cells else 0
            
            return {'cells': cells, 'num_rows': nr, 'num_cols': nc}
        except Exception as e:
            logger.error(f"Docling parse failed: {e}")
            return {'cells': [], 'num_rows': 0, 'num_cols': 0}

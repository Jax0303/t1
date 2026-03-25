"""
Comprehensive Metrics for Table Structure Recognition

Implements 4 core metrics:
- M1: Relation/Adjacency F1
- M2: Row Boundary Score
- M3: Col Boundary Score
- M4: Span F1
"""

import logging
from typing import List, Dict, Any, Set, Tuple
import numpy as np

from .utils import safe_divide, compute_iou, cells_to_grid

logger = logging.getLogger(__name__)


class DiagnosisMetrics:
    """
    Comprehensive TSR evaluation metrics
    """
    
    def __init__(self):
        """Initialize metrics calculator"""
        pass
    
    def compute_all_metrics(
        self,
        pred_cells: List[Dict[str, Any]],
        gt_cells: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute all 4 metrics
        
        Args:
            pred_cells: Predicted cells
            gt_cells: Ground truth cells
        
        Returns:
            Dict with metric scores
        """
        m1 = self.compute_relation_f1(pred_cells, gt_cells)
        m2 = self.compute_row_boundary_score(pred_cells, gt_cells)
        m3 = self.compute_col_boundary_score(pred_cells, gt_cells)
        m4 = self.compute_span_f1(pred_cells, gt_cells)
        
        return {
            'm1_relation_f1': m1,
            'm2_row_boundary': m2,
            'm3_col_boundary': m3,
            'm4_span_f1': m4,
            'average': (m1 + m2 + m3 + m4) / 4.0
        }
    
    def compute_relation_f1(
        self,
        pred_cells: List[Dict[str, Any]],
        gt_cells: List[Dict[str, Any]]
    ) -> float:
        """
        M1: Relation/Adjacency F1
        
        Extract adjacency edges from cell structure and compute F1.
        Edge = (cell_i, cell_j, direction) where direction ∈ {right, below}
        
        Args:
            pred_cells: Predicted cells
            gt_cells: Ground truth cells
        
        Returns:
            F1 score
        """
        pred_edges = self._extract_adjacency_edges(pred_cells)
        gt_edges = self._extract_adjacency_edges(gt_cells)
        
        if not gt_edges:
            return 1.0 if not pred_edges else 0.0
        
        tp = len(pred_edges & gt_edges)
        fp = len(pred_edges - gt_edges)
        fn = len(gt_edges - pred_edges)
        
        precision = safe_divide(tp, tp + fp, 0.0)
        recall = safe_divide(tp, tp + fn, 0.0)
        f1 = safe_divide(2 * precision * recall, precision + recall, 0.0)
        
        return f1
    
    def _extract_adjacency_edges(self, cells: List[Dict[str, Any]]) -> Set[Tuple]:
        """
        Extract adjacency edges from cells
        
        For each cell, find cells directly to the right and below.
        
        Args:
            cells: List of cells
        
        Returns:
            Set of edge tuples: (cell_id1, cell_id2, direction)
        """
        if not cells:
            return set()
        
        # Build grid
        num_rows, num_cols, grid = cells_to_grid(cells)
        
        edges = set()
        
        # Create cell ID mapping
        cell_ids = {id(cell): i for i, cell in enumerate(cells)}
        
        # For each position in grid
        for r in range(num_rows):
            for c in range(num_cols):
                if (r, c) not in grid:
                    continue
                
                current_cell = grid[(r, c)]
                current_id = cell_ids.get(id(current_cell))
                
                if current_id is None:
                    continue
                
                # Check right neighbor (skip if same cell due to spans)
                c_right = c + 1
                if c_right < num_cols and (r, c_right) in grid:
                    right_cell = grid[(r, c_right)]
                    if id(right_cell) != id(current_cell):  # Different cell
                        right_id = cell_ids.get(id(right_cell))
                        if right_id is not None:
                            edges.add((current_id, right_id, 'right'))
                
                # Check below neighbor
                r_below = r + 1
                if r_below < num_rows and (r_below, c) in grid:
                    below_cell = grid[(r_below, c)]
                    if id(below_cell) != id(current_cell):  # Different cell
                        below_id = cell_ids.get(id(below_cell))
                        if below_id is not None:
                            edges.add((current_id, below_id, 'below'))
        
        return edges
    
    def compute_row_boundary_score(
        self,
        pred_cells: List[Dict[str, Any]],
        gt_cells: List[Dict[str, Any]]
    ) -> float:
        """
        M2: Row Boundary Score
        
        Compares row count and row boundary positions.
        Uses F1 of detected row boundaries.
        
        Args:
            pred_cells: Predicted cells
            gt_cells: Ground truth cells
        
        Returns:
            Boundary F1 score
        """
        if not gt_cells:
            return 1.0 if not pred_cells else 0.0
        
        # Extract row boundaries (y-coordinates)
        gt_boundaries = self._extract_row_boundaries(gt_cells)
        pred_boundaries = self._extract_row_boundaries(pred_cells)
        
        # Match boundaries with tolerance
        tolerance = 10.0  # pixels
        matched_gt = set()
        matched_pred = set()
        
        for pred_y in pred_boundaries:
            for gt_y in gt_boundaries:
                if abs(pred_y - gt_y) <= tolerance and gt_y not in matched_gt:
                    matched_gt.add(gt_y)
                    matched_pred.add(pred_y)
                    break
        
        tp = len(matched_pred)
        fp = len(pred_boundaries) - tp
        fn = len(gt_boundaries) - len(matched_gt)
        
        precision = safe_divide(tp, tp + fp, 0.0)
        recall = safe_divide(tp, tp + fn, 0.0)
        f1 = safe_divide(2 * precision * recall, precision + recall, 0.0)
        
        return f1
    
    def _extract_row_boundaries(self, cells: List[Dict[str, Any]]) -> List[float]:
        """
        Extract row boundary y-coordinates from cells
        
        Args:
            cells: List of cells
        
        Returns:
            List of unique y-coordinates representing row boundaries
        """
        if not cells:
            return []
        
        y_coords = set()
        
        for cell in cells:
            box = cell.get('box', cell.get('bbox', [0, 0, 0, 0]))
            if len(box) >= 4:
                y1 = box[1]
                y2 = box[3]
                y_coords.add(y1)
                y_coords.add(y2)
        
        return sorted(list(y_coords))
    
    def compute_col_boundary_score(
        self,
        pred_cells: List[Dict[str, Any]],
        gt_cells: List[Dict[str, Any]]
    ) -> float:
        """
        M3: Col Boundary Score
        
        Compares column count and column boundary positions.
        Uses F1 of detected column boundaries.
        
        Args:
            pred_cells: Predicted cells
            gt_cells: Ground truth cells
        
        Returns:
            Boundary F1 score
        """
        if not gt_cells:
            return 1.0 if not pred_cells else 0.0
        
        # Extract column boundaries (x-coordinates)
        gt_boundaries = self._extract_col_boundaries(gt_cells)
        pred_boundaries = self._extract_col_boundaries(pred_cells)
        
        # Match boundaries with tolerance
        tolerance = 10.0  # pixels
        matched_gt = set()
        matched_pred = set()
        
        for pred_x in pred_boundaries:
            for gt_x in gt_boundaries:
                if abs(pred_x - gt_x) <= tolerance and gt_x not in matched_gt:
                    matched_gt.add(gt_x)
                    matched_pred.add(pred_x)
                    break
        
        tp = len(matched_pred)
        fp = len(pred_boundaries) - tp
        fn = len(gt_boundaries) - len(matched_gt)
        
        precision = safe_divide(tp, tp + fp, 0.0)
        recall = safe_divide(tp, tp + fn, 0.0)
        f1 = safe_divide(2 * precision * recall, precision + recall, 0.0)
        
        return f1
    
    def _extract_col_boundaries(self, cells: List[Dict[str, Any]]) -> List[float]:
        """
        Extract column boundary x-coordinates from cells
        
        Args:
            cells: List of cells
        
        Returns:
            List of unique x-coordinates representing column boundaries
        """
        if not cells:
            return []
        
        x_coords = set()
        
        for cell in cells:
            box = cell.get('box', cell.get('bbox', [0, 0, 0, 0]))
            if len(box) >= 4:
                x1 = box[0]
                x2 = box[2]
                x_coords.add(x1)
                x_coords.add(x2)
        
        return sorted(list(x_coords))
    
    def compute_span_f1(
        self,
        pred_cells: List[Dict[str, Any]],
        gt_cells: List[Dict[str, Any]],
        iou_threshold: float = 0.5
    ) -> float:
        """
        M4: Span F1
        
        Treat each cell as a span (r0, c0, r1, c1) and match using IoU.
        
        Args:
            pred_cells: Predicted cells
            gt_cells: Ground truth cells
            iou_threshold: IoU threshold for matching spans
        
        Returns:
            F1 score
        """
        if not gt_cells:
            return 1.0 if not pred_cells else 0.0
        
        # Convert cells to span representations
        pred_spans = self._cells_to_spans(pred_cells)
        gt_spans = self._cells_to_spans(gt_cells)
        
        # Match spans
        matched_gt = set()
        matched_pred = set()
        
        for i, pred_span in enumerate(pred_spans):
            best_iou = 0.0
            best_j = -1
            
            for j, gt_span in enumerate(gt_spans):
                if j in matched_gt:
                    continue
                
                iou = self._compute_span_iou(pred_span, gt_span)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_iou >= iou_threshold and best_j >= 0:
                matched_pred.add(i)
                matched_gt.add(best_j)
        
        tp = len(matched_pred)
        fp = len(pred_spans) - tp
        fn = len(gt_spans) - len(matched_gt)
        
        precision = safe_divide(tp, tp + fp, 0.0)
        recall = safe_divide(tp, tp + fn, 0.0)
        f1 = safe_divide(2 * precision * recall, precision + recall, 0.0)
        
        return f1
    
    def _cells_to_spans(self, cells: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
        """
        Convert cells to span tuples (r0, c0, r1, c1)
        
        Args:
            cells: List of cells
        
        Returns:
            List of (start_row, start_col, end_row, end_col) tuples
        """
        spans = []
        
        for cell in cells:
            row = cell.get('row', cell.get('start_row', 0))
            col = cell.get('col', cell.get('start_col', 0))
            row_span = cell.get('row_span', cell.get('end_row', row) - row + 1)
            col_span = cell.get('col_span', cell.get('end_col', col) - col + 1)
            
            end_row = row + row_span - 1
            end_col = col + col_span - 1
            
            spans.append((row, col, end_row, end_col))
        
        return spans
    
    def _compute_span_iou(
        self,
        span1: Tuple[int, int, int, int],
        span2: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute IoU between two spans in grid coordinates
        
        Args:
            span1: (r0, c0, r1, c1)
            span2: (r0, c0, r1, c1)
        
        Returns:
            IoU score
        """
        r1_0, c1_0, r1_1, c1_1 = span1
        r2_0, c2_0, r2_1, c2_1 = span2
        
        # Compute intersection
        r_i0 = max(r1_0, r2_0)
        c_i0 = max(c1_0, c2_0)
        r_i1 = min(r1_1, r2_1)
        c_i1 = min(c1_1, c2_1)
        
        if r_i1 < r_i0 or c_i1 < c_i0:
            return 0.0
        
        intersection = (r_i1 - r_i0 + 1) * (c_i1 - c_i0 + 1)
        
        # Compute union
        area1 = (r1_1 - r1_0 + 1) * (c1_1 - c1_0 + 1)
        area2 = (r2_1 - r2_0 + 1) * (c2_1 - c2_0 + 1)
        union = area1 + area2 - intersection
        
        return safe_divide(intersection, union, 0.0)

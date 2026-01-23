"""
Intersection-based Grid Restorer.

This component is the final step of the RCA pipeline.
It uses the stable global grid (Rows x Cols) defined by the RCAModel
to recover individual cells.

Key Innovation: 
Unlike 'Merge-then-Split' approaches, this uses 'Intersection-Validation'.
It checks if the intersection of Row i and Col j contains content.
If multiple intersections share content without dividers, they are merged.
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class IntersectionRestorer:
    """
    Component 3: Intersection Restorer
    
    Restores cells by mapping the global grid intersections back to the image content.
    Includes Text-Aware Grid Refinement (TAGR) to handle Row/Col alignment errors.
    """
    
    def _refine_grid_with_ocr(self, boundaries: List[float], ocr_boxes: List[Dict[str, Any]], axis: str = 'y') -> List[float]:
        """
        Refines grid boundaries by snapping them to text-free gaps.
        ocr_boxes: List of cells with 'box' [xmin, ymin, xmax, ymax]
        axis: 'y' for rows, 'x' for columns
        """
        if not ocr_boxes:
            return boundaries
            
        refined = []
        # Convert boxes to intervals on the target axis
        idx_start = 1 if axis == 'y' else 0
        idx_end = 3 if axis == 'y' else 2
        
        intervals = sorted([(b['box'][idx_start], b['box'][idx_end]) for b in ocr_boxes if 'box' in b])
        
        for line in boundaries:
            # Check if this line intersects any text box
            intersecting = [i for i in intervals if i[0] < line < i[1]]
            
            if not intersecting:
                refined.append(line)
                continue
                
            # If intersecting, find the nearest gap
            # This is a simplified TAGR: snap to the edge of the intersecting box
            # Ideally, we look for the largest gap in a neighborhood
            best_snap = line
            min_dist = float('inf')
            
            for start, end in intersecting:
                if abs(line - start) < min_dist:
                    min_dist = abs(line - start)
                    best_snap = start
                if abs(line - end) < min_dist:
                    min_dist = abs(line - end)
                    best_snap = end
            
            refined.append(best_snap)
            
        return sorted(list(set(refined)))

    def restore_cells(self, rca_result: Dict[str, Any], ocr_boxes: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Reconstructs the cell structure using TAGR-refined boundaries.
        """
        logger.info("[RCA] Restoring Cells with TAGR Refinement...")
        
        row_bounds = rca_result["row_boundaries"]
        col_bounds = rca_result["col_boundaries"]
        
        if ocr_boxes:
            # Normalize OCR boxes if they aren't already (assuming [0, 1] for now or providing orig_size)
            # For SciTSR, we might need the image dimensions.
            row_bounds = self._refine_grid_with_ocr(row_bounds, ocr_boxes, axis='y')
            col_bounds = self._refine_grid_with_ocr(col_bounds, ocr_boxes, axis='x')
        
        cells = []
        for r in range(len(row_bounds) - 1):
            y_min, y_max = row_bounds[r], row_bounds[r+1]
            for c in range(len(col_bounds) - 1):
                x_min, x_max = col_bounds[c], col_bounds[c+1]
                
                # Check which OCR boxes fall into this intersection
                contained_content = []
                for box in (ocr_boxes or []):
                    b = box['box']
                    # Simple center-point check
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    if x_min <= cx <= x_max and y_min <= cy <= y_max:
                        contained_content.append(box.get('content', ''))
                
                if contained_content:
                    cells.append({
                        "row_idx": r, "col_idx": c,
                        "row_span": 1, "col_span": 1,
                        "content": " ".join(contained_content),
                        "bbox": [x_min, y_min, x_max, y_max]
                    })
                    
        return cells

    def refine_merges(self, cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optional pass to merge adjacent cells if no visual separator exists.
        """
        # (Placeholder for post-processing logic)
        return cells

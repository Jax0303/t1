"""
Common utilities for diagnosis module
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """
    Configure logging for diagnosis module
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('diagnosis')


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that returns default value when denominator is zero
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero
    
    Returns:
        Division result or default
    """
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return default
    return numerator / denominator


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU score (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return safe_divide(intersection, union, 0.0)


def cells_to_grid(cells: List[Dict[str, Any]]) -> Tuple[int, int, Dict[Tuple[int, int], Dict]]:
    """
    Convert cell list to 2D grid representation
    
    Args:
        cells: List of cell dicts with row, col, row_span, col_span
    
    Returns:
        (num_rows, num_cols, grid_dict) where grid_dict maps (r, c) to cell
    """
    if not cells:
        return 0, 0, {}
    
    # Compute grid dimensions
    max_row = 0
    max_col = 0
    
    for cell in cells:
        row = cell.get('row', cell.get('start_row', 0))
        col = cell.get('col', cell.get('start_col', 0))
        row_span = cell.get('row_span', cell.get('end_row', row) - row + 1)
        col_span = cell.get('col_span', cell.get('end_col', col) - col + 1)
        
        max_row = max(max_row, row + row_span)
        max_col = max(max_col, col + col_span)
    
    # Build grid dict
    grid = {}
    for cell in cells:
        row = cell.get('row', cell.get('start_row', 0))
        col = cell.get('col', cell.get('start_col', 0))
        row_span = cell.get('row_span', cell.get('end_row', row) - row + 1)
        col_span = cell.get('col_span', cell.get('end_col', col) - col + 1)
        
        # Store cell at all positions it spans
        for r in range(row, row + row_span):
            for c in range(col, col + col_span):
                grid[(r, c)] = cell
    
    return max_row, max_col, grid


def normalize_bbox(bbox: Any) -> List[float]:
    """
    Normalize bbox to [x1, y1, x2, y2] format
    
    Args:
        bbox: Can be [x1,y1,x2,y2] or [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    
    Returns:
        [x1, y1, x2, y2] format
    """
    if not bbox:
        return [0, 0, 0, 0]
    
    # If already in correct format
    if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
        return list(bbox)
    
    # If in points format [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    if len(bbox) == 4 and all(isinstance(p, (list, tuple)) for p in bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return [min(xs), min(ys), max(xs), max(ys)]
    
    # Fallback
    return [0, 0, 0, 0]


def extract_text_content(cell: Dict[str, Any]) -> str:
    """
    Extract text content from cell in various formats
    
    Args:
        cell: Cell dict that may have 'text', 'content', or 'tex' field
    
    Returns:
        Normalized text string
    """
    text = cell.get('text', cell.get('content', cell.get('tex', '')))
    
    if isinstance(text, list):
        text = ' '.join(str(t) for t in text)
    elif not isinstance(text, str):
        text = str(text)
    
    return text.strip()

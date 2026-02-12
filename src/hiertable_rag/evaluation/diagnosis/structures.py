"""
Common Data Structures for Diagnosis Module

Defines intermediate representations (IR) for table structures and OCR results.
Used as common interface between different TSR/OCR engines.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class Cell:
    """
    Table cell with span information
    
    Coordinate Convention:
    - Row/col indices use exclusive upper bound: [r0, r1), [c0, c1)
    - r1 and c1 are NOT included in the cell span
    - Example: Cell(r0=0, c0=0, r1=2, c1=3) spans rows 0-1 and cols 0-2
    
    Attributes:
        r0: Start row index (inclusive)
        c0: Start column index (inclusive)
        r1: End row index (exclusive)
        c1: End column index (exclusive)
        bbox_xyxy: Bounding box in (x0, y0, x1, y1) format
        text: Cell text content
        confidence: Confidence score (0-1)
        is_header: Whether cell is a header
    """
    r0: int
    c0: int
    r1: int
    c1: int
    bbox_xyxy: Tuple[float, float, float, float]
    text: str = ""
    confidence: float = 1.0
    is_header: bool = False
    
    @property
    def row_span(self) -> int:
        """Number of rows spanned"""
        return self.r1 - self.r0
    
    @property
    def col_span(self) -> int:
        """Number of columns spanned"""
        return self.c1 - self.c0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for compatibility with existing metrics"""
        return {
            'row': self.r0,
            'col': self.c0,
            'row_span': self.row_span,
            'col_span': self.col_span,
            'box': list(self.bbox_xyxy),
            'text': self.text,
            'confidence': self.confidence
        }


@dataclass
class TableStruct:
    """
    Unified table structure representation
    
    Serves as common intermediate representation (IR) between different
    TSR engines and the metrics computation layer.
    
    Attributes:
        cells: List of table cells with spans
        n_rows: Total number of rows
        n_cols: Total number of columns
        row_boundaries: Y-coordinates of row boundaries (sorted)
        col_boundaries: X-coordinates of column boundaries (sorted)
        meta: Additional metadata (engine name, confidence, etc.)
    """
    cells: List[Cell]
    n_rows: int
    n_cols: int
    row_boundaries: List[float] = field(default_factory=list)
    col_boundaries: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for compatibility with existing code"""
        return {
            'cells': [cell.to_dict() for cell in self.cells],
            'num_rows': self.n_rows,
            'num_cols': self.n_cols
        }
    
    @classmethod
    def from_cells(cls, cells: List[Cell], **kwargs) -> 'TableStruct':
        """
        Create TableStruct from cells, auto-computing grid dimensions
        
        Args:
            cells: List of Cell objects
            **kwargs: Additional metadata
        
        Returns:
            TableStruct instance
        """
        if not cells:
            return cls(
                cells=[],
                n_rows=0,
                n_cols=0,
                row_boundaries=[],
                col_boundaries=[],
                meta=kwargs
            )
        
        # Compute grid dimensions
        n_rows = max(cell.r1 for cell in cells)
        n_cols = max(cell.c1 for cell in cells)
        
        # Extract boundaries from bboxes
        y_coords = set()
        x_coords = set()
        for cell in cells:
            x0, y0, x1, y1 = cell.bbox_xyxy
            y_coords.add(y0)
            y_coords.add(y1)
            x_coords.add(x0)
            x_coords.add(x1)
        
        row_boundaries = sorted(list(y_coords))
        col_boundaries = sorted(list(x_coords))
        
        return cls(
            cells=cells,
            n_rows=n_rows,
            n_cols=n_cols,
            row_boundaries=row_boundaries,
            col_boundaries=col_boundaries,
            meta=kwargs
        )
    
    @classmethod
    def empty(cls, reason: str = "") -> 'TableStruct':
        """Create empty table structure"""
        return cls(
            cells=[],
            n_rows=0,
            n_cols=0,
            row_boundaries=[],
            col_boundaries=[],
            meta={'empty': True, 'reason': reason}
        )


@dataclass
class OcrResult:
    """
    OCR output representation
    
    Standardized format for OCR results from different engines.
    
    Attributes:
        tokens: List of recognized text strings
        bboxes: List of bounding boxes in (x0, y0, x1, y1) format
        confidences: Confidence scores for each token (0-1)
        meta: Additional metadata (engine name, etc.)
    """
    tokens: List[str]
    bboxes: List[Tuple[float, float, float, float]]
    confidences: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize data"""
        # Ensure same length
        if len(self.tokens) != len(self.bboxes):
            raise ValueError(f"Tokens ({len(self.tokens)}) and bboxes ({len(self.bboxes)}) must have same length")
        
        # Fill confidences if empty
        if not self.confidences:
            self.confidences = [1.0] * len(self.tokens)
        
        # Validate confidences length
        if len(self.confidences) != len(self.tokens):
            raise ValueError(f"Confidences ({len(self.confidences)}) must match tokens ({len(self.tokens)})")
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dicts for compatibility with existing code"""
        result = []
        for token, bbox, conf in zip(self.tokens, self.bboxes, self.confidences):
            # Convert xyxy to points format for backward compatibility
            x0, y0, x1, y1 = bbox
            bbox_points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            
            result.append({
                'bbox': bbox_points,
                'text': token,
                'confidence': conf
            })
        return result
    
    @classmethod
    def empty(cls) -> 'OcrResult':
        """Create empty OCR result"""
        return cls(tokens=[], bboxes=[], confidences=[], meta={'empty': True})


def convert_polygon_to_xyxy(polygon: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert polygon coordinates to axis-aligned bounding box
    
    Args:
        polygon: List of coordinates, either [x1,y1,x2,y2,x3,y3,x4,y4]
                or [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    
    Returns:
        (x0, y0, x1, y1) tuple
    """
    # Flatten if nested
    if polygon and isinstance(polygon[0], (list, tuple)):
        coords = []
        for point in polygon:
            coords.extend(point)
        polygon = coords
    
    # Extract x and y coordinates
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_iou(box1: Tuple[float, float, float, float], 
             box2: Tuple[float, float, float, float]) -> float:
    """
    Compute IoU between two bounding boxes
    
    Args:
        box1: (x0, y0, x1, y1)
        box2: (x0, y0, x1, y1)
    
    Returns:
        IoU score (0-1)
    """
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    
    # Intersection
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)
    
    if x1_i < x0_i or y1_i < y0_i:
        return 0.0
    
    intersection = (x1_i - x0_i) * (y1_i - y0_i)
    
    # Union
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

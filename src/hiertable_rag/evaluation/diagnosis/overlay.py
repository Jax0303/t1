"""
Visual Overlay Generator for Error Atlas

Creates GT/Pred/Diff overlay images for visual error analysis.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import normalize_bbox

logger = logging.getLogger(__name__)


class OverlayGenerator:
    """
    Generate visual overlays for error analysis
    """
    
    def __init__(
        self,
        gt_color: Tuple[int, int, int] = (0, 255, 0),      # Green
        pred_color: Tuple[int, int, int] = (0, 0, 255),    # Blue
        tp_color: Tuple[int, int, int] = (128, 0, 128),    # Purple
        fp_color: Tuple[int, int, int] = (255, 0, 0),      # Red
        fn_color: Tuple[int, int, int] = (255, 165, 0),    # Orange
        line_width: int = 2,
        alpha: int = 128
    ):
        """
        Initialize overlay generator
        
        Args:
            gt_color: Color for GT boxes (R, G, B)
            pred_color: Color for prediction boxes
            tp_color: Color for true positives
            fp_color: Color for false positives
            fn_color: Color for false negatives
            line_width: Line width for boxes
            alpha: Transparency (0-255, higher = more opaque)
        """
        self.gt_color = gt_color
        self.pred_color = pred_color
        self.tp_color = tp_color
        self.fp_color = fp_color
        self.fn_color = fn_color
        self.line_width = line_width
        self.alpha = alpha
    
    def create_gt_overlay(
        self,
        image: np.ndarray,
        gt_cells: List[Dict[str, Any]]
    ) -> Image.Image:
        """
        Create GT overlay (green boxes)
        
        Args:
            image: Original image array (H, W, 3)
            gt_cells: Ground truth cells
        
        Returns:
            PIL Image with GT overlay
        """
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = image.copy()
        
        # Create transparent overlay
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        for cell in gt_cells:
            box = normalize_bbox(cell.get('box', cell.get('bbox', [])))
            if len(box) >= 4:
                x1, y1, x2, y2 = box
                
                # Draw rectangle
                color = self.gt_color + (self.alpha,)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)
        
        # Composite
        img = img.convert('RGBA')
        result = Image.alpha_composite(img, overlay)
        
        return result.convert('RGB')
    
    def create_pred_overlay(
        self,
        image: np.ndarray,
        pred_cells: List[Dict[str, Any]]
    ) -> Image.Image:
        """
        Create prediction overlay (blue boxes)
        
        Args:
            image: Original image array (H, W, 3)
            pred_cells: Predicted cells
        
        Returns:
            PIL Image with prediction overlay
        """
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = image.copy()
        
        # Create transparent overlay
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        for cell in pred_cells:
            box = normalize_bbox(cell.get('box', cell.get('bbox', [])))
            if len(box) >= 4:
                x1, y1, x2, y2 = box
                
                # Draw rectangle
                color = self.pred_color + (self.alpha,)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)
        
        # Composite
        img = img.convert('RGBA')
        result = Image.alpha_composite(img, overlay)
        
        return result.convert('RGB')
    
    def create_diff_overlay(
        self,
        image: np.ndarray,
        gt_cells: List[Dict[str, Any]],
        pred_cells: List[Dict[str, Any]],
        iou_threshold: float = 0.5
    ) -> Image.Image:
        """
        Create diff overlay showing TP/FP/FN
        
        Args:
            image: Original image array (H, W, 3)
            gt_cells: Ground truth cells
            pred_cells: Predicted cells
            iou_threshold: IoU threshold for matching cells
        
        Returns:
            PIL Image with diff overlay
            - Purple: True Positive (matched)
            - Red: False Positive (predicted but no match)
            - Orange: False Negative (GT but not predicted)
        """
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = image.copy()
        
        # Match cells using box IoU
        matched_gt, matched_pred = self._match_cells(gt_cells, pred_cells, iou_threshold)
        
        # Create transparent overlay
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw False Negatives (GT not matched) - Orange
        for i, cell in enumerate(gt_cells):
            if i not in matched_gt:
                box = normalize_bbox(cell.get('box', cell.get('bbox', [])))
                if len(box) >= 4:
                    x1, y1, x2, y2 = box
                    color = self.fn_color + (self.alpha,)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)
        
        # Draw False Positives (Pred not matched) - Red
        for i, cell in enumerate(pred_cells):
            if i not in matched_pred:
                box = normalize_bbox(cell.get('box', cell.get('bbox', [])))
                if len(box) >= 4:
                    x1, y1, x2, y2 = box
                    color = self.fp_color + (self.alpha,)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)
        
        # Draw True Positives (Matched) - Purple
        for i in matched_pred:
            cell = pred_cells[i]
            box = normalize_bbox(cell.get('box', cell.get('bbox', [])))
            if len(box) >= 4:
                x1, y1, x2, y2 = box
                color = self.tp_color + (self.alpha,)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=self.line_width)
        
        # Composite
        img = img.convert('RGBA')
        result = Image.alpha_composite(img, overlay)
        
        return result.convert('RGB')
    
    def _match_cells(
        self,
        gt_cells: List[Dict[str, Any]],
        pred_cells: List[Dict[str, Any]],
        iou_threshold: float
    ) -> Tuple[set, set]:
        """
        Match GT and predicted cells using box IoU
        
        Args:
            gt_cells: Ground truth cells
            pred_cells: Predicted cells
            iou_threshold: IoU threshold for matching
        
        Returns:
            (matched_gt_indices, matched_pred_indices)
        """
        from .utils import compute_iou
        
        matched_gt = set()
        matched_pred = set()
        
        for i, pred_cell in enumerate(pred_cells):
            pred_box = normalize_bbox(pred_cell.get('box', pred_cell.get('bbox', [])))
            
            best_iou = 0.0
            best_j = -1
            
            for j, gt_cell in enumerate(gt_cells):
                if j in matched_gt:
                    continue
                
                gt_box = normalize_bbox(gt_cell.get('box', gt_cell.get('bbox', [])))
                
                iou = compute_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_iou >= iou_threshold and best_j >= 0:
                matched_pred.add(i)
                matched_gt.add(best_j)
        
        return matched_gt, matched_pred
    
    def create_atlas_images(
        self,
        image: np.ndarray,
        gt_cells: List[Dict[str, Any]],
        pred_cells: List[Dict[str, Any]]
    ) -> Dict[str, Image.Image]:
        """
        Create all atlas images at once
        
        Args:
            image: Original image
            gt_cells: GT cells
            pred_cells: Predicted cells
        
        Returns:
            Dict with keys: 'original', 'gt_overlay', 'pred_overlay', 'diff_overlay'
        """
        original = Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
        
        return {
            'original': original,
            'gt_overlay': self.create_gt_overlay(image, gt_cells),
            'pred_overlay': self.create_pred_overlay(image, pred_cells),
            'diff_overlay': self.create_diff_overlay(image, gt_cells, pred_cells)
        }

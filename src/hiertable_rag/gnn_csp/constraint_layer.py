"""
Constraint Layer for Table Structure Recognition

Defines hard and soft constraints for table structure:
- Hard Constraints: Physical laws that must be satisfied
- Soft Constraints: Preferences learned from data
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TableConstraints:
    """Container for all constraints"""
    # Hard constraints
    row_height_tolerance: float = 0.1  # 10% tolerance
    col_width_tolerance: float = 0.1
    overlap_tolerance: float = 0.05   # 5% IoU allowed
    coverage_threshold: float = 0.95  # 95% coverage required
    
    # Soft constraint weights
    semantic_weight: float = 0.3
    visual_weight: float = 0.5
    layout_weight: float = 0.2


class HardConstraints:
    """
    Enforces physical table structure rules that must be satisfied.
    """
    
    def __init__(self, constraints: TableConstraints):
        self.constraints = constraints
    
    def check_row_height_consistency(
        self,
        cells: List[Dict[str, Any]],
        row_assignment: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if all cells in the same row have consistent heights.
        
        Args:
            cells: List of cell dictionaries with 'box' key
            row_assignment: [N] array of row indices
            
        Returns:
            is_valid: Whether constraint is satisfied
            violation_score: Degree of violation (0 = perfect, higher = worse)
        """
        if len(cells) == 0:
            return True, 0.0
        
        unique_rows = np.unique(row_assignment)
        max_violation = 0.0
        
        for row_idx in unique_rows:
            row_mask = row_assignment == row_idx
            row_cells = [cells[i] for i in np.where(row_mask)[0]]
            
            if len(row_cells) < 2:
                continue
            
            heights = [cell['box'][3] - cell['box'][1] for cell in row_cells]
            mean_height = np.mean(heights)
            max_height = np.max(heights)
            min_height = np.min(heights)
            
            if mean_height > 0:
                variation = (max_height - min_height) / mean_height
                max_violation = max(max_violation, variation)
        
        is_valid = max_violation <= self.constraints.row_height_tolerance
        
        return is_valid, max_violation
    
    def check_col_width_consistency(
        self,
        cells: List[Dict[str, Any]],
        col_assignment: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if all cells in the same column have consistent widths.
        """
        if len(cells) == 0:
            return True, 0.0
        
        unique_cols = np.unique(col_assignment)
        max_violation = 0.0
        
        for col_idx in unique_cols:
            col_mask = col_assignment == col_idx
            col_cells = [cells[i] for i in np.where(col_mask)[0]]
            
            if len(col_cells) < 2:
                continue
            
            widths = [cell['box'][2] - cell['box'][0] for cell in col_cells]
            mean_width = np.mean(widths)
            max_width = np.max(widths)
            min_width = np.min(widths)
            
            if mean_width > 0:
                variation = (max_width - min_width) / mean_width
                max_violation = max(max_violation, variation)
        
        is_valid = max_violation <= self.constraints.col_width_tolerance
        
        return is_valid, max_violation
    
    def check_no_overlap(
        self,
        cells: List[Dict[str, Any]]
    ) -> Tuple[bool, float]:
        """
        Check that no two cells overlap significantly.
        """
        num_cells = len(cells)
        max_overlap = 0.0
        
        for i in range(num_cells):
            box_i = cells[i]['box']
            for j in range(i + 1, num_cells):
                box_j = cells[j]['box']
                
                iou = self._compute_iou(box_i, box_j)
                max_overlap = max(max_overlap, iou)
        
        is_valid = max_overlap <= self.constraints.overlap_tolerance
        
        return is_valid, max_overlap
    
    def check_full_coverage(
        self,
        cells: List[Dict[str, Any]],
        table_bbox: List[float]
    ) -> Tuple[bool, float]:
        """
        Check that cells cover the table area sufficiently.
        
        Args:
            cells: List of cells
            table_bbox: [x1, y1, x2, y2] of entire table
            
        Returns:
            is_valid: Whether coverage is sufficient
            coverage_ratio: Actual coverage ratio
        """
        if len(cells) == 0:
            return False, 0.0
        
        # Compute total cell area
        total_cell_area = sum([
            (cell['box'][2] - cell['box'][0]) * (cell['box'][3] - cell['box'][1])
            for cell in cells
        ])
        
        # Compute table area
        table_area = (table_bbox[2] - table_bbox[0]) * (table_bbox[3] - table_bbox[1])
        
        if table_area == 0:
            return False, 0.0
        
        coverage_ratio = total_cell_area / table_area
        is_valid = coverage_ratio >= self.constraints.coverage_threshold
        
        return is_valid, coverage_ratio
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute Intersection over Union of two boxes."""
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def validate_all(
        self,
        cells: List[Dict[str, Any]],
        row_assignment: np.ndarray,
        col_assignment: np.ndarray,
        table_bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Validate all hard constraints.
        
        Returns:
            Dictionary with:
                - 'all_valid': bool
                - 'violations': dict of constraint violations
        """
        row_valid, row_violation = self.check_row_height_consistency(cells, row_assignment)
        col_valid, col_violation = self.check_col_width_consistency(cells, col_assignment)
        overlap_valid, overlap_score = self.check_no_overlap(cells)
        coverage_valid, coverage_ratio = self.check_full_coverage(cells, table_bbox)
        
        all_valid = row_valid and col_valid and overlap_valid and coverage_valid
        
        return {
            'all_valid': all_valid,
            'violations': {
                'row_height': row_violation,
                'col_width': col_violation,
                'overlap': overlap_score,
                'coverage': coverage_ratio
            }
        }


class SoftConstraints:
    """
    Computes soft constraint scores (preferences, not strict rules).
    """
    
    def __init__(self, constraints: TableConstraints):
        self.constraints = constraints
    
    def semantic_alignment_score(
        self,
        cells: List[Dict[str, Any]],
        node_embeddings: torch.Tensor,
        row_assignment: np.ndarray,
        col_assignment: np.ndarray
    ) -> float:
        """
        Compute semantic alignment between headers and data cells.
        
        Higher score = better semantic coherence within columns.
        """
        if len(cells) < 2:
            return 1.0
        
        unique_cols = np.unique(col_assignment)
        alignment_scores = []
        
        for col_idx in unique_cols:
            col_mask = col_assignment == col_idx
            col_indices = np.where(col_mask)[0]
            
            if len(col_indices) < 2:
                continue
            
            # Get embeddings for this column
            col_embeddings = node_embeddings[col_indices]
            
            # Compute pairwise cosine similarity
            col_embeddings_norm = F.normalize(col_embeddings, p=2, dim=-1)
            similarity_matrix = torch.mm(col_embeddings_norm, col_embeddings_norm.t())
            
            # Average similarity (exclude diagonal)
            num_cells = len(col_indices)
            total_similarity = similarity_matrix.sum().item() - num_cells
            avg_similarity = total_similarity / (num_cells * (num_cells - 1)) if num_cells > 1 else 1.0
            
            alignment_scores.append(avg_similarity)
        
        if len(alignment_scores) == 0:
            return 1.0
        
        return np.mean(alignment_scores)
    
    def visual_confidence_score(
        self,
        visual_confidences: torch.Tensor
    ) -> float:
        """
        Average visual confidence from RCA model.
        """
        if len(visual_confidences) == 0:
            return 0.5
        
        return visual_confidences.mean().item()
    
    def layout_regularity_score(
        self,
        cells: List[Dict[str, Any]],
        row_assignment: np.ndarray,
        col_assignment: np.ndarray
    ) -> float:
        """
        Measure how well-aligned cell boundaries are.
        
        Higher score = more regular grid structure.
        """
        if len(cells) < 2:
            return 1.0
        
        # Check horizontal alignment (same y-coordinates for same row)
        unique_rows = np.unique(row_assignment)
        row_alignment_scores = []
        
        for row_idx in unique_rows:
            row_mask = row_assignment == row_idx
            row_cells = [cells[i] for i in np.where(row_mask)[0]]
            
            if len(row_cells) < 2:
                continue
            
            y_coords = [(cell['box'][1] + cell['box'][3]) / 2 for cell in row_cells]
            y_std = np.std(y_coords)
            y_mean = np.mean(y_coords)
            
            if y_mean > 0:
                alignment = 1.0 / (1.0 + y_std / y_mean)
                row_alignment_scores.append(alignment)
        
        # Check vertical alignment (same x-coordinates for same column)
        unique_cols = np.unique(col_assignment)
        col_alignment_scores = []
        
        for col_idx in unique_cols:
            col_mask = col_assignment == col_idx
            col_cells = [cells[i] for i in np.where(col_mask)[0]]
            
            if len(col_cells) < 2:
                continue
            
            x_coords = [(cell['box'][0] + cell['box'][2]) / 2 for cell in col_cells]
            x_std = np.std(x_coords)
            x_mean = np.mean(x_coords)
            
            if x_mean > 0:
                alignment = 1.0 / (1.0 + x_std / x_mean)
                col_alignment_scores.append(alignment)
        
        all_scores = row_alignment_scores + col_alignment_scores
        
        if len(all_scores) == 0:
            return 1.0
        
        return np.mean(all_scores)
    
    def compute_all(
        self,
        cells: List[Dict[str, Any]],
        node_embeddings: torch.Tensor,
        visual_confidences: torch.Tensor,
        row_assignment: np.ndarray,
        col_assignment: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all soft constraint scores.
        
        Returns:
            Dictionary of scores (all in [0, 1], higher is better)
        """
        semantic = self.semantic_alignment_score(
            cells, node_embeddings, row_assignment, col_assignment
        )
        visual = self.visual_confidence_score(visual_confidences)
        layout = self.layout_regularity_score(cells, row_assignment, col_assignment)
        
        # Weighted combination
        combined = (
            self.constraints.semantic_weight * semantic +
            self.constraints.visual_weight * visual +
            self.constraints.layout_weight * layout
        )
        
        return {
            'semantic': semantic,
            'visual': visual,
            'layout': layout,
            'combined': combined
        }


class ConstraintLayer:
    """
    Unified interface for both hard and soft constraints.
    """
    
    def __init__(self, constraints: TableConstraints = None):
        if constraints is None:
            constraints = TableConstraints()
        
        self.constraints = constraints
        self.hard = HardConstraints(constraints)
        self.soft = SoftConstraints(constraints)
    
    def evaluate(
        self,
        cells: List[Dict[str, Any]],
        row_assignment: np.ndarray,
        col_assignment: np.ndarray,
        table_bbox: List[float],
        node_embeddings: torch.Tensor,
        visual_confidences: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Evaluate both hard and soft constraints.
        
        Returns:
            Dictionary with:
                - 'hard_valid': bool
                - 'hard_violations': dict
                - 'soft_scores': dict
                - 'total_score': float (higher is better)
        """
        hard_result = self.hard.validate_all(
            cells, row_assignment, col_assignment, table_bbox
        )
        
        soft_scores = self.soft.compute_all(
            cells, node_embeddings, visual_confidences,
            row_assignment, col_assignment
        )
        
        # Total score (penalize hard constraint violations heavily)
        if hard_result['all_valid']:
            total_score = soft_scores['combined']
        else:
            # Heavy penalty for hard constraint violations
            penalty = sum(hard_result['violations'].values())
            total_score = max(0.0, soft_scores['combined'] - penalty)
        
        return {
            'hard_valid': hard_result['all_valid'],
            'hard_violations': hard_result['violations'],
            'soft_scores': soft_scores,
            'total_score': total_score
        }


# Import at the end to avoid circular dependency
import torch.nn.functional as F

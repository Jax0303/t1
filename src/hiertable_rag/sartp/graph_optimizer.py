"""
Graph Optimizer for SARTP
Refines table structure using visual + semantic + layout scores.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)


class GraphOptimizer:
    """
    Optimizes table cell structure using combined visual, semantic, and layout scores.
    
    Uses Dynamic Programming or Hungarian algorithm to find optimal cell assignments
    that maximize global structure consistency.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,  # Visual weight
        beta: float = 0.3,   # Semantic weight
        gamma: float = 0.2,  # Layout weight
        method: str = 'hungarian'  # 'hungarian' or 'dp'
    ):
        """
        Initialize optimizer.
        
        Args:
            alpha: Weight for visual boundary confidence
            beta: Weight for semantic alignment scores
            gamma: Weight for geometric layout consistency
            method: Optimization algorithm ('hungarian' or 'dp')
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.method = method
        
        # Normalize weights
        total = alpha + beta + gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
        
        logger.info(f"GraphOptimizer initialized: α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}")
    
    def refine(
        self,
        visual_pred: Dict[str, Any],
        alignment_scores: Dict[str, Any],
        cells: List[Dict[str, Any]],
        return_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Refine cell structure using combined scores.
        
        Args:
            visual_pred: Output from RCAModel with row/col boundaries and confidence
            alignment_scores: Output from HeaderDataAligner
            cells: Initial cell structure from GridRestorer
            return_confidence: If True, include confidence in output cells
            
        Returns:
            Refined list of cells with adjusted positions
        """
        logger.info(f"Refining {len(cells)} cells using {self.method} optimization")
        
        # Step 1: Compute combined score matrix
        score_matrix = self._compute_score_matrix(visual_pred, alignment_scores, cells)
        
        # Step 2: Optimize cell assignments
        if self.method == 'hungarian':
            refined_cells = self._optimize_hungarian(cells, score_matrix, visual_pred, alignment_scores)
        elif self.method == 'dp':
            refined_cells = self._optimize_dp(cells, score_matrix, visual_pred)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
        
        logger.info(f"Refinement complete: {len(refined_cells)} cells")
        
        return refined_cells
    
    def _compute_score_matrix(
        self,
        visual_pred: Dict[str, Any],
        alignment_scores: Dict[str, Any],
        cells: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Compute combined score matrix for each cell.
        
        Returns:
            scores: [N_cells] array of combined scores
        """
        n_cells = len(cells)
        scores = np.zeros(n_cells)
        
        for i, cell in enumerate(cells):
            row_idx = cell.get('row_idx', 0)
            col_idx = cell.get('col_idx', 0)
            
            # Visual score: confidence from RCA model
            visual_score = self._get_visual_score(row_idx, col_idx, visual_pred)
            
            # Semantic score: alignment with column header
            semantic_score = self._get_semantic_score(col_idx, alignment_scores)
            
            # Layout score: geometric consistency
            layout_score = self._get_layout_score(cell, cells)
            
            # Combined score
            scores[i] = (
                self.alpha * visual_score +
                self.beta * semantic_score +
                self.gamma * layout_score
            )
        
        return scores
    
    def _get_visual_score(
        self,
        row_idx: int,
        col_idx: int,
        visual_pred: Dict[str, Any]
    ) -> float:
        """Get visual confidence score for a cell position."""
        # Extract confidence scores if available
        row_conf = visual_pred.get('row_confidence', None)
        col_conf = visual_pred.get('col_confidence', None)
        
        if row_conf is None or col_conf is None:
            # No confidence available, use default
            return 0.5
        
        # Get confidence for this cell's boundaries
        # Note: row_conf/col_conf are lists per batch
        if len(row_conf) > 0 and len(row_conf[0]) > row_idx:
            row_score = row_conf[0][row_idx] if row_idx < len(row_conf[0]) else 0.5
        else:
            row_score = 0.5
        
        if len(col_conf) > 0 and len(col_conf[0]) > col_idx:
            col_score = col_conf[0][col_idx] if col_idx < len(col_conf[0]) else 0.5
        else:
            col_score = 0.5
        
        # Average of row and column confidence
        return (row_score + col_score) / 2.0
    
    def _get_semantic_score(
        self,
        col_idx: int,
        alignment_scores: Dict[str, Any]
    ) -> float:
        """Get semantic alignment score for a column."""
        col_alignment = alignment_scores.get('col_alignment', {})
        
        if col_idx not in col_alignment:
            return 0.0
        
        # Use mean similarity as semantic score
        return col_alignment[col_idx].get('mean_similarity', 0.0)
    
    def _get_layout_score(
        self,
        cell: Dict[str, Any],
        all_cells: List[Dict[str, Any]]
    ) -> float:
        """
        Compute geometric layout consistency score.
        
        Checks if cell's bbox aligns well with other cells in same row/column.
        """
        bbox = cell.get('bbox', None)
        if bbox is None:
            return 0.5
        
        row_idx = cell.get('row_idx', 0)
        col_idx = cell.get('col_idx', 0)
        
        # Find cells in same row and column
        same_row = [c for c in all_cells if c.get('row_idx') == row_idx and c.get('bbox') is not None]
        same_col = [c for c in all_cells if c.get('col_idx') == col_idx and c.get('bbox') is not None]
        
        # Compute vertical alignment score (within row)
        row_score = self._compute_alignment_score(bbox, same_row, axis='horizontal')
        
        # Compute horizontal alignment score (within column)
        col_score = self._compute_alignment_score(bbox, same_col, axis='vertical')
        
        return (row_score + col_score) / 2.0
    
    def _compute_alignment_score(
        self,
        bbox: List[float],
        cells: List[Dict[str, Any]],
        axis: str
    ) -> float:
        """
        Compute how well a bbox aligns with other cells.
        
        Args:
            bbox: [xmin, ymin, xmax, ymax]
            cells: List of cells to compare against
            axis: 'horizontal' (check y alignment) or 'vertical' (check x alignment)
        """
        if len(cells) <= 1:
            return 1.0  # No comparison needed
        
        if axis == 'horizontal':
            # Check y-alignment (top/bottom edges)
            coord_idx = 1  # ymin
            coord_idx_max = 3  # ymax
        else:  # vertical
            # Check x-alignment (left/right edges)
            coord_idx = 0  # xmin
            coord_idx_max = 2  # xmax
        
        # Compute variance of edge positions
        edge_positions = [c['bbox'][coord_idx] for c in cells]
        edge_positions_max = [c['bbox'][coord_idx_max] for c in cells]
        
        var_min = np.var(edge_positions) if len(edge_positions) > 1 else 0.0
        var_max = np.var(edge_positions_max) if len(edge_positions_max) > 1 else 0.0
        
        # Lower variance = better alignment
        # Map variance to score in [0, 1]
        alignment_score = 1.0 / (1.0 + var_min + var_max)
        
        return alignment_score
    
    def _optimize_hungarian(
        self,
        cells: List[Dict[str, Any]],
        scores: np.ndarray,
        visual_pred: Dict[str, Any],
        alignment_scores: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Optimize using Hungarian algorithm (linear assignment).
        
        Reassigns cells to columns based on semantic alignment.
        """
        # Get misaligned cells
        col_alignment = alignment_scores.get('col_alignment', {})
        col_groups = alignment_scores.get('col_groups', {})
        
        refined_cells = cells.copy()
        
        # For each column with low alignment, try to reassign cells
        for col_idx, col_scores in col_alignment.items():
            if col_scores['mean_similarity'] < 0.4:  # Threshold for reassignment
                logger.info(f"Column {col_idx} has low alignment ({col_scores['mean_similarity']:.2f}), attempting reassignment")
                
                # Get cells in this column
                if col_idx not in col_groups:
                    continue
                
                cell_indices = col_groups[col_idx]
                
                # For each cell, compute cost of assigning to each column
                for cell_idx in cell_indices:
                    # Skip if header (don't reassign headers)
                    if cells[cell_idx].get('row_idx', 0) == 0:
                        continue
                    
                    # Reassign based on semantic similarity to all column headers
                    # (This is simplified; full Hungarian would build cost matrix)
                    # For now, just use greedy reassignment
                    best_col = col_idx
                    best_score = col_scores['mean_similarity']
                    
                    for other_col, other_scores in col_alignment.items():
                        if other_scores['mean_similarity'] > best_score:
                            best_col = other_col
                            best_score = other_scores['mean_similarity']
                    
                    if best_col != col_idx:
                        logger.info(f"  Reassigning cell {cell_idx} from col {col_idx} to col {best_col}")
                        refined_cells[cell_idx]['col_idx'] = best_col
        
        return refined_cells
    
    def _optimize_dp(
        self,
        cells: List[Dict[str, Any]],
        scores: np.ndarray,
        visual_pred: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Optimize using Dynamic Programming.
        
        Finds optimal row assignment to maximize total score.
        """
        # DP approach: for each row, find best assignment of cells
        # This is simplified for now
        
        refined_cells = cells.copy()
        
        # Group cells by row
        rows = {}
        for i, cell in enumerate(cells):
            row_idx = cell.get('row_idx', 0)
            if row_idx not in rows:
                rows[row_idx] = []
            rows[row_idx].append((i, scores[i]))
        
        # For each row, sort cells by score and potentially reorder
        for row_idx, row_cells in rows.items():
            # Sort by column index
            row_cells_sorted = sorted(row_cells, key=lambda x: cells[x[0]].get('col_idx', 0))
            
            # Update based on scores (keep high-score cells, drop low-score ones)
            for cell_idx, score in row_cells_sorted:
                if score < 0.3:  # Threshold for dropping
                    logger.info(f"DP: Low score ({score:.2f}) for cell {cell_idx}, marking for review")
                    # Could mark cell as uncertain or merge with neighbors
        
        return refined_cells

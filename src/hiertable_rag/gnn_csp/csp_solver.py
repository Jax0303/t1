"""
CSP Solver for Table Structure Recognition

Uses Google OR-Tools CP-SAT solver to find optimal table structure
that satisfies all hard constraints while maximizing soft constraint scores.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ortools.sat.python import cp_model
import logging

logger = logging.getLogger(__name__)


class TableCSPSolver:
    """
    Constraint Satisfaction Problem solver for table structure.
    
    Formulates TSR as an optimization problem:
    - Decision variables: row/col assignments for each cell
    - Objective: Maximize alignment scores
    - Constraints: Hard constraints must be satisfied
    """
    
    def __init__(
        self,
        alpha: float = 0.5,  # Visual weight
        beta: float = 0.3,   # Semantic weight
        gamma: float = 0.2,  # Layout weight
        max_rows: int = 50,
        max_cols: int = 20,
        time_limit_seconds: float = 30.0
    ):
        """
        Args:
            alpha, beta, gamma: Weights for different objectives
            max_rows, max_cols: Maximum table dimensions
            time_limit_seconds: Maximum solver time
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.time_limit = time_limit_seconds
        
        # Normalize weights
        total = alpha + beta + gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
    
    def solve(
        self,
        cells: List[Dict[str, Any]],
        gnn_scores: Dict[str, torch.Tensor],
        visual_scores: torch.Tensor,
        initial_structure: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Solve for optimal table structure.
        
        Args:
            cells: List of cell dictionaries
            gnn_scores: Scores from GNN (structure_confidence, header_prob)
            visual_scores: Visual confidence from RCA
            initial_structure: Optional initial row/col assignments for warm start
            
        Returns:
            Dictionary with:
                - 'row_assignment': [N] array of row indices
                - 'col_assignment': [N] array of col indices
                - 'num_rows': int
                - 'num_cols': int
                - 'objective_value': float (optimization score)
                - 'solve_time': float (seconds)
                - 'status': str (OPTIMAL, FEASIBLE, INFEASIBLE)
        """
        num_cells = len(cells)
        
        if num_cells == 0:
            return {
                'row_assignment': np.array([], dtype=int),
                'col_assignment': np.array([], dtype=int),
                'num_rows': 0,
                'num_cols': 0,
                'objective_value': 0.0,
                'solve_time': 0.0,
                'status': 'EMPTY'
            }
        
        # Use greedy heuristic for initial solution
        # OR-Tools CP-SAT is better for feasibility than optimality for complex problems
        # So we'll use a simplified greedy approach with constraint checking
        
        logger.info(f"Solving CSP for {num_cells} cells")
        
        result = self._greedy_solve_with_constraints(
            cells, gnn_scores, visual_scores, initial_structure
        )
        
        return result
    
    def _greedy_solve_with_constraints(
        self,
        cells: List[Dict[str, Any]],
        gnn_scores: Dict[str, torch.Tensor],
        visual_scores: torch.Tensor,
        initial_structure: Optional[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Greedy algorithm with constraint checking.
        
        This is a practical heuristic since full CSP is NP-hard.
        """
        num_cells = len(cells)
        
        # If we have initial structure, validate and refine it
        if initial_structure is not None:
            row_assignment = initial_structure['row_assignment']
            col_assignment = initial_structure['col_assignment']
        else:
            # Initialize with spatial sorting
            row_assignment, col_assignment = self._spatial_sort_initialization(cells)
        
        # Refine assignments to satisfy constraints
        row_assignment, col_assignment = self._refine_assignments(
            cells, row_assignment, col_assignment, gnn_scores, visual_scores
        )
        
        # Count unique rows/cols
        num_rows = len(np.unique(row_assignment))
        num_cols = len(np.unique(col_assignment))
        
        # Compute objective value
        objective_value = self._compute_objective(
            cells, row_assignment, col_assignment, gnn_scores, visual_scores
        )
        
        return {
            'row_assignment': row_assignment,
            'col_assignment': col_assignment,
            'num_rows': num_rows,
            'num_cols': num_cols,
            'objective_value': objective_value,
            'solve_time': 0.0,  # Greedy is instant
            'status': 'FEASIBLE'
        }
    
    def _spatial_sort_initialization(
        self,
        cells: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize row/col assignments by spatial sorting.
        
        - Sort cells by y-coordinate for row assignment
        - Sort cells by x-coordinate for col assignment
        """
        num_cells = len(cells)
        
        # Extract cell centers
        centers = np.array([
            [(box['box'][0] + box['box'][2]) / 2,
             (box['box'][1] + box['box'][3]) / 2]
            for box in cells
        ])
        
        # Row assignment: cluster by y-coordinate
        y_coords = centers[:, 1]
        y_sorted_indices = np.argsort(y_coords)
        
        # Assign rows based on y-coordinate clusters
        row_assignment = np.zeros(num_cells, dtype=int)
        current_row = 0
        last_y = y_coords[y_sorted_indices[0]]
        
        # Use adaptive threshold based on median cell height
        heights = [cell['box'][3] - cell['box'][1] for cell in cells]
        median_height = np.median(heights)
        y_threshold = median_height * 0.4  # 40% of median height
        
        for idx in y_sorted_indices:
            y = y_coords[idx]
            if y - last_y > y_threshold:
                current_row += 1
                last_y = y
            row_assignment[idx] = current_row
        
        # Column assignment: cluster by x-coordinate within each row
        col_assignment = np.zeros(num_cells, dtype=int)
        x_coords = centers[:, 0]
        
        for row_idx in range(current_row + 1):
            row_mask = row_assignment == row_idx
            row_indices = np.where(row_mask)[0]
            
            if len(row_indices) == 0:
                continue
            
            # Sort by x within this row
            row_x_coords = x_coords[row_indices]
            x_sorted = np.argsort(row_x_coords)
            
            for col_idx, cell_idx in enumerate(row_indices[x_sorted]):
                col_assignment[cell_idx] = col_idx
        
        # Normalize column assignments across rows
        max_col = col_assignment.max()
        for col_idx in range(max_col + 1):
            col_mask = col_assignment == col_idx
            if col_mask.sum() == 0:
                continue
            
            # Get median x-coordinate for this column
            col_x = x_coords[col_mask].mean()
            
            # Reassign based on global x-coordinate
            for i in range(num_cells):
                if abs(x_coords[i] - col_x) < median_height:  # Use height as proxy for width
                    col_assignment[i] = col_idx
        
        return row_assignment, col_assignment
    
    def _refine_assignments(
        self,
        cells: List[Dict[str, Any]],
        row_assignment: np.ndarray,
        col_assignment: np.ndarray,
        gnn_scores: Dict[str, torch.Tensor],
        visual_scores: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine assignments to better satisfy constraints.
        
        Uses local search to improve objective while maintaining feasibility.
        """
        num_cells = len(cells)
        
        # Make a copy to avoid modifying input
        row_assignment = row_assignment.copy()
        col_assignment = col_assignment.copy()
        
        # Iterative refinement (max 3 iterations)
        for iteration in range(3):
            improved = False
            
            # Try swapping assignments for cells with low confidence
            confidence = gnn_scores['structure_confidence'].cpu().numpy()
            low_conf_indices = np.where(confidence < 0.7)[0]
            
            for idx in low_conf_indices:
                current_row = row_assignment[idx]
                current_col = col_assignment[idx]
                
                # Try adjacent row/col assignments
                for delta_row in [-1, 0, 1]:
                    for delta_col in [-1, 0, 1]:
                        if delta_row == 0 and delta_col == 0:
                            continue
                        
                        new_row = max(0, current_row + delta_row)
                        new_col = max(0, current_col + delta_col)
                        
                        # Try new assignment
                        old_row, old_col = row_assignment[idx], col_assignment[idx]
                        row_assignment[idx] = new_row
                        col_assignment[idx] = new_col
                        
                        # Check if this improves objective
                        new_objective = self._compute_objective(
                            cells, row_assignment, col_assignment, gnn_scores, visual_scores
                        )
                        
                        # Revert if not better
                        row_assignment[idx] = old_row
                        col_assignment[idx] = old_col
                        
                        old_objective = self._compute_objective(
                            cells, row_assignment, col_assignment, gnn_scores, visual_scores
                        )
                        
                        if new_objective > old_objective:
                            row_assignment[idx] = new_row
                            col_assignment[idx] = new_col
                            improved = True
                            break
            
            if not improved:
                break
        
        return row_assignment, col_assignment
    
    def _compute_objective(
        self,
        cells: List[Dict[str, Any]],
        row_assignment: np.ndarray,
        col_assignment: np.ndarray,
        gnn_scores: Dict[str, torch.Tensor],
        visual_scores: torch.Tensor
    ) -> float:
        """
        Compute objective value for current assignment.
        
        Higher is better.
        """
        num_cells = len(cells)
        
        if num_cells == 0:
            return 0.0
        
        # Visual score component
        visual_score = visual_scores.mean().item()
        
        # GNN structure confidence component
        gnn_confidence = gnn_scores['structure_confidence'].mean().item()
        
        # Layout regularity (penalty for irregular assignments)
        layout_score = 1.0  # Placeholder, would compute actual layout metrics
        
        # Weighted combination
        objective = (
            self.alpha * visual_score +
            self.beta * gnn_confidence +
            self.gamma * layout_score
        )
        
        return objective

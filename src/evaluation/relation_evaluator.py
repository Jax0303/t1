"""
Relation-based Structure Evaluator for Error Atlas Framework

Evaluates table structure using adjacency relations instead of direct cell comparison.
This approach is more robust to coordinate system differences between GT and predictions.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Any


class RelationEvaluator:
    """Evaluate table structure using adjacency relation comparison"""
    
    def __init__(self):
        pass
    
    def cells_to_relations(self, cells: List[Dict[str, Any]]) -> Set[Tuple]:
        """
        Convert cell list to adjacency relation set
        
        Args:
            cells: List of cells with structure info
                   Each cell should have: row, col, row_span, col_span (or start_row, end_row, start_col, end_col)
        
        Returns:
            Set of tuples: (cell_i_idx, cell_j_idx, relation_type)
            relation_type: 'horizontal' or 'vertical'
        """
        relations = set()
        
        # Build grid map: (row, col) -> node_id (start_row, start_col)
        grid = {}
        
        for i, cell in enumerate(cells):
            # Handle different cell formats
            if 'start_row' in cell and 'end_row' in cell:
                # SciTSR format
                s_r, e_r = cell['start_row'], cell['end_row']
                s_c, e_c = cell['start_col'], cell['end_col']
            elif 'row' in cell and 'row_span' in cell:
                # Standard format
                s_r = cell['row']
                e_r = cell['row'] + cell.get('row_span', 1) - 1
                s_c = cell['col']
                e_c = cell['col'] + cell.get('col_span', 1) - 1
            else:
                # Fallback
                s_r = cell.get('row', 0)
                e_r = s_r
                s_c = cell.get('col', 0)
                e_c = s_c
            
            # Node ID is the top-left coordinate of the cell
            node_id = (s_r, s_c)
            
            # Fill grid
            for r in range(s_r, e_r + 1):
                for c in range(s_c, e_c + 1):
                    grid[(r, c)] = node_id
        
        # Extract horizontal adjacency (same row, adjacent columns)
        for (r, c), node_i in grid.items():
            if (r, c + 1) in grid:
                node_j = grid[(r, c + 1)]
                if node_i != node_j:  # Different cells
                    # Use sorted tuple to avoid duplicates
                    # node_i and node_j are tuples (r,c), so they are sortable
                    relation = tuple(sorted([node_i, node_j])) + ('horizontal',)
                    relations.add(relation)
        
        # Extract vertical adjacency (same column, adjacent rows)
        for (r, c), node_i in grid.items():
            if (r + 1, c) in grid:
                node_j = grid[(r + 1, c)]
                if node_i != node_j:  # Different cells
                    relation = tuple(sorted([node_i, node_j])) + ('vertical',)
                    relations.add(relation)
        
        return relations
    
    def evaluate_structure(self, 
                          pred_relations: Set[Tuple], 
                          gt_relations: Set[Tuple]) -> Dict[str, Any]:
        """
        Evaluate structure using precision, recall, F1 on relations
        
        Args:
            pred_relations: Predicted adjacency relations
            gt_relations: Ground truth adjacency relations
        
        Returns:
            Dict with metrics: precision, recall, f1, TP, FP, FN, edge_FP, edge_FN
        """
        # Calculate set operations
        TP = len(pred_relations & gt_relations)
        FP = len(pred_relations - gt_relations)
        FN = len(gt_relations - pred_relations)
        
        # Calculate metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Separate horizontal and vertical metrics
        h_pred = {r for r in pred_relations if r[2] == 'horizontal'}
        h_gt = {r for r in gt_relations if r[2] == 'horizontal'}
        v_pred = {r for r in pred_relations if r[2] == 'vertical'}
        v_gt = {r for r in gt_relations if r[2] == 'vertical'}
        
        h_tp = len(h_pred & h_gt)
        h_fp = len(h_pred - h_gt)
        h_fn = len(h_gt - h_pred)
        
        v_tp = len(v_pred & v_gt)
        v_fp = len(v_pred - v_gt)
        v_fn = len(v_gt - v_pred)
        
        h_prec = h_tp / (h_tp + h_fp) if (h_tp + h_fp) > 0 else 0.0
        h_rec = h_tp / (h_tp + h_fn) if (h_tp + h_fn) > 0 else 0.0
        h_f1 = 2 * h_prec * h_rec / (h_prec + h_rec) if (h_prec + h_rec) > 0 else 0.0
        
        v_prec = v_tp / (v_tp + v_fp) if (v_tp + v_fp) > 0 else 0.0
        v_rec = v_tp / (v_tp + v_fn) if (v_tp + v_fn) > 0 else 0.0
        v_f1 = 2 * v_prec * v_rec / (v_prec + v_rec) if (v_prec + v_rec) > 0 else 0.0
        
        return {
            # Overall
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            
            # Directional
            'horizontal_f1': h_f1,
            'horizontal_precision': h_prec,
            'horizontal_recall': h_rec,
            'vertical_f1': v_f1,
            'vertical_precision': v_prec,
            'vertical_recall': v_rec,
            
            # Error details
            'edge_FP': list(pred_relations - gt_relations),
            'edge_FN': list(gt_relations - pred_relations),
            
            # Counts
            'total_pred_relations': len(pred_relations),
            'total_gt_relations': len(gt_relations)
        }
    
    def evaluate_cells(self, 
                       pred_cells: List[Dict], 
                       gt_cells: List[Dict]) -> Dict[str, Any]:
        """
        High-level evaluation: convert cells to relations and evaluate
        
        Args:
            pred_cells: Predicted cells
            gt_cells: Ground truth cells
        
        Returns:
            Evaluation metrics
        """
        pred_relations = self.cells_to_relations(pred_cells)
        gt_relations = self.cells_to_relations(gt_cells)
        
        metrics = self.evaluate_structure(pred_relations, gt_relations)
        
        # Add cell-level stats
        metrics['num_pred_cells'] = len(pred_cells)
        metrics['num_gt_cells'] = len(gt_cells)
        metrics['cell_count_diff'] = len(pred_cells) - len(gt_cells)
        
        return metrics
    
    def analyze_error_patterns(self, 
                               edge_FP: List[Tuple], 
                               edge_FN: List[Tuple]) -> Dict[str, Any]:
        """
        Analyze error patterns from false positive and false negative edges
        
        Args:
            edge_FP: False positive edges
            edge_FN: False negative edges
        
        Returns:
            Pattern analysis results
        """
        patterns = {
            'row_merge_split_likelihood': 0.0,
            'col_merge_split_likelihood': 0.0,
            'shift_pattern_likelihood': 0.0,
            'span_error_likelihood': 0.0
        }
        
        if not edge_FP and not edge_FN:
            return patterns
        
        # Count directional errors
        h_fp = sum(1 for e in edge_FP if e[2] == 'horizontal')
        h_fn = sum(1 for e in edge_FN if e[2] == 'horizontal')
        v_fp = sum(1 for e in edge_FP if e[2] == 'vertical')
        v_fn = sum(1 for e in edge_FN if e[2] == 'vertical')
        
        total_errors = len(edge_FP) + len(edge_FN)
        
        # Heuristic pattern detection
        # Row merge/split: many vertical edges broken
        if v_fn > v_fp and v_fn > h_fn:
            patterns['row_merge_split_likelihood'] = v_fn / total_errors
        
        # Col merge/split: many horizontal edges broken
        if h_fn > h_fp and h_fn > v_fn:
            patterns['col_merge_split_likelihood'] = h_fn / total_errors
        
        # Shift: balanced FP and FN in same direction
        if abs(h_fp - h_fn) < 5 and (h_fp + h_fn) > total_errors * 0.5:
            patterns['shift_pattern_likelihood'] = (h_fp + h_fn) / total_errors
        
        # Span error: cross-directional errors
        if min(h_fp, h_fn, v_fp, v_fn) > 0:
            patterns['span_error_likelihood'] = min(h_fp, h_fn, v_fp, v_fn) / total_errors
        
        return patterns
    
    def compute_grid_stats(self, cells: List[Dict]) -> Dict[str, int]:
        """
        Compute grid statistics from cells
        
        Args:
            cells: List of cells
        
        Returns:
            Dict with num_rows, num_cols
        """
        if not cells:
            return {'num_rows': 0, 'num_cols': 0}
        
        # Find max row and col indices
        max_row = 0
        max_col = 0
        
        for cell in cells:
            if 'end_row' in cell:
                max_row = max(max_row, cell['end_row'])
                max_col = max(max_col, cell['end_col'])
            elif 'row' in cell and 'row_span' in cell:
                max_row = max(max_row, cell['row'] + cell.get('row_span', 1) - 1)
                max_col = max(max_col, cell['col'] + cell.get('col_span', 1) - 1)
            else:
                max_row = max(max_row, cell.get('row', 0))
                max_col = max(max_col, cell.get('col', 0))
        
        return {
            'num_rows': max_row + 1,
            'num_cols': max_col + 1
        }

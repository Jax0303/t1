"""
Table uncertainty estimation for adaptive routing.
"""

from typing import Dict, Any, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TableUncertaintyEstimator:
    """
    Estimates the 'Structural Uncertainty' (SU) of a table.
    SU is defined as a weighted combination of geometric and topological complexity:
    SU = α * Dense_merged + β * Depth_hier + γ * Var_structure
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def estimate(self, table_raw: Dict[str, Any]) -> float:
        """
        Calculates the Structural Uncertainty (SU) score ∈ [0, 1].
        """
        rows = table_raw.get("rows", [])
        if not rows:
            return 1.0
            
        num_rows = len(rows)
        num_cols = len(rows[0]) if rows else 0
        total_grid_area = num_rows * num_cols if num_rows > 0 else 1
        
        # 1. Structural Density (Merged Cells)
        merged_area = 0
        has_explicit_spans = False
        for row in rows:
            for cell in row:
                if isinstance(cell, dict):
                    rs = cell.get("row_span", 1)
                    cs = cell.get("col_span", 1)
                    if rs > 1 or cs > 1:
                        merged_area += (rs * cs)
                        has_explicit_spans = True
        
        if not has_explicit_spans:
            dense_merged = table_raw.get("metadata", {}).get("merged_cell_ratio", 0.0)
        else:
            dense_merged = min(merged_area / total_grid_area, 1.0)
        
        # 2. Hierarchy Depth
        headers = table_raw.get("headers", [])
        max_depth = 1
        if headers:
            depths = {}
            for h in headers:
                h_id = h.get("id")
                parent = h.get("parent")
                if not parent or parent not in depths:
                    depths[h_id] = 1
                else:
                    depths[h_id] = depths.get(parent, 1) + 1
                max_depth = max(max_depth, depths[h_id])
        
        depth_hier = min((max_depth - 1) / 3.0, 1.0) if max_depth > 1 else 0.0
        
        # 3. Size Variance
        row_lengths = [len(r) for r in rows]
        if len(row_lengths) > 1:
            var_structure = np.std(row_lengths) / (np.mean(row_lengths) + 1e-6)
        else:
            var_structure = 0.0
        
        var_structure = min(var_structure, 1.0)
        
        su = (self.alpha * dense_merged) + (self.beta * depth_hier) + (self.gamma * var_structure)
        return float(np.clip(su, 0.0, 1.0))

class TTAUncertaintyEstimator:
    """
    Estimates uncertainty using Test-Time Augmentation (TTA) consistency.
    Higher consistency across augments -> Higher Confidence -> Lower Uncertainty.
    """
    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
        self.structural_estimator = TableUncertaintyEstimator()

    def estimate_confidence(self, table_raw: Dict[str, Any]) -> float:
        """
        Simulates TTA confidence by checking consistency of structural complexity
        under simulated 'noise' (representing image augmentations).
        
        Confidence = 1.0 - (StdDev of SU over samples * scaling_factor)
        """
        base_su = self.structural_estimator.estimate(table_raw)
        
        # Simulate TTA by adding noise proportional to the base complexity
        # More complex tables are more sensitive to 'augmentation noise'
        samples = []
        for _ in range(self.num_samples):
            noise = np.random.normal(0, 0.1 * base_su)
            samples.append(np.clip(base_su + noise, 0.0, 1.0))
            
        # Consistency is inversely proportional to standard deviation
        consistency = 1.0 - np.std(samples) * 5.0 # Scale to emphasize variance
        
        # Baseline confidence is also affected by absolute complexity
        # High complexity tables (high SU) have lower ceiling for confidence
        complexity_ceiling = 1.0 - (base_su * 0.5)
        
        confidence = float(np.clip(consistency * complexity_ceiling, 0.0, 1.0))
        
        logger.info(f"TTA Confidence: {confidence:.4f} (Base SU: {base_su:.2f})")
        return confidence

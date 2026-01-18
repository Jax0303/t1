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
    Based on Ajayi et al. (2024), consistency across multiple 'augmented' 
    interpretations of the table indicates higher confidence.
    """
    def __init__(self, num_samples: int = 7, perturbation_rate: float = 0.15):
        self.num_samples = num_samples
        self.perturbation_rate = perturbation_rate
        self.structural_estimator = TableUncertaintyEstimator()

    def _perturb_structure(self, table_raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates OCR/TSR noise by perturbing the table structure.
        """
        import copy
        perturbed = copy.deepcopy(table_raw)
        rows = perturbed.get("rows", [])
        if not rows:
            return perturbed
            
        # Simulating cell merge/split noise or dropouts
        for r_idx, row in enumerate(rows):
            for c_idx, _ in enumerate(row):
                if np.random.random() < self.perturbation_rate:
                    # Randomly toggle a span or content to simulate uncertainty
                    if isinstance(row[c_idx], dict):
                        row[c_idx]["row_span"] = max(1, row[c_idx].get("row_span", 1) + np.random.randint(-1, 2))
                        row[c_idx]["col_span"] = max(1, row[c_idx].get("col_span", 1) + np.random.randint(-1, 2))
        
        return perturbed

    def estimate_confidence(self, table_raw: Dict[str, Any]) -> float:
        """
        Calculates confidence based on the variance of structural uncertainty 
        across multiple perturbed samples.
        
        Confidence = 1.0 / (1.0 + Var(SU) * sensitivity)
        """
        base_su = self.structural_estimator.estimate(table_raw)
        
        su_samples = []
        for _ in range(self.num_samples):
            perturbed_table = self._perturb_structure(table_raw)
            sample_su = self.structural_estimator.estimate(perturbed_table)
            su_samples.append(sample_su)
            
        # Variance of samples indicates instability (uncertainty)
        variance = float(np.var(su_samples))
        
        # Stability-based confidence
        # If variance is 0, stability is 1.0
        stability = 1.0 / (1.0 + variance * 20.0)
        
        # Combine with absolute complexity: 
        # High base SU already suggests lower inherent confidence
        complexity_penalty = 1.0 - (base_su * 0.3)
        
        confidence = float(np.clip(stability * complexity_penalty, 0.0, 1.0))
        
        logger.info(
            f"TTA Analysis: Base SU={base_su:.3f}, Var={variance:.5f}, "
            f"Stability={stability:.3f}, Final Confidence={confidence:.3f}"
        )
        return confidence

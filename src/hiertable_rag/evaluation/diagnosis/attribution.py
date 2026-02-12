"""
Attribution Calculator for S1-S4 Scenarios

Computes OCR vs TSR error attribution with bootstrap confidence intervals.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

from .utils import safe_divide

logger = logging.getLogger(__name__)


class AttributionCalculator:
    """
    Calculate attribution between OCR and TSR errors using S1-S4 scenarios
    """
    
    # S1-S4 Scenario Definitions
    SCENARIOS = {
        'S1': {'ocr': 'noisy', 'tsr': 'engine'},     # Baseline
        'S2': {'ocr': 'perfect', 'tsr': 'engine'},   # Isolate TSR error
        'S3': {'ocr': 'noisy', 'tsr': 'gt'},         # Isolate OCR error
        'S4': {'ocr': 'perfect', 'tsr': 'gt'}        # Upper bound
    }
    
    def __init__(self, metric_key: str = 'm1_relation_f1'):
        """
        Initialize attribution calculator
        
        Args:
            metric_key: Which metric to use for attribution (default: m1_relation_f1)
        """
        self.metric_key = metric_key
    
    def compute_attribution(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute attribution from S1-S4 results
        
        Args:
            results: Dict with keys 'S1', 'S2', 'S3', 'S4' containing metrics
        
        Returns:
            Attribution dict with:
            - f1_s1, f1_s2, f1_s3, f1_s4: Individual scenario scores
            - delta_tsr: TSR improvement (S2 - S1)
            - delta_ocr: OCR improvement (S3 - S1)
            - tsr_attribution: TSR error percentage
            - ocr_attribution: OCR error percentage
            - primary_cause: 'TSR' or 'OCR'
        """
        # Extract F1 scores
        f1_s1 = results.get('S1', {}).get(self.metric_key, 0.0)  # Noisy OCR + Engine TSR
        f1_s2 = results.get('S2', {}).get(self.metric_key, 0.0)  # Perfect OCR + Engine TSR
        f1_s3 = results.get('S3', {}).get(self.metric_key, 0.0)  # Noisy OCR + GT TSR
        f1_s4 = results.get('S4', {}).get(self.metric_key, 1.0)  # Perfect OCR + GT TSR (upper bound)
        
        # CORRECTED: Calculate deltas with proper attribution
        # S2 vs S1: Perfect OCR effect (OCR contribution)
        # S3 vs S1: GT TSR effect (TSR contribution)
        delta_ocr = f1_s2 - f1_s1  # How much perfect OCR helps (isolates OCR error)
        delta_tsr = f1_s3 - f1_s1  # How much GT TSR helps (isolates TSR error)
        
        total_gap = f1_s4 - f1_s1  # Total improvement possible
        total_delta = delta_ocr + delta_tsr
        
        # Calculate attribution percentages
        # Handle near-zero total_delta to avoid division by zero
        if abs(total_delta) > 1e-6:
            ocr_attribution = delta_ocr / total_delta
            tsr_attribution = delta_tsr / total_delta
            
            # Check for structure-only TSR engines (OCR doesn't affect structure)
            if abs(delta_ocr) < 1e-6:
                logger.info("Structure-only TSR engine: OCR does not affect structure metrics (S1≈S2)")
        else:
            # No improvement possible or already perfect
            logger.warning(f"Total delta near zero ({total_delta:.6f}), cannot compute attribution reliably")
            ocr_attribution = float('nan')
            tsr_attribution = float('nan')
        
        # Determine primary cause (handle NaN)  
        if not np.isnan(tsr_attribution) and not np.isnan(ocr_attribution):
            primary_cause = 'TSR' if tsr_attribution > ocr_attribution else 'OCR'
        else:
            primary_cause = 'UNKNOWN'
        
        return {
            'f1_s1': f1_s1,
            'f1_s2': f1_s2,
            'f1_s3': f1_s3,
            'f1_s4': f1_s4,
            'delta_ocr': delta_ocr,
            'delta_tsr': delta_tsr,
            'total_gap': total_gap,
            'ocr_attribution': ocr_attribution,
            'tsr_attribution': tsr_attribution,
            'primary_cause': primary_cause
        }
    
    def compute_batch_attribution(
        self,
        batch_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute attribution across batch of tables
        
        Args:
            batch_results: List of per-table results with S1-S4 scenarios
        
        Returns:
            Aggregated attribution statistics
        """
        attributions = []
        
        for table_result in batch_results:
            table_id = table_result.get('table_id')
            scenarios = table_result.get('scenarios', {})
            
            # Extract metrics from scenarios
            s1_metrics = scenarios.get('S1', {}).get('metrics', {})
            s2_metrics = scenarios.get('S2', {}).get('metrics', {})
            s3_metrics = scenarios.get('S3', {}).get('metrics', {})
            s4_metrics = scenarios.get('S4', {}).get('metrics', {})
            
            attr = self.compute_attribution({
                'S1': s1_metrics,
                'S2': s2_metrics,
                'S3': s3_metrics,
                'S4': s4_metrics
            })
            
            attr['table_id'] = table_id
            attributions.append(attr)
        
        # Aggregate statistics
        if not attributions:
            return {}
        
        return {
            'n_tables': len(attributions),
            'mean_tsr_attribution': np.mean([a['tsr_attribution'] for a in attributions]),
            'mean_ocr_attribution': np.mean([a['ocr_attribution'] for a in attributions]),
            'mean_delta_tsr': np.mean([a['delta_tsr'] for a in attributions]),
            'mean_delta_ocr': np.mean([a['delta_ocr'] for a in attributions]),
            'tsr_dominant_count': sum(1 for a in attributions if a['primary_cause'] == 'TSR'),
            'ocr_dominant_count': sum(1 for a in attributions if a['primary_cause'] == 'OCR'),
            'per_table': attributions
        }
    
    def compute_bootstrap_ci(
        self,
        batch_results: List[Dict[str, Any]],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for attribution
        
        Args:
            batch_results: List of per-table results
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (default: 0.95 for 95% CI)
        
        Returns:
            Dict with (lower, upper) bounds for each metric
        """
        if not batch_results:
            return {}
        
        n_tables = len(batch_results)
        
        tsr_attrs = []
        ocr_attrs = []
        
        # Bootstrap sampling
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample_indices = np.random.choice(n_tables, size=n_tables, replace=True)
            sample = [batch_results[i] for i in sample_indices]
            
            # Compute attribution for sample
            sample_attr = self.compute_batch_attribution(sample)
            
            tsr_attrs.append(sample_attr.get('mean_tsr_attribution', 0.5))
            ocr_attrs.append(sample_attr.get('mean_ocr_attribution', 0.5))
        
        # Compute percentile confidence intervals
        alpha = (1 - confidence) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100
        
        tsr_ci = (
            np.percentile(tsr_attrs, lower_percentile),
            np.percentile(tsr_attrs, upper_percentile)
        )
        
        ocr_ci = (
            np.percentile(ocr_attrs, lower_percentile),
            np.percentile(ocr_attrs, upper_percentile)
        )
        
        return {
            'tsr_attribution_ci': tsr_ci,
            'ocr_attribution_ci': ocr_ci
        }
    
    def aggregate_by_type(
        self,
        batch_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate attribution by table type
        
        Args:
            batch_results: List of per-table results with 'type_labels' field
        
        Returns:
            Dict mapping type label to attribution statistics
        """
        type_groups = defaultdict(list)
        
        # Group by type
        for result in batch_results:
            type_labels = result.get('type_labels', [])
            
            # Assign to primary type (first label)
            if type_labels:
                primary_type = type_labels[0]
                type_groups[primary_type].append(result)
        
        # Compute attribution for each type
        type_attributions = {}
        
        for type_label, type_results in type_groups.items():
            type_attr = self.compute_batch_attribution(type_results)
            type_attributions[type_label] = type_attr
        
        return type_attributions

"""
Error Classifier with Heuristic Triage

Automatically classifies errors based on metrics and patterns.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class ErrorClassifier:
    """Automatic error classification based on metrics and patterns"""
    
    # Error type definitions
    ERROR_TYPES = [
        'Row Merge/Split',
        'Col Merge/Split',
        'Row/Col Shift',
        'Span Error',
        'Table Boundary Error',
        'OCR Content Error',
        'Pure TSR Error',
        'Combined Error',
        'Success',
        'Unknown'
    ]
    
    # Processing stage definitions
    STAGES = [
        'OCR Detection',
        'OCR Recognition',
        'OCR-to-TSR Transform',
        'TSR Grid Detection',
        'TSR Spanning/Merge',
        'Post-Processing',
        'Cell Assignment'
    ]
    
    def __init__(self):
        pass
    
    def classify(self, 
                 table_id: str,
                 metrics: Dict[str, Any],
                 edge_FP: List[Tuple],
                 edge_FN: List[Tuple]) -> Dict[str, Any]:
        """
        Classify error type and stage
        
        Args:
            table_id: Table identifier
            metrics: Evaluation metrics
            edge_FP: False positive edges
            edge_FN: False negative edges
        
        Returns:
            Classification result with type, stage, severity, confidence
        """
        struct_f1 = metrics.get('f1', 0.0)
        cer = metrics.get('cer', 0.0)
        
        # Default values
        error_type = 'Unknown'
        stage = 'Unknown'
        severity = 0.0
        confidence = 0.0
        secondary_types = []
        
        # Success check
        if struct_f1 > 0.98 and cer < 0.02:
            return {
                'primary_type': 'Success',
                'secondary_types': [],
                'stage': 'N/A',
                'severity': 0.0,
                'confidence': 0.95
            }
        
        # Count directional errors
        h_fp = sum(1 for e in edge_FP if len(e) > 2 and e[2] == 'horizontal')
        h_fn = sum(1 for e in edge_FN if len(e) > 2 and e[2] == 'horizontal')
        v_fp = sum(1 for e in edge_FP if len(e) > 2 and e[2] == 'vertical')
        v_fn = sum(1 for e in edge_FN if len(e) > 2 and e[2] == 'vertical')
        
        total_errors = len(edge_FP) + len(edge_FN)
        
        if total_errors == 0:
            # No structure errors but might have content errors
            if cer > 0.05:
                return {
                    'primary_type': 'OCR Content Error',
                    'secondary_types': [],
                    'stage': 'OCR Recognition',
                    'severity': cer,
                    'confidence': 0.9
                }
            else:
                return {
                    'primary_type': 'Success',
                    'secondary_types': [],
                    'stage': 'N/A',
                    'severity': 0.0,
                    'confidence': 0.9
                }
        
        # Classification rules
        
        # 1. Row Merge/Split: many vertical edges broken
        row_merge_split_score = 0.0
        if v_fn > v_fp and v_fn > h_fn * 1.5:
            row_merge_split_score = v_fn / total_errors
        
        # 2. Col Merge/Split: many horizontal edges broken
        col_merge_split_score = 0.0
        if h_fn > h_fp and h_fn > v_fn * 1.5:
            col_merge_split_score = h_fn / total_errors
        
        # 3. Shift pattern: balanced FP and FN
        shift_score = 0.0
        if abs(h_fp - h_fn) < 5 and (h_fp + h_fn) > total_errors * 0.4:
            shift_score = (h_fp + h_fn) / total_errors
        
        # 4. Span error: cross-directional errors
        span_score = 0.0
        if min(h_fp, h_fn, v_fp, v_fn) > 0:
            span_score = min(h_fp, h_fn, v_fp, v_fn) / total_errors
        
        # 5. Boundary error: cell count mismatch
        boundary_score = 0.0
        cell_diff = abs(metrics.get('num_pred_cells', 0) - metrics.get('num_gt_cells', 1))
        if cell_diff > metrics.get('num_gt_cells', 1) * 0.3:
            boundary_score = min(1.0, cell_diff / metrics.get('num_gt_cells', 1))
        
        # 6. OCR error: good structure, bad content
        ocr_error_score = 0.0
        if struct_f1 > 0.9 and cer > 0.05:
            ocr_error_score = cer
        
        # 7. Pure TSR error: bad structure, good OCR
        tsr_error_score = 0.0
        if struct_f1 < 0.7 and cer < 0.05:
            # Scale down slightly to allow specific structural errors to win if they are identified
            tsr_error_score = (1 - struct_f1) * 0.9
        
        # 8. Combined error: both bad
        combined_error_score = 0.0
        if struct_f1 < 0.7 and cer > 0.1:
            combined_error_score = (1 - struct_f1) * cer
        
        # Select primary type (highest score)
        scores = {
            'Row Merge/Split': row_merge_split_score,
            'Col Merge/Split': col_merge_split_score,
            'Row/Col Shift': shift_score,
            'Span Error': span_score,
            'Table Boundary Error': boundary_score,
            'OCR Content Error': ocr_error_score,
            'Pure TSR Error': tsr_error_score,
            'Combined Error': combined_error_score
        }
        
        if max(scores.values()) > 0:
            # Prioritize specific structural errors over generic 'Pure TSR Error'
            # If a specific error score is close to the max score (e.g. within 20%), prefer the specific one
            max_score = max(scores.values())
            candidates = [t for t, s in scores.items() if s >= max_score * 0.8]
            
            # Priority order (Specific -> Generic)
            priority = [
                'OCR Content Error', # Most specific
                'Combined Error',
                'Table Boundary Error',
                'Row Merge/Split',
                'Col Merge/Split',
                'Row/Col Shift', 
                'Span Error',
                'Pure TSR Error' # Least specific
            ]
            
            # Pick the highest priority candidate
            error_type = 'Unknown'
            for p in priority:
                if p in candidates:
                    error_type = p
                    break
            
            if error_type == 'Unknown':
                error_type = max(scores, key=scores.get)

            severity = scores[error_type]
            confidence = min(0.9, severity * 1.2)
            
            # Add secondary types (scores > 0.2)
            secondary_types = [t for t, s in scores.items() 
                             if s > 0.2 and t != error_type]
        else:
            error_type = 'Unknown'
            severity = 1 - struct_f1
            confidence = 0.3
        
        # Determine processing stage
        if error_type in ['Row Merge/Split', 'Col Merge/Split']:
            stage = 'TSR Grid Detection'
        elif error_type == 'Span Error':
            stage = 'TSR Spanning/Merge'
        elif error_type == 'Row/Col Shift':
            stage = 'Cell Assignment'
        elif error_type == 'Table Boundary Error':
            stage = 'TSR Grid Detection'
        elif error_type == 'OCR Content Error':
            stage = 'OCR Recognition'
        elif error_type == 'Pure TSR Error':
            stage = 'TSR Grid Detection'
        elif error_type == 'Combined Error':
            stage = 'Multiple'
        else:
            stage = 'Unknown'
        
        return {
            'primary_type': error_type,
            'secondary_types': secondary_types,
            'stage': stage,
            'severity': severity,
            'confidence': confidence,
            'scores': scores  # For debugging
        }
    
    def analyze_error_patterns(self, 
                               edge_FP: List[Tuple], 
                               edge_FN: List[Tuple]) -> Dict[str, float]:
        """Analyze error patterns for detailed diagnosis"""
        
        if not edge_FP and not edge_FN:
            return {}
        
        # Count by direction
        h_fp = sum(1 for e in edge_FP if len(e) > 2 and e[2] == 'horizontal')
        h_fn = sum(1 for e in edge_FN if len(e) > 2 and e[2] == 'horizontal')
        v_fp = sum(1 for e in edge_FP if len(e) > 2 and e[2] == 'vertical')
        v_fn = sum(1 for e in edge_FN if len(e) > 2 and e[2] == 'vertical')
        
        total_errors = len(edge_FP) + len(edge_FN)
        
        return {
            'horizontal_fp_ratio': h_fp / total_errors if total_errors > 0 else 0,
            'horizontal_fn_ratio': h_fn / total_errors if total_errors > 0 else 0,
            'vertical_fp_ratio': v_fp / total_errors if total_errors > 0 else 0,
            'vertical_fn_ratio': v_fn / total_errors if total_errors > 0 else 0,
            'total_errors': total_errors,
            'fp_fn_balance': abs(len(edge_FP) - len(edge_FN)) / total_errors if total_errors > 0 else 0
        }


if __name__ == '__main__':
    # Test classifier
    classifier = ErrorClassifier()
    
    # Test case 1: Success
    metrics1 = {'f1': 0.99, 'cer': 0.01}
    result1 = classifier.classify('test1', metrics1, [], [])
    print("Test 1 (Success):", result1['primary_type'])
    
    # Test case 2: Row merge/split
    metrics2 = {'f1': 0.5, 'cer': 0.01, 'num_pred_cells': 10, 'num_gt_cells': 12}
    edge_fn2 = [(1, 2, 'vertical'), (3, 4, 'vertical'), (5, 6, 'vertical')]
    result2 = classifier.classify('test2', metrics2, [], edge_fn2)
    print("Test 2 (Row Merge/Split):", result2['primary_type'], f"(severity: {result2['severity']:.2f})")
    
    # Test case 3: OCR error
    metrics3 = {'f1': 0.95, 'cer': 0.15}
    result3 = classifier.classify('test3', metrics3, [], [])
    print("Test 3 (OCR Error):", result3['primary_type'], f"(severity: {result3['severity']:.2f})")
    
    print("\n✓ Error classifier test complete")

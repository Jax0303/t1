"""
Automated Error Labeler for Table Extraction
"""
from typing import Dict, List, Any

class AutoLabeler:
    @staticmethod
    def classify(metrics: Dict[str, Any], tsr_res: Dict[str, Any], gt_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify error into TSR, OCR, or Assignment based on metrics
        """
        f1 = metrics.get('f1', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        
        # 1. Structure (TSR) check
        n_row_p = tsr_res.get('num_rows', 0)
        n_col_p = tsr_res.get('num_cols', 0)
        
        # SciTSR GT might have different row/col definition, but we can check if it's wildly different
        # For evidence, we often check if Relation F1 is low.
        
        error_type = "None"
        fault_stage = "N/A"
        evidence_note = ""
        
        if f1 < 0.9:
            fault_stage = "TSR"
            if precision < recall:
                error_type = "TSR: Split/Extra (FP Relations)"
            elif recall < precision:
                error_type = "TSR: Merge/Missing (FN Relations)"
            else:
                error_type = "TSR: General Structural Failure"
            evidence_note = f"Relation F1={f1:.2f} indicates structural misalignment."
        else:
            # Structure is good, check content
            # (In a real scenario, we'd calculate CER here, but for now we look at S1 vs S3 results)
            # This logic will be used in the main runner to compare S1 and S3
            error_type = "Content: TBD"
            fault_stage = "OCR/Assignment"
            
        return {
            'error_type': error_type,
            'fault_stage': fault_stage,
            'evidence_note': evidence_note
        }

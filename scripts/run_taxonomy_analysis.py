
import sys
import json
import os
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_tsr.enhanced_pipeline import EnhancedOCRTSRPipeline
from src.evaluation.relation_evaluator import RelationEvaluator

def run_taxonomy_experiment():
    # 1. Selection based on Taxonomy
    taxonomy = {
        "Type 1: Simple Grid": "1003.0628v1.3",
        "Type 2: Hierarchical Col Headers": "1504.01806v1.4",
        "Type 3: Hierarchical Row Headers": "1805.02036v1.2",
        "Type 4: Sparse/Irregular": "1704.01419v1.3",
        "Type 5: Complex Internal Spans": "1807.01801v1.6"
    }
    
    evaluator = RelationEvaluator()
    results = []
    
    # Initialize Pipelines
    print("Initializing Pipelines...")
    pipeline_full = EnhancedOCRTSRPipeline(ocr_mode='paddleocr', tsr_mode='table_transformer')
    pipeline_pure_tsr = EnhancedOCRTSRPipeline(ocr_mode='chunk', tsr_mode='table_transformer')
    pipeline_pure_ocr = EnhancedOCRTSRPipeline(ocr_mode='paddleocr', tsr_mode='gt')
    
    output_dir = Path("results/taxonomy_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for label, tid in taxonomy.items():
        print(f"\nProcessing {label} (ID: {tid})...")
        img_path = f"data/SciTSR/test/img/{tid}.png"
        gt_path = f"data/SciTSR/test/structure/{tid}.json"
        
        if not os.path.exists(img_path):
            print(f"Skipping {tid}, image not found.")
            continue
            
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
            
        # Scenario A: Full (PaddleOCR + TATR)
        print(" [A] Running Full Pipeline...")
        ocr_a, tsr_a = pipeline_full.process(img_path, tid, gt_data)
        metrics_a = evaluator.evaluate_cells(tsr_a['cells'], gt_data['cells'])
        
        # Scenario B: Pure TSR (Perfect OCR + TATR)
        print(" [B] Running Pure TSR...")
        ocr_b, tsr_b = pipeline_pure_tsr.process(img_path, tid, gt_data)
        metrics_b = evaluator.evaluate_cells(tsr_b['cells'], gt_data['cells'])
        
        # Scenario C: Pure OCR (PaddleOCR + Perfect TSR)
        print(" [C] Running Pure OCR...")
        ocr_c, tsr_c = pipeline_pure_ocr.process(img_path, tid, gt_data)
        metrics_c = evaluator.evaluate_cells(tsr_c['cells'], gt_data['cells'])
        
        # Save Contrast Data for User Verification
        # Calculate specific error metrics
        cell_loss = max(0, len(gt_data['cells']) - len(tsr_a['cells']))
        # Structural shift is estimated by relationship False Positives (misplaced connections)
        shift_count = metrics_a.get('FP', 0)
        
        contrast_data = {
            "taxonomy": label,
            "table_id": tid,
            "error_metrics": {
                "cell_loss": cell_loss,
                "shift_count": shift_count
            },
            "scenarios": {
                "A_Full": {"f1": metrics_a['f1'], "fp_edges": metrics_a['edge_FP'][:5], "fn_edges": metrics_a['edge_FN'][:5]},
                "B_PureTSR": {"f1": metrics_b['f1'], "fp_edges": metrics_b['edge_FP'][:5], "fn_edges": metrics_b['edge_FN'][:5]},
                "C_PureOCR": {"f1": metrics_c['f1'], "fp_edges": metrics_c['edge_FP'][:5], "fn_edges": metrics_c['edge_FN'][:5]}
            }
        }
        
        with open(output_dir / f"{tid}_contrast.json", 'w') as f:
            json.dump(contrast_data, f, indent=2)
            
        results.append({
            "Type": label,
            "ID": tid,
            "Full_F1": metrics_a['f1'],
            "Cell_Loss": cell_loss,
            "Shift_Count": shift_count,
            "TSR_Loss": metrics_c['f1'] - metrics_a['f1'],
            "OCR_Loss": metrics_b['f1'] - metrics_a['f1'],
            "Raw_Metrics": {"A": metrics_a, "B": metrics_b, "C": metrics_c}
        })

    # Save summary results
    with open(output_dir / "taxonomy_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTaxonomy Analysis Complete.")
    return results

if __name__ == "__main__":
    run_taxonomy_experiment()

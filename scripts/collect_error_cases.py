"""
Collect Error Cases from Validation Set
Runs RCA model on validation data and collects all misclassified cases for detailed analysis.
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from src.hiertable_rag.core.rca_model import RCAModel
from src.hiertable_rag.core.grid_restorer import IntersectionRestorer
from scripts.universal_dataset import UniversalTableDataset

from src.evaluation.table_classifier import TableClassifier

def calculate_teds_simple(pred_cells, gt_cells):
    """Simplified TEDS calculation."""
    if not gt_cells:
        return 0.0
    
    correct = 0
    for p in pred_cells:
        for g in gt_cells:
            if (p.get('row_idx') == g.get('row_idx') and 
                p.get('col_idx') == g.get('col_idx')):
                correct += 1
                break
    
    return correct / max(len(pred_cells), len(gt_cells), 1)

def identify_failure_mode(pred_cells, gt_cells, pred_rows, pred_cols, gt_rows, gt_cols):
    """
    Identify the primary failure mode for this error case.
    """
    failure_modes = []
    
    # Row alignment check
    if len(pred_rows) != len(gt_rows):
        failure_modes.append("Row-Count-Mismatch")
    else:
        # Check alignment more carefully
        # Calculate max shift
        if gt_rows is not None and len(gt_rows) > 0:
             # Match closest
             shifts = []
             for gr in gt_rows:
                 closest = min([abs(pr - gr) for pr in pred_rows]) if pred_rows else 1.0
                 shifts.append(closest)
             
             if np.mean(shifts) > 0.02: # Threshold for noticeable shift
                 failure_modes.append("Row-Shift")
    
    # Column alignment check
    if len(pred_cols) != len(gt_cols):
        failure_modes.append("Col-Count-Mismatch")
    else:
        if gt_cols is not None and len(gt_cols) > 0:
             shifts = []
             for gc in gt_cols:
                 closest = min([abs(pc - gc) for pc in pred_cols]) if pred_cols else 1.0
                 shifts.append(closest)
             
             if np.mean(shifts) > 0.02:
                 failure_modes.append("Col-Misalignment")
    
    # Cell-level checks
    if len(pred_cells) != len(gt_cells):
        failure_modes.append("Cell-Count-Mismatch")
    
    # Check for merge errors
    pred_merged = sum(1 for c in pred_cells if c.get('row_span', 1) > 1 or c.get('col_span', 1) > 1)
    gt_merged = sum(1 for c in gt_cells if c.get('row_span', 1) > 1 or c.get('col_span', 1) > 1)
    if abs(pred_merged - gt_merged) > 2:
        failure_modes.append("Cell-Merge-Error")
    
    # Content loss check
    pred_content_cells = sum(1 for c in pred_cells if c.get('content', '').strip())
    gt_content_cells = sum(1 for c in gt_cells if c.get('content', '').strip())
    if pred_content_cells < gt_content_cells * 0.8:
        failure_modes.append("Content-Loss")
    
    return failure_modes if failure_modes else ["Unknown-Error"]

def collect_errors(dataset_name="scitsr", split="val", teds_threshold=0.8, limit=None):
    """Collect all error cases from validation set."""
    
    print(f"\n{'='*70}")
    print(f"COLLECTING ERROR CASES: {dataset_name.upper()} {split.upper()}")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = RCAModel().to(device)
    checkpoint_path = "/root/t1-9/outputs/rca_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: No trained model found, using random initialization")
    
    model.eval()
    restorer = IntersectionRestorer()
    classifier = TableClassifier()
    
    # Load dataset
    base_dir = "/root/t1-9/data/external"
    try:
        dataset = UniversalTableDataset(base_dir, dataset_name, split=split)
        print(f"Dataset loaded: {len(dataset)} samples\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    error_cases = []
    all_teds_scores = []
    
    # Initialize taxonomy counters
    taxonomy_counts = {
        'macro_type': defaultdict(int),
        'header_structure': defaultdict(int),
        'cell_characteristics': defaultdict(int)
    }
    
    num_samples = len(dataset)
    if limit:
        num_samples = min(num_samples, limit)
        print(f"Limiting to first {num_samples} samples.")
    
    print("Processing samples...")
    for i in tqdm(range(num_samples)):
        try:
            batch = dataset[i]
            img = batch['image'].unsqueeze(0).to(device)
            gt_cells = batch['cells']
            raw_cells = batch.get('raw_cells', [])
            
            # Get ground truth boundaries if available
            gt_row_bounds = batch.get('row_gt', None)
            gt_col_bounds = batch.get('col_gt', None)
            
            # Predict
            with torch.no_grad():
                pred = model.predict_boundaries(img)
            
            pred_row_bounds = pred["row_boundaries"][0]
            pred_col_bounds = pred["col_boundaries"][0]
            
            rca_res = {
                "row_boundaries": pred_row_bounds,
                "col_boundaries": pred_col_bounds
            }
            
            # Restore cells
            pred_cells = restorer.restore_cells(rca_res, ocr_boxes=raw_cells)
            
            # Calculate TEDS
            teds = calculate_teds_simple(pred_cells, gt_cells)
            all_teds_scores.append(teds)
            
            # Match taxonomy for all samples to get denominators
            tags = classifier.classify({"cells": gt_cells})
            table_type = tags["macro"]
            header_structure = tags["structural"]
            cell_chars = tags["micro"]
            
            # Update counts
            taxonomy_counts['macro_type'][table_type] += 1
            taxonomy_counts['header_structure'][header_structure] += 1
            taxonomy_counts['cell_characteristics'][cell_chars] += 1

            # Collect error cases
            if teds < teds_threshold:
                # Identify failure mode
                failure_modes = identify_failure_mode(
                    pred_cells, gt_cells,
                    pred_row_bounds, pred_col_bounds,
                    gt_row_bounds if gt_row_bounds is not None else [],
                    gt_col_bounds if gt_col_bounds is not None else []
                )
                
                error_case = {
                    'index': i,
                    'teds': float(teds),
                    'table_type': table_type,
                    'header_structure': header_structure,
                    'cell_characteristics': cell_chars,
                    'failure_modes': failure_modes,
                    'num_pred_cells': len(pred_cells),
                    'num_gt_cells': len(gt_cells),
                    'num_pred_rows': len(pred_row_bounds),
                    'num_gt_rows': len(gt_row_bounds) if gt_row_bounds is not None else 0,
                    'num_pred_cols': len(pred_col_bounds),
                    'num_gt_cols': len(gt_col_bounds) if gt_col_bounds is not None else 0,
                }
                
                error_cases.append(error_case)
        
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples processed: {num_samples}")
    print(f"Error cases collected (TEDS < {teds_threshold}): {len(error_cases)}")
    print(f"Error rate: {len(error_cases)/num_samples*100:.2f}%")
    if all_teds_scores:
        print(f"Average TEDS (all): {np.mean(all_teds_scores):.4f}")
    if error_cases:
        print(f"Average TEDS (errors only): {np.mean([e['teds'] for e in error_cases]):.4f}")
    

    
    # Save taxonomy counts separate or included? Included is better but let's save separate or just inside error_cases.json
    # Updating json dump above to include counts
    output_path = Path("outputs/error_cases.json")
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'split': split,
            'teds_threshold': teds_threshold,
            'total_samples': num_samples,
            'error_count': len(error_cases),
            'error_rate': len(error_cases)/num_samples if num_samples > 0 else 0,
            'avg_teds_all': float(np.mean(all_teds_scores)) if all_teds_scores else 0.0,
            'avg_teds_errors': float(np.mean([e['teds'] for e in error_cases])) if error_cases else 0.0,
            'taxonomy_counts': {
                'macro_type': dict(taxonomy_counts['macro_type']),
                'header_structure': dict(taxonomy_counts['header_structure']),
                'cell_characteristics': dict(taxonomy_counts['cell_characteristics'])
            },
            'error_cases': error_cases
        }, f, indent=2)

    print(f"\n✓ Saved error cases to: {output_path}")
    print(f"{'='*70}\n")
    
    return error_cases

if __name__ == "__main__":
    collect_errors(dataset_name="scitsr", split="val", teds_threshold=0.8, limit=100)

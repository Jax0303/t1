"""
Root Cause Analysis: Why is the model always predicting 13 rows?
Investigates the RCA model's boundary prediction mechanism.
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from src.hiertable_rag.core.rca_model import RCAModel
from scripts.universal_dataset import UniversalTableDataset

def analyze_prediction_distribution():
    """Analyze the distribution of predicted row/col counts."""
    
    print(f"\n{'='*80}")
    print(f"ROOT CAUSE ANALYSIS: Boundary Prediction Distribution")
    print(f"{'='*80}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCAModel().to(device)
    
    checkpoint_path = "/root/t1-9/outputs/rca_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.eval()
    
    # Load dataset
    base_dir = "/root/t1-9/data/external"
    dataset = UniversalTableDataset(base_dir, "scitsr", split="val")
    
    pred_row_counts = []
    pred_col_counts = []
    gt_row_counts = []
    gt_col_counts = []
    
    # Sample 100 cases
    print("Analyzing 100 samples...")
    for i in range(min(100, len(dataset))):
        batch = dataset[i]
        img = batch['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model.predict_boundaries(img)
        
        pred_row_bounds = pred["row_boundaries"][0]
        pred_col_bounds = pred["col_boundaries"][0]
        
        pred_row_counts.append(len(pred_row_bounds))
        pred_col_counts.append(len(pred_col_bounds))
        
        # Get GT if available
        gt_cells = batch.get('cells', [])
        if gt_cells:
            gt_rows = set(c.get('row_idx', 0) for c in gt_cells)
            gt_cols = set(c.get('col_idx', 0) for c in gt_cells)
            gt_row_counts.append(len(gt_rows) + 1)  # +1 for boundaries
            gt_col_counts.append(len(gt_cols) + 1)
    
    # Analyze distribution
    print(f"\n{'='*80}")
    print(f"PREDICTION DISTRIBUTION")
    print(f"{'='*80}\n")
    
    print("ROW COUNT DISTRIBUTION:")
    row_counter = Counter(pred_row_counts)
    for count, freq in sorted(row_counter.items()):
        pct = freq / len(pred_row_counts) * 100
        print(f"   {count:>3} rows: {freq:>4} samples ({pct:>5.1f}%)")
    
    print(f"\nCOLUMN COUNT DISTRIBUTION:")
    col_counter = Counter(pred_col_counts)
    for count, freq in sorted(col_counter.items()):
        pct = freq / len(pred_col_counts) * 100
        print(f"   {count:>3} cols: {freq:>4} samples ({pct:>5.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"STATISTICS")
    print(f"{'='*80}\n")
    
    print(f"Predicted Rows:  Mean={np.mean(pred_row_counts):.2f}, Std={np.std(pred_row_counts):.2f}, "
          f"Min={min(pred_row_counts)}, Max={max(pred_row_counts)}")
    print(f"Predicted Cols:  Mean={np.mean(pred_col_counts):.2f}, Std={np.std(pred_col_counts):.2f}, "
          f"Min={min(pred_col_counts)}, Max={max(pred_col_counts)}")
    
    if gt_row_counts:
        print(f"Ground Truth Rows: Mean={np.mean(gt_row_counts):.2f}, Std={np.std(gt_row_counts):.2f}, "
              f"Min={min(gt_row_counts)}, Max={max(gt_row_counts)}")
        print(f"Ground Truth Cols: Mean={np.mean(gt_col_counts):.2f}, Std={np.std(gt_col_counts):.2f}, "
              f"Min={min(gt_col_counts)}, Max={max(gt_col_counts)}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(pred_row_counts, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Predicted Row Count Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Number of Rows')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(pred_row_counts), color='red', linestyle='--', label=f'Mean={np.mean(pred_row_counts):.1f}')
    axes[0, 0].legend()
    
    axes[0, 1].hist(pred_col_counts, bins=20, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Predicted Column Count Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Columns')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(pred_col_counts), color='red', linestyle='--', label=f'Mean={np.mean(pred_col_counts):.1f}')
    axes[0, 1].legend()
    
    if gt_row_counts:
        axes[1, 0].hist(gt_row_counts, bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Ground Truth Row Count Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Number of Rows')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(gt_row_counts), color='red', linestyle='--', label=f'Mean={np.mean(gt_row_counts):.1f}')
        axes[1, 0].legend()
        
        axes[1, 1].hist(gt_col_counts, bins=20, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Ground Truth Column Count Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Number of Columns')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(gt_col_counts), color='red', linestyle='--', label=f'Mean={np.mean(gt_col_counts):.1f}')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/rigorous_analysis/boundary_count_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: outputs/rigorous_analysis/boundary_count_distribution.png")
    
    # Diagnosis
    print(f"\n{'='*80}")
    print(f"DIAGNOSIS")
    print(f"{'='*80}\n")
    
    if np.std(pred_row_counts) < 1.0:
        print("⚠️  WARNING: Predicted row counts have VERY LOW variance!")
        print("   → Model is outputting nearly constant predictions")
        print("   → Possible causes:")
        print("      1. Model is not properly trained")
        print("      2. Threshold in predict_boundaries() is too strict")
        print("      3. Feature extraction is failing")
    
    if np.std(pred_col_counts) < 2.0:
        print("\n⚠️  WARNING: Predicted column counts have LOW variance!")
        print("   → Model struggles to adapt to different table widths")
    
    # Check if model is stuck at a specific value
    most_common_rows = row_counter.most_common(1)[0]
    if most_common_rows[1] / len(pred_row_counts) > 0.5:
        print(f"\n🔴 CRITICAL: {most_common_rows[1]/len(pred_row_counts)*100:.1f}% of predictions have {most_common_rows[0]} rows!")
        print("   → Model has collapsed to a single output")
        print("   → This is a TRAINING FAILURE, not a post-processing issue")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    analyze_prediction_distribution()

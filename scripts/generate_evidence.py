"""
Evidence Generator for Error Atlas

Generates 6-panel visual evidence for error cases:
1. Input Image
2. OCR Bboxes
3. Predicted TSR
4. Ground Truth TSR
5. Difference Overlay
6. Final Output Table (Text)
"""

import cv2
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def generate_6_panel_evidence(table_id: str, output_path: str = 'results/evidence'):
    """Generate a 6-panel mosaic for a specific error case"""
    base_path = Path(f'artifacts/{table_id}')
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load components
    img_input = plt.imread(base_path / 'A_input/input.png')
    img_ocr = plt.imread(base_path / 'B_ocr/ocr_overlay.png')
    img_tsr_pred = plt.imread(base_path / 'D_tsr/pred_overlay.png')
    img_tsr_gt = plt.imread(base_path / 'F_eval/gt_overlay.png')
    img_diff = plt.imread(base_path / 'F_eval/diff_overlay.png')
    
    # Create mosaic
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Error Evidence Snapshot: {table_id}", fontsize=20, weight='bold')
    
    axes[0, 0].imshow(img_input)
    axes[0, 0].set_title("1. Original Image", fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_ocr)
    axes[0, 1].set_title("2. OCR Detections", fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_tsr_pred)
    axes[0, 2].set_title("3. Predicted Structure", fontsize=14)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_tsr_gt)
    axes[1, 0].set_title("4. Ground Truth Structure", fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_diff)
    axes[1, 1].set_title("5. Structure Diff (GT vs Pred)", fontsize=14)
    axes[1, 1].axis('off')
    
    # Panel 6: Quantitative Summary
    with open(base_path / 'F_eval/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    summary_text = f"Table ID: {table_id}\n\n"
    summary_text += f"Structure F1: {metrics.get('f1', 0):.4f}\n"
    summary_text += f"Precision: {metrics.get('precision', 0):.4f}\n"
    summary_text += f"Recall: {metrics.get('recall', 0):.4f}\n\n"
    summary_text += f"GT Grid: {metrics.get('gt_rows', 0)}x{metrics.get('gt_cols', 0)}\n"
    summary_text += f"Pred Grid: {metrics.get('pred_rows', 0)}x{metrics.get('pred_cols', 0)}\n\n"
    summary_text += f"Edge FP: {metrics.get('edge_FP', 0)}\n"
    summary_text += f"Edge FN: {metrics.get('edge_FN', 0)}\n"
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=14, family='monospace', verticalalignment='center')
    axes[1, 2].set_title("6. Quantitative Summary", fontsize=14)
    axes[1, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / f'{table_id}_evidence.png', dpi=150)
    plt.close()
    
    print(f"✓ Evidence generated for {table_id}")

if __name__ == '__main__':
    # Load error DB to find worst cases
    import pandas as pd
    df = pd.read_csv('results/real_error_db.csv')
    
    # Sample different types if available
    worst_cases = df.nsmallest(5, 'struct_f1')['table_id'].tolist()
    
    print(f"Generating evidence for {len(worst_cases)} cases...")
    for tid in worst_cases:
        generate_6_panel_evidence(tid)

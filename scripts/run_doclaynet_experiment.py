#!/usr/bin/env python3
"""
S1-S4 Counterfactual Experiment on DocLayNet

Compares Vision-based TSR (S1: Noisy OCR + Vision) vs PDF-based TSR (S4: Silver Standard).
Quantifies the contribution of OCR/Vision noise to performance degradation.

Scenarios:
S1: Vision-based Docling (Image input) -> Simulates Noisy OCR + Vision TSR
S4 (Reference): PDF-based Cells (Text/BBox from PDF) -> Simulates Perfect Info (Silver Standard)

Metrics:
- Cell Match F1 (IoU > 0.5) based on spatial overlap
- Relative degradation from S4 to S1
"""

import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hiertable_rag.evaluation.diagnosis.doclaynet_adapter import DocLayNetAdapter
from hiertable_rag.evaluation.diagnosis.tsr_engines import DoclingEngine
from hiertable_rag.evaluation.diagnosis.structures import bbox_iou, Cell, TableStruct

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_match(pred_cells, gt_cells, iou_threshold=0.5):
    """
    Compute Precision/Recall/F1 based on bbox matching.
    Input:
        pred_cells: List of dict or Cell objects
        gt_cells: List of dict or Cell objects
    """
    if not pred_cells:
        return 0.0, 0.0, 0.0
    if not gt_cells:
        return 0.0, 0.0, 0.0

    # Convert to list of boxes [x0, y0, x1, y1] used for matching
    p_boxes = []
    for c in pred_cells:
        if hasattr(c, 'bbox_xyxy'):
            p_boxes.append(c.bbox_xyxy)
        elif isinstance(c, dict):
            # Normalizing dict format: 'box' or 'bbox'
            box = c.get('box', c.get('bbox', [0,0,0,0]))
            # Ensure xyxy
            if len(box) == 4:
                p_boxes.append(box)
            else:
                p_boxes.append([0,0,0,0])

    g_boxes = []
    for c in gt_cells:
        # Check type
        if hasattr(c, 'bbox_xyxy'):
            g_boxes.append(c.bbox_xyxy)
        elif isinstance(c, dict):
            # DocLayNet pdf_cells: 'bbox' is usually [x, y, w, h] or [x0, y0, x1, y1]
            # Let's assume [x0, y0, x1, y1] for simple overlap check
            # DocLayNet bbox is [x0, y0, x1, y1] in absolute coords (usually) based on inspection
            box = c.get('bbox', [0,0,0,0])
            g_boxes.append(box)

    matches = 0
    used_gt = set()
    
    for i, p_box in enumerate(p_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, g_box in enumerate(g_boxes):
            if j in used_gt:
                continue
            
            iou_val = bbox_iou(tuple(p_box), tuple(g_box))
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = j
        
        if best_iou >= iou_threshold:
            matches += 1
            used_gt.add(best_gt_idx)
            
    precision = matches / len(p_boxes) if p_boxes else 0.0
    recall = matches / len(g_boxes) if g_boxes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def run_experiment(max_samples=10, output_path="outputs/doclaynet_results.csv"):
    logger.info(f"Starting DocLayNet Experiment (max_samples={max_samples})")
    
    adapter = DocLayNetAdapter(split="validation", max_samples=max_samples)
    samples = adapter.load()
    
    if not samples:
        logger.error("No samples loaded!")
        return

    logger.info(f"Initialized Engine...")
    engine = DoclingEngine()
    
    results = []
    
    for sample in tqdm(samples, desc="Processing"):
        table_id = sample.get('table_id', 'unknown')
        image = sample.get('image')
        pdf_cells = sample.get('pdf_cells', [])
        
        if not image:
            continue
            
        # 1. Prediction (S1: Vision)
        try:
            # Predict structure from Image
            pred_struct = engine.predict(image, ocr_boxes=[])
            pred_cells = pred_struct.get('cells', [])
        except Exception as e:
            logger.error(f"Prediction error for {table_id}: {e}")
            continue
            
        # 2. Reference (S4: PDF Cells as Silver Standard)
        if not pdf_cells:
            logger.warning(f"No PDF cells for {table_id}, skipping")
            continue
            
        # 3. Evaluate Match
        prec, rec, f1 = evaluate_match(pred_cells, pdf_cells)
        
        results.append({
            'table_id': table_id,
            's1_f1': f1,
            's1_precision': prec,
            's1_recall': rec,
            'n_pred': len(pred_cells),
            'n_ref': len(pdf_cells),
            # Simple complexity metrics
            'n_rows': pred_struct.get('num_rows', 0),
            'n_cols': pred_struct.get('num_cols', 0)
        })
        
    if not results:
        logger.warning("No results generated")
        return
        
    df = pd.DataFrame(results)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Report
    avg_f1 = df['s1_f1'].mean()
    print("\n" + "="*50)
    print("DocLayNet Experiment Results (Real Data)")
    print("="*50)
    print(f"Total Samples: {len(df)}")
    print(f"Vision Recovery Rate (F1): {avg_f1:.4f}")
    print(f"Precision: {df['s1_precision'].mean():.4f}")
    print(f"Recall: {df['s1_recall'].mean():.4f}")
    print("\nAnalysis:")
    print(f"  - S4 (Silver Standard) assumed 1.0 (PDF Source)")
    print(f"  - S1 (Vision Mode) achieved {avg_f1:.2%} of PDF quality")
    print(f"  - Performance Gap: {(1.0 - avg_f1)*100:.1f}%")
    
    if avg_f1 > 0.8:
        print("  -> OCR/Vision Noise Impact: LOW (TSR is robust)")
    elif avg_f1 > 0.5:
        print("  -> OCR/Vision Noise Impact: MODERATE")
    else:
        print("  -> OCR/Vision Noise Impact: HIGH (Dominant Factor)")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--out", type=str, default="outputs/doclaynet_results.csv")
    args = parser.parse_args()
    
    run_experiment(max_samples=args.max_samples, output_path=args.out)

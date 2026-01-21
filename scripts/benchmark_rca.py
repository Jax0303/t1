import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from universal_dataset import UniversalTableDataset
import sys
import os
from pathlib import Path
import numpy as np
from typing import List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.hiertable_rag.core.rca_model import RCAModel

def compute_metrics(pred_lines: List[float], gt_lines: List[float], threshold: float = 0.05):
    """
    Computes precision, recall, and F1 for predicted lines compared to GT.
    A predicted line is 'correct' if it's within 'threshold' distance of a GT line.
    """
    if not pred_lines and not gt_lines: return 1.0, 1.0, 1.0
    if not pred_lines or not gt_lines: return 0.0, 0.0, 0.0
    
    tp = 0
    matched_gt = set()
    
    for pl in pred_lines:
        for i, gl in enumerate(gt_lines):
            if i not in matched_gt and abs(pl - gl) < threshold:
                tp += 1
                matched_gt.add(i)
                break
                
    precision = tp / len(pred_lines)
    recall = tp / len(gt_lines)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = "/home/user/t1-4/data/external"
    img_size = 448
    
    model = RCAModel().to(device)
    # Load weights if available
    weights_path = "/home/user/t1-4/models/rca_baseline.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print("Warning: No weights found. Running with random initialization.")
    
    for dtype in ["scitsr", "fintabnet"]:
        print(f"\n--- Benchmarking {dtype} ---")
        dataset = UniversalTableDataset(base_dir, dtype, split="val", img_size=img_size)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        all_row_f1 = []
        all_col_f1 = []
        
        limit = 50
        for i, batch in enumerate(loader):
            images = batch["image"].to(device)
            gt_rows = batch["row_lines"] # This is a list of tensors or padded tensor
            gt_cols = batch["col_lines"]
            
            predictions = model.predict_boundaries(images)
            
            # Since batching with varying number of lines is tricky, 
            # we iterate through the batch
            for b in range(images.shape[0]):
                p_rows = predictions["row_boundaries"][b]
                p_cols = predictions["col_boundaries"][b]
                
                # Get raw GT lines (need to handle padding if DataLoader did it)
                # UniversalTableDataset returns raw tensors, but DataLoader might pad if we use custom collate.
                # Here, we assume batch contains tensors.
                g_rows = gt_rows[b].tolist() if isinstance(gt_rows[b], torch.Tensor) else gt_rows[b]
                g_cols = gt_cols[b].tolist() if isinstance(gt_cols[b], torch.Tensor) else gt_cols[b]
                
                _, _, r_f1 = compute_metrics(p_rows, g_rows)
                _, _, c_f1 = compute_metrics(p_cols, g_cols)
                
                all_row_f1.append(r_f1)
                all_col_f1.append(c_f1)
            
            if i * 4 >= limit: break
            
        print(f"{dtype} Row F1: {np.mean(all_row_f1):.4f}")
        print(f"{dtype} Col F1: {np.mean(all_col_f1):.4f}")

if __name__ == "__main__":
    benchmark()

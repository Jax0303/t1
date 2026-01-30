import argparse
import sys
import os
import torch
import json
import logging
from tqdm import tqdm
from hiertable_rag.gnn_csp import GNNCSPPipeline
from hiertable_rag.evaluation.metrics import compute_teds  # Assuming this exists based on file list
from glob import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(data_dir, pipeline, use_csp=True, evaluation_limit=None):
    """
    Evaluate pipeline on JSON files in data_dir.
    """
    files = glob(os.path.join(data_dir, "*.json"))
    if evaluation_limit:
        files = files[:evaluation_limit]
        
    logger.info(f"Evaluating on {len(files)} files (CSP={use_csp})...")
    
    total_files = 0
    total_valid_structure_match = 0
    total_time = 0
    
    # We might need to load ground truth to compare
    
    for fpath in tqdm(files):
        with open(fpath) as f:
            gt_data = json.load(f)
            
        # Prepare inputs
        # In real scenario we interpret GT boxes as OCR input
        ocr_boxes = []
        for cell in gt_data['cells']:
            ocr_boxes.append({
                'box': cell['box'],
                'content': cell.get('text', ''),
                'id': cell.get('id', None)
            })
            
        # Dummy image (since we are focusing on structure from boxes)
        # Note: In real inference, we need the image for RCA. 
        # Here we assume we are testing the GNN+CSP part given OCR boxes.
        image = torch.zeros((3, 100, 100)) 
        
        result = pipeline.parse(
            image, 
            ocr_boxes, 
            use_csp=use_csp,
            use_semantic=False # Faster for dev, enable if models are present
        )
        
        # Metrics
        # 1. Structure Match (Num rows/cols)
        pred_rows = result['num_rows']
        pred_cols = result['num_cols']
        gt_rows = gt_data['num_rows']
        gt_cols = gt_data['num_cols']
        
        if pred_rows == gt_rows and pred_cols == gt_cols:
            total_valid_structure_match += 1
            
        total_time += result['solve_time']
        total_files += 1
        
    avg_acc = total_valid_structure_match / total_files if total_files > 0 else 0
    avg_time = total_time / total_files if total_files > 0 else 0
    
    return {
        'structure_accuracy': avg_acc,
        'avg_time': avg_time,
        'num_samples': total_files
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Path to processed JSON files")
    parser.add_argument('--limit', type=int, default=50, help="Max samples to evaluate")
    args = parser.parse_args()
    
    # Initialize Pipeline
    pipeline = GNNCSPPipeline(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Experiment 1: Baseline (No CSP) - strictly using GNN heuristics? 
    # Actually current pipeline implementation with use_csp=False might still do some solving or just return raw.
    # Let's assume use_csp=False skips the rigorous solver.
    results_baseline = evaluate(args.data_dir, pipeline, use_csp=False, evaluation_limit=args.limit)
    
    # Experiment 2: GNN-CSP
    results_csp = evaluate(args.data_dir, pipeline, use_csp=True, evaluation_limit=args.limit)
    
    print("\n" + "="*40)
    print("Experiment Results")
    print("="*40)
    print(f"Baseline (Vision/GNN only):")
    print(f"  Structure Accuracy: {results_baseline['structure_accuracy']:.2%}")
    print(f"  Avg Time: {results_baseline['avg_time']:.3f}s")
    
    print(f"\nGNN-CSP (Ours):")
    print(f"  Structure Accuracy: {results_csp['structure_accuracy']:.2%}")
    print(f"  Avg Time: {results_csp['avg_time']:.3f}s")
    print("="*40)
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/pubtables_results.json', 'w') as f:
        json.dump({
            'baseline': results_baseline,
            'gnn_csp': results_csp
        }, f, indent=2)

if __name__ == '__main__':
    main()

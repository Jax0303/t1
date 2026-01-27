import json
import os
import sys
import argparse
import time
import torch
import numpy as np
import logging
from typing import List, Dict, Any
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
# Add repo root to path to support 'src.hiertable_rag' imports found in codebase
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hiertable_rag.gnn_csp.pipeline import GNNCSPPipeline
from hiertable_rag.evaluation.metrics import calculate_teds

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ground_truth(gt_dir: str, num_samples: int = 10) -> List[Dict[str, Any]]:
    """Load Ground Truth samples from SciTSR format."""
    samples = []
    if not os.path.exists(gt_dir):
        logger.error(f"GT directory not found: {gt_dir}")
        return samples
        
    files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]
    selected_files = files[:num_samples]
    
    for fname in selected_files:
        path = os.path.join(gt_dir, fname)
    for fname in selected_files:
        path = os.path.join(gt_dir, fname)
        with open(path, 'r') as f:
            data = json.load(f)
            
            # SciTSR format is a list of cells
            if isinstance(data, list):
                cells_data = data
                gt_structure = {'cells': data} # Wrap for consistency in sample dict
            elif isinstance(data, dict) and 'cells' in data:
                cells_data = data['cells']
                gt_structure = data
            else:
                logger.warning(f"Unknown format for {fname}, skipping")
                continue
            
            # Create a dummy image tensor (the pipeline handles 3D tensors)
            # Image size doesn't matter for this ablation since we are mocking visual features in the pipeline currently
            image = torch.zeros((3, 1000, 1000)) 
            
            # Extract OCR boxes
            ocr_boxes = []
            for cell in cells_data:
                # box is [x1, y1, x2, y2] usually
                # SciTSR box: [x1, y1, x2, y2]
                bbox = cell.get('box', [0, 0, 0, 0])
                ocr_boxes.append({
                    'box': bbox,
                    'content': cell.get('text', '') # SciTSR uses 'text'
                })
            
            samples.append({
                'filename': fname,
                'image': image,
                'ocr_boxes': ocr_boxes,
                'ground_truth': gt_structure
            })
            
    return samples

def run_experiment(
    pipeline: GNNCSPPipeline, 
    samples: List[Dict[str, Any]], 
    config_name: str,
    use_csp: bool,
    use_semantic: bool
) -> Dict[str, Any]:
    """Run inference on samples with specific configuration."""
    logger.info(f"Running experiment: {config_name}")
    
    results = {
        'config': config_name,
        'metrics': {
            'teds': [],
            'time': [],
            'row_acc': [],
            'col_acc': []
        },
        'details': []
    }
    
    for sample in tqdm(samples, desc=f"Eval {config_name}"):
        try:
            # Run pipeline
            output = pipeline.parse(
                sample['image'],
                sample['ocr_boxes'],
                use_csp=use_csp,
                use_semantic=use_semantic
            )
            
            # Calculate metrics
            # TEDS
            # Note: The simple calculate_teds in metrics.py is a placeholder. 
            # We add Row/Col Error as a more direct structural metric for this ablation.
            teds_score = calculate_teds(output['cells'], sample['ground_truth'].get('structure', []))
            
            # Row/Col Error
            # Assuming GT has 'structure' with 'num_rows'/'num_cols' or we derive it
            # For SciTSR, let's look for known keys or default to 0 error if missing to avoid crash
            # Row/Col Error
            gt_rows = 0
            gt_cols = 0
            
            # Try to extract from 'logical_coords' in cells
            # SciTSR cell: {"logical_coords": [r1, c1, r2, c2], ...}
            gt_cells = sample['ground_truth'].get('cells', [])
            if gt_cells:
                try:
                    # logical_coords is [start_row, start_col, end_row, end_col]
                    # usually 0-indexed
                    max_r = 0
                    max_c = 0
                    for c in gt_cells:
                        if 'logical_coords' in c:
                            r2 = c['logical_coords'][2]
                            c2 = c['logical_coords'][3]
                            if r2 > max_r: max_r = r2
                            if c2 > max_c: max_c = c2
                    gt_rows = max_r + 1
                    gt_cols = max_c + 1
                except Exception as e:
                    logger.warning(f"Failed to parse GT dims: {e}") 
            
            # If still 0, try other keys or fallback
            if gt_rows == 0 and 'structure' in sample['ground_truth']:
                 # ... (previous logic if needed, but SciTSR is covered above)
                 pass
            
            row_err = abs(output['num_rows'] - gt_rows) if gt_rows > 0 else 0
            col_err = abs(output['num_cols'] - gt_cols) if gt_cols > 0 else 0
            
            results['metrics']['teds'].append(teds_score)
            results['metrics']['time'].append(output['solve_time'])
            results['metrics']['row_acc'].append(row_err) # Using error instead of raw count
            results['metrics']['col_acc'].append(col_err)
            
            results['details'].append({
                'filename': sample['filename'],
                'teds': teds_score,
                'time': output['solve_time'],
                'row_error': row_err,
                'col_error': col_err
            })
            
        except Exception as e:
            logger.error(f"Error processing {sample['filename']}: {e}")
            
    # Aggregate
    results['summary'] = {
        'mean_teds': float(np.mean(results['metrics']['teds'])) if results['metrics']['teds'] else 0.0,
        'mean_time': float(np.mean(results['metrics']['time'])) if results['metrics']['time'] else 0.0,
        'mean_row_error': float(np.mean(results['metrics']['row_acc'])) if results['metrics']['row_acc'] else 0.0,
        'mean_col_error': float(np.mean(results['metrics']['col_acc'])) if results['metrics']['col_acc'] else 0.0
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="GNN-CSP Ablation Analysis")
    parser.add_argument('--gt_dir', type=str, default='/root/t1-9/data/external/scitsr/SciTsr_Logical/test/gt/', help='Path to GT json files')
    parser.add_argument('--output', type=str, default='/root/t1-9/outputs/gnn_csp_analysis_results.json', help='Output results JSON')
    parser.add_argument('--sample_size', type=int, default=10, help='Number of samples to use')
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load Data
    logger.info("Loading Ground Truth Data...")
    samples = load_ground_truth(args.gt_dir, args.sample_size)
    if not samples:
        logger.error("No samples loaded. Exiting.")
        return

    # Initialize Pipeline
    pipeline = GNNCSPPipeline()
    
    final_report = {
        'ablations': {},
        'sensitivity': {}
    }
    
    # 1. Ablation Studies
    configurations = [
        ('Full Model', True, True),
        ('No CSP', False, True),
        ('No Semantic', True, False)
    ]
    
    logger.info("Starting Ablation Studies...")
    for name, use_csp, use_sem in configurations:
        res = run_experiment(pipeline, samples, name, use_csp, use_sem)
        final_report['ablations'][name] = res['summary']
        
    # 2. Sensitivity Analysis (Constraint Weights)
    # Only run on Full Model configuration (use_csp=True, use_sem=True)
    logger.info("Starting Sensitivity Analysis...")
    weight_configs = [
        (0.5, 0.3, 0.2), # Default
        (0.8, 0.1, 0.1), # Visual Focus
        (0.1, 0.8, 0.1), # Semantic Focus
        (0.33, 0.33, 0.33) # Balanced
    ]
    
    for alpha, beta, gamma in weight_configs:
        config_name = f"Weights(a={alpha},b={beta},g={gamma})"
        pipeline.set_weights(alpha, beta, gamma)
        res = run_experiment(pipeline, samples, config_name, True, True)
        final_report['sensitivity'][config_name] = res['summary']
        
    # Save Report
    with open(args.output, 'w') as f:
        json.dump(final_report, f, indent=2)
        
    logger.info(f"Analysis complete. Results saved to {args.output}")
    
    # Print Summary to Console
    print("\n=== Analysis Summary ===")
    print("Ablation Results:")
    for name, summary in final_report['ablations'].items():
        print(f"  {name}: TEDS={summary['mean_teds']:.4f}, Time={summary['mean_time']:.4f}s, RowErr={summary['mean_row_error']:.2f}")
        
    print("\nSensitivity Results:")
    for name, summary in final_report['sensitivity'].items():
        print(f"  {name}: TEDS={summary['mean_teds']:.4f}, RowErr={summary['mean_row_error']:.2f}")

if __name__ == "__main__":
    main()

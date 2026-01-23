"""
SARTP Benchmark Evaluation Script
Compares baseline RCA vs SARTP on error cases from SciTSR dataset.
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
import matplotlib.pyplot as plt
import seaborn as sns

from src.hiertable_rag.core.rca_model import RCAModel
from src.hiertable_rag.core.grid_restorer import IntersectionRestorer
from src.hiertable_rag.sartp import SARTPPipeline
from scripts.universal_dataset import UniversalTableDataset


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


def calculate_hdma(pred_cells, gt_cells, header_mask):
    """
    Calculate Header-Data Matching Accuracy.
    Measures if data cells are in correct columns based on header semantics.
    """
    if not pred_cells or not gt_cells:
        return 0.0
    
    # For simplicity, check if column assignments match
    correct = 0
    total = 0
    
    for i, (pred, gt) in enumerate(zip(pred_cells, gt_cells)):
        if i < len(header_mask) and not header_mask[i]:  # Only data cells
            if pred.get('col_idx') == gt.get('col_idx'):
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def calculate_scs(cells, embeddings, header_mask):
    """
    Calculate Semantic Coherence Score.
    Measures semantic consistency within columns.
    """
    if embeddings is None or len(cells) == 0:
        return 0.0
    
    # Group by column
    col_groups = defaultdict(list)
    for i, cell in enumerate(cells):
        if i < len(header_mask) and not header_mask[i]:  # Only data cells
            col_idx = cell.get('col_idx', 0)
            col_groups[col_idx].append(i)
    
    coherence_scores = []
    for col_cells in col_groups.values():
        if len(col_cells) < 2:
            continue
        
        # Get embeddings for this column
        col_embs = embeddings[col_cells]
        
        # Compute pairwise cosine similarity
        from torch.nn.functional import normalize
        col_embs_norm = normalize(col_embs, p=2, dim=1)
        similarity = torch.mm(col_embs_norm, col_embs_norm.t())
        
        # Average of upper triangle
        n = len(col_cells)
        triu_indices = torch.triu_indices(n, n, offset=1)
        coherence = similarity[triu_indices[0], triu_indices[1]].mean().item()
        coherence_scores.append(coherence)
    
    return np.mean(coherence_scores) if coherence_scores else 0.0


def calculate_brg(teds_before, teds_after):
    """
    Calculate Boundary Refinement Gain.
    """
    if teds_before >= 1.0:
        return 0.0
    
    return (teds_after - teds_before) / (1.0 - teds_before)


def run_benchmark(
    dataset_name="scitsr",
    split="val",
    num_samples=50,
    output_dir="outputs/sartp_benchmark"
):
    """
    Run SARTP benchmark on error cases.
    
    Args:
        dataset_name: Dataset to use
        split: Data split (train/val/test)
        num_samples: Number of samples to evaluate
        output_dir: Directory to save results
    """
    print(f"\n{'='*70}")
    print(f"SARTP BENCHMARK EVALUATION")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load error cases
    error_cases_path = Path("outputs/error_cases.json")
    if not error_cases_path.exists():
        print(f"Error: {error_cases_path} not found. Run collect_error_cases.py first.")
        return
    
    with open(error_cases_path, 'r') as f:
        error_data = json.load(f)
    
    error_cases = error_data['error_cases'][:num_samples]
    print(f"Loaded {len(error_cases)} error cases\n")
    
    # Initialize models
    print("[1/3] Initializing models...")
    
    # Baseline: RCA only
    rca_model = RCAModel().to(device)
    checkpoint_path = "/root/t1-9/outputs/rca_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"  Loading RCA weights from {checkpoint_path}")
        rca_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("  Warning: No trained RCA model, using random initialization")
    rca_model.eval()
    
    grid_restorer = IntersectionRestorer()
    
    # SARTP: Full pipeline
    sartp_pipeline = SARTPPipeline(
        rca_checkpoint=checkpoint_path if os.path.exists(checkpoint_path) else None,
        semantic_model="roberta-base",
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        device=device
    )
    print("  Models initialized\n")
    
    # Load dataset
    print("[2/3] Loading dataset...")
    base_dir = "/root/t1-9/data/external"
    try:
        dataset = UniversalTableDataset(base_dir, dataset_name, split=split)
        print(f"  Dataset loaded: {len(dataset)} samples\n")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return
    
    # Run evaluation
    print("[3/3] Running evaluation...")
    
    results = {
        'baseline': {
            'teds': [],
            'hdma': [],
            'scs': []
        },
        'sartp': {
            'teds': [],
            'hdma': [],
            'scs': []
        },
        'brg': []
    }
    
    comparison_cases = []
    
    for error_case in tqdm(error_cases, desc="Evaluating"):
        idx = error_case['index']
        
        try:
            batch = dataset[idx]
            img = batch['image'].unsqueeze(0).to(device)
            gt_cells = batch['cells']
            raw_cells = batch.get('raw_cells', [])
            
            # === Baseline: RCA only ===
            with torch.no_grad():
                visual_pred = rca_model.predict_boundaries(img, threshold=0.5)
            
            baseline_cells = grid_restorer.restore_cells(visual_pred, raw_cells)
            baseline_teds = calculate_teds_simple(baseline_cells, gt_cells)
            
            # Mock header mask for baseline (assume first row is header)
            baseline_header_mask = [c.get('row_idx', 0) == 0 for c in baseline_cells]
            baseline_hdma = calculate_hdma(baseline_cells, gt_cells, baseline_header_mask)
            baseline_scs = 0.0  # No semantic info in baseline
            
            results['baseline']['teds'].append(baseline_teds)
            results['baseline']['hdma'].append(baseline_hdma)
            results['baseline']['scs'].append(baseline_scs)
            
            # === SARTP: Full pipeline ===
            sartp_result = sartp_pipeline.parse(
                img, 
                raw_cells, 
                return_intermediate=True
            )
            
            sartp_cells = sartp_result['cells']
            sartp_teds = calculate_teds_simple(sartp_cells, gt_cells)
            
            sartp_header_mask = sartp_result.get('header_mask', [])
            sartp_hdma = calculate_hdma(sartp_cells, gt_cells, sartp_header_mask)
            
            sartp_embeddings = sartp_result.get('embeddings', None)
            sartp_scs = calculate_scs(sartp_cells, sartp_embeddings, sartp_header_mask)
            
            results['sartp']['teds'].append(sartp_teds)
            results['sartp']['hdma'].append(sartp_hdma)
            results['sartp']['scs'].append(sartp_scs)
            
            # BRG
            brg = calculate_brg(baseline_teds, sartp_teds)
            results['brg'].append(brg)
            
            # Track interesting cases (big improvements)
            if sartp_teds - baseline_teds > 0.1:
                comparison_cases.append({
                    'index': idx,
                    'baseline_teds': baseline_teds,
                    'sartp_teds': sartp_teds,
                    'improvement': sartp_teds - baseline_teds,
                    'failure_modes': error_case.get('failure_modes', [])
                })
        
        except Exception as e:
            print(f"\n  Error processing sample {idx}: {e}")
            continue
    
    # Calculate summary statistics
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}\n")
    
    summary = {
        'baseline': {
            'teds_mean': np.mean(results['baseline']['teds']),
            'teds_std': np.std(results['baseline']['teds']),
            'hdma_mean': np.mean(results['baseline']['hdma']),
            'scs_mean': np.mean(results['baseline']['scs'])
        },
        'sartp': {
            'teds_mean': np.mean(results['sartp']['teds']),
            'teds_std': np.std(results['sartp']['teds']),
            'hdma_mean': np.mean(results['sartp']['hdma']),
            'scs_mean': np.mean(results['sartp']['scs'])
        },
        'brg_mean': np.mean(results['brg'])
    }
    
    print("Configuration          TEDS ↑     HDMA ↑     SCS ↑      BRG ↑")
    print("-" * 70)
    print(f"Baseline (RCA only)    {summary['baseline']['teds_mean']:.3f}      "
          f"{summary['baseline']['hdma_mean']:.3f}      "
          f"{summary['baseline']['scs_mean']:.3f}      -")
    print(f"SARTP (Ours)           {summary['sartp']['teds_mean']:.3f}      "
          f"{summary['sartp']['hdma_mean']:.3f}      "
          f"{summary['sartp']['scs_mean']:.3f}      "
          f"{summary['brg_mean']:.3f}")
    print("-" * 70)
    print(f"Improvement            +{summary['sartp']['teds_mean'] - summary['baseline']['teds_mean']:.3f}     "
          f"+{summary['sartp']['hdma_mean'] - summary['baseline']['hdma_mean']:.3f}     "
          f"+{summary['sartp']['scs_mean']:.3f}     -")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': results,
            'comparison_cases': comparison_cases,
            'config': {
                'dataset': dataset_name,
                'split': split,
                'num_samples': len(error_cases),
                'alpha': 0.5,
                'beta': 0.3,
                'gamma': 0.2
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'benchmark_results.json'}")
    
    # Generate visualizations
    generate_plots(results, summary, comparison_cases, output_dir)
    
    print(f"\n✓ Visualizations saved to {output_dir}/")
    print(f"{'='*70}\n")
    
    return summary, results


def generate_plots(results, summary, comparison_cases, output_dir):
    """Generate visualization plots."""
    
    # 1. TEDS comparison boxplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # TEDS
    axes[0].boxplot([results['baseline']['teds'], results['sartp']['teds']], 
                     labels=['Baseline', 'SARTP'])
    axes[0].set_ylabel('TEDS Score')
    axes[0].set_title('TEDS Comparison')
    axes[0].grid(axis='y', alpha=0.3)
    
    # HDMA
    axes[1].boxplot([results['baseline']['hdma'], results['sartp']['hdma']], 
                     labels=['Baseline', 'SARTP'])
    axes[1].set_ylabel('HDMA Score')
    axes[1].set_title('Header-Data Matching Accuracy')
    axes[1].grid(axis='y', alpha=0.3)
    
    # BRG
    axes[2].hist(results['brg'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[2].axvline(summary['brg_mean'], color='red', linestyle='--', 
                     label=f'Mean: {summary["brg_mean"]:.3f}')
    axes[2].set_xlabel('Boundary Refinement Gain')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of BRG')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Improvement scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_teds = results['baseline']['teds']
    sartp_teds = results['sartp']['teds']
    
    ax.scatter(baseline_teds, sartp_teds, alpha=0.6, s=50)
    ax.plot([0, 1], [0, 1], 'r--', label='No improvement line', linewidth=2)
    ax.set_xlabel('Baseline TEDS', fontsize=12)
    ax.set_ylabel('SARTP TEDS', fontsize=12)
    ax.set_title('TEDS: Baseline vs SARTP', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'teds_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Generated metrics_comparison.png")
    print(f"  Generated teds_scatter.png")


if __name__ == "__main__":
    summary, results = run_benchmark(
        dataset_name="scitsr",
        split="val",
        num_samples=50,  # Start with 50 samples
        output_dir="outputs/sartp_benchmark"
    )

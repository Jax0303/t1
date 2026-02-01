"""
Sample Dataset for Error Atlas

Samples and freezes SciTSR dataset for reproducible experiments.
Separates COMP (complex) and normal test samples.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple


def load_comp_list(comp_list_path: str = 'data/SciTSR/SciTSR-COMP.list') -> List[str]:
    """Load list of complex table IDs"""
    with open(comp_list_path, 'r') as f:
        comp_ids = [line.strip() for line in f if line.strip()]
    return comp_ids


def load_test_ids(test_dir: str = 'data/SciTSR/test/structure') -> List[str]:
    """Load all test table IDs"""
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Get all JSON files
    json_files = list(test_path.glob('*.json'))
    table_ids = [f.stem for f in json_files]
    
    return table_ids


def sample_scitsr(n_comp: int = 100, 
                  n_normal: int = 100,
                  seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Sample and freeze dataset
    
    Args:
        n_comp: Number of complex tables to sample
        n_normal: Number of normal tables to sample
        seed: Random seed for reproducibility
    
    Returns:
        (comp_samples, normal_samples)
    """
    random.seed(seed)
    
    # Load data
    comp_list = load_comp_list()
    all_test = load_test_ids()
    
    print(f"Total test tables: {len(all_test)}")
    print(f"Complex (COMP) tables: {len(comp_list)}")
    print(f"Normal tables: {len(all_test) - len(comp_list)}")
    
    # Sample complex tables
    available_comp = [tid for tid in comp_list if tid in all_test]
    n_comp_actual = min(n_comp, len(available_comp))
    comp_samples = random.sample(available_comp, n_comp_actual)
    
    print(f"\nSampling {n_comp_actual} complex tables...")
    
    # Sample normal tables
    normal_candidates = [tid for tid in all_test if tid not in comp_list]
    n_normal_actual = min(n_normal, len(normal_candidates))
    normal_samples = random.sample(normal_candidates, n_normal_actual)
    
    print(f"Sampling {n_normal_actual} normal tables...")
    
    return comp_samples, normal_samples


def save_samples(comp_samples: List[str], 
                 normal_samples: List[str],
                 output_dir: str = 'samples'):
    """
    Save sampled table IDs to files
    
    Args:
        comp_samples: List of complex table IDs
        normal_samples: List of normal table IDs
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save comp samples
    with open(output_path / 'comp_samples.txt', 'w') as f:
        f.write('\n'.join(comp_samples))
    
    # Save normal samples
    with open(output_path / 'normal_samples.txt', 'w') as f:
        f.write('\n'.join(normal_samples))
    
    # Save combined
    all_samples = comp_samples + normal_samples
    with open(output_path / 'all_samples.txt', 'w') as f:
        f.write('\n'.join(all_samples))
    
    # Save metadata
    metadata = {
        'seed': 42,
        'n_comp': len(comp_samples),
        'n_normal': len(normal_samples),
        'total': len(all_samples),
        'comp_samples': comp_samples,
        'normal_samples': normal_samples
    }
    
    with open(output_path / 'sample_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Samples saved to: {output_path}")
    print(f"  - comp_samples.txt: {len(comp_samples)} tables")
    print(f"  - normal_samples.txt: {len(normal_samples)} tables")
    print(f"  - all_samples.txt: {len(all_samples)} tables")
    print(f"  - sample_metadata.json")


def load_samples(sample_file: str = 'samples/all_samples.txt') -> List[str]:
    """Load previously saved samples"""
    with open(sample_file, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]
    return samples


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample SciTSR dataset')
    parser.add_argument('--n_comp', type=int, default=100,
                       help='Number of complex tables')
    parser.add_argument('--n_normal', type=int, default=100,
                       help='Number of normal tables')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("="*50)
    print("SciTSR Dataset Sampling")
    print("="*50)
    
    # Sample
    comp_samples, normal_samples = sample_scitsr(
        n_comp=args.n_comp,
        n_normal=args.n_normal,
        seed=args.seed
    )
    
    # Save
    save_samples(comp_samples, normal_samples)
    
    print("\n" + "="*50)
    print(f"Total sampled: {len(comp_samples) + len(normal_samples)} tables")
    print("="*50)

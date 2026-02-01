"""
Classify All Errors and Populate Error DB

Loads pipeline results, classifies errors, and populates error database.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from classification.error_classifier import ErrorClassifier
from evaluation.relation_evaluator import RelationEvaluator


def load_pipeline_results(results_file: str = 'results/pipeline_results.json') -> list:
    """Load pipeline execution results"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def load_artifact_metrics(table_id: str) -> dict:
    """Load metrics from artifact directory"""
    metrics_path = Path(f'artifacts/{table_id}/F_eval/metrics.json')
    
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def load_sample_metadata() -> dict:
    """Load sample metadata to get split info"""
    with open('samples/sample_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create lookup dict
    split_lookup = {}
    for tid in metadata['comp_samples']:
        split_lookup[tid] = 'comp'
    for tid in metadata['normal_samples']:
        split_lookup[tid] = 'normal'
    
    return split_lookup


def classify_all(results: list, error_db_path: str = 'results/error_db.csv'):
    """
    Classify all results and populate error DB
    
    Args:
        results: List of pipeline results
        error_db_path: Path to error database CSV
    """
    classifier = ErrorClassifier()
    split_lookup = load_sample_metadata()
    
    error_records = []
    
    print("\nClassifying errors...")
    print("="*50)
    
    for result in tqdm(results, desc="Classifying"):
        if result['status'] != 'success':
            # Handle failed runs
            continue
        
        table_id = result['table_id']
        metrics = result.get('metrics', {})
        
        # Load detailed metrics from artifacts
        artifact_metrics = load_artifact_metrics(table_id)
        if artifact_metrics:
            metrics.update(artifact_metrics)
        
        # Extract edge errors
        edge_FP = metrics.get('edge_FP', [])
        edge_FN = metrics.get('edge_FN', [])
        
        # Classify
        classification = classifier.classify(table_id, metrics, edge_FP, edge_FN)
        
        # Build error record
        record = {
            # Identification
            'table_id': table_id,
            'split': split_lookup.get(table_id, 'unknown'),
            
            # Classification
            'primary_type': classification['primary_type'],
            'secondary_types': ','.join(classification.get('secondary_types', [])),
            'stage': classification['stage'],
            'severity': classification['severity'],
            
            # Structure metrics
            'struct_f1': metrics.get('f1', 0.0),
            'struct_precision': metrics.get('precision', 0.0),
            'struct_recall': metrics.get('recall', 0.0),
            'horizontal_f1': metrics.get('horizontal_f1', 0.0),
            'vertical_f1': metrics.get('vertical_f1', 0.0),
            'edge_FP': len(edge_FP),
            'edge_FN': len(edge_FN),
            
            # Content metrics (placeholder)
            'cer': metrics.get('cer', 0.0),
            'wer': metrics.get('wer', 0.0),
            'cer_numeric': 0.0,
            'cer_alpha': 0.0,
            
            # Grid stats
            'num_pred_cells': metrics.get('num_pred_cells', 0),
            'num_gt_cells': metrics.get('num_gt_cells', 0),
            'pred_rows': metrics.get('pred_rows', 0),
            'pred_cols': metrics.get('pred_cols', 0),
            'gt_rows': metrics.get('gt_rows', 0),
            'gt_cols': metrics.get('gt_cols', 0),
            
            # Attribution (placeholder - to be filled by counterfactual)
            'ocr_impact': 0.0,
            'tsr_impact': 0.0,
            'ocr_attribution': 0.0,
            'tsr_attribution': 0.0,
            
            # Labels
            'auto_label': classification['primary_type'],
            'manual_label': '',
            'verified': False,
            
            # Notes
            'notes': '',
            'timestamp': datetime.now().isoformat()
        }
        
        error_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(error_records)
    
    # Save to CSV
    df.to_csv(error_db_path, index=False)
    
    print("\n" + "="*50)
    print(f"Error DB populated!")
    print(f"  Total records: {len(df)}")
    print(f"  Saved to: {error_db_path}")
    print("="*50)
    
    # Print distribution
    print("\nError Type Distribution:")
    print(df['primary_type'].value_counts())
    
    print("\nProcessing Stage Distribution:")
    print(df['stage'].value_counts())
    
    print(f"\nAverage Severity: {df['severity'].mean():.4f}")
    print(f"Average Structure F1: {df['struct_f1'].mean():.4f}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify errors and populate error DB')
    parser.add_argument('--results', type=str, default='results/pipeline_results.json',
                       help='Pipeline results file')
    parser.add_argument('--output', type=str, default='results/error_db.csv',
                       help='Error DB output path')
    
    args = parser.parse_args()
    
    # Load results
    results = load_pipeline_results(args.results)
    
    # Classify
    df = classify_all(results, args.output)
    
    # Save summary stats
    summary = {
        'total_tables': len(df),
        'error_types': df['primary_type'].value_counts().to_dict(),
        'stages': df['stage'].value_counts().to_dict(),
        'avg_severity': float(df['severity'].mean()),
        'avg_struct_f1': float(df['struct_f1'].mean()),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('results/classification_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: results/classification_summary.json")

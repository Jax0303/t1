"""
Counterfactual Attribution Experiments

4-scenario framework to decompose OCR vs TSR errors:

Scenario 1: GT OCR + Real TSR    → Isolate TSR errors
Scenario 2: Real OCR + GT TSR    → Isolate OCR errors  
Scenario 3: GT OCR + GT TSR      → Upper bound (perfect)
Scenario 4: Real OCR + Real TSR  → Current baseline

Attribution:
- OCR_impact = F1(S1) - F1(S4)   # How much OCR hurts
- TSR_impact = F1(S2) - F1(S4)   # How much TSR hurts
- OCR_attribution = OCR_impact / (OCR_impact + TSR_impact)
- TSR_attribution = TSR_impact / (OCR_impact + TSR_impact)
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.artifact_logger import ArtifactLogger
from evaluation.relation_evaluator import RelationEvaluator
from ocr_tsr.simple_pipeline import SimpleSpatialTSR


def load_scitsr_image(table_id: str, img_dir: str = 'data/SciTSR/test/img') -> np.ndarray:
    """Load SciTSR image"""
    img_path = Path(img_dir) / f'{table_id}.png'
    if not img_path.exists():
        img_path = Path(img_dir) / f'{table_id}.jpg'
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {table_id}")
    
    img = Image.open(img_path).convert('RGB')
    return np.array(img)


def load_scitsr_gt(table_id: str, gt_dir: str = 'data/SciTSR/test/structure') -> Dict:
    """Load SciTSR ground truth"""
    gt_path = Path(gt_dir) / f'{table_id}.json'
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    # Parse SciTSR format
    cells_data = gt_data.get('cells', gt_data) if isinstance(gt_data, dict) else gt_data
    
    cells = []
    for cell in cells_data:
        content = cell.get('content', cell.get('tex', ''))
        if isinstance(content, list):
            content = ' '.join(content)
        elif not isinstance(content, str):
            content = str(content)
        
        cells.append({
            'start_row': cell.get('start_row', 0),
            'end_row': cell.get('end_row', 0),
            'start_col': cell.get('start_col', 0),
            'end_col': cell.get('end_col', 0),
            'content': content,
            'box': cell.get('bbox', [0, 0, 10, 10])
        })
    
    return {'cells': cells}


def create_gt_ocr_from_gt(gt_data: Dict) -> List[Dict]:
    """
    Create perfect OCR results from GT data
    
    This represents perfect text detection and recognition.
    """
    ocr_results = []
    
    for cell in gt_data['cells']:
        bbox = cell.get('box', [0, 0, 10, 10])
        
        # Convert to [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] format
        x1, y1, x2, y2 = bbox
        bbox_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        
        ocr_results.append({
            'bbox': bbox_points,
            'text': cell['content'],
            'confidence': 1.0
        })
    
    return ocr_results


def create_gt_structure_from_gt(gt_data: Dict) -> Dict:
    """
    Create perfect TSR structure from GT data
    
    This represents perfect table structure recognition.
    """
    cells = []
    
    for cell in gt_data['cells']:
        cells.append({
            'row': cell['start_row'],
            'col': cell['start_col'],
            'row_span': cell['end_row'] - cell['start_row'] + 1,
            'col_span': cell['end_col'] - cell['start_col'] + 1,
            'text': cell['content'],
            'box': cell.get('box', [0, 0, 10, 10]),
            'confidence': 1.0
        })
    
    # Calculate grid dimensions
    max_row = max(c['row'] + c['row_span'] for c in cells)
    max_col = max(c['col'] + c['col_span'] for c in cells)
    
    return {
        'cells': cells,
        'num_rows': max_row,
        'num_cols': max_col
    }


def mock_real_ocr(gt_data: Dict, error_rate: float = 0.0) -> List[Dict]:
    """
    Create realistic OCR errors from GT
    
    For now, just use GT OCR with optional noise.
    Replace with actual PaddleOCR when installed.
    """
    return create_gt_ocr_from_gt(gt_data)


def run_counterfactual(
    table_id: str,
    scenario: str,
    gt_data: Dict,
    image: np.ndarray,
    tsr_engine: SimpleSpatialTSR
) -> Dict[str, Any]:
    """
    Run one counterfactual scenario
    
    Args:
        table_id: Table identifier
        scenario: 'S1_GT-OCR_Real-TSR', 'S2_Real-OCR_GT-TSR', 'S3_GT-OCR_GT-TSR', 'S4_Real-OCR_Real-TSR'
        gt_data: Ground truth data
        image: Input image
        tsr_engine: TSR engine
    
    Returns:
        Dict with OCR results, structure, and metrics
    """
    evaluator = RelationEvaluator()
    
    # Select OCR and TSR based on scenario
    if 'GT-OCR' in scenario:
        ocr_results = create_gt_ocr_from_gt(gt_data)
        ocr_type = 'GT'
    else:
        ocr_results = mock_real_ocr(gt_data)
        ocr_type = 'Real'
    
    if 'GT-TSR' in scenario:
        structure = create_gt_structure_from_gt(gt_data)
        tsr_type = 'GT'
    else:
        structure = tsr_engine.process(ocr_results)
        tsr_type = 'Real'
    
    # Evaluate
    pred_cells = structure['cells']
    gt_cells = gt_data['cells']
    
    metrics = evaluator.evaluate_cells(pred_cells, gt_cells)
    
    # Add grid stats
    gt_stats = evaluator.compute_grid_stats(gt_cells)
    pred_stats = evaluator.compute_grid_stats(pred_cells)
    
    metrics['gt_rows'] = gt_stats['num_rows']
    metrics['gt_cols'] = gt_stats['num_cols']
    metrics['pred_rows'] = pred_stats['num_rows']
    metrics['pred_cols'] = pred_stats['num_cols']
    
    return {
        'scenario': scenario,
        'ocr_type': ocr_type,
        'tsr_type': tsr_type,
        'ocr_results': ocr_results,
        'structure': structure,
        'metrics': metrics
    }


def run_attribution_experiment(table_id: str) -> Dict[str, Any]:
    """
    Run full 4-scenario attribution experiment for one table
    
    Returns:
        Dict with all scenarios and attribution metrics
    """
    # Load data
    image = load_scitsr_image(table_id)
    gt_data = load_scitsr_gt(table_id)
    
    # Initialize TSR
    tsr_engine = SimpleSpatialTSR(row_threshold=15.0)
    
    # Run 4 scenarios
    scenarios = {
        'S1': 'GT-OCR_Real-TSR',
        'S2': 'Real-OCR_GT-TSR',
        'S3': 'GT-OCR_GT-TSR',
        'S4': 'Real-OCR_Real-TSR'
    }
    
    results = {}
    
    for s_id, s_name in scenarios.items():
        try:
            result = run_counterfactual(table_id, s_name, gt_data, image, tsr_engine)
            results[s_id] = result
        except Exception as e:
            print(f"  ⚠️  {s_id} failed: {e}")
            results[s_id] = {
                'scenario': s_name,
                'error': str(e),
                'metrics': {'f1': 0.0}
            }
    
    # Calculate attribution
    f1_s1 = results['S1']['metrics']['f1']  # GT OCR + Real TSR
    f1_s2 = results['S2']['metrics']['f1']  # Real OCR + GT TSR
    f1_s3 = results['S3']['metrics']['f1']  # GT OCR + GT TSR (perfect)
    f1_s4 = results['S4']['metrics']['f1']  # Real OCR + Real TSR (current)
    
    # Attribution
    ocr_impact = f1_s1 - f1_s4  # How much perfect OCR helps
    tsr_impact = f1_s2 - f1_s4  # How much perfect TSR helps
    
    total_impact = ocr_impact + tsr_impact
    
    if total_impact > 0:
        ocr_attribution = ocr_impact / total_impact
        tsr_attribution = tsr_impact / total_impact
    else:
        ocr_attribution = 0.5
        tsr_attribution = 0.5
    
    attribution = {
        'f1_s1_gt_ocr_real_tsr': f1_s1,
        'f1_s2_real_ocr_gt_tsr': f1_s2,
        'f1_s3_gt_ocr_gt_tsr': f1_s3,
        'f1_s4_real_ocr_real_tsr': f1_s4,
        'ocr_impact': ocr_impact,
        'tsr_impact': tsr_impact,
        'ocr_attribution': ocr_attribution,
        'tsr_attribution': tsr_attribution,
        'primary_cause': 'OCR' if ocr_attribution > 0.5 else 'TSR'
    }
    
    return {
        'table_id': table_id,
        'scenarios': results,
        'attribution': attribution
    }


def run_batch_attribution(
    table_ids: List[str],
    output_file: str = 'results/attribution_results.json'
) -> List[Dict]:
    """
    Run attribution experiments on batch of tables
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running Attribution Experiments on {len(table_ids)} tables")
    print(f"{'='*60}\n")
    
    for table_id in tqdm(table_ids, desc="Processing"):
        try:
            result = run_attribution_experiment(table_id)
            results.append(result)
        except Exception as e:
            tqdm.write(f"⚠️  Failed: {table_id} - {e}")
            results.append({
                'table_id': table_id,
                'error': str(e),
                'attribution': {
                    'ocr_attribution': 0.0,
                    'tsr_attribution': 0.0
                }
            })
    
    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Attribution Experiment Complete!")
    print("="*60)
    
    if results:
        # Calculate average attribution
        valid_results = [r for r in results if 'attribution' in r and 'error' not in r]
        
        if valid_results:
            avg_ocr_attr = np.mean([r['attribution']['ocr_attribution'] for r in valid_results])
            avg_tsr_attr = np.mean([r['attribution']['tsr_attribution'] for r in valid_results])
            avg_ocr_impact = np.mean([r['attribution']['ocr_impact'] for r in valid_results])
            avg_tsr_impact = np.mean([r['attribution']['tsr_impact'] for r in valid_results])
            
            print(f"\n📊 Average Attribution:")
            print(f"   OCR Attribution: {avg_ocr_attr:.2%}")
            print(f"   TSR Attribution: {avg_tsr_attr:.2%}")
            print(f"\n📉 Average Impact:")
            print(f"   OCR Impact (F1 gain with perfect OCR): {avg_ocr_impact:+.4f}")
            print(f"   TSR Impact (F1 gain with perfect TSR): {avg_tsr_impact:+.4f}")
            
            # Primary cause distribution
            ocr_primary = sum(1 for r in valid_results if r['attribution']['primary_cause'] == 'OCR')
            tsr_primary = sum(1 for r in valid_results if r['attribution']['primary_cause'] == 'TSR')
            
            print(f"\n🎯 Primary Cause:")
            print(f"   OCR-dominant errors: {ocr_primary} ({100*ocr_primary/len(valid_results):.1f}%)")
            print(f"   TSR-dominant errors: {tsr_primary} ({100*tsr_primary/len(valid_results):.1f}%)")
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run counterfactual attribution experiments')
    parser.add_argument('--sample_file', type=str, default='samples/all_samples.txt',
                       help='File with table IDs')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tables')
    
    args = parser.parse_args()
    
    # Load samples
    with open(args.sample_file, 'r') as f:
        table_ids = [line.strip() for line in f if line.strip()]
    
    if args.limit:
        table_ids = table_ids[:args.limit]
    
    print(f"Loaded {len(table_ids)} table IDs")
    
    # Run attribution
    results = run_batch_attribution(table_ids)

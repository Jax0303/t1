"""
Enhanced Attribution Experiment with Real PaddleOCR + SciTSR Chunks

2-Scenario Counterfactual:
- S1: Real PaddleOCR + SimpleSpatialTSR  (Current baseline)
- S2: SciTSR Chunk + SimpleSpatialTSR     (Perfect OCR)

Attribution:
- OCR_impact = F1(S2) - F1(S1)
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ocr_tsr.enhanced_pipeline import EnhancedOCRTSRPipeline
from evaluation.relation_evaluator import RelationEvaluator
from pipeline.artifact_logger import ArtifactLogger


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


def run_enhanced_attribution(table_id: str) -> Dict[str, Any]:
    """
    Run 2-scenario attribution for one table
    
    Returns:
        {
          'table_id': str,
          'scenarios': {
            'S1_Real': {...},
            'S2_Chunk': {...}
          },
          'attribution': {
            'f1_s1_real': float,
            'f1_s2_chunk': float,
            'ocr_impact': float,
            'ocr_attribution': float
          }
        }
    """
    # Load data
    image = load_scitsr_image(table_id)
    gt_data = load_scitsr_gt(table_id)
    img_path = Path('data/SciTSR/test/img') / f'{table_id}.png'
    if not img_path.exists():
        img_path = Path('data/SciTSR/test/img') / f'{table_id}.jpg'
    
   # Initialize pipelines
    pipeline_real = EnhancedOCRTSRPipeline(ocr_mode='paddleocr', use_gpu=False)
    pipeline_chunk = EnhancedOCRTSRPipeline(ocr_mode='chunk')
    
    evaluator = RelationEvaluator()
    
    results = {}
    
    # S1: Real PaddleOCR
    try:
        ocr_real, tsr_real = pipeline_real.process(str(img_path), table_id)
        metrics_real = evaluator.evaluate_cells(tsr_real['cells'], gt_data['cells'])
        results['S1_Real'] = {
            'ocr_mode': 'paddleocr',
            'ocr_tokens': len(ocr_real),
            'tsr_cells': len(tsr_real['cells']),
            'metrics': metrics_real
        }
    except Exception as e:
        print(f"  S1 failed: {e}")
        results['S1_Real'] = {'error': str(e), 'metrics': {'f1': 0.0}}
    
    # S2: SciTSR Chunk (Perfect OCR)
    try:
        ocr_chunk, tsr_chunk = pipeline_chunk.process(str(img_path), table_id)
        metrics_chunk = evaluator.evaluate_cells(tsr_chunk['cells'], gt_data['cells'])
        results['S2_Chunk'] = {
            'ocr_mode': 'chunk',
            'ocr_tokens': len(ocr_chunk),
            'tsr_cells': len(tsr_chunk['cells']),
            'metrics': metrics_chunk
        }
    except Exception as e:
        print(f"  S2 failed: {e}")
        results['S2_Chunk'] = {'error': str(e), 'metrics': {'f1': 0.0}}
    
    # Calculate attribution
    f1_s1 = results['S1_Real']['metrics']['f1']
    f1_s2 = results['S2_Chunk']['metrics']['f1']
    
    ocr_impact = f1_s2 - f1_s1  # How much perfect OCR helps
    
    attribution = {
        'f1_s1_real': f1_s1,
        'f1_s2_chunk': f1_s2,
        'ocr_impact': ocr_impact,
        'ocr_attribution': 1.0 if ocr_impact > 0 else 0.0  # Binary for now
    }
    
    return {
        'table_id': table_id,
        'scenarios': results,
        'attribution': attribution
    }


def run_batch_enhanced_attribution(
    table_ids: List[str],
    output_file: str = 'results/enhanced_attribution.json'
) -> List[Dict]:
    """Run enhanced attribution on batch"""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running Enhanced Attribution (PaddleOCR + Chunks)")
    print(f"Tables: {len(table_ids)}")
    print(f"{'='*60}\n")
    
    for table_id in tqdm(table_ids, desc="Processing"):
        try:
            result = run_enhanced_attribution(table_id)
            results.append(result)
        except Exception as e:
            tqdm.write(f"⚠️  Failed: {table_id} - {e}")
            results.append({
                'table_id': table_id,
                'error': str(e)
            })
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    valid = [r for r in results if 'error' not in r]
    
    if valid:
        avg_f1_real = np.mean([r['attribution']['f1_s1_real'] for r in valid])
        avg_f1_chunk = np.mean([r['attribution']['f1_s2_chunk'] for r in valid])
        avg_ocr_impact = np.mean([r['attribution']['ocr_impact'] for r in valid])
        
        print("\n" + "="*60)
        print("Enhanced Attribution Complete!")
        print("="*60)
        print(f"\n📊 Average F1 Scores:")
        print(f"   S1 (Real PaddleOCR): {avg_f1_real:.4f}")
        print(f"   S2 (SciTSR Chunk):   {avg_f1_chunk:.4f}")
        print(f"\n📉 OCR Impact:")
        print(f"   Average OCR Impact: {avg_ocr_impact:+.4f}")
        
        if avg_ocr_impact > 0.1:
            print(f"\n✅ OCR is a significant contributor to errors!")
        elif avg_ocr_impact < -0.1:
            print(f"\n⚠️  Real OCR performs BETTER than chunks (unexpected)")
        else:
            print(f"\n✅ OCR is NOT the main problem - TSR is the bottleneck")
        
        print(f"\n✓ Results saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced attribution')
    parser.add_argument('--sample_file', type=str, default='samples/all_samples.txt')
    parser.add_argument('--limit', type=int, default=None)
    
    args = parser.parse_args()
    
    with open(args.sample_file, 'r') as f:
        table_ids = [line.strip() for line in f if line.strip()]
    
    if args.limit:
        table_ids = table_ids[:args.limit]
    
    print(f"Loaded {len(table_ids)} table IDs")
    
    results = run_batch_enhanced_attribution(table_ids)

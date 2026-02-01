"""
Updated Pipeline Runner with Real OCR+TSR

Uses SimpleOCRTSRPipeline (PaddleOCR + Spatial TSR) for actual error analysis.
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.artifact_logger import ArtifactLogger
from evaluation.relation_evaluator import RelationEvaluator
from ocr_tsr.simple_pipeline import SimpleOCRTSRPipeline


class ProcessCheckpoint:
    """Checkpoint logging for process attribution"""
    
    STAGES = [
        'OCR_Detection',
        'OCR_Recognition',
        'OCR_to_TSR_Transform',
        'TSR_Grid_Detection',
        'TSR_Spanning',
        'Post_Processing',
        'Cell_Assignment'
    ]
    
    def __init__(self, table_id: str):
        self.table_id = table_id
        self.checkpoints = {}
    
    def log(self, stage: str, status: str, details: Dict):
        """Log checkpoint"""
        self.checkpoints[stage] = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
    
    def save(self, output_path: Path):
        """Save checkpoints to file"""
        with open(output_path, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)


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


def run_one_real(table_id: str, 
                 split: str = 'test',
                 pipeline: Optional[SimpleOCRTSRPipeline] = None) -> Dict[str, Any]:
    """
    Run complete pipeline for one table with REAL OCR+TSR
    
    Args:
        table_id: Table identifier
        split: Dataset split ('test' or 'comp')
        pipeline: OCR+TSR pipeline instance
    
    Returns:
        Dict with status and metrics
    """
    if pipeline is None:
        pipeline = SimpleOCRTSRPipeline()
    
    logger = ArtifactLogger(table_id)
    checkpoint = ProcessCheckpoint(table_id)
    evaluator = RelationEvaluator()
    
    try:
        # A. Load input
        image = load_scitsr_image(table_id)
        gt_data = load_scitsr_gt(table_id)
        img_path = Path('data/SciTSR/test/img') / f'{table_id}.png'
        if not img_path.exists():
            img_path = Path('data/SciTSR/test/img') / f'{table_id}.jpg'
        
        metadata = {
            'table_id': table_id,
            'split': split,
            'image_size': image.shape[:2],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.save_input(image, metadata)
        checkpoint.log('Input', 'pass', {'image_loaded': True})
        
        # B. Run REAL OCR+TSR Pipeline
        print(f"  Processing {table_id}...")
        ocr_results, structure = pipeline.process(str(img_path))
        
        # Log OCR results
        ocr_tokens = []
        for det in ocr_results:
            bbox = det['bbox']
            # Normalize bbox
            if isinstance(bbox[0], (list, tuple)):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            
            ocr_tokens.append({
                'text': det['text'],
                'bbox': bbox,
                'confidence': det.get('confidence', 1.0),
                'line_id': -1,
                'block_id': -1
            })
        
        logger.save_ocr(ocr_tokens, image, draw_overlay=True)
        checkpoint.log('OCR_Recognition', 'pass', {
            'num_tokens': len(ocr_tokens),
            'avg_confidence': np.mean([t['confidence'] for t in ocr_tokens]) if ocr_tokens else 0.0
        })
        
        # C. TSR Input
        tsr_input = {'ocr_results': ocr_results}
        logger.save_tsr_input(tsr_input)
        checkpoint.log('OCR_to_TSR_Transform', 'pass', {
            'input_tokens': len(ocr_results)
        })
        
        # D. TSR Output
        tsr_log = {
            'num_cells': len(structure['cells']),
            'num_rows': structure.get('num_rows', 0),
            'num_cols': structure.get('num_cols', 0),
            'algorithm': 'Spatial Sorting'
        }
        
        logger.save_tsr(structure, image, log=tsr_log, draw_overlay=True)
        checkpoint.log('TSR_Grid_Detection', 'pass', {
            'detected_cells': len(structure['cells']),
            'detected_rows': structure.get('num_rows', 0),
            'detected_cols': structure.get('num_cols', 0)
        })
        
        # E. Result
        logger.save_result(structure, postprocess_log=None)
        
        # F. Evaluate
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
        
        # Add content metrics (CER placeholder)
        metrics['cer'] = 0.0  # TODO: Implement proper CER calculation
        metrics['wer'] = 0.0
        
        logger.save_eval(metrics, gt_structure=gt_data, pred_structure=structure, image=image)
        
        # Save checkpoint
        checkpoint.save(logger.dirs['eval'] / 'checkpoint.json')
        
        return {
            'status': 'success',
            'table_id': table_id,
            'metrics': metrics
        }
        
    except Exception as e:
        # Log failure
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        
        fail_path = logger.artifact_dir / 'fail_reason.json'
        with open(fail_path, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        print(f"  ⚠️  Error: {str(e)}")
        
        return {
            'status': 'failed',
            'table_id': table_id,
            'error': str(e)
        }


def run_batch_real(table_ids: List[str], 
                   output_file: str = 'results/real_pipeline_results.json'):
    """
    Run REAL OCR+TSR pipeline on batch of tables
    
    Args:
        table_ids: List of table IDs
        output_file: Path to save results
    """
    # Initialize pipeline once (for efficiency)
    pipeline = SimpleOCRTSRPipeline(use_gpu=False, lang='en')
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running REAL OCR+TSR Pipeline on {len(table_ids)} tables")
    print(f"{'='*60}")
    print(f"OCR: PaddleOCR (English)")
    print(f"TSR: Spatial Sorting\n")
    
    for table_id in tqdm(table_ids, desc="Processing"):
        result = run_one_real(table_id, pipeline=pipeline)
        results.append(result)
        
        if result['status'] == 'failed':
            tqdm.write(f"⚠️  Failed: {table_id} - {result['error']}")
    
    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successes = sum(1 for r in results if r['status'] == 'success')
    failures = len(results) - successes
    
    print("\n" + "="*60)
    print(f"Pipeline Complete!")
    print(f"  Success: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")
    print(f"  Failed: {failures}/{len(results)} ({100*failures/len(results) if len(results) > 0 else 0:.1f}%)")
    print(f"  Results saved to: {output_file}")
    print("="*60)
    
    if successes > 0:
        # Calculate average metrics
        avg_f1 = np.mean([r['metrics']['f1'] for r in results if r['status'] == 'success'])
        print(f"\n📊 Average Structure F1: {avg_f1:.4f}")
        
        # Show distribution
        high_f1 = sum(1 for r in results if r['status'] == 'success' and r['metrics']['f1'] > 0.8)
        med_f1 = sum(1 for r in results if r['status'] == 'success' and 0.5 < r['metrics']['f1'] <= 0.8)
        low_f1 = sum(1 for r in results if r['status'] == 'success' and r['metrics']['f1'] <= 0.5)
        
        print(f"   High (>0.8): {high_f1}")
        print(f"   Med (0.5-0.8): {med_f1}")
        print(f"   Low (<=0.5): {low_f1}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run REAL OCR+TSR pipeline')
    parser.add_argument('--sample_file', type=str, default='samples/all_samples.txt',
                       help='File with table IDs to process')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tables to process')
    
    args = parser.parse_args()
    
    # Load samples
    with open(args.sample_file, 'r') as f:
        table_ids = [line.strip() for line in f if line.strip()]
    
    if args.limit:
        table_ids = table_ids[:args.limit]
    
    print(f"\nLoaded {len(table_ids)} table IDs from {args.sample_file}")
    
    # Run
    results = run_batch_real(table_ids)

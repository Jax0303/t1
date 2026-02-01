"""
Pipeline Runner with Artifact Logging

Runs complete OCR+TSR pipeline with comprehensive artifact logging
for each table in the sample set.
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
from ocr_tsr.table_transformer import TableTransformerTSR
from utils.ocr_normalizer import RobustNormalizer


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


def mock_ocr(image: np.ndarray, gt_data: Dict) -> List[Dict]:
    """
    Mock OCR using GT data (for testing)
    In real implementation, replace with actual OCR
    """
    tokens = []
    
    for i, cell in enumerate(gt_data['cells']):
        bbox = cell.get('box', [0, 0, 10, 10])
        
        tokens.append({
            'text': cell['content'],
            'bbox': bbox,
            'confidence': 1.0,
            'line_id': cell['start_row'],
            'block_id': i
        })
    
    return tokens


def mock_tsr(ocr_tokens: List[Dict], gt_data: Dict) -> Dict:
    """
    Mock TSR using GT data (for testing)
    In real implementation, replace with actual TSR
    """
    # Convert GT to TSR format  
    cells = []
    
    for cell in gt_data['cells']:
        cells.append({
            'row': cell['start_row'],
            'col': cell['start_col'],
            'row_span': cell['end_row'] - cell['start_row'] + 1,
            'col_span': cell['end_col'] - cell['start_col'] + 1,
            'text': cell['content'],
            'box': cell.get('box', [0, 0, 10, 10])
        })
    
    return {'cells': cells}


def run_one(table_id: str, 
            split: str = 'test',
            use_mock: bool = True,
            **kwargs) -> Dict[str, Any]:
    """
    Run complete pipeline for one table with artifact logging
    
    Args:
        table_id: Table identifier
        split: Dataset split ('test' or 'comp')
        use_mock: Use mock OCR/TSR (for testing)
    
    Returns:
        Dict with status and metrics
    """
    logger = ArtifactLogger(table_id)
    checkpoint = ProcessCheckpoint(table_id)
    evaluator = RelationEvaluator()
    
    try:
        # A. Load input
        image = load_scitsr_image(table_id)
        gt_data = load_scitsr_gt(table_id)
        
        metadata = {
            'table_id': table_id,
            'split': split,
            'image_size': image.shape[:2],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.save_input(image, metadata)
        checkpoint.log('Input', 'pass', {'image_loaded': True})
        
        # B. OCR
        if use_mock:
            ocr_tokens = mock_ocr(image, gt_data)
        elif kwargs.get('mode') == 's2':
            # S2: Perfect OCR (Chunks) + Normalization
            chunk_path = Path('data/SciTSR/test/chunk') / f'{table_id}.chunk'
            if not chunk_path.exists():
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
            
            with open(chunk_path, 'r') as f:
                chunks = json.load(f)['chunks']
                
            norm = RobustNormalizer()
            norm.learn_scale_offset(chunks, image.shape[:2][::-1]) # image.shape is (H, W), we need (W, H)
            
            ocr_tokens = []
            for c in chunks:
                box = norm.transform(c['pos'])
                ocr_tokens.append({
                    'text': c['text'],
                    'bbox': box,
                    'confidence': 1.0, # Ground truth
                    'original_pos': c['pos']
                })
        else:
            # TODO: Replace with real OCR
            ocr_tokens = mock_ocr(image, gt_data)
        
        logger.save_ocr(ocr_tokens, image, draw_overlay=True)
        checkpoint.log('OCR_Recognition', 'pass', {
            'num_tokens': len(ocr_tokens),
            'avg_confidence': np.mean([t['confidence'] for t in ocr_tokens])
        })
        
        # C. TSR Input (OCR -> TSR transformation)
        tsr_input = {'tokens': ocr_tokens}  # Simplified
        logger.save_tsr_input(tsr_input)
        checkpoint.log('OCR_to_TSR_Transform', 'pass', {
            'input_tokens': len(ocr_tokens)
        })
        
        # D. TSR
        if use_mock:
            structure = mock_tsr(ocr_tokens, gt_data)
        elif kwargs.get('mode') == 's2':
            # S2: TableTransformer TSR + Alignment
            model = kwargs.get('tsr_model')
            if model is None:
                raise ValueError("TSR Model not provided for S2 mode")
                
            # 1. Detect Structure
            structure_res = model.process_structure(Image.fromarray(image))
            
            # 2. Align OCR to Structure
            structure = model.align_ocr_to_structure(ocr_tokens, structure_res)
        else:
            # TODO: Replace with real TSR
            structure = mock_tsr(ocr_tokens, gt_data)
        
        tsr_log = {
            'num_cells': len(structure['cells']),
            'processing_time': 0.1
        }
        
        logger.save_tsr(structure, image, log=tsr_log, draw_overlay=True)
        checkpoint.log('TSR_Grid_Detection', 'pass', {
            'detected_cells': len(structure['cells'])
        })
        
        # E. Result (no postprocessing for now)
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
        
        return {
            'status': 'failed',
            'table_id': table_id,
            'error': str(e)
        }


def run_batch(table_ids: List[str], 
              output_file: str = 'results/pipeline_results.json',
              use_mock: bool = True,
              mode: str = 'mock'):
    """
    Run pipeline on batch of tables
    """
    results = []
    
    # Initialize shared models if needed
    tsr_model = None
    if mode == 's2':
        print("Loading TableTransformerTSR...")
        tsr_model = TableTransformerTSR()
    
    print(f"\nRunning pipeline on {len(table_ids)} tables... Mode: {mode}")
    print("="*50)
    
    for table_id in tqdm(table_ids, desc="Processing tables"):
        result = run_one(table_id, use_mock=(mode=='mock'), mode=mode, tsr_model=tsr_model)
        results.append(result)
        
        if result['status'] == 'failed':
            print(f"\n⚠️  Failed: {table_id} - {result['error']}")
    
    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successes = sum(1 for r in results if r['status'] == 'success')
    failures = len(results) - successes
    
    print("\n" + "="*50)
    print(f"Pipeline Complete!")
    print(f"  Success: {successes}/{len(results)}")
    print(f"  Failed: {failures}/{len(results)}")
    print(f"  Results saved to: {output_file}")
    print("="*50)
    
    if successes > 0:
        # Calculate average metrics
        avg_f1 = np.mean([r['metrics']['f1'] for r in results if r['status'] == 'success'])
        print(f"\nAverage Structure F1: {avg_f1:.4f}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run pipeline with artifact logging')
    parser.add_argument('--sample_file', type=str, default='samples/all_samples.txt',
                       help='File with table IDs to process')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tables to process')
    parser.add_argument('--use_mock', action='store_true', default=False,
                       help='Use mock OCR/TSR (DEPRECATED: Use --mode mock)')
    parser.add_argument('--mode', type=str, default='mock', choices=['mock', 's2'],
                       help='Pipeline mode: mock or s2 (Perfect OCR + TATR)')
    
    args = parser.parse_args()
    
    # Backward compatibility
    if args.use_mock and args.mode == 'mock':
        args.mode = 'mock'
    
    # Load samples
    with open(args.sample_file, 'r') as f:
        table_ids = [line.strip() for line in f if line.strip()]
    
    if args.limit:
        table_ids = table_ids[:args.limit]
    
    print(f"Loaded {len(table_ids)} table IDs from {args.sample_file}")
    
    # Run
    results = run_batch(table_ids, mode=args.mode)

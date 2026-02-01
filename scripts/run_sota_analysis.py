import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add project root and src to path
sys.path.insert(0, str(Path('/home/user/t1-7')))
sys.path.insert(0, str(Path('/home/user/t1-7/src')))
from src.ocr_tsr.enhanced_pipeline import EnhancedOCRTSRPipeline
from src.evaluation.relation_evaluator import RelationEvaluator
from src.classification.error_classifier import ErrorClassifier

def run_sota_batch(limit=20):
    print(f"Starting SOTA Pipeline (PaddleOCR + TATR) for {limit} tables...")
    pipeline = EnhancedOCRTSRPipeline(ocr_mode='paddleocr', tsr_mode='table_transformer')
    evaluator = RelationEvaluator()
    classifier = ErrorClassifier()
    
    # Load sample IDs
    try:
        with open('samples/all_samples.txt', 'r') as f:
            table_ids = [line.strip() for line in f if line.strip()][:limit]
    except:
        # Fallback to listing data dir
        table_ids = [p.stem for p in Path('data/SciTSR/test/img').glob('*.png')][:limit]

    results = []
    error_records = []
    
    for tid in tqdm(table_ids, desc="Processing"):
        img_path = f'data/SciTSR/test/img/{tid}.png'
        if not Path(img_path).exists(): img_path = f'data/SciTSR/test/img/{tid}.jpg'
        
        try:
            # 1. Inference
            ocr_res, tsr_res = pipeline.process(str(img_path), tid)
            
            # 2. Get Ground Truth for Evaluation
            # We use the pipeline internal loader or similar
            gt_data = pipeline._load_scitsr_gt(tid)
            
            # 3. Evaluate
            metrics = evaluator.evaluate_cells(tsr_res['cells'], gt_data['cells'])
            
            # 4. Classify Error
            edge_FP = metrics.get('edge_FP', [])
            edge_FN = metrics.get('edge_FN', [])
            classification = classifier.classify(tid, metrics, edge_FP, edge_FN)
            
            # 5. Record
            record = {
                'table_id': tid,
                'primary_type': classification['primary_type'],
                'secondary_types': ','.join(classification.get('secondary_types', [])),
                'stage': classification['stage'],
                'severity': classification['severity'],
                'struct_f1': metrics.get('f1', 0.0),
                'cer': metrics.get('cer', 0.0),
                'num_pred_cells': len(tsr_res['cells']),
                'num_gt_cells': len(gt_data['cells']),
                'timestamp': datetime.now().isoformat()
            }
            error_records.append(record)
            results.append({'table_id': tid, 'metrics': metrics, 'status': 'success'})
            
        except Exception as e:
            print(f"Failed {tid}: {e}")
            
    # Save SOTA DB
    df = pd.DataFrame(error_records)
    output_path = 'results/sota_error_db.csv'
    df.to_csv(output_path, index=False)
    print(f"SOTA Error DB saved to {output_path}")
    
    # Update classification summary for heatmap script to use
    with open('results/sota_pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_sota_batch(20)

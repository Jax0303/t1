"""
Optimized Production Attribution Script with Model Reuse
"""
import os
import json
import argparse
from pathlib import Path
import tqdm
from PIL import Image
from typing import Dict, List, Any

# Re-importing locally to ensure everything is in scope
from src.ocr_tsr.enhanced_pipeline import EnhancedOCRTSRPipeline
from src.evaluation.relation_evaluator import RelationEvaluator
from src.evaluation.auto_labeler import AutoLabeler
from src.visualization.overlay import draw_ocr_overlay, draw_tsr_overlay
from src.ocr_tsr.olmocr_wrapper import OlmOCROCR

def save_evidence(table_id: str, img_path: str, ocr_res: List[Dict], tsr_res: Dict, gt_data: Dict, metrics: Dict, label: Dict):
    evidence_dir = Path(f"results/evidence/{table_id}")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    img = Image.open(img_path).convert("RGB")
    img.save(evidence_dir / "input.png")
    
    ocr_overlay = draw_ocr_overlay(img, ocr_res)
    ocr_overlay.save(evidence_dir / "ocr_overlay.png")
    
    tsr_overlay = draw_tsr_overlay(img, tsr_res.get('detections', []))
    tsr_overlay.save(evidence_dir / "tatr_overlay.png")
    
    with open(evidence_dir / "ocr_tokens.json", 'w') as f:
        json.dump(ocr_res, f, indent=2)
    with open(evidence_dir / "pred_table.json", 'w') as f:
        json.dump({'cells': tsr_res.get('cells', []), 'num_rows': tsr_res.get('num_rows'), 'num_cols': tsr_res.get('num_cols')}, f, indent=2)
    with open(evidence_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(evidence_dir / "error_label.json", 'w') as f:
        json.dump(label, f, indent=2)

def run_attribution_for_table(table_id: str, olmocr_engine, split: str = 'test') -> Dict[str, Any]:
    img_path = f"data/SciTSR/{split}/img/{table_id}.png"
    gt_struct_path = f"data/SciTSR/{split}/structure/{table_id}.json"
    
    if not os.path.exists(img_path): return {'status': 'skip'}
    with open(gt_struct_path, 'r') as f: gt_data = json.load(f)
        
    evaluator = RelationEvaluator()
    labeler = AutoLabeler()
    results = {'table_id': table_id, 'scenarios': {}, 'status': 'success'}
    
    # Pre-cache OCR results to avoid redundant VLM calls
    # S1 & S3 share the same Real OCR (olmOCR)
    # S2 & S4 share the same Perfect OCR (chunk)
    
    # 1. Real OCR (olmOCR)
    pipeline_real = EnhancedOCRTSRPipeline(ocr_mode='olmocr', ocr_engine=olmocr_engine)
    real_ocr_res = pipeline_real.process_ocr(img_path, table_id)
    
    # 2. Perfect OCR (chunk)
    pipeline_perfect = EnhancedOCRTSRPipeline(ocr_mode='chunk')
    perfect_ocr_res = pipeline_perfect.process_ocr(img_path, table_id)
    
    scenarios = [
        ('S1_RealReal', real_ocr_res, 'table_transformer'),
        ('S2_ChunkReal', perfect_ocr_res, 'table_transformer'),
        ('S3_RealGT', real_ocr_res, 'gt'),
        ('S4_ChunkGT', perfect_ocr_res, 'gt')
    ]
    
    s1_metrics = None
    s1_tsr_res = None

    for name, ocr_res, tsr_mode in scenarios:
        pipeline = EnhancedOCRTSRPipeline(tsr_mode=tsr_mode)
        # Use our pre-cached OCR results
        _, tsr_res = pipeline.process_with_ocr_res(ocr_res, img_path, table_id, gt_data)
        
        metrics = evaluator.evaluate_cells(tsr_res['cells'], gt_data['cells'])
        results['scenarios'][name] = {'metrics': metrics, 'ocr_tokens': len(ocr_res)}
        
        if name == 'S1_RealReal':
            s1_metrics = metrics
            s1_tsr_res = tsr_res
            
    # Auto-label based on S1
    label = labeler.classify(s1_metrics, s1_tsr_res, gt_data)
    results['error_label'] = label
    save_evidence(table_id, img_path, real_ocr_res, s1_tsr_res, gt_data, s1_metrics, label)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    
    # Initialize VLM once
    olmocr_engine = OlmOCROCR()
    
    chunk_dir = Path("data/SciTSR/test/chunk")
    table_ids = [f.stem for f in chunk_dir.glob("*.chunk")][:args.limit]
    
    all_results = []
    for tid in tqdm.tqdm(table_ids):
        res = run_attribution_for_table(tid, olmocr_engine)
        if res['status'] == 'success': all_results.append(res)
            
    with open("results/production_attribution_final.json", 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()

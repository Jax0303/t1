import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.utils import resample
import random

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.dataset_loader import get_dataset_loader
from models import get_table_structure_recognizer, get_ocr_engine
from models.oracle_providers import OracleOCRProvider, OracleTSRProvider
from utils.metrics import ComprehensiveEvaluator, TEDSCalculator
from models.base_model import BoundingBox

class OracleSwapExperiment:
    def __init__(self, dataset_name, data_dir, scenarios, num_samples, oracle_ocr_mode="TOKEN"):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.scenarios = scenarios
        self.num_samples = num_samples
        self.oracle_ocr_mode = oracle_ocr_mode
        self.evaluator = ComprehensiveEvaluator()
        self.output_dir = Path("results/oracle_swap")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.real_tsr = None
        self.real_ocr = None
        self.oracle_tsr = OracleTSRProvider()
        self.oracle_ocr = OracleOCRProvider(mode=oracle_ocr_mode)

    def load_models(self):
        print("Initializing Real Models...")
        # Real TSR (TATR)
        self.real_tsr = get_table_structure_recognizer('huggingface', {'model_name': 'microsoft/table-transformer-structure-recognition'})
        self.real_tsr.load_model()
        
        # Real OCR (Tesseract as baseline)
        self.real_ocr = get_ocr_engine('tesseract', {'lang': 'eng'})
        self.real_ocr.load_model()

    def map_ocr_to_cells(self, ocr_results, cells):
        """OCR 결과를 셀에 매핑"""
        for cell in cells:
            if cell.bbox is None:
                continue
            
            c_x1, c_y1, c_x2, c_y2 = cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2
            matched_texts = []
            
            for ocr_item in ocr_results:
                ocr_bbox = ocr_item.get('bbox')
                if not ocr_bbox: continue
                
                # Center point matching
                ox1, oy1, ox2, oy2 = ocr_bbox.x1, ocr_bbox.y1, ocr_bbox.x2, ocr_bbox.y2
                cx = (ox1 + ox2) / 2
                cy = (oy1 + oy2) / 2
                
                if (c_x1 <= cx <= c_x2 and c_y1 <= cy <= c_y2):
                    matched_texts.append(ocr_item.get('text', ''))
            
            cell.text = ' '.join(matched_texts)

    def run_inference(self, sample, scenario):
        img = sample.load_image()
        gt = sample.ground_truth
        
        # Scenario Logic
        tsr_provider = self.oracle_tsr if scenario in ['S3', 'S4'] else self.real_tsr
        ocr_provider = self.oracle_ocr if scenario in ['S2', 'S4'] else self.real_ocr
        
        if hasattr(tsr_provider, 'set_gt'): tsr_provider.set_gt(gt)
        if hasattr(ocr_provider, 'set_gt'): ocr_provider.set_gt(gt)
        
        # 1. TSR
        structure = tsr_provider.predict(img)
        
        # 2. OCR
        ocr_results = ocr_provider.predict(img)
        
        # 3. Mapping (unless OracleOCR-CELL mode)
        if scenario in ['S2', 'S4'] and self.oracle_ocr_mode == "CELL":
            # Already has text in cells if using OracleTSR, 
            # but if using RealTSR + OracleOCR-CELL, we need to map GT cells to Real cells based on IoU
            if scenario == 'S2':
                gt_cells = gt.get('cells', [])
                for pred_cell in structure.cells:
                    best_iou = 0
                    best_text = ""
                    for gt_cell in gt_cells:
                        gt_bbox = gt_cell.get('bbox')
                        if gt_bbox:
                            if hasattr(gt_bbox, 'iou'):
                                iou = pred_cell.bbox.iou(gt_bbox)
                            elif isinstance(gt_bbox, dict):
                                iou = pred_cell.bbox.iou(BoundingBox(gt_bbox['x1'], gt_bbox['y1'], gt_bbox['x2'], gt_bbox['y2']))
                            else:
                                iou = pred_cell.bbox.iou(BoundingBox(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]))
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_text = gt_cell.get('text', '')
                    if best_iou > 0.5:
                        pred_cell.text = best_text
        else:
            self.map_ocr_to_cells(ocr_results, structure.cells)
            
        return structure

    def evaluate_scenario(self, samples):
        results = []
        for scenario in self.scenarios:
            print(f"Running Scenario {scenario}...")
            scenario_results = []
            for sample in tqdm(samples, desc=f"Samples {scenario}"):
                try:
                    pred_structure = self.run_inference(sample, scenario)
                    
                    # GT structure for metrics
                    self.oracle_tsr.set_gt(sample.ground_truth)
                    gt_structure = self.oracle_tsr.predict(sample.load_image())
                    
                    # Metrics
                    eval_res = self.evaluator.evaluate(
                        pred_html=pred_structure.to_html(),
                        gt_html=gt_structure.to_html()
                    )
                    
                    # Error Tagging (heuristic)
                    error_tags = []
                    if eval_res.teds < 1.0:
                        if pred_structure.num_rows != gt_structure.num_rows: error_tags.append("row_shift")
                        if pred_structure.num_cols != gt_structure.num_cols: error_tags.append("col_shift")
                        if len(pred_structure.cells) != len(gt_structure.cells): error_tags.append("merged_split")
                    
                    scenario_results.append({
                        'sample_id': sample.sample_id,
                        'scenario': scenario,
                        'ocr_mode': self.oracle_ocr_mode,
                        'teds': eval_res.teds,
                        's_teds': eval_res.s_teds,
                        'error_tags': error_tags
                    })
                except Exception as e:
                    print(f"Error on sample {sample.sample_id}: {e}")
            
            results.extend(scenario_results)
        return results

    def bootstrap_stats(self, df):
        summary = []
        scenarios = df['scenario'].unique()
        baseline_scores = df[df['scenario'] == 'S1']['teds'].values
        
        for scen in scenarios:
            scores = df[df['scenario'] == scen]['teds'].values
            mean_score = np.mean(scores)
            
            # Bootstrap for 95% CI
            boot_means = []
            for _ in range(1000):
                boot_sample = resample(scores)
                boot_means.append(np.mean(boot_sample))
            
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            
            delta = 0
            delta_ci = (0, 0)
            if scen != 'S1':
                deltas = scores - baseline_scores
                delta = np.mean(deltas)
                boot_deltas = []
                for _ in range(1000):
                    boot_deltas.append(np.mean(resample(deltas)))
                delta_ci = np.percentile(boot_deltas, [2.5, 97.5])

            summary.append({
                'scenario': scen,
                'mean_teds': mean_score,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'delta': delta,
                'delta_ci_low': delta_ci[0],
                'delta_ci_high': delta_ci[1]
            })
        return pd.DataFrame(summary)

    def run(self):
        # 1. Load Data
        loader = get_dataset_loader(self.dataset_name, self.data_dir)
        samples = loader.load_samples(split='test', max_samples=self.num_samples)
        
        # 2. Load Models
        self.load_models()
        
        # 3. Run Experiments
        all_results = self.evaluate_scenario(samples)
        
        # 4. Save per-sample results
        with open(self.output_dir / "per_sample_results.jsonl", "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")
        
        # 5. Summary Statistics
        df = pd.DataFrame(all_results)
        summary_df = self.bootstrap_stats(df)
        summary_df.to_csv(self.output_dir / "summary_statistics.csv", index=False)
        
        print("\n=== Experiment Summary ===")
        print(summary_df)
        
        # 6. Interpretation
        try:
            s1 = summary_df[summary_df['scenario'] == 'S1']['mean_teds'].values[0] if 'S1' in summary_df['scenario'].values else 0
            s2 = summary_df[summary_df['scenario'] == 'S2']['mean_teds'].values[0] if 'S2' in summary_df['scenario'].values else None
            s3 = summary_df[summary_df['scenario'] == 'S3']['mean_teds'].values[0] if 'S3' in summary_df['scenario'].values else None
            
            if s2 is not None and s3 is not None:
                delta_ocr = s2 - s1
                delta_tsr = s3 - s1
                
                print(f"\nDelta_OCR (S2-S1): {delta_ocr:.4f}")
                print(f"Delta_TSR (S3-S1): {delta_tsr:.4f}")
                
                if delta_tsr > delta_ocr:
                    # Check overlap of delta CI
                    s2_delta_ci = (summary_df[summary_df['scenario'] == 'S2']['delta_ci_low'].values[0], 
                                   summary_df[summary_df['scenario'] == 'S2']['delta_ci_high'].values[0])
                    s3_delta_ci = (summary_df[summary_df['scenario'] == 'S3']['delta_ci_low'].values[0], 
                                   summary_df[summary_df['scenario'] == 'S3']['delta_ci_high'].values[0])
                    
                    if s3_delta_ci[0] > s2_delta_ci[1]:
                        print(">> RESULT: TSR bottleneck likely (statistically consistent)")
                    else:
                        print(">> RESULT: TSR bottleneck is higher, but CI overlaps slightly.")
                else:
                    print(">> RESULT: OCR bottleneck may be more significant or comparable.")
            else:
                print("\nInterpretation skipped: Not all scenarios (S1, S2, S3) were run.")
        except Exception as e:
            print(f"\nError during interpretation: {e}")

        # 7. Error Tags Count
        error_df = df[df['error_tags'].map(len) > 0]
        all_tags = [tag for tags in error_df['error_tags'] for tag in tags]
        tag_counts = pd.Series(all_tags).value_counts().head(20)
        print("\n=== Top Error Tags ===")
        print(tag_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3", "S4"])
    parser.add_argument("--oracle_ocr_mode", type=str, choices=["TOKEN", "CELL"], default="TOKEN")
    
    args = parser.parse_args()
    
    exp = OracleSwapExperiment(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        scenarios=args.scenarios,
        num_samples=args.num_samples,
        oracle_ocr_mode=args.oracle_ocr_mode
    )
    exp.run()

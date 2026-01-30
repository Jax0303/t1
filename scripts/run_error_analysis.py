import os
import json
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any

# Root setup
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.hiertable_rag.gnn_csp.pipeline import GNNCSPPipeline
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.utils.table_converter import table_data_to_html

def load_scitsr_comp_list(path: str) -> List[str]:
    if not os.path.exists(path):
        print(f"Warning: COMP list not found at {path}")
        return []
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def sample_scitsr_data(gt_dir: str, comp_list: List[str], num_samples: int = 100) -> Dict[str, List[str]]:
    all_files = [f.replace('.json', '') for f in os.listdir(gt_dir) if f.endswith('.json')]
    
    comp_files = [f for f in all_files if f in comp_list]
    normal_files = [f for f in all_files if f not in comp_list]
    
    selected_comp = random.sample(comp_files, min(num_samples, len(comp_files))) if comp_files else []
    selected_normal = random.sample(normal_files, min(num_samples, len(normal_files))) if normal_files else []
    
    return {
        "complex": selected_comp,
        "normal": selected_normal
    }

def simulate_ocr_noise(ocr_boxes: List[Dict], coord_noise: float = 0.05, text_noise: float = 0.1) -> List[Dict]:
    noisy_boxes = []
    for box_data in ocr_boxes:
        box = list(box_data["box"])
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        # Add noise to coords
        box[0] += np.random.normal(0, coord_noise * w)
        box[1] += np.random.normal(0, coord_noise * h)
        box[2] += np.random.normal(0, coord_noise * w)
        box[3] += np.random.normal(0, coord_noise * h)
        
        # Add text noise
        text = str(box_data["content"])
        if text and np.random.random() < text_noise:
            chars = list(text)
            if chars:
                idx = np.random.randint(0, len(chars))
                chars[idx] = '*' 
                text = "".join(chars)
                
        # Ensure valid box
        if not np.isnan(box).any():
            if box[2] <= box[0]: box[2] = box[0] + 1
            if box[3] <= box[1]: box[3] = box[1] + 1
        else:
            box = list(box_data["box"])

        noisy_boxes.append({
            "box": box,
            "content": text
        })
    return noisy_boxes

class TSRModelWrapper:
    """Wrapper to switch between different TSR 'models' or logic."""
    def __init__(self, pipeline: GNNCSPPipeline, mode: str = "spatial", gt_cells: List[Dict] = None):
        self.pipeline = pipeline
        self.mode = mode
        self.gt_cells = gt_cells # For 'gt' mode

    def parse(self, image, ocr_boxes):
        if self.mode == "gt":
            # Use GT structure but with provided ocr_boxes (which might have noise)
            cells = []
            for i, ocr_box in enumerate(ocr_boxes):
                # We assume ocr_boxes matches gt_cells order (which it does in our loop)
                cells.append({
                    "row": self.gt_cells[i]["row"],
                    "col": self.gt_cells[i]["col"],
                    "row_span": self.gt_cells[i]["row_span"],
                    "col_span": self.gt_cells[i]["col_span"],
                    "text": ocr_box["content"],
                    "box": ocr_box["box"]
                })
            return {
                'cells': cells,
                'num_rows': max([c['row'] for c in cells]) + 1,
                'num_cols': max([c['col'] for c in cells]) + 1
            }
        elif self.mode == "spatial":
            # Use pipeline without CSP/GNN (defaults to spatial sort in GNNCSPPipeline)
            return self.pipeline.parse(image, ocr_boxes, use_csp=False)
        elif self.mode == "gnn_csp":
            # Real pipeline
            return self.pipeline.parse(image, ocr_boxes, use_csp=True)
        else: # Default
            return self.pipeline.parse(image, ocr_boxes, use_csp=False)

def run_experiment(
    pipeline_wrapper: TSRModelWrapper,
    analyzer: ErrorAnalyzer,
    gt_dir: str,
    img_dir: str,
    file_list: List[str],
    label: str,
    use_noise: bool = False,
    noise_level: float = 0.05
) -> List[Dict]:
    results = []
    
    for file_id in tqdm(file_list, desc=f"Analyzing {label} tables"):
        try:
            gt_path = os.path.join(gt_dir, f"{file_id}.json")
            img_path = os.path.join(img_dir, f"{file_id}.png")
            
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            
            gt_cells = []
            for cell in gt_data:
                gt_cells.append({
                    "row": cell["logical_coords"][0], # Consistent key 'row'
                    "col": cell["logical_coords"][2],
                    "row_span": cell["logical_coords"][1] - cell["logical_coords"][0] + 1,
                    "col_span": cell["logical_coords"][3] - cell["logical_coords"][2] + 1,
                    "text": cell.get("text", cell.get("content", "")),
                    "box": cell["box"]
                })
            
            gt_html = table_data_to_html(gt_cells)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((448, 448), Image.BILINEAR)
            image = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            
            ocr_boxes = [{"box": c["box"], "content": c["text"]} for c in gt_cells]
            if use_noise:
                ocr_boxes = simulate_ocr_noise(ocr_boxes, coord_noise=noise_level)
            
            # Update wrapper with current GT for 'gt' mode
            pipeline_wrapper.gt_cells = gt_cells
            
            output = pipeline_wrapper.parse(image, ocr_boxes)
            pred_cells = output['cells']
            # Ensure pred_cells have consistent keys
            for c in pred_cells:
                if 'text' not in c and 'content' in c: c['text'] = c['content']
                if 'row' not in c and 'row_idx' in c: c['row'] = c['row_idx']
                if 'col' not in c and 'col_idx' in c: c['col'] = c['col_idx']
                
            pred_html = table_data_to_html(pred_cells)
            
            analysis = analyzer.analyze_sample(pred_html, gt_html, pred_cells, gt_cells)
            analysis["filename"] = file_id
            analysis["label"] = label
            
            results.append(analysis)
            
        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SciTSR Error Analysis & Comparison Framework")
    parser.add_argument('--num_samples', type=int, default=100, help='Samples per category')
    args = parser.parse_args()

    scitsr_root = Path("/root/t1-9/data/external/scitsr/SciTsr_Logical/test")
    gt_dir = str(scitsr_root / "gt")
    img_dir = str(scitsr_root / "images")
    comp_list_path = "/root/t1-11/data/scitsr/SciTSR/SciTSR-COMP.list"
    checkpoint_path = "/root/t1-10/outputs/rca_best.pth"
    output_report_path = "/root/t1-10/outputs/refined_comparison_report.json"
    
    os.makedirs("/root/t1-10/outputs", exist_ok=True)
    
    print(f"Sampling {args.num_samples} tables (Normal & Complex)...")
    comp_list = load_scitsr_comp_list(comp_list_path)
    sampled_data = sample_scitsr_data(gt_dir, comp_list, num_samples=args.num_samples)
    
    pipeline = GNNCSPPipeline(rca_checkpoint=checkpoint_path)
    analyzer = ErrorAnalyzer()
    
    # Scenarios for User's Comparative Experiments
    scenarios = [
        # Scenario 1: TSR Limit (Perfect OCR + Spatial TSR)
        {"name": "Exp 1: Perfect OCR + Spatial TSR", "tsr_mode": "spatial", "use_noise": False},
        
        # Scenario 2: OCR Impact (Noisy OCR + Spatial TSR)
        {"name": "Exp 2: Noisy OCR + Spatial TSR", "tsr_mode": "spatial", "use_noise": True},
        
        # Scenario 3: TSR Reliability with Noise (Noisy OCR + RCA/GNN TSR)
        # Assuming gnn_csp represents our "better" model
        {"name": "Exp 3: Noisy OCR + GNN-CSP TSR", "tsr_mode": "gnn_csp", "use_noise": True},
        
        # Scenario 4: Pure OCR Error (Noisy OCR + Perfect TSR)
        {"name": "Exp 4: Noisy OCR + Perfect TSR", "tsr_mode": "gt", "use_noise": True},
    ]
    
    final_comparisons = {}
    
    for sc in scenarios:
        print(f"\n>>> Running Scenario: {sc['name']}")
        wrapper = TSRModelWrapper(pipeline, mode=sc['tsr_mode'])
        
        exp_results = []
        for label in ["Normal", "Complex"]:
            file_list = sampled_data[label.lower()]
            res = run_experiment(wrapper, analyzer, gt_dir, img_dir, file_list, label, 
                                 use_noise=sc['use_noise'], noise_level=0.05)
            exp_results.extend(res)
            
        final_comparisons[sc['name']] = {
            "summary": analyzer.get_summary_stats(exp_results),
            "details_sample": exp_results[:10] # Save only small sample of details to keep JSON light
        }
    
    with open(output_report_path, 'w') as f:
        json.dump(final_comparisons, f, indent=2)
    
    print(f"\nRefined report saved to: {output_report_path}")
    
    # Final Table
    print("\n" + "="*80)
    print(f"{'Scenario':<35} | {'TEDS-S':<8} | {'CER':<8} | {'Num CER':<8} | {'Alpha CER':<8}")
    print("-" * 80)
    for name, data in final_comparisons.items():
        s = data["summary"]
        print(f"{name:<35} | {s['mean_teds_s']:.4f} | {s['mean_cer']:.4f} | {s.get('cer_numeric', 0):.4f} | {s.get('cer_alpha', 0):.4f}")
    print("="*80)

if __name__ == "__main__":
    main()

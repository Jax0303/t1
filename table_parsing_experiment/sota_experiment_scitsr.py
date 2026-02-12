
import sys
import os
import time
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
import builtins

# Colab 호환성을 위한 패치
if not hasattr(builtins, 'cv2_imshow'):
    def cv2_imshow(img):
        pass
    builtins.cv2_imshow = cv2_imshow

from data.dataset_loader import get_dataset_loader
from models.base_model import BoundingBox, TableStructure

# 결과 저장 디렉토리
RESULTS_DIR = Path('results/sota_experiment_scitsr')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("======================================================================")
print("SOTA MODEL COMPARISON with SciTSR Dataset (Real Data)")
print("======================================================================")

# 1. SciTSR 데이터 로드
print("\n[1/5] Loading SciTSR samples...")
try:
    loader = get_dataset_loader('scitsr', 'data')
    # 테스트셋에서 10개 샘플 로드
    samples = loader.load_samples(split='test', max_samples=10)
    print(f"✓ Loaded {len(samples)} SciTSR samples")
except Exception as e:
    print(f"Error loading SciTSR: {e}")
    print("Please make sure SciTSR dataset is downloaded and extracted in data/SciTSR")
    sys.exit(1)

# 실험 결과 저장용
experiment_results = {}

# 2. Table Transformer (TD + TSR)
print("\n[2/5] Testing Table Transformer (Baseline)...")
try:
    from models.table_detection import HuggingFaceTableDetector
    from models.table_structure import HuggingFaceTableStructureRecognizer
    
    tt_td = HuggingFaceTableDetector({'model_name': 'microsoft/table-transformer-detection', 'threshold': 0.5})
    tt_tsr = HuggingFaceTableStructureRecognizer({'model_name': 'microsoft/table-transformer-structure-recognition'})
    
    tt_td.load_model()
    tt_tsr.load_model()
    
    tt_results = []
    for sample in samples:
        img = sample.load_image()
        
        # Detection
        tables = tt_td.detect_tables(img)
        
        if tables:
            # Crop first table (assuming 1 table per image in simple test)
            # In real pipeline, we should crop. But TT TSR can take full image too if cropped properly.
            # Here we just pass full image to TSR for simplicity as SciTSR images are usually cropped tables or pages.
            # Actually SciTSR 'img' folder are cropped tables usually? No, full pages or regions?
            # SciTSR images are converted from PDF, might be full pages.
            # Use detected bbox to crop.
            
            bbox = tables[0]
            # Padding
            pad = 20
            crop_box = (
                max(0, bbox.x1 - pad),
                max(0, bbox.y1 - pad),
                min(img.width, bbox.x2 + pad),
                min(img.height, bbox.y2 + pad)
            )
            cropped_img = img.crop(crop_box)
            
            structure = tt_tsr.recognize_structure(cropped_img)
            
            # Simple metric: number of cells detected
            tt_results.append({
                'id': sample.sample_id,
                'status': 'success',
                'detected_cells': len(structure.cells)
            })
        else:
            tt_results.append({'id': sample.sample_id, 'status': 'failed_detection'})
            
    experiment_results['TableTransformer'] = tt_results
    print(f"✓ Table Transformer tested")
except Exception as e:
    print(f"✗ Table Transformer failed: {e}")
    experiment_results['TableTransformer'] = {'error': str(e)}

# 3. DocTR (OCR)
print("\n[3/5] Testing DocTR (OCR)...")
try:
    from models.ocr_engines import DocTROCREngine
    
    doctr = DocTROCREngine({'pretrained': True})
    doctr.load_model()
    
    doctr_results = []
    for sample in samples:
        img = sample.load_image()
        res = doctr.recognize_text(img)
        
        # Ground truth comparison
        gt_texts = [c['text'] for c in sample.ground_truth.get('cells', [])]
        pred_texts = [r['text'] for r in res]
        
        detected_text_count = len(pred_texts)
        gt_text_count = len(gt_texts)
        
        # Simple text matching (inclusion)
        matched = 0
        for pt in pred_texts:
            for gt in gt_texts:
                if pt in gt or gt in pt: # Loose matching
                    matched += 1
                    break
        
        accuracy = matched / max(1, len(pred_texts)) # Precision-like
        coverage = matched / max(1, len(gt_texts))   # Recall-like
        
        doctr_results.append({
            'id': sample.sample_id,
            'precision': accuracy,
            'recall': coverage,
            'text_count': detected_text_count
        })
        
    experiment_results['DocTR'] = doctr_results
    print(f"✓ DocTR tested")
except Exception as e:
    print(f"✗ DocTR failed: {e}")
    experiment_results['DocTR'] = {'error': str(e)}

# 4. Tesseract (Comparison)
print("\n[4/5] Testing Tesseract (OCR)...")
try:
    from models.ocr_engines import TesseractEngine
    
    tess = TesseractEngine({})
    
    tess_results = []
    for sample in samples:
        img = sample.load_image()
        res = tess.recognize_text(img)
        
        gt_texts = [c['text'] for c in sample.ground_truth.get('cells', [])]
        pred_texts = [r['text'] for r in res]
        
        matched = 0
        for pt in pred_texts:
            for gt in gt_texts:
                if pt in gt or gt in pt:
                    matched += 1
                    break
                    
        accuracy = matched / max(1, len(pred_texts))
        coverage = matched / max(1, len(gt_texts))
        
        tess_results.append({
            'id': sample.sample_id,
            'precision': accuracy,
            'recall': coverage
        })
        
    experiment_results['Tesseract'] = tess_results
    print(f"✓ Tesseract tested")
except Exception as e:
    print(f"✗ Tesseract failed: {e}")

# 5. Report
print("\n[5/5] Generating Report...")

def safe_mean(lst):
    if not lst: return 0
    return sum(lst) / len(lst)

doctr_prec = safe_mean([r['precision'] for r in experiment_results.get('DocTR', [])])
doctr_rec = safe_mean([r['recall'] for r in experiment_results.get('DocTR', [])])
tess_prec = safe_mean([r['precision'] for r in experiment_results.get('Tesseract', [])])
tess_rec = safe_mean([r['recall'] for r in experiment_results.get('Tesseract', [])])

tt_success = len([r for r in experiment_results.get('TableTransformer', []) if r.get('status') == 'success'])

html_content = f"""
<html>
<head>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
</style>
</head>
<body>
<h1>SciTSR Real Data Evaluation</h1>
<p>Tested on {len(samples)} random samples from SciTSR test set.</p>

<h2>Performance Summary</h2>
<table border="1">
<tr>
    <th>Metric</th>
    <th>Table Transformer</th>
    <th>DocTR</th>
    <th>Tesseract</th>
</tr>
<tr>
    <td>Success/Recall</td>
    <td>{tt_success}/{len(samples)} (Detection)</td>
    <td>{doctr_rec:.2%} (Text Recall)</td>
    <td>{tess_rec:.2%} (Text Recall)</td>
</tr>
<tr>
    <td>Precision</td>
    <td>N/A</td>
    <td>{doctr_prec:.2%}</td>
    <td>{tess_prec:.2%}</td>
</tr>
</table>

<h2>Detailed Results</h2>
<pre>
{json.dumps(experiment_results, indent=2)}
</pre>
</body>
</html>
"""

with open(RESULTS_DIR / 'scitsr_report.html', 'w') as f:
    f.write(html_content)

print(f"Report saved to {RESULTS_DIR}/scitsr_report.html")

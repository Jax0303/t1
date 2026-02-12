"""
SOTA 모델 비교 실험
DocTR OCR vs Table Transformer를 사용한 종합 실험
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw, ImageFont
import json
import random
from datetime import datetime
import time

print("="*70)
print("SOTA MODEL COMPARISON EXPERIMENT")
print("="*70)

# 출력 디렉토리
output_dir = Path('results/sota_experiment')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 테스트 이미지 생성
print("\n[1/5] Generating test table images...")
images_dir = output_dir / 'test_images'
images_dir.mkdir(exist_ok=True)

table_types = ['fully_bordered', 'unbordered', 'partially_bordered']
samples = []

for i in range(5):
    table_type = random.choice(table_types)
    
    # 표 생성
    width, height = 800, 400
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    rows = random.randint(3, 5)
    cols = random.randint(3, 4)
    cell_width = width // cols
    cell_height = height // rows
    
    # 테두리
    if table_type == 'fully_bordered':
        for r in range(rows + 1):
            y = r * cell_height
            draw.line([(0, y), (width, y)], fill='black', width=2)
        for c in range(cols + 1):
            x = c * cell_width
            draw.line([(x, 0), (x, height)], fill='black', width=2)
    elif table_type == 'partially_bordered':
        draw.rectangle([(0, 0), (width-1, height-1)], outline='black', width=2)
        draw.line([(0, cell_height), (width, cell_height)], fill='black', width=2)
    
    # 텍스트
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    cells_data = []
    for r in range(rows):
        for c in range(cols):
            x = c * cell_width + 15
            y = r * cell_height + 15
            text = f"Cell{r}{c}" if r == 0 else f"Data{r}{c}"
            draw.text((x, y), text, fill='black', font=font)
            
            cells_data.append({
                'row': r,
                'col': c,
                'text': text,
                'is_header': (r == 0)
            })
    
    img_path = images_dir / f'table_{table_type}_{i}.png'
    img.save(img_path)
    
    samples.append({
        'id': i,
        'path': str(img_path),
        'type': table_type,
        'rows': rows,
        'cols': cols,
        'ground_truth': cells_data
    })

print(f"✓ Generated {len(samples)} test images")

# 2. Table Transformer 모델 테스트
print("\n[2/5] Testing Table Transformer models...")

try:
    from models.table_detection import HuggingFaceTableDetector
    from models.table_structure import HuggingFaceTableStructureRecognizer
    
    td_model = HuggingFaceTableDetector({'model_name': 'microsoft/table-transformer-detection'})
    tsr_model = HuggingFaceTableStructureRecognizer({'model_name': 'microsoft/table-transformer-structure-recognition'})
    
    td_model.load_model()
    tsr_model.load_model()
    
    tt_results = []
    for sample in samples:
        img = Image.open(sample['path'])
        
        start = time.time()
        tables = td_model.detect_tables(img)
        td_time = time.time() - start
        
        start = time.time()
        if tables:
            structure = tsr_model.recognize_structure(img, tables[0])
            tsr_time = time.time() - start
        else:
            structure = None
            tsr_time = 0
        
        tt_results.append({
            'sample_id': sample['id'],
            'type': sample['type'],
            'tables_detected': len(tables),
            'td_time': td_time,
            'tsr_time': tsr_time,
            'success': len(tables) > 0
        })
    
    print(f"✓ Table Transformer tested on {len(tt_results)} samples")
    tt_available = True
except Exception as e:
    print(f"✗ Table Transformer failed: {e}")
    tt_results = []
    tt_available = False

# 3. DocTR OCR 테스트
print("\n[3/5] Testing DocTR OCR...")

try:
    from models.ocr_engines import DocTROCREngine
    
    doctr_engine = DocTROCREngine({'pretrained': True})
    doctr_engine.load_model()
    
    doctr_results = []
    for sample in samples:
        img = Image.open(sample['path'])
        
        start = time.time()
        ocr_output = doctr_engine.recognize_text(img)
        ocr_time = time.time() - start
        
        # Ground truth와 비교
        gt_texts = [cell['text'] for cell in sample['ground_truth']]
        detected_texts = [item['text'] for item in ocr_output]
        
        matched = sum(1 for t in detected_texts if any(t in gt for gt in gt_texts))
        precision = matched / len(detected_texts) if detected_texts else 0
        recall = matched / len(gt_texts) if gt_texts else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        doctr_results.append({
            'sample_id': sample['id'],
            'type': sample['type'],
            'num_detected': len(detected_texts),
            'num_expected': len(gt_texts),
            'matched': matched,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ocr_time': ocr_time
        })
    
    print(f"✓ DocTR tested on {len(doctr_results)} samples")
    doctr_available = True
except Exception as e:
    print(f"✗ DocTR failed: {e}")
    doctr_results = []
    doctr_available = False

# 4. 결과 분석
print("\n[4/5] Analyzing results...")

analysis = {
    'experiment_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
    'models_tested': {
        'table_transformer': tt_available,
        'doctr': doctr_available
    },
    'total_samples': len(samples),
    'samples': samples,
    'table_transformer_results': tt_results,
    'doctr_results': doctr_results,
    'summary': {}
}

# 요약 통계
if tt_available and tt_results:
    avg_td_time = sum(r['td_time'] for r in tt_results) / len(tt_results)
    success_rate = sum(1 for r in tt_results if r['success']) / len(tt_results)
    analysis['summary']['table_transformer'] = {
        'avg_detection_time': avg_td_time,
        'detection_success_rate': success_rate
    }

if doctr_available and doctr_results:
    avg_f1 = sum(r['f1'] for r in doctr_results) / len(doctr_results)
    avg_precision = sum(r['precision'] for r in doctr_results) / len(doctr_results)
    avg_recall = sum(r['recall'] for r in doctr_results) / len(doctr_results)
    avg_time = sum(r['ocr_time'] for r in doctr_results) / len(doctr_results)
    
    analysis['summary']['doctr'] = {
        'avg_f1': avg_f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_ocr_time': avg_time
    }

# 결과 저장
results_file = output_dir / 'sota_comparison_results.json'
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print(f"✓ Results saved to {results_file}")

# 5. HTML 리포트 생성
print("\n[5/5] Generating report...")

html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SOTA Model Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .metric {{ display: inline-block; padding: 20px; margin: 10px; background: #ecf0f1; 
                   border-radius: 5px; text-align: center; min-width: 150px; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 SOTA Model Comparison Report</h1>
        <p><strong>Experiment ID:</strong> {analysis['experiment_id']} </p>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Models Tested</h2>
"""

if tt_available:
    html += f"""
        <div class="metric">
            <div class="metric-label">Table Transformer</div>
            <div class="metric-value success">✓ Available</div>
        </div>
"""
else:
    html += f"""
        <div class="metric">
            <div class="metric-label">Table Transformer</div>
            <div class="metric-value warning">✗ Unavailable</div>
        </div>
"""

if doctr_available:
    html += f"""
        <div class="metric">
            <div class="metric-label">DocTR OCR</div>
            <div class="metric-value success">✓ Available</div>
        </div>
"""
else:
    html += f"""
        <div class="metric">
            <div class="metric-label">DocTR OCR</div>
            <div class="metric-value warning">✗ Unavailable</div>
        </div>
"""

# 성능 요약
if 'table_transformer' in analysis['summary']:
    tt_sum = analysis['summary']['table_transformer']
    html += f"""
        <h2>Table Transformer Performance</h2>
        <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Detection Success Rate</td><td>{tt_sum['detection_success_rate']:.1%}</td></tr>
                <tr><td>Avg Detection Time</td><td>{tt_sum['avg_detection_time']:.3f}s</td></tr>
            </tbody>
        </table>
"""

if 'doctr' in analysis['summary']:
    doctr_sum = analysis['summary']['doctr']
    html += f"""
        <h2>DocTR OCR Performance</h2>
        <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Average F1 Score</td><td>{doctr_sum['avg_f1']:.2%}</td></tr>
                <tr><td>Average Precision</td><td>{doctr_sum['avg_precision']:.2%}</td></tr>
                <tr><td>Average Recall</td><td>{doctr_sum['avg_recall']:.2%}</td></tr>
                <tr><td>Avg OCR Time</td><td>{doctr_sum['avg_ocr_time']:.3f}s</td></tr>
            </tbody>
        </table>
"""

html += """
    </div>
</body>
</html>
"""

report_path = output_dir / 'sota_comparison_report.html'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✓ Report saved to {report_path}")

# 최종 요약
print("\n" + "="*70)
print("SOTA Model Comparison Complete!")
print("="*70)
print(f"\n📁 Results: {output_dir}")
print(f"📊 Report: {report_path}")
print(f"🖼️  Images: {images_dir}")

if 'table_transformer' in analysis['summary']:
    tt_sum = analysis['summary']['table_transformer']
    print(f"\n📈 Table Transformer: Success Rate = {tt_sum['detection_success_rate']:.1%}")

if 'doctr' in analysis['summary']:
    doctr_sum = analysis['summary']['doctr']
    print(f"📈 DocTR OCR: F1 = {doctr_sum['avg_f1']:.2%}, Precision = {doctr_sum['avg_precision']:.2%}, Recall = {doctr_sum['avg_recall']:.2%}")

print("\n💡 To view the report:")
print(f"   xdg-open {report_path}")

print("\n" + "="*70)

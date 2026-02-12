"""
Real Model Experiment - 실제 모델로 표 파싱 실험
Tesseract만 사용하는 간단한 버전 (transformers 설치 대기 중)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw, ImageFont
import json
import random
from datetime import datetime

print("="*60)
print("Real Model Table Parsing Experiment")
print("="*60)

# 출력 디렉토리
output_dir = Path('results/real_experiment')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 테스트 이미지 생성
print("\n[1/4] Generating test table images...")
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

# 2. Tesseract OCR 테스트
print("\n[2/4] Testing OCR (Tesseract)...")
ocr_results = []

try:
    import pytesseract
    tesseract_available = True
    
    for sample in samples:
        img = Image.open(sample['path'])
        
        try:
            # OCR 실행
            ocr_text = pytesseract.image_to_string(img)
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # 텍스트 추출
            detected_texts = [
                ocr_data['text'][i] 
                for i in range(len(ocr_data['text'])) 
                if ocr_data['text'][i].strip() and int(ocr_data['conf'][i]) > 0
            ]
            
            # Ground truth와 비교
            gt_texts = [cell['text'] for cell in sample['ground_truth']]
            matched = sum(1 for t in detected_texts if t in gt_texts)
            precision = matched / len(detected_texts) if detected_texts else 0
            recall = matched / len(gt_texts) if gt_texts else 0
            
            ocr_results.append({
                'sample_id': sample['id'],
                'type': sample['type'],
                'detected_texts': detected_texts,
                'num_detected': len(detected_texts),
                'num_expected': len(gt_texts),
                'matched': matched,
                'precision': precision,
                'recall': recall,
                'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            })
            
        except Exception as e:
            print(f"  Error processing sample {sample['id']}: {e}")
            ocr_results.append({
                'sample_id': sample['id'],
                'type': sample['type'],
                'error': str(e)
            })
    
    print(f"✓ OCR completed on {len(ocr_results)} samples")
    
except ImportError:
    tesseract_available = False
    print("  ⚠️  Tesseract not available - skipping OCR")
    ocr_results = [{'note': 'Tesseract not installed'}]

# 3. 결과 분석
print("\n[3/4] Analyzing results...")

analysis = {
    'experiment_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
    'total_samples': len(samples),
    'ocr_available': tesseract_available,
    'samples': samples,
    'ocr_results': ocr_results,
    'summary': {}
}

if tesseract_available and ocr_results:
    # 타입별 집계
    by_type = {}
    for result in ocr_results:
        if 'error' in result:
            continue
        
        ttype = result['type']
        if ttype not in by_type:
            by_type[ttype] = {'count': 0, 'precision': [], 'recall': [], 'f1': []}
        
        by_type[ttype]['count'] += 1
        by_type[ttype]['precision'].append(result['precision'])
        by_type[ttype]['recall'].append(result['recall'])
        by_type[ttype]['f1'].append(result['f1'])
    
    # 평균 계산
    for ttype, data in by_type.items():
        data['avg_precision'] = sum(data['precision']) / len(data['precision']) if data['precision'] else 0
        data['avg_recall'] = sum(data['recall']) / len(data['recall']) if data['recall'] else 0
        data['avg_f1'] = sum(data['f1']) / len(data['f1']) if data['f1'] else 0
    
    analysis['summary']['by_type'] = by_type

# 결과 저장
results_file = output_dir / 'experiment_results.json'
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print(f"✓ Results saved to {results_file}")

# 4. 간단한 리포트 생성
print("\n[4/4] Generating report...")

html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Real Model Experiment Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #e74c3c; color: white; }}
        .metric {{ display: inline-block; padding: 20px; margin: 10px; background: #ecf0f1; 
                   border-radius: 5px; text-align: center; min-width: 120px; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
        .note {{ background: #fff3cd; padding: 10px; border-left: 4px solid #f39c12; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Real Model Experiment Results</h1>
        <p><strong>Experiment ID:</strong> {analysis['experiment_id']}</p>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overview</h2>
        <div class="metric">
            <div class="metric-label">Total Samples</div>
            <div class="metric-value">{analysis['total_samples']}</div>
        </div>
        <div class="metric">
            <div class="metric-label">OCR Status</div>
            <div class="metric-value {'success' if tesseract_available else 'error'}">
                {'✓ Available' if tesseract_available else '✗ Unavailable'}
            </div>
        </div>
"""

if tesseract_available and 'by_type' in analysis['summary']:
    html += """
        <h2>OCR Performance by Table Type</h2>
        <table>
            <thead>
                <tr><th>Table Type</th><th>Samples</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
            </thead>
            <tbody>
"""
    for ttype, data in analysis['summary']['by_type'].items():
        html += f"""
                <tr>
                    <td>{ttype.replace('_', ' ').title()}</td>
                    <td>{data['count']}</td>
                    <td>{data['avg_precision']:.2%}</td>
                    <td>{data['avg_recall']:.2%}</td>
                    <td>{data['avg_f1']:.2%}</td>
                </tr>
"""
    html += """
            </tbody>
        </table>
"""
else:
    html += """
        <div class="note">
            <strong>⚠️ Note:</strong> Tesseract OCR is not installed. Install it with:
            <code>sudo apt-get install tesseract-ocr</code>
        </div>
"""

html += """
        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr><th>Sample ID</th><th>Type</th><th>Detected</th><th>Expected</th><th>Matched</th></tr>
            </thead>
            <tbody>
"""

for result in ocr_results:
    if 'error' not in result and 'note' not in result:
        html += f"""
                <tr>
                    <td>{result['sample_id']}</td>
                    <td>{result['type']}</td>
                    <td>{result['num_detected']}</td>
                    <td>{result['num_expected']}</td>
                    <td>{result['matched']}</td>
                </tr>
"""

html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

report_path = output_dir / 'experiment_report.html'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✓ Report saved to {report_path}")

# 최종 요약
print("\n" + "="*60)
print("Experiment Complete!")
print("="*60)
print(f"\n📁 Results: {output_dir}")
print(f"📊 Report: {report_path}")
print(f"🖼️  Images: {images_dir}")

if tesseract_available and 'by_type' in analysis['summary']:
    print(f"\n📈 OCR Performance Summary:")
    for ttype, data in analysis['summary']['by_type'].items():
        print(f"  {ttype}: F1={data['avg_f1']:.2%}, Precision={data['avg_precision']:.2%}, Recall={data['avg_recall']:.2%}")
else:
    print(f"\n⚠️  Install Tesseract OCR for full functionality:")
    print(f"   sudo apt-get install tesseract-ocr")

print(f"\n💡 To view the report:")
print(f"   xdg-open {report_path}")
print(f"   # or open it in any web browser")

print("\n" + "="*60)

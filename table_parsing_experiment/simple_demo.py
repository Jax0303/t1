"""
간소화된 데모 실험
의존성 없이 실행 가능한 버전
"""
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 간단한 합성 데이터 생성
from PIL import Image, ImageDraw, ImageFont
import random


def create_test_table_images(output_dir, num_samples=10):
    """테스트용 표 이미지 생성"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    samples = []
    table_types = ['fully_bordered', 'unbordered', 'partially_bordered']
    
    print(f"Generating {num_samples} test table images...")
    
    for i in range(num_samples):
        table_type = random.choice(table_types)
        
        # 표 이미지 생성
        width, height = 800, 400
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        rows = random.randint(3, 5)
        cols = random.randint(3, 4)
        cell_width = width // cols
        cell_height = height // rows
        
        # 테두리 그리기
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
        
        # 텍스트 추가
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        for r in range(rows):
            for c in range(cols):
                x = c * cell_width + 10
                y = r * cell_height + 10
                text = f"R{r}C{c}"
                draw.text((x, y), text, fill='black', font=font)
        
        # 이미지 저장
        img_path = output_path / f'table_{table_type}_{i}.png'
        img.save(img_path)
        
        samples.append({
            'id': i,
            'path': str(img_path),
            'type': table_type,
            'rows': rows,
            'cols': cols
        })
    
    print(f"✓ Generated {len(samples)} images")
    return samples


def analyze_tables_simple(samples):
    """간단한 표 분석 (모델 없이 통계만)"""
    print("\nAnalyzing tables...")
    
    results = {
        'samples': [],
        'summary': {
            'total': len(samples),
            'by_type': {}
        }
    }
    
    for sample in samples:
        # 타입별 집계
        table_type = sample['type']
        if table_type not in results['summary']['by_type']:
            results['summary']['by_type'][table_type] = {
                'count': 0,
                'avg_cells': 0,
                'total_cells': 0
            }
        
        num_cells = sample['rows'] * sample['cols']
        results['summary']['by_type'][table_type]['count'] += 1
        results['summary']['by_type'][table_type]['total_cells'] += num_cells
        
        results['samples'].append({
            'id': sample['id'],
            'type': table_type,
            'cells': num_cells,
            'complexity': 'simple' if num_cells < 12 else 'medium' if num_cells < 20 else 'complex'
        })
    
    # 평균 계산
    for table_type, data in results['summary']['by_type'].items():
        if data['count'] > 0:
            data['avg_cells'] = data['total_cells'] / data['count']
    
    print("✓ Analysis complete")
    return results


def generate_simple_report(results, output_dir):
    """간단한 HTML 리포트 생성"""
    output_path = Path(output_dir)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Simple Table Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .metric {{ display: inline-block; padding: 20px; margin: 10px; background: #ecf0f1; 
                   border-radius: 5px; min-width: 150px; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); 
                   gap: 20px; margin: 20px 0; }}
        .gallery img {{ width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Simple Table Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overview</h2>
        <div class="metric">
            <div class="metric-label">Total Samples</div>
            <div class="metric-value">{results['summary']['total']}</div>
        </div>
"""
    
    # 타입별 통계
    for table_type, data in results['summary']['by_type'].items():
        html += f"""
        <div class="metric">
            <div class="metric-label">{table_type.replace('_', ' ').title()}</div>
            <div class="metric-value">{data['count']}</div>
        </div>
"""
    
    # 상세 표
    html += """
        <h2>Sample Details</h2>
        <table>
            <thead>
                <tr><th>ID</th><th>Type</th><th>Cells</th><th>Complexity</th></tr>
            </thead>
            <tbody>
"""
    
    for sample in results['samples']:
        html += f"""
                <tr>
                    <td>{sample['id']}</td>
                    <td>{sample['type']}</td>
                    <td>{sample['cells']}</td>
                    <td>{sample['complexity']}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
        
        <h2>Sample Images</h2>
        <div class="gallery">
"""
    
    # 이미지 갤러리 (처음 5개만)
    for i in range(min(5, len(results['samples']))):
        img_path = Path(output_dir) / 'synthetic_data' / f"table_*_{i}.png"
        matching = list(Path(output_dir).glob(f'synthetic_data/table_*_{i}.png'))
        if matching:
            rel_path = matching[0].relative_to(output_dir)
            html += f'<img src="{rel_path}" alt="Sample {i}">'
    
    html += """
        </div>
    </div>
</body>
</html>
"""
    
    report_path = output_path / 'simple_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Report saved to {report_path}")
    return report_path


def main():
    """메인 실행"""
    print("="*60)
    print("Simple Table Parsing Demo")
    print("="*60)
    
    # 출력 디렉토리
    output_dir = Path('results/simple_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 테스트 이미지 생성
    print("\n[Step 1/3] Creating test images...")
    samples = create_test_table_images(output_dir / 'synthetic_data', num_samples=10)
    
    # 2. 간단한 분석
    print("\n[Step 2/3] Analyzing tables...")
    results = analyze_tables_simple(samples)
    
    # 결과 저장
    results_file = output_dir / 'results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_file}")
    
    # 3. 리포트 생성
    print("\n[Step 3/3] Generating report...")
    report_path = generate_simple_report(results, output_dir)
    
    # 요약 출력
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 HTML Report: {report_path}")
    print(f"🖼️  Images: {output_dir / 'synthetic_data'}")
    print(f"\n📈 Summary:")
    for table_type, data in results['summary']['by_type'].items():
        print(f"  - {table_type}: {data['count']} samples, avg {data['avg_cells']:.1f} cells")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

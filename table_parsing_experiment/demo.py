"""
Demo Script - 간단한 표 이미지로 파이프라인 테스트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image, ImageDraw, ImageFont
import json

from config import MODEL_CONFIG, METRICS_CONFIG
from experiments.pipeline import TableParsingPipeline


def create_sample_table_image(width=800, height=400):
    """간단한 테스트용 표 이미지 생성"""
    # 흰색 배경
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # 표 그리기 (3x3)
    rows, cols = 3, 3
    cell_width = width // cols
    cell_height = height // rows
    
    # 격자선 그리기
    for i in range(rows + 1):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill='black', width=2)
    
    for j in range(cols + 1):
        x = j * cell_width
        draw.line([(x, 0), (x, height)], fill='black', width=2)
    
    # 텍스트 추가
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # 헤더 행
    headers = ['Name', 'Age', 'City']
    for j, header in enumerate(headers):
        x = j * cell_width + 10
        y = 10
        draw.text((x, y), header, fill='black', font=font)
    
    # 데이터 행
    data = [
        ['Alice', '30', 'Seoul'],
        ['Bob', '25', 'Busan']
    ]
    
    for i, row_data in enumerate(data):
        for j, text in enumerate(row_data):
            x = j * cell_width + 10
            y = (i + 1) * cell_height + 10
            draw.text((x, y), text, fill='black', font=font)
    
    return img


def main():
    """메인 데모 함수"""
    print("=" * 60)
    print("Table Parsing Pipeline Demo")
    print("=" * 60)
    
    # 파이프라인 설정
    config = {
        'ocr': {
            'type': 'tesseract',
            **MODEL_CONFIG['ocr_engines']['tesseract']
        },
        'table_detection': {
            'type': 'huggingface',
            **MODEL_CONFIG['table_detection']
        },
        'table_structure': {
            'type': 'huggingface',
            **MODEL_CONFIG['table_structure']
        },
        'metrics': METRICS_CONFIG
    }
    
    print("\n1. Initializing Pipeline...")
    try:
        pipeline = TableParsingPipeline(config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nNote: This demo requires:")
        print("  - PyTorch and transformers installed")
        print("  - Tesseract OCR installed")
        print("  - Internet connection to download models")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        print("  sudo apt-get install tesseract-ocr")
        return
    
    print("\n2. Creating Sample Table Image...")
    sample_image = create_sample_table_image()
    
    # 이미지 저장
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    sample_image_path = output_dir / "sample_table.png"
    sample_image.save(sample_image_path)
    print(f"   Saved to: {sample_image_path}")
    
    print("\n3. Parsing Table...")
    results = pipeline.parse_image(sample_image, return_intermediates=True)
    
    print("\n4. Results:")
    print(f"   Success: {results['success']}")
    print(f"   Number of tables detected: {len(results.get('tables', []))}")
    
    if results.get('tables'):
        for idx, table in enumerate(results['tables']):
            print(f"\n   Table {idx + 1}:")
            print(f"     - Bounding Box: {table['bbox']}")
            print(f"     - Rows: {table['structure']['num_rows']}")
            print(f"     - Cols: {table['structure']['num_cols']}")
            print(f"     - Cells: {len(table['structure']['cells'])}")
            print(f"\n     HTML Output:")
            print(f"     {table['html'][:200]}...")  # 처음 200자만 출력
            
            # 실행 시간
            print(f"\n     Execution Time:")
            for step, time_val in table['execution_time'].items():
                print(f"       - {step}: {time_val:.3f}s")
    
    # 결과를 JSON으로 저장
    results_copy = results.copy()
    # PIL Image는 JSON으로 직렬화할 수 없으므로 제거
    for table in results_copy.get('tables', []):
        table.pop('table_image', None)
    
    results_path = output_dir / "demo_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_copy, f, indent=2, ensure_ascii=False)
    print(f"\n5. Full results saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

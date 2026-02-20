import os
import sys
from pathlib import Path
import json
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

# 프로젝트 루트를 경로에 추가
sys.path.append(os.getcwd())

from data.dataset_loader import get_dataset_loader
from models.table_structure import HuggingFaceTableStructureRecognizer

def visualize_cells(image, cells, output_path, title):
    """셀 경계 시각화"""
    draw_img = image.copy().convert('RGB')
    draw = ImageDraw.Draw(draw_img)
    
    for cell in cells:
        bbox = cell.get('bbox')
        if not bbox: continue
        
        # BoundingBox 객체이거나 dict일 수 있음
        if hasattr(bbox, 'x1'):
            coords = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
        else:
            coords = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
            
        draw.rectangle(coords, outline='red', width=2)
    
    draw_img.save(output_path)

def main():
    # 설정
    data_dir = "data"
    results_dir = Path("results/complexity_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    config = {
        'model_name': 'microsoft/table-transformer-structure-recognition',
        'threshold': 0.7
    }
    recognizer = HuggingFaceTableStructureRecognizer(config)
    recognizer.load_model()
    
    # 1. 샘플 선택 전략
    complexity_samples = {
        "Level 1 (Simple Grid)": [],
        "Level 2 (Normal)": [],
        "Level 3 (Medium - Merged)": [],
        "Level 4 (High - Real World)": []
    }
    
    print("Loading Level 1 samples (Synthetic)...")
    # 1-1. Level 1: Synthetic
    synthetic_loader = get_dataset_loader('synthetic', 'data/synthetic')
    synthetic_samples = synthetic_loader.generate_samples(num_samples=10, table_types=['fully_bordered'])
    complexity_samples["Level 1 (Simple Grid)"] = synthetic_samples[:2]
    
    print("Loading Level 2 & 3 samples (SciTSR)...")
    # 1-2. Level 2 & 3: SciTSR
    scitsr_loader = get_dataset_loader('scitsr', 'data/SciTSR')
    scitsr_all = scitsr_loader.load_samples(split='test', max_samples=50) # Reduced to 50 for speed
    
    normal_samples = [s for s in scitsr_all if s.metadata.get('complexity') == 'normal']
    complex_samples = [s for s in scitsr_all if s.metadata.get('complexity') == 'complex']
    
    complexity_samples["Level 2 (Normal)"] = normal_samples[:1] # Take 1 to save time
    complexity_samples["Level 3 (Medium - Merged)"] = complex_samples[:1]
    
    print("Loading Level 4 samples (DocLayNet via streaming)...")
    # 1-3. Level 4: DocLayNet (Custom streaming load to avoid large download)
    from datasets import load_dataset
    from data.dataset_loader import DatasetSample
    try:
        ds = load_dataset('docling-project/DocLayNet-v1.2', split='test', streaming=True)
        count = 0
        for item in ds:
            if count >= 1: break
            if 10 in item['category_id']:
                # Save temp image
                import tempfile
                temp_dir = Path(tempfile.gettempdir()) / "doclaynet_analysis_cache"
                temp_dir.mkdir(exist_ok=True)
                img_path = temp_dir / f"level4_sample_{count}.png"
                item['image'].save(img_path)
                
                sample = DatasetSample(
                    sample_id=f"doclaynet_stream_{count}",
                    image_path=str(img_path),
                    ground_truth={'tables': []}, # Minimal for analysis
                    metadata={'source': 'DocLayNet', 'complexity': 'high'}
                )
                complexity_samples["Level 4 (High - Real World)"].append(sample)
                count += 1
    except Exception as e:
        print(f"Error loading DocLayNet: {e}")
    
    # 2. TSR 실행 및 결과 분석
    analysis_results = []
    
    for level, samples in complexity_samples.items():
        print(f"\nAnalyzing {level}...")
        for i, sample in enumerate(samples):
            img = sample.load_image()
            
            # TSR 수행
            # recognizer.predict는 TableStructure 객체를 반환
            # TableStructure.cells는 Cell 객체 리스트
            structure = recognizer.predict(img)
            
            # 시각화 저장
            output_name = f"level_{level[6]}_sample_{i}.png"
            output_path = results_dir / output_name
            
            # Cell 객체를 dict 스타일로 변환하여 시각화 함수에 전달
            viz_cells = []
            for cell in structure.cells:
                viz_cells.append({'bbox': cell.bbox})
            
            visualize_cells(img, viz_cells, output_path, level)
            
            # 결과 기록
            analysis_results.append({
                'level': level,
                'sample_id': sample.sample_id,
                'image_path': str(output_path),
                'num_cells_detected': len(structure.cells)
            })
            
    # 결과 저장
    with open(results_dir / "analysis_report.json", "w", encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main()

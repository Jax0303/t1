#!/usr/bin/env python3
"""
HiTab 데이터셋으로부터 실험용 데이터셋 생성
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def create_hitab_dataset(
    hitab_data_dir: str,
    output_path: str,
    max_samples: int = 50
):
    """HiTab 데이터로부터 실험용 데이터셋 생성"""
    
    hitab_path = Path(hitab_data_dir)
    data_dir = hitab_path / "data"
    
    # JSONL 파일 로드
    train_file = data_dir / "train_samples.jsonl"
    dev_file = data_dir / "dev_samples.jsonl"
    test_file = data_dir / "test_samples.jsonl"
    
    # 테이블 파일 로드
    tables_dir = data_dir / "tables" / "raw"
    table_files = {}
    
    if tables_dir.exists():
        for table_file in list(tables_dir.glob("*.json"))[:1000]:  # 처음 1000개만
            try:
                with open(table_file, 'r', encoding='utf-8') as f:
                    table_data = json.load(f)
                    table_id = table_file.stem
                    table_files[table_id] = table_data
            except Exception as e:
                print(f"Error loading {table_file}: {e}")
                continue
    
    print(f"Loaded {len(table_files)} table files")
    
    # 샘플 로드
    qa_pairs = []
    tables_dict = {}
    
    # Train 데이터에서 샘플 추출
    sample_count = 0
    for jsonl_file in [train_file, dev_file, test_file]:
        if not jsonl_file.exists():
            continue
        
        split_name = jsonl_file.stem.replace('_samples', '')
        print(f"Processing {split_name}...")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if sample_count >= max_samples:
                    break
                
                if not line.strip():
                    continue
                
                try:
                    item = json.loads(line)
                    table_id = item.get('table_id', '')
                    
                    # 테이블 데이터 가져오기
                    table_info = table_files.get(table_id)
                    if not table_info:
                        continue
                    
                    # 테이블을 딕셔너리에 저장
                    if table_id not in tables_dict:
                        tables_dict[table_id] = {
                            'table_id': table_id,
                            'data': table_info,
                            'metadata': {
                                'table_source': item.get('table_source', ''),
                            }
                        }
                    
                    # QA 쌍 생성
                    qa_pairs.append({
                        'question': item.get('question', ''),
                        'answer': item.get('answer', []),
                        'required_cells': _extract_required_cells(item),
                        'question_type': 'factual',
                        'metadata': {
                            'table_id': table_id,
                            'id': item.get('id', ''),
                            'sub_sentence': item.get('sub_sentence', ''),
                            'split': split_name,
                            'linked_cells': item.get('linked_cells', {}),
                            'aggregation': item.get('aggregation', []),
                        }
                    })
                    
                    sample_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
        
        if sample_count >= max_samples:
            break
    
    # 데이터셋 생성
    dataset = {
        'qa_pairs': qa_pairs,
        'tables': tables_dict,
        'metadata': {
            'source': 'hitab',
            'total_tables': len(tables_dict),
            'total_qa_pairs': len(qa_pairs)
        }
    }
    
    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n데이터셋 생성 완료: {output_path}")
    print(f"  표 개수: {len(tables_dict)}")
    print(f"  QA 쌍 개수: {len(qa_pairs)}")
    
    return dataset

def _extract_required_cells(item: dict) -> list:
    """linked_cells에서 required_cells 추출"""
    required_cells = []
    linked_cells = item.get('linked_cells', {})
    
    # quantity_link의 [ANSWER] 셀 찾기
    quantity_link = linked_cells.get('quantity_link', {})
    answer_cells = quantity_link.get('[ANSWER]', {})
    
    for coord, value in answer_cells.items():
        # coord 형식: "(row, col)"
        if coord.startswith('(') and coord.endswith(')'):
            coord_str = coord[1:-1]
            parts = coord_str.split(',')
            if len(parts) == 2:
                try:
                    row = int(parts[0].strip())
                    col = int(parts[1].strip())
                    required_cells.append({
                        'row': row,
                        'col': col,
                        'format': 'tuple',
                        'original_text': str(value)
                    })
                except ValueError:
                    continue
    
    return required_cells

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HiTab 실험용 데이터셋 생성")
    parser.add_argument("--input", type=str, default="data/collected/hitab/HiTab",
                       help="HiTab 데이터 디렉토리")
    parser.add_argument("--output", type=str, default="data/hitab_experiment_dataset.json",
                       help="출력 데이터셋 경로")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="최대 샘플 개수")
    
    args = parser.parse_args()
    
    create_hitab_dataset(
        args.input,
        args.output,
        args.max_samples
    )


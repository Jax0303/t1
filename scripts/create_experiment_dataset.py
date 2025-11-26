#!/usr/bin/env python3
"""
추출된 표 데이터로부터 실험용 데이터셋 생성
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def create_dataset_from_extracted_tables(
    extracted_tables_path: str,
    output_path: str,
    num_qa_pairs: int = 10
):
    """추출된 표 데이터로부터 실험용 데이터셋 생성"""
    
    # 추출된 표 로드
    with open(extracted_tables_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    # 표를 딕셔너리로 변환 (table_id를 키로)
    tables_dict = {}
    qa_pairs = []
    
    for idx, table in enumerate(tables_data[:50]):  # 처음 50개만 사용
        table_id = table.get('table_id', f'table_{idx}')
        
        # 표 데이터 변환
        table_obj = {
            'table_id': table_id,
            'rows': table.get('rows', []),
            'row_count': table.get('row_count', 0),
            'col_count': table.get('col_count', 0),
            'source_file': table.get('source_file', ''),
            'metadata': {
                'corp_name': table.get('corp_name', ''),
                'report_nm': table.get('report_nm', ''),
            }
        }
        tables_dict[table_id] = table_obj
        
        # 간단한 QA 쌍 생성 (예시)
        if idx < num_qa_pairs:
            # 표의 첫 번째 행에서 질문 생성
            rows = table.get('rows', [])
            if rows and len(rows) > 0:
                first_row = rows[0]
                if isinstance(first_row, list) and len(first_row) > 0:
                    # 첫 번째 셀의 텍스트를 사용
                    first_cell_text = ""
                    if isinstance(first_row[0], dict):
                        first_cell_text = first_row[0].get('text', '')
                    elif isinstance(first_row[0], str):
                        first_cell_text = first_row[0]
                    
                    if first_cell_text and first_cell_text != "None":
                        qa_pairs.append({
                            'question': f"{first_cell_text}에 대한 정보를 알려주세요.",
                            'answer': f"{first_cell_text} 관련 정보입니다.",
                            'required_cells': [
                                {
                                    'row': 0,
                                    'col': 0,
                                    'format': 'tuple',
                                    'original_text': first_cell_text
                                }
                            ],
                            'question_type': 'factual',
                            'metadata': {
                                'table_id': table_id,
                                'source': 'dart'
                            }
                        })
    
    # 데이터셋 생성
    dataset = {
        'qa_pairs': qa_pairs,
        'tables': tables_dict,
        'metadata': {
            'source': 'dart',
            'total_tables': len(tables_dict),
            'total_qa_pairs': len(qa_pairs)
        }
    }
    
    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"데이터셋 생성 완료: {output_path}")
    print(f"  표 개수: {len(tables_dict)}")
    print(f"  QA 쌍 개수: {len(qa_pairs)}")
    
    return dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="실험용 데이터셋 생성")
    parser.add_argument("--input", type=str, default="data/dart_extracted_tables.json",
                       help="추출된 표 데이터 경로")
    parser.add_argument("--output", type=str, default="data/experiment_dataset.json",
                       help="출력 데이터셋 경로")
    parser.add_argument("--num-qa", type=int, default=10,
                       help="생성할 QA 쌍 개수")
    
    args = parser.parse_args()
    
    create_dataset_from_extracted_tables(
        args.input,
        args.output,
        args.num_qa
    )


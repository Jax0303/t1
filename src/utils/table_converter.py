"""
표 데이터 변환 유틸리티
다양한 형식의 표 데이터를 통일된 형식으로 변환
"""
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd


def convert_hwp5_extractor_json_to_rag_format(json_file: str) -> List[Dict]:
    """
    hwp5-table-extractor로 추출한 JSON을 RAG 시스템 형식으로 변환
    
    Args:
        json_file: JSON 파일 경로
        
    Returns:
        RAG 시스템 형식의 표 리스트
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tables = []
    
    for idx, table_data in enumerate(data):
        try:
            # rows 데이터가 이미 있으면 그대로 사용
            if 'rows' in table_data:
                table_info = {
                    'table_id': f"hwp5_table_{idx}",
                    'rows': table_data['rows'],
                    'row_count': table_data.get('row_count', len(table_data.get('rows', []))),
                    'col_count': table_data.get('col_count', 0),
                    'source_file': json_file,
                    'extraction_method': 'hwp5-table-extractor',
                    'caption': table_data.get('caption')
                }
                tables.append(table_info)
            else:
                # DataFrame 형식으로 변환 필요
                rows = []
                for row_data in table_data.get('rows', []):
                    row_values = []
                    for cell in row_data:
                        cell_text = cell.get('text', '')
                        if not cell_text and cell.get('lines'):
                            cell_text = '\n'.join(cell.get('lines', []))
                        row_values.append(cell_text)
                    
                    if row_values:
                        rows.append(row_values)
                
                if not rows:
                    continue
                
                max_cols = max(len(row) for row in rows) if rows else 0
                if max_cols == 0:
                    continue
                
                for row in rows:
                    while len(row) < max_cols:
                        row.append("")
                
                if rows:
                    headers = rows[0] if len(rows) > 1 else [f"열{i+1}" for i in range(max_cols)]
                    data_rows = rows[1:] if len(rows) > 1 else []
                    
                    if not headers or all(not str(h).strip() for h in headers):
                        headers = [f"열{i+1}" for i in range(max_cols)]
                    
                    if data_rows:
                        df = pd.DataFrame(data_rows, columns=headers[:max_cols])
                    else:
                        df = pd.DataFrame(columns=headers[:max_cols])
                    
                    table_info = {
                        'table_id': f"hwp5_table_{idx}",
                        'dataframe': df,
                        'source_file': json_file,
                        'extraction_method': 'hwp5-table-extractor',
                        'row_count': table_data.get('row_count', len(df)),
                        'col_count': table_data.get('col_count', len(df.columns))
                    }
                    
                    tables.append(table_info)
        except Exception as e:
            print(f"  경고: 표 {idx} 변환 실패: {e}")
            continue
    
    return tables


def dataframe_to_table_data(df: pd.DataFrame, table_id: str) -> Dict:
    """
    DataFrame을 표 데이터 형식으로 변환
    
    Args:
        df: DataFrame
        table_id: 표 식별자
        
    Returns:
        표 데이터 딕셔너리
    """
    rows = []
    
    # 헤더 행 추가
    header_row = []
    for col_idx, col_name in enumerate(df.columns):
        header_row.append({
            'row': 0,
            'col': col_idx,
            'row_span': 1,
            'col_span': 1,
            'text': str(col_name),
            'lines': [str(col_name)]
        })
    rows.append(header_row)
    
    # 데이터 행 추가
    for row_idx, (_, row_data) in enumerate(df.iterrows(), start=1):
        data_row = []
        for col_idx, value in enumerate(row_data):
            data_row.append({
                'row': row_idx,
                'col': col_idx,
                'row_span': 1,
                'col_span': 1,
                'text': str(value),
                'lines': [str(value)]
            })
        rows.append(data_row)
    
    return {
        'rows': rows,
        'row_count': len(rows),
        'col_count': len(df.columns)
    }




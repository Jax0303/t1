#!/usr/bin/env python3
"""
HWP5 파일에서 표를 추출하는 테스트 스크립트
pyhwp 라이브러리를 사용하여 HWP5 파일 파싱
"""
import sys
from pathlib import Path
import pandas as pd
import json

def extract_tables_with_pyhwp(hwp_path: str):
    """pyhwp를 사용하여 HWP 파일에서 표 추출"""
    try:
        from pyhwp import hwp5
        from pyhwp.hwp5 import plat
        
        print(f"HWP 파일 열기: {hwp_path}")
        hwp5file = hwp5.open(hwp_path)
        
        tables = []
        table_idx = 0
        
        # 본문 스트림 찾기
        if not hasattr(hwp5file, 'bodytext') or hwp5file.bodytext is None:
            print("경고: bodytext를 찾을 수 없습니다.")
            return tables
        
        bodytext = hwp5file.bodytext
        print(f"섹션 개수: {len(bodytext.sections)}")
        
        # 섹션별로 처리
        for section_idx, section in enumerate(bodytext.sections):
            print(f"\n섹션 {section_idx} 처리 중...")
            
            # 레코드 순회
            for record_idx, record in enumerate(section.records):
                # 표 관련 레코드 찾기
                if hasattr(record, 'tagname'):
                    tagname = record.tagname
                    
                    # 표 관련 태그 확인
                    if 'TABLE' in tagname.upper() or 'TBL' in tagname.upper():
                        print(f"  표 레코드 발견: {tagname} (레코드 {record_idx})")
                        
                        try:
                            table_data = parse_table_record(record)
                            if table_data is not None and not table_data.empty:
                                tables.append({
                                    'table_id': f"table_{table_idx}",
                                    'dataframe': table_data,
                                    'section': section_idx,
                                    'record_idx': record_idx,
                                    'tagname': tagname
                                })
                                print(f"    표 추출 성공: {len(table_data)}행 x {len(table_data.columns)}열")
                                table_idx += 1
                        except Exception as e:
                            print(f"    표 파싱 오류: {e}")
                            import traceback
                            traceback.print_exc()
        
        return tables
    
    except ImportError as e:
        print(f"오류: pyhwp 라이브러리를 찾을 수 없습니다: {e}")
        return []
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
        return []

def parse_table_record(record):
    """표 레코드를 DataFrame으로 변환"""
    try:
        rows = []
        
        # 레코드 구조 확인
        print(f"      레코드 타입: {type(record)}")
        print(f"      레코드 속성: {dir(record)}")
        
        # 다양한 방법으로 표 데이터 추출 시도
        if hasattr(record, 'rows'):
            for row in record.rows:
                cells = []
                if hasattr(row, 'cells'):
                    for cell in row.cells:
                        cell_text = extract_cell_text(cell)
                        cells.append(cell_text)
                if cells:
                    rows.append(cells)
        
        elif hasattr(record, 'children'):
            # 자식 요소에서 표 구조 찾기
            for child in record.children:
                if hasattr(child, 'tagname') and 'ROW' in child.tagname.upper():
                    cells = []
                    if hasattr(child, 'cells'):
                        for cell in child.cells:
                            cells.append(extract_cell_text(cell))
                    if cells:
                        rows.append(cells)
        
        elif hasattr(record, 'data'):
            # 바이너리 데이터에서 직접 파싱 시도
            data = record.data
            if isinstance(data, bytes):
                # 간단한 텍스트 추출 시도
                text = data.decode('utf-16-le', errors='ignore')
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                for line in lines:
                    if '\t' in line:
                        cells = line.split('\t')
                        rows.append(cells)
        
        if rows:
            # 행 길이 맞추기
            max_cols = max(len(row) for row in rows) if rows else 0
            for row in rows:
                while len(row) < max_cols:
                    row.append("")
            
            return pd.DataFrame(rows)
        
        return None
    
    except Exception as e:
        print(f"      레코드 파싱 오류: {e}")
        return None

def extract_cell_text(cell):
    """셀에서 텍스트 추출"""
    texts = []
    
    try:
        # 다양한 방법으로 텍스트 추출 시도
        if hasattr(cell, 'paragraphs'):
            for para in cell.paragraphs:
                if hasattr(para, 'chars'):
                    for char in para.chars:
                        if hasattr(char, 'text'):
                            texts.append(char.text)
                        elif hasattr(char, 'char'):
                            texts.append(str(char.char))
        
        elif hasattr(cell, 'text'):
            texts.append(str(cell.text))
        
        elif hasattr(cell, 'content'):
            texts.append(str(cell.content))
        
        # 정규화
        text = " ".join(str(t) for t in texts if t)
        return text.strip()
    
    except Exception as e:
        return ""

def main():
    """메인 함수"""
    # HWP 파일 경로
    hwp_file = Path("data/raw/dataset2/개정 표준취업규칙(2025년, 배포).hwp")
    
    if not hwp_file.exists():
        print(f"오류: HWP 파일을 찾을 수 없습니다: {hwp_file}")
        return
    
    print("=" * 60)
    print("HWP5 표 추출 테스트")
    print("=" * 60)
    
    # 표 추출
    tables = extract_tables_with_pyhwp(str(hwp_file))
    
    print("\n" + "=" * 60)
    print(f"추출 결과: {len(tables)}개 표 발견")
    print("=" * 60)
    
    # 결과 출력
    for idx, table_info in enumerate(tables):
        print(f"\n표 {idx + 1}:")
        print(f"  ID: {table_info['table_id']}")
        print(f"  섹션: {table_info['section']}")
        print(f"  레코드: {table_info['record_idx']}")
        print(f"  태그명: {table_info['tagname']}")
        print(f"  크기: {len(table_info['dataframe'])}행 x {len(table_info['dataframe'].columns)}열")
        print("\n  데이터 미리보기:")
        print(table_info['dataframe'].head().to_string())
        print("-" * 60)
    
    # JSON으로 저장
    output_file = Path("extracted_tables_hwp5_pyhwp.json")
    output_data = []
    
    for table_info in tables:
        df = table_info['dataframe']
        output_data.append({
            'table_id': table_info['table_id'],
            'section': table_info['section'],
            'record_idx': table_info['record_idx'],
            'tagname': table_info['tagname'],
            'rows': len(df),
            'cols': len(df.columns),
            'data': df.to_dict('records'),
            'columns': list(df.columns)
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()


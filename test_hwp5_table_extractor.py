#!/usr/bin/env python3
"""
hwp5-table-extractor를 사용하여 HWP 파일에서 표를 추출하는 테스트 스크립트
"""
import sys
from pathlib import Path
import json

# hwp5-table-extractor 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'hwp5-table-extractor'))

from hwp5_table import HwpFile

def main():
    """메인 함수"""
    # HWP 파일 경로
    hwp_file = Path("data/raw/dataset2/개정 표준취업규칙(2025년, 배포).hwp")
    
    if not hwp_file.exists():
        print(f"오류: HWP 파일을 찾을 수 없습니다: {hwp_file}")
        return
    
    print("=" * 60)
    print("hwp5-table-extractor를 사용한 HWP 표 추출 테스트")
    print("=" * 60)
    print(f"파일: {hwp_file}")
    print()
    
    # HWP 파일 열기
    print("HWP 파일 열기 중...")
    try:
        with open(hwp_file, 'rb') as f:
            hwp = HwpFile(f)
            
            print(f"압축 여부: {hwp.compressed}")
            print()
            
            # 모든 섹션에서 표 추출
            tables = []
            section_idx = 0
            
            while hwp.ole.exists('BodyText/Section%d' % section_idx):
                print(f"섹션 {section_idx} 처리 중...")
                try:
                    section_tables = hwp.get_tables(section_idx)
                    print(f"  → {len(section_tables)}개 표 발견")
                    tables.extend(section_tables)
                except Exception as e:
                    print(f"  → 오류 발생 (무시하고 계속): {e}")
                section_idx += 1
            
            print()
            print("=" * 60)
            print(f"총 {len(tables)}개 표 추출 완료")
            print("=" * 60)
            
            # 결과를 JSON으로 저장
            results = []
            for idx, table in enumerate(tables):
                table_data = {
                    'table_index': idx,
                    'row_count': table.row_cnt,
                    'col_count': table.col_cnt,
                    'caption': table.caption,
                    'rows': []
                }
                
                for row_idx, row in enumerate(table.rows):
                    row_data = []
                    for cell in row:
                        cell_data = {
                            'row': cell.row,
                            'col': cell.col,
                            'row_span': cell.row_span,
                            'col_span': cell.col_span,
                            'text': '\n'.join(cell.lines),
                            'lines': cell.lines
                        }
                        row_data.append(cell_data)
                    table_data['rows'].append(row_data)
                
                results.append(table_data)
            
            # JSON 파일로 저장
            output_file = Path("extracted_tables_hwp5_extractor.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n결과가 {output_file}에 저장되었습니다.")
            
            # 첫 번째 표의 일부 내용 출력
            if tables:
                print("\n첫 번째 표 미리보기:")
                print("-" * 60)
                first_table = tables[0]
                print(f"행 수: {first_table.row_cnt}, 열 수: {first_table.col_cnt}")
                print("\n처음 3행의 내용:")
                for i, row in enumerate(first_table.rows[:3]):
                    print(f"\n행 {i}:")
                    for cell in row:
                        text = ' '.join(cell.lines) if cell.lines else '(빈 셀)'
                        print(f"  [{cell.row},{cell.col}] (rowspan={cell.row_span}, colspan={cell.col_span}): {text[:50]}")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
hwp5-table-extractor를 개선하여 모든 표를 추출하는 테스트 스크립트
오류 처리 강화 및 디버깅 정보 추가
"""
import sys
from pathlib import Path
import json

# hwp5-table-extractor 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'hwp5-table-extractor'))

from hwp5_table import HwpFile, Record
import struct

def safe_get_tables(hwp, section_idx):
    """안전하게 표를 추출하는 함수 (오류 처리 강화)"""
    try:
        record_tree_root = hwp.get_record_tree(section_idx)
        tables = make_tables_safe(record_tree_root)
        return tables
    except Exception as e:
        print(f"    오류 상세: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []

def make_tables_safe(record_tree_root):
    """make_tables 함수의 안전한 버전 (오류 처리 강화)"""
    def traverse(record, depth=0):
        try:
            if record.tag_name == 'HWPTAG_TABLE':
                if 'current_table_idx' not in ctx:
                    ctx['current_table_idx'] = 0
                else:
                    ctx['current_table_idx'] += 1

                try:
                    row_cnt = struct.unpack('<H', record.payload[4:6])[0]
                    col_cnt = struct.unpack('<H', record.payload[6:8])[0]
                    ctx['tables'].append(Table(None, row_cnt, col_cnt))
                    ctx['table_info'][ctx['current_table_idx']] = {'row_cnt': row_cnt, 'col_cnt': col_cnt}
                except Exception as e:
                    print(f"      표 헤더 파싱 오류: {e}")
                    # 빈 표라도 추가
                    ctx['tables'].append(Table(None, 0, 0))
                    ctx['table_info'][ctx['current_table_idx']] = {'row_cnt': 0, 'col_cnt': 0}
                    
            elif (record.tag_name == 'HWPTAG_LIST_HEADER'
                  and record.parent.tag_name == 'HWPTAG_CTRL_HEADER'
                  and record.parent.payload[:4][::-1] == b'tbl '):
                try:
                    if 'current_table_idx' not in ctx or ctx['current_table_idx'] < 0:
                        return
                    
                    paragraph_count = struct.unpack('<H', record.payload[:2])[0]
                    col = struct.unpack('<H', record.payload[8:10])[0]
                    row = struct.unpack('<H', record.payload[10:12])[0]
                    col_span = struct.unpack('<H', record.payload[12:14])[0]
                    row_span = struct.unpack('<H', record.payload[14:16])[0]

                    # 현재 표의 행 수 확인
                    if ctx['current_table_idx'] >= len(ctx['tables']):
                        print(f"      경고: current_table_idx({ctx['current_table_idx']})가 표 개수({len(ctx['tables'])})를 초과")
                        return
                    
                    current_table = ctx['tables'][ctx['current_table_idx']]
                    
                    # 행 인덱스 범위 체크
                    if row >= len(current_table.rows):
                        print(f"      경고: row({row})가 표의 행 수({len(current_table.rows)})를 초과. 표 확장 중...")
                        # 행 수 확장
                        while len(current_table.rows) <= row:
                            current_table.rows.append([])
                        current_table.row_cnt = len(current_table.rows)

                    lines = []
                    for sibling in record.get_next_siblings(paragraph_count):
                        for child in sibling.children:
                            if child.tag_name == 'HWPTAG_PARA_TEXT':
                                lines.extend(child.get_text().strip().splitlines())
                                break

                    current_table.rows[row].append(
                        TableCell(lines, row, col, row_span, col_span)
                    )
                except Exception as e:
                    print(f"      셀 파싱 오류: {e}")
                    # 오류가 발생해도 계속 진행

            for child in record.children:
                traverse(child, depth + 1)
        except Exception as e:
            print(f"      traverse 오류: {e}")

    from hwp5_table import Table, TableCell
    
    ctx = {'tables': [], 'table_info': {}}
    traverse(record_tree_root)
    
    return ctx['tables']

def main():
    """메인 함수"""
    # HWP 파일 경로
    hwp_file = Path("data/raw/dataset2/개정 표준취업규칙(2025년, 배포).hwp")
    
    if not hwp_file.exists():
        print(f"오류: HWP 파일을 찾을 수 없습니다: {hwp_file}")
        return
    
    print("=" * 60)
    print("hwp5-table-extractor 개선 버전 - 모든 표 추출")
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
                section_tables = safe_get_tables(hwp, section_idx)
                print(f"  → {len(section_tables)}개 표 발견")
                tables.extend(section_tables)
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
            output_file = Path("extracted_tables_hwp5_extractor_improved.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n결과가 {output_file}에 저장되었습니다.")
            
            # 표 크기 통계
            print("\n표 크기 통계:")
            row_counts = [t.row_cnt for t in tables]
            col_counts = [t.col_cnt for t in tables]
            print(f"  행 수: 최소={min(row_counts) if row_counts else 0}, 최대={max(row_counts) if row_counts else 0}, 평균={sum(row_counts)/len(row_counts) if row_counts else 0:.1f}")
            print(f"  열 수: 최소={min(col_counts) if col_counts else 0}, 최대={max(col_counts) if col_counts else 0}, 평균={sum(col_counts)/len(col_counts) if col_counts else 0:.1f}")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()


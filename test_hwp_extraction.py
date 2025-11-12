#!/usr/bin/env python3
"""
HWP 파일에서 표를 추출하는 테스트 스크립트
기존 HWPTableExtractor 사용
"""
import sys
from pathlib import Path
import json

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.extractors.hwp_extractor import HWPTableExtractor

def main():
    """메인 함수"""
    # HWP 파일 경로
    hwp_file = Path("data/raw/dataset2/개정 표준취업규칙(2025년, 배포).hwp")
    
    if not hwp_file.exists():
        print(f"오류: HWP 파일을 찾을 수 없습니다: {hwp_file}")
        return
    
    print("=" * 60)
    print("HWP 표 추출 테스트")
    print("=" * 60)
    print(f"파일: {hwp_file}")
    print()
    
    # 추출기 생성
    extractor = HWPTableExtractor(use_direct_parsing=True)
    
    # 표 추출
    print("표 추출 중...")
    tables = extractor.extract_tables(str(hwp_file), method='auto')
    
    print("\n" + "=" * 60)
    print(f"추출 결과: {len(tables)}개 표 발견")
    print("=" * 60)
    
    # 결과 출력
    for idx, table_info in enumerate(tables):
        print(f"\n표 {idx + 1}:")
        print(f"  ID: {table_info.get('table_id', 'N/A')}")
        print(f"  추출 방법: {table_info.get('extraction_method', 'N/A')}")
        df = table_info.get('dataframe')
        if df is not None:
            print(f"  크기: {len(df)}행 x {len(df.columns)}열")
            print("\n  데이터 미리보기:")
            print(df.head(10).to_string())
        else:
            print("  경고: DataFrame이 없습니다.")
        print("-" * 60)
    
    # JSON으로 저장
    output_file = Path("extracted_tables_hwp_test.json")
    output_data = []
    
    for table_info in tables:
        df = table_info.get('dataframe')
        if df is not None:
            output_data.append({
                'table_id': table_info.get('table_id', ''),
                'extraction_method': table_info.get('extraction_method', ''),
                'source_file': table_info.get('source_file', ''),
                'rows': len(df),
                'cols': len(df.columns),
                'data': df.to_dict('records'),
                'columns': list(df.columns)
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 {output_file}에 저장되었습니다.")
    print(f"총 {len(output_data)}개 표가 추출되었습니다.")

if __name__ == "__main__":
    main()


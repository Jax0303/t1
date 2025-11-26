#!/usr/bin/env python3
"""
단일 PDF 파일 표 추출 실험 스크립트
복잡한 레이아웃 문제가 있는 파일 테스트용
"""
import sys
import json
from pathlib import Path
from typing import List, Dict
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.pdf_extractor import PDFTableExtractor
from src.utils.table_converter import dataframe_to_table_data


def analyze_table_quality(tables: List[Dict]) -> Dict:
    """표 품질 분석"""
    stats = {
        'total_tables': len(tables),
        'empty_tables': 0,
        'single_row_tables': 0,
        'high_none_ratio_tables': 0,
        'tables_by_page': {},
        'none_ratio_details': []
    }
    
    for table in tables:
        df = table.get('dataframe')
        if df is None or df.empty:
            stats['empty_tables'] += 1
            continue
        
        # 1행 표 체크
        if len(df) == 1:
            stats['single_row_tables'] += 1
        
        # None 비율 계산
        total_cells = len(df) * len(df.columns)
        none_count = df.isna().sum().sum()
        none_ratio = none_count / total_cells if total_cells > 0 else 0
        
        if none_ratio >= 0.3:  # 30% 이상 None
            stats['high_none_ratio_tables'] += 1
            stats['none_ratio_details'].append({
                'table_id': table.get('table_id', ''),
                'page': table.get('page_number', 0),
                'none_ratio': round(none_ratio, 3),
                'shape': f"{len(df)}x{len(df.columns)}"
            })
        
        # 페이지별 통계
        page = table.get('page_number', 0)
        if page not in stats['tables_by_page']:
            stats['tables_by_page'][page] = 0
        stats['tables_by_page'][page] += 1
    
    return stats


def extract_single_pdf(
    pdf_path: str,
    method: str = "pdfplumber",
    output_path: str = None
) -> Dict:
    """
    단일 PDF 파일에서 표 추출 및 분석
    
    Args:
        pdf_path: PDF 파일 경로
        method: 추출 방법
        output_path: 출력 JSON 파일 경로 (None이면 자동 생성)
        
    Returns:
        추출 결과 및 통계
    """
    print("=" * 80)
    print(f"단일 PDF 표 추출 실험")
    print("=" * 80)
    print(f"파일: {pdf_path}")
    print(f"방법: {method}")
    print("=" * 80)
    
    # 파일 존재 확인
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {pdf_path}")
        return {}
    
    # PDF 추출기 초기화
    print(f"\nPDF 추출기 초기화: {method}")
    extractor = PDFTableExtractor(method=method)
    
    # 표 추출
    print("\n표 추출 시작...")
    start_time = time.time()
    tables = extractor.extract_tables(pdf_path)
    extraction_time = time.time() - start_time
    
    print(f"\n추출 완료! (소요 시간: {extraction_time:.2f}초)")
    print(f"  추출된 표: {len(tables)}개")
    
    # 품질 분석
    print("\n표 품질 분석 중...")
    quality_stats = analyze_table_quality(tables)
    
    print(f"\n품질 통계:")
    print(f"  총 표: {quality_stats['total_tables']}개")
    print(f"  빈 표: {quality_stats['empty_tables']}개")
    print(f"  1행 표: {quality_stats['single_row_tables']}개")
    print(f"  None 비율 30% 이상: {quality_stats['high_none_ratio_tables']}개")
    
    if quality_stats['none_ratio_details']:
        print(f"\nNone 비율이 높은 표 (상위 10개):")
        for detail in sorted(quality_stats['none_ratio_details'], 
                           key=lambda x: x['none_ratio'], reverse=True)[:10]:
            print(f"  - {detail['table_id']} (페이지 {detail['page']}, {detail['shape']}): "
                  f"None 비율 {detail['none_ratio']*100:.1f}%")
    
    # RAG 형식으로 변환
    print("\nRAG 형식으로 변환 중...")
    rag_tables = []
    for table_info in tables:
        df = table_info.get('dataframe')
        if df is not None and not df.empty:
            table_data = dataframe_to_table_data(df, table_info.get('table_id', ''))
            rag_table = {
                **table_data,
                'table_id': table_info.get('table_id', ''),
                'source_file': table_info.get('source_file', pdf_path),
                'page_number': table_info.get('page_number', 0),
                'extraction_method': table_info.get('extraction_method', method),
                'extraction_time': table_info.get('extraction_time', extraction_time)
            }
            rag_tables.append(rag_table)
    
    # 결과 저장
    if output_path is None:
        output_path = f"extracted_{Path(pdf_path).stem}_{method}.json"
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'source_file': pdf_path,
        'extraction_method': method,
        'extraction_time': extraction_time,
        'total_tables': len(rag_tables),
        'quality_stats': quality_stats,
        'tables': rag_tables
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n결과 저장: {output_path}")
    print(f"  파일 크기: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    return result


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='단일 PDF 파일 표 추출 실험')
    parser.add_argument(
        '--pdf',
        type=str,
        default='data/dart_pdfs/DMS_[기재정정]사업보고서 (2023.12)_20250321000022.pdf',
        help='PDF 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 JSON 파일 경로 (기본값: 자동 생성)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='pdfplumber',
        choices=['pdfplumber', 'camelot', 'tabula', 'pymupdf', 'pypdf', 
                'docling', 'unstructured', 'marker', 'nougat'],
        help='추출 방법 (기본값: pdfplumber)'
    )
    
    args = parser.parse_args()
    
    result = extract_single_pdf(
        pdf_path=args.pdf,
        method=args.method,
        output_path=args.output
    )
    
    return result


if __name__ == "__main__":
    main()


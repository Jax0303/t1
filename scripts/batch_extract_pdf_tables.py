#!/usr/bin/env python3
"""
DART PDF 파일에서 일괄 표 추출 스크립트
다운로드된 모든 PDF 파일에서 표를 추출하여 JSON 파일로 저장
"""
import sys
import json
from pathlib import Path
from typing import List, Dict

# 선택적 import
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # tqdm 없을 때는 단순 반복자 사용
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(desc)
        return iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.pdf_extractor import PDFTableExtractor
from src.utils.table_converter import dataframe_to_table_data


def extract_tables_from_pdf(pdf_path: str, extractor: PDFTableExtractor) -> List[Dict]:
    """
    단일 PDF 파일에서 표 추출
    
    Args:
        pdf_path: PDF 파일 경로
        extractor: PDF 추출기 인스턴스
        
    Returns:
        추출된 표 리스트 (RAG 형식)
    """
    try:
        # 표 추출
        tables = extractor.extract_tables(pdf_path)
        
        # RAG 형식으로 변환
        rag_tables = []
        for table_info in tables:
            df = table_info.get('dataframe')
            if df is not None and not df.empty:
                # DataFrame을 table_data 형식으로 변환
                table_data = dataframe_to_table_data(df, table_info.get('table_id', ''))
                
                # 메타데이터 추가
                rag_table = {
                    **table_data,
                    'table_id': table_info.get('table_id', ''),
                    'source_file': table_info.get('source_file', pdf_path),
                    'page_number': table_info.get('page_number', 0),
                    'extraction_method': table_info.get('extraction_method', 'pdfplumber'),
                    'extraction_time': table_info.get('extraction_time', 0)
                }
                rag_tables.append(rag_table)
        
        return rag_tables
    
    except Exception as e:
        print(f"  오류: {e}")
        return []


def batch_extract_pdf_tables(
    download_list_path: str = "data/dart_pdfs/download_list.json",
    output_path: str = "extracted_tables_dart_pdfs.json",
    method: str = "pdfplumber",
    max_files: int = None
) -> Dict:
    """
    다운로드된 모든 PDF 파일에서 표를 일괄 추출
    
    Args:
        download_list_path: 다운로드 목록 JSON 파일 경로
        output_path: 출력 JSON 파일 경로
        method: 추출 방법 ('pdfplumber', 'camelot', 'tabula')
        max_files: 최대 처리 파일 수 (None이면 모두 처리)
        
    Returns:
        추출 결과 통계
    """
    print("=" * 80)
    print("DART PDF 일괄 표 추출")
    print("=" * 80)
    
    # 다운로드 목록 로드
    download_list_file = Path(download_list_path)
    if not download_list_file.exists():
        print(f"오류: {download_list_path} 파일을 찾을 수 없습니다.")
        return {}
    
    with open(download_list_file, 'r', encoding='utf-8') as f:
        download_list = json.load(f)
    
    print(f"\n총 {len(download_list)}개 PDF 파일 발견")
    
    if max_files:
        download_list = download_list[:max_files]
        print(f"처리할 파일 수: {max_files}개 (제한)")
    
    # PDF 추출기 초기화
    print(f"\nPDF 추출기 초기화: {method}")
    extractor = PDFTableExtractor(method=method)
    
    # 일괄 추출
    all_tables = []
    stats = {
        'total_files': len(download_list),
        'processed_files': 0,
        'failed_files': 0,
        'total_tables': 0,
        'total_pages': 0,
        'extraction_method': method,
        'file_stats': []
    }
    
    print("\n표 추출 시작...")
    for file_info in tqdm(download_list, desc="PDF 처리"):
        pdf_path = file_info.get('filepath', '')
        corp_name = file_info.get('corp_name', '')
        report_nm = file_info.get('report_nm', '')
        
        # 상대 경로를 절대 경로로 변환
        if not Path(pdf_path).is_absolute():
            pdf_path = Path(__file__).parent.parent / pdf_path
        
        pdf_path = str(pdf_path)
        
        if not Path(pdf_path).exists():
            print(f"\n경고: 파일을 찾을 수 없습니다: {pdf_path}")
            stats['failed_files'] += 1
            continue
        
        # 표 추출
        try:
            tables = extract_tables_from_pdf(pdf_path, extractor)
            
            if tables:
                # 메타데이터 추가
                for table in tables:
                    table['corp_name'] = corp_name
                    table['report_nm'] = report_nm
                    table['rcept_no'] = file_info.get('rcept_no', '')
                    table['rcept_dt'] = file_info.get('rcept_dt', '')
                
                all_tables.extend(tables)
                stats['total_tables'] += len(tables)
                stats['total_pages'] += len(set(t.get('page_number', 0) for t in tables))
                stats['processed_files'] += 1
                
                stats['file_stats'].append({
                    'filepath': pdf_path,
                    'corp_name': corp_name,
                    'report_nm': report_nm,
                    'tables_extracted': len(tables),
                    'status': 'success'
                })
            else:
                stats['failed_files'] += 1
                stats['file_stats'].append({
                    'filepath': pdf_path,
                    'corp_name': corp_name,
                    'report_nm': report_nm,
                    'tables_extracted': 0,
                    'status': 'no_tables'
                })
        
        except Exception as e:
            print(f"\n오류 ({pdf_path}): {e}")
            stats['failed_files'] += 1
            stats['file_stats'].append({
                'filepath': pdf_path,
                'corp_name': corp_name,
                'report_nm': report_nm,
                'tables_extracted': 0,
                'status': 'error',
                'error': str(e)
            })
    
    # 결과 저장
    print(f"\n추출 완료!")
    print(f"  처리된 파일: {stats['processed_files']}개")
    print(f"  실패한 파일: {stats['failed_files']}개")
    print(f"  추출된 표: {stats['total_tables']}개")
    print(f"  총 페이지: {stats['total_pages']}개")
    
    if all_tables:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_tables, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n결과 저장: {output_path}")
        print(f"  파일 크기: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 통계 정보도 별도 저장
        stats_file = output_path.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        print(f"통계 저장: {stats_file}")
    else:
        print("\n경고: 추출된 표가 없습니다.")
    
    return stats


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DART PDF 파일에서 일괄 표 추출')
    parser.add_argument(
        '--input',
        type=str,
        default='data/dart_pdfs/download_list.json',
        help='다운로드 목록 JSON 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='extracted_tables_dart_pdfs.json',
        help='출력 JSON 파일 경로'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='pdfplumber',
        choices=['pdfplumber', 'camelot', 'tabula'],
        help='추출 방법 (기본값: pdfplumber)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='최대 처리 파일 수 (테스트용)'
    )
    
    args = parser.parse_args()
    
    stats = batch_extract_pdf_tables(
        download_list_path=args.input,
        output_path=args.output,
        method=args.method,
        max_files=args.max_files
    )
    
    return stats


if __name__ == "__main__":
    main()


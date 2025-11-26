#!/usr/bin/env python3
"""
PDF 표 추출 도구 비교 실험 스크립트
보고서에서 언급한 여러 파서를 동시에 비교하여 성능을 평가합니다.
"""
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# 선택적 import
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(desc)
        return iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.pdf_extractor import PDFTableExtractor


def extract_with_method(pdf_path: str, method: str) -> Dict[str, Any]:
    """
    특정 방법으로 PDF에서 표 추출
    
    Args:
        pdf_path: PDF 파일 경로
        method: 추출 방법
        
    Returns:
        추출 결과 딕셔너리
    """
    result = {
        'method': method,
        'pdf_path': pdf_path,
        'success': False,
        'table_count': 0,
        'extraction_time': 0,
        'error': None,
        'tables': []
    }
    
    try:
        start_time = time.time()
        extractor = PDFTableExtractor(method=method)
        tables = extractor.extract_tables(pdf_path)
        extraction_time = time.time() - start_time
        
        result['success'] = True
        result['table_count'] = len(tables)
        result['extraction_time'] = extraction_time
        result['tables'] = tables
        
        # 표 크기 통계
        if tables:
            total_rows = 0
            total_cols = 0
            for table in tables:
                df = table.get('dataframe')
                if df is not None:
                    total_rows += len(df)
                    total_cols = max(total_cols, len(df.columns))
            
            result['total_rows'] = total_rows
            result['avg_rows_per_table'] = total_rows / len(tables) if tables else 0
            result['max_cols'] = total_cols
        
    except Exception as e:
        result['error'] = str(e)
        result['extraction_time'] = time.time() - start_time
    
    return result


def compare_extractors(
    pdf_paths: List[str],
    methods: List[str] = None,
    max_files: int = None
) -> Dict[str, Any]:
    """
    여러 PDF 파일에 대해 여러 추출 방법을 비교
    
    Args:
        pdf_paths: PDF 파일 경로 리스트
        methods: 비교할 추출 방법 리스트 (None이면 모두 비교)
        max_files: 최대 처리 파일 수
        
    Returns:
        비교 결과 딕셔너리
    """
    if methods is None:
        methods = ['pdfplumber', 'camelot', 'tabula', 'pymupdf', 'pypdf', 'docling', 'unstructured', 'marker', 'nougat']
    
    if max_files:
        pdf_paths = pdf_paths[:max_files]
    
    print("=" * 80)
    print("PDF 표 추출 도구 비교 실험")
    print("=" * 80)
    print(f"\n비교할 방법: {', '.join(methods)}")
    print(f"테스트 파일 수: {len(pdf_paths)}개")
    print()
    
    # 결과 저장
    comparison_results = {
        'methods': methods,
        'total_files': len(pdf_paths),
        'results': [],
        'summary': {}
    }
    
    # 파일별 결과
    file_results = []
    
    for pdf_path in tqdm(pdf_paths, desc="PDF 파일 처리"):
        if not Path(pdf_path).exists():
            print(f"경고: 파일을 찾을 수 없습니다: {pdf_path}")
            continue
        
        file_result = {
            'pdf_path': pdf_path,
            'pdf_name': Path(pdf_path).name,
            'methods': {}
        }
        
        for method in methods:
            print(f"\n[{Path(pdf_path).name}] {method} 추출 중...")
            result = extract_with_method(pdf_path, method)
            file_result['methods'][method] = result
            
            if result['success']:
                print(f"  ✓ 성공: {result['table_count']}개 표 추출 ({result['extraction_time']:.2f}초)")
            else:
                print(f"  ✗ 실패: {result['error']}")
        
        file_results.append(file_result)
    
    comparison_results['results'] = file_results
    
    # 통계 계산
    summary = defaultdict(lambda: {
        'success_count': 0,
        'failure_count': 0,
        'total_tables': 0,
        'total_time': 0,
        'avg_tables_per_file': 0,
        'avg_time_per_file': 0,
        'total_rows': 0,
        'avg_rows_per_table': 0
    })
    
    for file_result in file_results:
        for method in methods:
            method_result = file_result['methods'].get(method, {})
            stats = summary[method]
            
            if method_result.get('success'):
                stats['success_count'] += 1
                stats['total_tables'] += method_result.get('table_count', 0)
                stats['total_time'] += method_result.get('extraction_time', 0)
                stats['total_rows'] += method_result.get('total_rows', 0)
            else:
                stats['failure_count'] += 1
                stats['total_time'] += method_result.get('extraction_time', 0)
    
    # 평균 계산
    for method in methods:
        stats = summary[method]
        total_files = stats['success_count'] + stats['failure_count']
        if total_files > 0:
            stats['avg_tables_per_file'] = stats['total_tables'] / total_files
            stats['avg_time_per_file'] = stats['total_time'] / total_files
        if stats['total_tables'] > 0:
            stats['avg_rows_per_table'] = stats['total_rows'] / stats['total_tables']
    
    comparison_results['summary'] = dict(summary)
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("비교 결과 요약")
    print("=" * 80)
    
    for method in methods:
        stats = summary[method]
        print(f"\n[{method}]")
        print(f"  성공: {stats['success_count']}개 파일")
        print(f"  실패: {stats['failure_count']}개 파일")
        print(f"  총 추출 표: {stats['total_tables']}개")
        print(f"  파일당 평균 표: {stats['avg_tables_per_file']:.2f}개")
        print(f"  총 소요 시간: {stats['total_time']:.2f}초")
        print(f"  파일당 평균 시간: {stats['avg_time_per_file']:.2f}초")
        if stats['total_tables'] > 0:
            print(f"  표당 평균 행: {stats['avg_rows_per_table']:.2f}행")
    
    return comparison_results


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDF 표 추출 도구 비교 실험')
    parser.add_argument(
        '--input',
        type=str,
        default='data/dart_pdfs/download_list.json',
        help='다운로드 목록 JSON 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pdf_extractor_comparison.json',
        help='출력 JSON 파일 경로'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['pdfplumber', 'camelot', 'tabula', 'pymupdf', 'pypdf', 'docling', 'unstructured', 'marker', 'nougat'],
        choices=['pdfplumber', 'camelot', 'tabula', 'pymupdf', 'pypdf', 'docling', 'unstructured', 'marker', 'nougat'],
        help='비교할 추출 방법 (기본값: 모두)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=5,
        help='최대 처리 파일 수 (기본값: 5)'
    )
    parser.add_argument(
        '--single-file',
        type=str,
        default=None,
        help='단일 파일만 테스트 (경로 지정)'
    )
    
    args = parser.parse_args()
    
    # PDF 파일 경로 수집
    if args.single_file:
        pdf_paths = [args.single_file]
    else:
        # 다운로드 목록에서 읽기
        download_list_file = Path(args.input)
        if not download_list_file.exists():
            print(f"오류: {args.input} 파일을 찾을 수 없습니다.")
            return
        
        with open(download_list_file, 'r', encoding='utf-8') as f:
            download_list = json.load(f)
        
        # 절대 경로로 변환
        base_path = Path(__file__).parent.parent
        pdf_paths = []
        for file_info in download_list:
            pdf_path = file_info.get('filepath', '')
            if not Path(pdf_path).is_absolute():
                pdf_path = base_path / pdf_path
            pdf_paths.append(str(pdf_path))
    
    # 비교 실험 실행
    results = compare_extractors(
        pdf_paths=pdf_paths,
        methods=args.methods,
        max_files=args.max_files if not args.single_file else None
    )
    
    # 결과 저장
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n결과 저장: {output_file}")
    
    # 요약 리포트 생성
    report_file = output_file.with_suffix('.md')
    generate_report(results, report_file)
    print(f"리포트 저장: {report_file}")


def generate_report(results: Dict[str, Any], report_path: Path):
    """비교 결과 리포트 생성"""
    summary = results['summary']
    methods = results['methods']
    
    report_lines = [
        "# PDF 표 추출 도구 비교 실험 결과",
        "",
        f"**실험 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**테스트 파일 수**: {results['total_files']}개",
        f"**비교 방법**: {', '.join(methods)}",
        "",
        "## 요약",
        "",
        "| 방법 | 성공 파일 | 실패 파일 | 총 표 | 평균 표/파일 | 평균 시간/파일 |",
        "|------|----------|----------|-------|------------|--------------|"
    ]
    
    for method in methods:
        stats = summary.get(method, {})
        report_lines.append(
            f"| {method} | {stats.get('success_count', 0)} | "
            f"{stats.get('failure_count', 0)} | {stats.get('total_tables', 0)} | "
            f"{stats.get('avg_tables_per_file', 0):.2f} | "
            f"{stats.get('avg_time_per_file', 0):.2f}초 |"
        )
    
    report_lines.extend([
        "",
        "## 상세 결과",
        ""
    ])
    
    for file_result in results['results']:
        report_lines.append(f"### {file_result['pdf_name']}")
        report_lines.append("")
        
        for method in methods:
            method_result = file_result['methods'].get(method, {})
            if method_result.get('success'):
                report_lines.append(
                    f"- **{method}**: ✓ {method_result.get('table_count', 0)}개 표 "
                    f"({method_result.get('extraction_time', 0):.2f}초)"
                )
            else:
                report_lines.append(
                    f"- **{method}**: ✗ 실패 - {method_result.get('error', 'Unknown error')}"
                )
        
        report_lines.append("")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


if __name__ == "__main__":
    main()


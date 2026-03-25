#!/usr/bin/env python3
"""
한국어 PDF 표 추출 시 발생하는 문제점 분석 스크립트
보고서에서 언급한 병합 셀, 다중 페이지, 한글 폰트 등의 문제를 분석합니다.
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.pdf_extractor import PDFTableExtractor


def analyze_merged_cells(table_df: pd.DataFrame) -> Dict[str, Any]:
    """
    병합 셀 패턴 분석
    
    Args:
        table_df: 추출된 표 DataFrame
        
    Returns:
        병합 셀 분석 결과
    """
    analysis = {
        'has_merged_patterns': False,
        'empty_cells': 0,
        'repeated_values': 0,
        'suspected_merged_rows': [],
        'suspected_merged_cols': []
    }
    
    if table_df.empty:
        return analysis
    
    # 빈 셀 개수 확인
    analysis['empty_cells'] = table_df.isna().sum().sum()
    
    # 행 단위로 반복되는 값 확인 (병합 셀 가능성)
    for idx, row in table_df.iterrows():
        # 같은 값이 연속으로 나타나는 경우
        row_values = row.values
        for i in range(len(row_values) - 1):
            if pd.notna(row_values[i]) and pd.notna(row_values[i+1]):
                if str(row_values[i]).strip() == str(row_values[i+1]).strip():
                    analysis['repeated_values'] += 1
                    analysis['suspected_merged_cols'].append({
                        'row': idx,
                        'col': i,
                        'value': str(row_values[i])
                    })
    
    # 열 단위로 반복되는 값 확인
    for col in table_df.columns:
        col_values = table_df[col].values
        for i in range(len(col_values) - 1):
            if pd.notna(col_values[i]) and pd.notna(col_values[i+1]):
                if str(col_values[i]).strip() == str(col_values[i+1]).strip():
                    analysis['suspected_merged_rows'].append({
                        'row': i,
                        'col': col,
                        'value': str(col_values[i])
                    })
    
    analysis['has_merged_patterns'] = (
        analysis['repeated_values'] > 0 or
        len(analysis['suspected_merged_rows']) > 0 or
        len(analysis['suspected_merged_cols']) > 0
    )
    
    return analysis


def analyze_korean_text(table_df: pd.DataFrame) -> Dict[str, Any]:
    """
    한글 텍스트 분석
    
    Args:
        table_df: 추출된 표 DataFrame
        
    Returns:
        한글 텍스트 분석 결과
    """
    analysis = {
        'korean_chars': 0,
        'total_chars': 0,
        'korean_ratio': 0.0,
        'has_vertical_text': False,
        'encoding_issues': []
    }
    
    if table_df.empty:
        return analysis
    
    korean_pattern = re.compile(r'[가-힣]')
    vertical_patterns = [
        re.compile(r'[\n\r].*[가-힣].*[\n\r]'),  # 줄바꿈이 많은 경우
    ]
    
    for col in table_df.columns:
        for idx, value in table_df[col].items():
            if pd.notna(value):
                text = str(value)
                analysis['total_chars'] += len(text)
                korean_matches = korean_pattern.findall(text)
                analysis['korean_chars'] += len(korean_matches)
                
                # 세로 쓰기 패턴 확인
                for pattern in vertical_patterns:
                    if pattern.search(text):
                        analysis['has_vertical_text'] = True
                        break
                
                # 인코딩 문제 확인 (깨진 문자)
                if '�' in text or '\ufffd' in text:
                    analysis['encoding_issues'].append({
                        'row': idx,
                        'col': col,
                        'text': text[:50]  # 처음 50자만
                    })
    
    if analysis['total_chars'] > 0:
        analysis['korean_ratio'] = analysis['korean_chars'] / analysis['total_chars']
    
    return analysis


def analyze_table_structure(table_df: pd.DataFrame) -> Dict[str, Any]:
    """
    표 구조 분석
    
    Args:
        table_df: 추출된 표 DataFrame
        
    Returns:
        표 구조 분석 결과
    """
    analysis = {
        'row_count': len(table_df),
        'col_count': len(table_df.columns),
        'is_empty': table_df.empty,
        'has_header': False,
        'irregular_structure': False,
        'column_widths': {}
    }
    
    if table_df.empty:
        return analysis
    
    # 헤더 확인 (첫 번째 행이 텍스트인 경우)
    if len(table_df) > 0:
        first_row = table_df.iloc[0]
        text_count = sum(1 for v in first_row.values if pd.notna(v) and str(v).strip())
        analysis['has_header'] = text_count > len(first_row) * 0.5
    
    # 불규칙한 구조 확인 (행마다 열 개수가 다른 경우)
    if len(table_df) > 1:
        col_counts = table_df.apply(lambda row: row.notna().sum(), axis=1)
        if col_counts.nunique() > 1:
            analysis['irregular_structure'] = True
    
    # 열 너비 분석 (텍스트 길이 기반)
    for col in table_df.columns:
        max_length = 0
        for value in table_df[col]:
            if pd.notna(value):
                max_length = max(max_length, len(str(value)))
        analysis['column_widths'][col] = max_length
    
    return analysis


def analyze_multi_page_issues(tables: List[Dict], source_file: str) -> Dict[str, Any]:
    """
    다중 페이지 표 문제 분석
    
    Args:
        tables: 추출된 표 리스트
        source_file: 원본 파일 경로
        
    Returns:
        다중 페이지 분석 결과
    """
    analysis = {
        'total_tables': len(tables),
        'pages_with_tables': set(),
        'suspected_continuation': [],
        'table_distribution': {}
    }
    
    # 페이지별 표 분포
    for table in tables:
        page_num = table.get('page_number', 0)
        analysis['pages_with_tables'].add(page_num)
        
        if page_num not in analysis['table_distribution']:
            analysis['table_distribution'][page_num] = 0
        analysis['table_distribution'][page_num] += 1
    
    # 연속된 페이지에 표가 있는 경우 (다중 페이지 표 가능성)
    pages = sorted(analysis['pages_with_tables'])
    for i in range(len(pages) - 1):
        if pages[i+1] - pages[i] == 1:
            analysis['suspected_continuation'].append({
                'start_page': pages[i],
                'end_page': pages[i+1]
            })
    
    analysis['pages_with_tables'] = sorted(list(analysis['pages_with_tables']))
    
    return analysis


def analyze_pdf_issues(
    pdf_path: str,
    methods: List[str] = None
) -> Dict[str, Any]:
    """
    PDF 파일의 표 추출 문제점 종합 분석
    
    Args:
        pdf_path: PDF 파일 경로
        methods: 분석할 추출 방법 리스트
        
    Returns:
        종합 분석 결과
    """
    if methods is None:
        methods = ['pdfplumber', 'camelot', 'tabula']
    
    results = {
        'pdf_path': pdf_path,
        'pdf_name': Path(pdf_path).name,
        'methods': {}
    }
    
    for method in methods:
        print(f"\n[{Path(pdf_path).name}] {method} 분석 중...")
        
        method_result = {
            'method': method,
            'success': False,
            'tables_analyzed': 0,
            'issues': {
                'merged_cells': [],
                'korean_text': {},
                'structure': [],
                'multi_page': {}
            }
        }
        
        try:
            extractor = PDFTableExtractor(method=method)
            tables = extractor.extract_tables(pdf_path)
            
            if tables:
                method_result['success'] = True
                method_result['tables_analyzed'] = len(tables)
                
                # 각 표 분석
                for table_info in tables:
                    df = table_info.get('dataframe')
                    if df is not None and not df.empty:
                        # 병합 셀 분석
                        merged_analysis = analyze_merged_cells(df)
                        method_result['issues']['merged_cells'].append(merged_analysis)
                        
                        # 한글 텍스트 분석
                        korean_analysis = analyze_korean_text(df)
                        method_result['issues']['korean_text'] = korean_analysis
                        
                        # 구조 분석
                        structure_analysis = analyze_table_structure(df)
                        method_result['issues']['structure'].append(structure_analysis)
                
                # 다중 페이지 분석
                multi_page_analysis = analyze_multi_page_issues(tables, pdf_path)
                method_result['issues']['multi_page'] = multi_page_analysis
                
                print(f"  ✓ {len(tables)}개 표 분석 완료")
            else:
                print(f"  ✗ 표를 추출하지 못함")
        
        except Exception as e:
            print(f"  ✗ 오류: {e}")
            method_result['error'] = str(e)
        
        results['methods'][method] = method_result
    
    return results


def generate_issue_report(analysis_results: Dict[str, Any]) -> str:
    """문제점 분석 리포트 생성"""
    report_lines = [
        "# 한국어 PDF 표 추출 문제점 분석 리포트",
        "",
        f"**파일**: {analysis_results['pdf_name']}",
        ""
    ]
    
    for method, result in analysis_results['methods'].items():
        report_lines.append(f"## {method}")
        report_lines.append("")
        
        if not result.get('success'):
            report_lines.append(f"- ❌ 표 추출 실패: {result.get('error', 'Unknown error')}")
            report_lines.append("")
            continue
        
        report_lines.append(f"- ✅ 분석된 표: {result['tables_analyzed']}개")
        report_lines.append("")
        
        # 병합 셀 문제
        merged_issues = result['issues']['merged_cells']
        if merged_issues:
            total_merged = sum(1 for m in merged_issues if m.get('has_merged_patterns'))
            if total_merged > 0:
                report_lines.append("### 병합 셀 문제")
                report_lines.append(f"- 병합 셀 패턴 발견: {total_merged}개 표")
                report_lines.append("")
        
        # 한글 텍스트 문제
        korean_issues = result['issues']['korean_text']
        if korean_issues.get('korean_chars', 0) > 0:
            report_lines.append("### 한글 텍스트 분석")
            report_lines.append(f"- 한글 비율: {korean_issues.get('korean_ratio', 0)*100:.1f}%")
            if korean_issues.get('has_vertical_text'):
                report_lines.append("- ⚠️ 세로 쓰기 패턴 발견")
            if korean_issues.get('encoding_issues'):
                report_lines.append(f"- ⚠️ 인코딩 문제: {len(korean_issues['encoding_issues'])}개 셀")
            report_lines.append("")
        
        # 구조 문제
        structure_issues = result['issues']['structure']
        irregular_count = sum(1 for s in structure_issues if s.get('irregular_structure'))
        if irregular_count > 0:
            report_lines.append("### 표 구조 문제")
            report_lines.append(f"- 불규칙한 구조: {irregular_count}개 표")
            report_lines.append("")
        
        # 다중 페이지 문제
        multi_page = result['issues']['multi_page']
        if multi_page.get('suspected_continuation'):
            report_lines.append("### 다중 페이지 표 문제")
            report_lines.append(f"- 연속 페이지 표: {len(multi_page['suspected_continuation'])}개")
            report_lines.append("")
    
    return '\n'.join(report_lines)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='한국어 PDF 표 추출 문제점 분석')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='분석할 PDF 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 JSON 파일 경로 (기본값: 자동 생성)'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['pdfplumber', 'camelot', 'tabula'],
        choices=['pdfplumber', 'camelot', 'tabula', 'pymupdf'],
        help='분석할 추출 방법'
    )
    
    args = parser.parse_args()
    
    pdf_path = args.input
    if not Path(pdf_path).exists():
        print(f"오류: 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    # 분석 실행
    results = analyze_pdf_issues(pdf_path, args.methods)
    
    # 결과 저장
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"korean_pdf_analysis_{Path(pdf_path).stem}.json")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n결과 저장: {output_file}")
    
    # 리포트 생성
    report_file = output_file.with_suffix('.md')
    report = generate_issue_report(results)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"리포트 저장: {report_file}")


if __name__ == "__main__":
    main()




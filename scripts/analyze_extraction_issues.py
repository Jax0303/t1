#!/usr/bin/env python3
"""
추출된 JSON 파일을 분석하여 원본 문서의 어떤 부분이 제대로 추출되지 않았는지 분석
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


def analyze_table_issues(table: Dict, page_num: int, pdf_path: str = None) -> Dict:
    """개별 표의 문제점 분석"""
    df = table.get('dataframe')
    if df is None:
        return {
            'issue_type': 'empty_dataframe',
            'description': 'DataFrame이 None입니다'
        }
    
    issues = []
    
    # None 비율 계산
    total_cells = len(df) * len(df.columns)
    none_count = df.isna().sum().sum()
    none_ratio = none_count / total_cells if total_cells > 0 else 0
    
    if none_ratio >= 0.3:
        issues.append({
            'issue_type': 'high_none_ratio',
            'severity': 'high' if none_ratio >= 0.5 else 'medium',
            'none_ratio': none_ratio,
            'none_count': int(none_count),
            'total_cells': total_cells,
            'description': f'{none_ratio*100:.1f}%의 셀이 None입니다'
        })
    
    # 빈 행/열 확인
    empty_rows = df[df.isna().all(axis=1)].shape[0]
    empty_cols = df[df.isna().all(axis=0)].shape[0]
    
    if empty_rows > 0:
        issues.append({
            'issue_type': 'empty_rows',
            'count': empty_rows,
            'description': f'{empty_rows}개의 빈 행이 있습니다'
        })
    
    if empty_cols > 0:
        issues.append({
            'issue_type': 'empty_columns',
            'count': empty_cols,
            'description': f'{empty_cols}개의 빈 열이 있습니다'
        })
    
    # 1행 표 확인
    if len(df) == 1:
        issues.append({
            'issue_type': 'single_row_table',
            'description': '헤더만 있고 데이터 행이 없습니다'
        })
    
    # 원본 PDF에서 해당 페이지의 텍스트 추출 (가능한 경우)
    original_text_snippet = None
    if pdf_path and HAS_PDFPLUMBER and Path(pdf_path).exists():
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    text = page.extract_text()
                    if text:
                        # 표 주변 텍스트 추출 (처음 500자)
                        original_text_snippet = text[:500] if len(text) > 500 else text
        except Exception as e:
            original_text_snippet = f"원본 텍스트 추출 실패: {e}"
    
    return {
        'issues': issues,
        'table_shape': f"{len(df)}x{len(df.columns)}",
        'original_text_snippet': original_text_snippet,
        'sample_data': df.head(3).to_dict() if len(df) > 0 else None
    }


def analyze_extraction_file(
    json_path: str,
    pdf_path: str = None,
    output_path: str = None
) -> Dict:
    """
    추출된 JSON 파일을 분석하여 문제점을 식별하고 주석 추가
    
    Args:
        json_path: 추출된 JSON 파일 경로
        pdf_path: 원본 PDF 파일 경로 (None이면 JSON에서 추출)
        output_path: 분석 결과 출력 경로 (None이면 자동 생성)
    """
    print("=" * 80)
    print("표 추출 문제 분석")
    print("=" * 80)
    
    # JSON 파일 로드
    print(f"\nJSON 파일 로드: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 원본 PDF 경로 확인
    if pdf_path is None:
        pdf_path = data.get('source_file')
    
    if pdf_path and not Path(pdf_path).is_absolute():
        pdf_path = Path(__file__).parent.parent / pdf_path
        pdf_path = str(pdf_path)
    
    print(f"원본 PDF: {pdf_path}")
    if pdf_path and not Path(pdf_path).exists():
        print(f"  경고: PDF 파일을 찾을 수 없습니다. 텍스트 추출을 건너뜁니다.")
        pdf_path = None
    
    # 품질 통계에서 문제 표 목록 가져오기
    quality_stats = data.get('quality_stats', {})
    none_ratio_details = quality_stats.get('none_ratio_details', [])
    
    print(f"\n총 표: {quality_stats.get('total_tables', 0)}개")
    print(f"문제 표: {len(none_ratio_details)}개")
    
    # 각 표 분석
    print("\n표별 상세 분석 중...")
    tables = data.get('tables', [])
    
    # 문제가 있는 표들만 필터링
    problem_tables = []
    for table in tables:
        table_id = table.get('table_id', '')
        page_num = table.get('page_number', 0)
        
        # None 비율이 높은 표인지 확인
        is_problem = any(
            detail.get('table_id') == table_id 
            for detail in none_ratio_details
        )
        
        if is_problem or len(table.get('table_data', {}).get('rows', [])) == 0:
            analysis = analyze_table_issues(table, page_num, pdf_path)
            problem_tables.append({
                'table_id': table_id,
                'page_number': page_num,
                'analysis': analysis,
                'original_table': table
            })
    
    # 분석 결과 정리
    analysis_result = {
        'source_file': json_path,
        'pdf_file': pdf_path,
        'total_tables': len(tables),
        'problem_tables_count': len(problem_tables),
        'problem_tables': problem_tables,
        'summary': {
            'high_none_ratio': len([t for t in problem_tables if any(
                issue.get('issue_type') == 'high_none_ratio' 
                for issue in t['analysis'].get('issues', [])
            )]),
            'single_row_tables': len([t for t in problem_tables if any(
                issue.get('issue_type') == 'single_row_table' 
                for issue in t['analysis'].get('issues', [])
            )]),
            'empty_tables': len([t for t in problem_tables if any(
                issue.get('issue_type') == 'empty_dataframe' 
                for issue in t['analysis'].get('issues', [])
            )])
        }
    }
    
    # 결과 저장
    if output_path is None:
        output_path = json_path.replace('.json', '_analysis.json')
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n분석 결과 저장: {output_path}")
    
    # 마크다운 리포트 생성
    markdown_path = output_path.replace('.json', '_report.md')
    generate_markdown_report(analysis_result, markdown_path)
    print(f"마크다운 리포트 생성: {markdown_path}")
    
    return analysis_result


def generate_markdown_report(analysis_result: Dict, output_path: str):
    """마크다운 형식의 분석 리포트 생성"""
    lines = []
    lines.append("# 표 추출 문제 분석 리포트\n")
    lines.append(f"**원본 파일**: `{analysis_result.get('pdf_file', 'N/A')}`\n")
    lines.append(f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n## 요약\n")
    
    summary = analysis_result.get('summary', {})
    lines.append(f"- 총 표: {analysis_result.get('total_tables', 0)}개")
    lines.append(f"- 문제 표: {analysis_result.get('problem_tables_count', 0)}개")
    lines.append(f"- None 비율 높은 표: {summary.get('high_none_ratio', 0)}개")
    lines.append(f"- 1행 표: {summary.get('single_row_tables', 0)}개")
    lines.append(f"- 빈 표: {summary.get('empty_tables', 0)}개")
    
    lines.append("\n## 문제 표 상세 분석\n")
    
    problem_tables = analysis_result.get('problem_tables', [])
    
    # 심각도별로 정렬
    problem_tables_sorted = sorted(
        problem_tables,
        key=lambda x: max(
            [0.5 if issue.get('severity') == 'high' else 0.3 if issue.get('severity') == 'medium' else 0.1
             for issue in x['analysis'].get('issues', [])],
            default=0
        ),
        reverse=True
    )
    
    for idx, table_info in enumerate(problem_tables_sorted, 1):
        table_id = table_info['table_id']
        page_num = table_info['page_number']
        analysis = table_info['analysis']
        issues = analysis.get('issues', [])
        
        lines.append(f"### {idx}. {table_id}\n")
        lines.append(f"- **페이지**: {page_num}")
        lines.append(f"- **표 크기**: {analysis.get('table_shape', 'N/A')}")
        lines.append(f"- **문제 유형**: {len(issues)}개\n")
        
        for issue in issues:
            issue_type = issue.get('issue_type', 'unknown')
            description = issue.get('description', '')
            severity = issue.get('severity', 'low')
            
            severity_emoji = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(severity, '⚪')
            
            lines.append(f"  {severity_emoji} **{issue_type}**: {description}")
            
            if issue_type == 'high_none_ratio':
                lines.append(f"    - None 비율: {issue.get('none_ratio', 0)*100:.1f}%")
                lines.append(f"    - None 셀 수: {issue.get('none_count', 0)}/{issue.get('total_cells', 0)}")
        
        # 샘플 데이터 표시
        sample_data = analysis.get('sample_data')
        if sample_data:
            lines.append("\n  **추출된 데이터 샘플 (상위 3행)**:")
            lines.append("  ```")
            try:
                df_sample = pd.DataFrame(sample_data)
                lines.append(f"  {df_sample.to_string().replace(chr(10), chr(10) + '  ')}")
            except:
                lines.append(f"  {str(sample_data)[:200]}...")
            lines.append("  ```")
        
        # 원본 텍스트 스니펫
        original_text = analysis.get('original_text_snippet')
        if original_text:
            lines.append("\n  **원본 페이지 텍스트 (일부)**:")
            lines.append("  ```")
            lines.append(f"  {original_text[:300]}...")
            lines.append("  ```")
        
        lines.append("\n---\n")
    
    # 권장 사항
    lines.append("\n## 권장 사항\n")
    lines.append("1. **None 비율이 높은 표**: 병합 셀이나 복잡한 레이아웃으로 인한 문제일 가능성이 높습니다.")
    lines.append("   - 대안 추출 방법 시도 (camelot, unstructured 등)")
    lines.append("   - OCR 기반 추출 고려 (Nougat, Marker 등)")
    lines.append("")
    lines.append("2. **1행 표**: 헤더만 있는 표는 실제 데이터가 다음 페이지에 있을 수 있습니다.")
    lines.append("   - 다음 페이지 확인 필요")
    lines.append("   - 페이지 경계를 넘는 표 처리 로직 개선 필요")
    lines.append("")
    lines.append("3. **빈 표**: 표 구조는 감지되었지만 데이터 추출 실패")
    lines.append("   - PDF 내부 구조 확인 필요")
    lines.append("   - 이미지 기반 표일 가능성")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='표 추출 문제 분석')
    parser.add_argument(
        '--json',
        type=str,
        default='extracted_DMS_[기재정정]사업보고서 (2023.12)_20250321000022_pdfplumber.json',
        help='추출된 JSON 파일 경로'
    )
    parser.add_argument(
        '--pdf',
        type=str,
        default=None,
        help='원본 PDF 파일 경로 (기본값: JSON에서 추출)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 파일 경로 (기본값: 자동 생성)'
    )
    
    args = parser.parse_args()
    
    result = analyze_extraction_file(
        json_path=args.json,
        pdf_path=args.pdf,
        output_path=args.output
    )
    
    return result


if __name__ == "__main__":
    main()


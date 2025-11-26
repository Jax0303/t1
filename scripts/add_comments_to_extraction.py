#!/usr/bin/env python3
"""
추출된 JSON 파일에 주석을 추가하여 원본 문서의 어떤 부분이 제대로 추출되지 않았는지 표시
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


def analyze_table_from_rows(rows: List[List[Dict]]) -> Dict:
    """rows 형식의 표 데이터를 분석"""
    if not rows:
        return {
            'issues': [{'issue_type': 'empty_table', 'description': '표가 비어있습니다'}],
            'table_shape': '0x0',
            'none_count': 0,
            'total_cells': 0,
            'none_ratio': 0.0
        }
    
    issues = []
    none_count = 0
    total_cells = 0
    
    # 행과 열 수 계산
    num_rows = len(rows)
    num_cols = max(len(row) for row in rows) if rows else 0
    
    # 각 셀 분석
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            total_cells += 1
            cell_text = cell.get('text', '')
            
            # None 또는 빈 값 체크
            if cell_text is None or cell_text == 'None' or cell_text == '' or str(cell_text).strip() == '':
                none_count += 1
    
    none_ratio = none_count / total_cells if total_cells > 0 else 0
    
    if none_ratio >= 0.3:
        issues.append({
            'issue_type': 'high_none_ratio',
            'severity': 'high' if none_ratio >= 0.5 else 'medium',
            'none_ratio': none_ratio,
            'none_count': none_count,
            'total_cells': total_cells,
            'description': f'{none_ratio*100:.1f}%의 셀이 None 또는 빈 값입니다'
        })
    
    # 1행 표 확인
    if num_rows == 1:
        issues.append({
            'issue_type': 'single_row_table',
            'description': '헤더만 있고 데이터 행이 없습니다'
        })
    
    # 샘플 데이터 추출
    sample_data = []
    for row in rows[:3]:  # 상위 3행만
        row_data = []
        for cell in row[:5]:  # 상위 5열만
            cell_text = cell.get('text', '')
            if cell_text == 'None':
                row_data.append(None)
            else:
                row_data.append(str(cell_text)[:30])  # 30자로 제한
        sample_data.append(row_data)
    
    return {
        'issues': issues,
        'table_shape': f"{num_rows}x{num_cols}",
        'none_count': none_count,
        'total_cells': total_cells,
        'none_ratio': none_ratio,
        'sample_data': sample_data
    }


def extract_page_text(pdf_path: str, page_num: int) -> str:
    """원본 PDF에서 특정 페이지의 텍스트 추출"""
    if not HAS_PDFPLUMBER or not Path(pdf_path).exists():
        return None
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num <= len(pdf.pages):
                page = pdf.pages[page_num - 1]
                text = page.extract_text()
                return text
    except Exception as e:
        return f"텍스트 추출 실패: {e}"
    
    return None


def add_comments_to_extraction(
    json_path: str,
    pdf_path: str = None,
    output_path: str = None
) -> Dict:
    """
    추출된 JSON 파일에 주석 추가
    
    Args:
        json_path: 추출된 JSON 파일 경로
        pdf_path: 원본 PDF 파일 경로
        output_path: 주석이 추가된 출력 파일 경로
    """
    print("=" * 80)
    print("표 추출 JSON 파일에 주석 추가")
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
    
    # 품질 통계에서 문제 표 목록 가져오기
    quality_stats = data.get('quality_stats', {})
    none_ratio_details = quality_stats.get('none_ratio_details', [])
    
    # 문제 표 ID 집합 생성
    problem_table_ids = {detail.get('table_id') for detail in none_ratio_details}
    
    print(f"\n총 표: {len(data.get('tables', []))}개")
    print(f"문제 표: {len(problem_table_ids)}개")
    
    # 각 표에 주석 추가
    print("\n표별 주석 추가 중...")
    tables = data.get('tables', [])
    annotated_tables = []
    
    for table in tables:
        table_id = table.get('table_id', '')
        page_num = table.get('page_number', 0)
        # table_data 또는 rows 직접 접근
        rows = table.get('table_data', {}).get('rows', []) if 'table_data' in table else table.get('rows', [])
        
        # 표 분석
        analysis = analyze_table_from_rows(rows)
        
        # 주석 생성
        comment = {
            'has_issues': table_id in problem_table_ids or len(analysis.get('issues', [])) > 0,
            'analysis': analysis,
            'original_page_text': extract_page_text(pdf_path, page_num) if pdf_path else None
        }
        
        # 표에 주석 추가
        annotated_table = table.copy()
        annotated_table['_comment'] = comment
        annotated_tables.append(annotated_table)
    
    # 결과 저장
    if output_path is None:
        output_path = json_path.replace('.json', '_annotated.json')
    
    output_data = data.copy()
    output_data['tables'] = annotated_tables
    output_data['_metadata'] = {
        'annotation_date': pd.Timestamp.now().isoformat(),
        'total_tables': len(annotated_tables),
        'problem_tables': len(problem_table_ids),
        'pdf_file': pdf_path
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n주석이 추가된 파일 저장: {output_path}")
    print(f"  파일 크기: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 요약 리포트 생성
    generate_summary_report(output_data, output_path.replace('.json', '_summary.md'))
    
    return output_data


def generate_summary_report(data: Dict, output_path: str):
    """요약 리포트 생성"""
    lines = []
    lines.append("# 표 추출 문제 요약 리포트\n")
    lines.append(f"**원본 파일**: `{data.get('_metadata', {}).get('pdf_file', 'N/A')}`\n")
    lines.append(f"**분석 일시**: {data.get('_metadata', {}).get('annotation_date', 'N/A')}\n")
    
    quality_stats = data.get('quality_stats', {})
    lines.append("\n## 전체 통계\n")
    lines.append(f"- 총 표: {quality_stats.get('total_tables', 0)}개")
    lines.append(f"- 빈 표: {quality_stats.get('empty_tables', 0)}개")
    lines.append(f"- 1행 표: {quality_stats.get('single_row_tables', 0)}개")
    lines.append(f"- None 비율 30% 이상: {quality_stats.get('high_none_ratio_tables', 0)}개")
    
    lines.append("\n## 문제 표 목록\n")
    
    tables = data.get('tables', [])
    problem_tables = [t for t in tables if t.get('_comment', {}).get('has_issues', False)]
    
    # 심각도별로 정렬
    problem_tables_sorted = sorted(
        problem_tables,
        key=lambda x: max(
            [0.5 if issue.get('severity') == 'high' else 0.3 if issue.get('severity') == 'medium' else 0.1
             for issue in x.get('_comment', {}).get('analysis', {}).get('issues', [])],
            default=0
        ),
        reverse=True
    )
    
    for idx, table in enumerate(problem_tables_sorted[:20], 1):  # 상위 20개만
        table_id = table.get('table_id', '')
        page_num = table.get('page_number', 0)
        comment = table.get('_comment', {})
        analysis = comment.get('analysis', {})
        issues = analysis.get('issues', [])
        
        lines.append(f"### {idx}. {table_id}\n")
        lines.append(f"- **페이지**: {page_num}")
        lines.append(f"- **표 크기**: {analysis.get('table_shape', 'N/A')}")
        lines.append(f"- **None 비율**: {analysis.get('none_ratio', 0)*100:.1f}%")
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
        
        # 샘플 데이터
        sample_data = analysis.get('sample_data', [])
        if sample_data:
            lines.append("\n  **추출된 데이터 샘플**:")
            lines.append("  ```")
            for row in sample_data:
                lines.append(f"  {row}")
            lines.append("  ```")
        
        lines.append("\n---\n")
    
    if len(problem_tables_sorted) > 20:
        lines.append(f"\n*총 {len(problem_tables_sorted)}개의 문제 표 중 상위 20개만 표시했습니다.*\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"요약 리포트 생성: {output_path}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='추출된 JSON 파일에 주석 추가')
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
    
    result = add_comments_to_extraction(
        json_path=args.json,
        pdf_path=args.pdf,
        output_path=args.output
    )
    
    return result


if __name__ == "__main__":
    main()


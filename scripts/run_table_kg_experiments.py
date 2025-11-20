#!/usr/bin/env python3
"""
다양한 유형의 표를 지식 그래프로 변환하여 매핑 방식을 비교하는 실험 스크립트
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kg.table_to_kg import TableToKnowledgeGraph


TYPE_DESCRIPTIONS = {
    'single_cell_notice': '단일 셀 또는 1x1 구조로 구성된 주의/안내형 표',
    'single_column_guideline': '여러 행이 있지만 단일 열로 서술형 문장을 나열한 표',
    'two_column_outline': '장·절 구성이나 목차를 두 개의 열로 정리한 표',
    'two_column_summary': '행 수가 적은 2열 요약 표',
    'multi_column_numeric_matrix': '5열 이상이고 숫자 비중이 높은 데이터 매트릭스',
    'multi_column_form_template': '5열 이상이면서 병합 셀 비율이 높은 양식/템플릿 표',
    'multi_column_text_matrix': '5열 이상이지만 텍스트 중심으로 구성된 표',
    'matrix_numeric': '3~4열의 숫자형 매트릭스',
    'matrix_text': '3~4열의 텍스트 중심 매트릭스',
    'misc_table': '위 분류에 속하지 않는 기타 표'
}


def extract_text(cell: Dict[str, Any]) -> str:
    text = cell.get('text', '') or ''
    if not text and cell.get('lines'):
        text = "\n".join(cell.get('lines', []))
    return text.strip()


def infer_row_count(rows: List[List[Dict[str, Any]]], fallback: int) -> int:
    max_row = 0
    for row in rows:
        for cell in row:
            row_idx = cell.get('row', 0)
            row_span = cell.get('row_span', 1)
            max_row = max(max_row, row_idx + row_span)
    return max(max_row, fallback, len(rows))


def infer_col_count(rows: List[List[Dict[str, Any]]], fallback: int) -> int:
    max_col = 0
    for row in rows:
        for cell in row:
            col_idx = cell.get('col', 0)
            col_span = cell.get('col_span', 1)
            max_col = max(max_col, col_idx + col_span)
    return max(max_col, fallback)


def compute_metrics(table: Dict[str, Any]) -> Dict[str, Any]:
    rows = table.get('rows', [])
    text_cells = []
    numeric_cells = 0
    merged_cells = 0
    total_cells = 0
    
    for row in rows:
        for cell in row:
            total_cells += 1
            if cell.get('row_span', 1) > 1 or cell.get('col_span', 1) > 1:
                merged_cells += 1
            text = extract_text(cell)
            if not text:
                continue
            text_cells.append(text)
            if any(ch.isdigit() for ch in text):
                numeric_cells += 1
    
    total_chars = sum(len(t) for t in text_cells)
    avg_chars = total_chars / len(text_cells) if text_cells else 0.0
    numeric_ratio = numeric_cells / len(text_cells) if text_cells else 0.0
    merged_ratio = merged_cells / total_cells if total_cells else 0.0
    
    return {
        'text_cells': len(text_cells),
        'total_chars': total_chars,
        'avg_chars': round(avg_chars, 2),
        'numeric_ratio': round(numeric_ratio, 3),
        'merged_ratio': round(merged_ratio, 3),
        'row_count': table['row_count'],
        'col_count': table['col_count']
    }


def classify_table(metrics: Dict[str, Any]) -> str:
    rc = metrics['row_count']
    cc = metrics['col_count']
    numeric_ratio = metrics['numeric_ratio']
    merged_ratio = metrics['merged_ratio']
    avg_chars = metrics['avg_chars']
    
    if cc <= 1:
        if rc <= 2 or avg_chars >= 120:
            return 'single_cell_notice'
        return 'single_column_guideline'
    
    if cc == 2:
        if rc > 30:
            return 'two_column_outline'
        return 'two_column_summary'
    
    if cc >= 5:
        if numeric_ratio >= 0.35:
            return 'multi_column_numeric_matrix'
        if merged_ratio >= 0.12:
            return 'multi_column_form_template'
        return 'multi_column_text_matrix'
    
    # 3~4열
    if numeric_ratio >= 0.3:
        return 'matrix_numeric'
    if avg_chars > 80:
        return 'matrix_text'
    
    return 'misc_table'


def collect_preview_lines(rows: List[List[Dict[str, Any]]], limit: int = 4) -> List[str]:
    preview = []
    for row in rows:
        for cell in row:
            text = extract_text(cell)
            if text:
                preview.append(text.replace("\n", " "))
            if len(preview) >= limit:
                return preview
    return preview


def build_column_lookup(kg) -> Dict[int, str]:
    lookup = {}
    for node_id, attrs in kg.nodes(data=True):
        if attrs.get('type') != 'Column':
            continue
        col_idx = attrs.get('index')
        if col_idx is None:
            continue
        header = attrs.get('header') or attrs.get('name') or f"열{col_idx + 1}"
        lookup[col_idx] = header
    return lookup


def build_row_label_lookup(table: Dict[str, Any]) -> Dict[int, str]:
    temp = {}
    for row in table.get('rows', []):
        for cell in row:
            row_idx = cell.get('row')
            col_idx = cell.get('col', 0)
            if row_idx is None:
                continue
            text = extract_text(cell)
            if not text:
                continue
            if row_idx not in temp or col_idx < temp[row_idx][0]:
                temp[row_idx] = (col_idx, text.splitlines()[0])
    return {row_idx: value for row_idx, (_, value) in temp.items()}


def sample_value_mappings(kg, table: Dict[str, Any], limit: int = 4) -> List[Dict[str, Any]]:
    samples = []
    column_lookup = build_column_lookup(kg)
    row_lookup = build_row_label_lookup(table)
    
    for node_id, attrs in kg.nodes(data=True):
        if attrs.get('type') != 'Value':
            continue
        
        value_text = attrs.get('text') or attrs.get('name') or ''
        source_cell = None
        for pred in kg.predecessors(node_id):
            if kg.nodes[pred].get('type') != 'Cell':
                continue
            edge_data = kg.get_edge_data(pred, node_id)
            if any(edge.get('type') == 'has_value' for edge in edge_data.values()):
                source_cell = kg.nodes[pred]
                break
        
        if not source_cell:
            continue
        
        col_idx = source_cell.get('col')
        row_idx = source_cell.get('row')
        header = column_lookup.get(col_idx, f"열{(col_idx + 1) if col_idx is not None else '?'}")
        row_label = row_lookup.get(row_idx)
        chain = f"{table['table_id']} -> {header}"
        chain += f" -> Row {row_idx}"
        if row_label:
            chain += f" ({row_label[:40]})"
        chain += f" -> '{value_text[:80]}'"
        
        samples.append({
            'value': value_text,
            'row': row_idx,
            'col': col_idx,
            'column_header': header,
            'row_label': row_label,
            'kg_chain': chain
        })
        
        if len(samples) >= limit:
            break
    
    return samples


def normalize_tables(raw_tables: List[Dict[str, Any]], source_file: str) -> List[Dict[str, Any]]:
    normalized = []
    for idx, raw in enumerate(raw_tables):
        rows = raw.get('rows', []) or []
        row_count = raw.get('row_count') or infer_row_count(rows, len(rows))
        col_count = raw.get('col_count') or infer_col_count(rows, 0)
        table = {
            'table_index': raw.get('table_index', idx),
            'table_id': raw.get('table_id') or f"hwp5_table_{idx}",
            'rows': rows,
            'row_count': row_count,
            'col_count': col_count,
            'caption': raw.get('caption'),
            'source_file': source_file
        }
        normalized.append(table)
    return normalized


def run_experiments(tables: List[Dict[str, Any]], samples_per_type: int,
                    output_dir: Path) -> Dict[str, Any]:
    kg_builder = TableToKnowledgeGraph()
    type_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_table_metadata = []
    
    for table in tables:
        metrics = compute_metrics(table)
        table_type = classify_table(metrics)
        preview = collect_preview_lines(table['rows'])
        meta = {
            'table': table,
            'metrics': metrics,
            'type': table_type,
            'preview': preview
        }
        type_groups[table_type].append(meta)
        per_table_metadata.append(meta)
    
    experiments = []
    rdf_dir = output_dir / "rdf_exports"
    rdf_dir.mkdir(parents=True, exist_ok=True)
    
    for table_type, entries in sorted(type_groups.items()):
        for entry in entries[:samples_per_type]:
            table = entry['table']
            table_payload = {
                'rows': table['rows'],
                'row_count': table['row_count'],
                'col_count': table['col_count'],
                'caption': table.get('caption')
            }
            table_id = table['table_id']
            
            kg = kg_builder.convert_table_to_kg(table_payload, table_id)
            kg_stats = kg_builder.get_graph_statistics()
            sample_paths = sample_value_mappings(kg, table)
            header_levels = sorted({attrs.get('level') for _, attrs in kg.nodes(data=True)
                                    if attrs.get('type') == 'Header' and 'level' in attrs})
            rdf_path = rdf_dir / f"{table_id}_{table_type}.ttl"
            rdf_exported = kg_builder.export_to_rdf(str(rdf_path))
            
            experiments.append({
                'table_type': table_type,
                'type_description': TYPE_DESCRIPTIONS.get(table_type, TYPE_DESCRIPTIONS['misc_table']),
                'table_id': table_id,
                'table_index': table['table_index'],
                'row_count': table['row_count'],
                'col_count': table['col_count'],
                'caption': table.get('caption'),
                'preview': entry['preview'],
                'metrics': entry['metrics'],
                'kg_stats': kg_stats,
                'header_levels': header_levels,
                'sample_mappings': sample_paths,
                'rdf_path': str(rdf_path),
                'rdf_exported': bool(rdf_exported)
            })
    
    type_summary = []
    for table_type, entries in sorted(type_groups.items()):
        type_summary.append({
            'type': table_type,
            'description': TYPE_DESCRIPTIONS.get(table_type, TYPE_DESCRIPTIONS['misc_table']),
            'count': len(entries),
            'row_range': [
                min(e['metrics']['row_count'] for e in entries),
                max(e['metrics']['row_count'] for e in entries)
            ],
            'col_range': [
                min(e['metrics']['col_count'] for e in entries),
                max(e['metrics']['col_count'] for e in entries)
            ]
        })
    
    return {
        'total_tables': len(tables),
        'type_summary': type_summary,
        'experiments': experiments
    }


def main():
    parser = argparse.ArgumentParser(description="표 유형별 지식 그래프 매핑 실험")
    parser.add_argument("--input", default="extracted_tables_hwp5_extractor_improved.json",
                        help="표 JSON 파일 경로")
    parser.add_argument("--samples-per-type", type=int, default=1,
                        help="유형별로 변환할 표 샘플 수")
    parser.add_argument("--output-dir", default="outputs/kg_table_experiments",
                        help="실험 결과 출력 디렉터리")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} 파일을 찾을 수 없습니다.")
    
    with open(input_path, "r", encoding="utf-8") as f:
        raw_tables = json.load(f)
    
    tables = normalize_tables(raw_tables, str(input_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = run_experiments(tables, args.samples_per_type, output_dir)
    report_path = output_dir / "kg_table_experiments.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            'source_file': str(input_path),
            **result
        }, f, ensure_ascii=False, indent=2)
    
    print(f"총 {result['total_tables']}개 표를 분석했습니다.")
    print("유형 분포:")
    for summary in result['type_summary']:
        print(f"  - {summary['type']} ({summary['count']}개)")
    
    print("\n실험 결과 요약:")
    for exp in result['experiments']:
        kg_stats = exp['kg_stats']
        print(f"  [{exp['table_type']}] {exp['table_id']} "
              f"(노드: {kg_stats['num_nodes']}, 엣지: {kg_stats['num_edges']}) -> RDF: {exp['rdf_exported']}")
    
    print(f"\nJSON 보고서: {report_path}")
    print(f"RDF 내보내기 경로: {output_dir / 'rdf_exports'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
다양한 유형의 표를 최신 KG 구축 라이브러리들을 사용하여 실험하는 확장 스크립트
- NetworkX (기본)
- RDFLib (RDF 변환)
- PyKEEN (KG Embeddings)
- SPARQL 쿼리 지원
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kg.table_to_kg import TableToKnowledgeGraph


def export_to_pykeen_format(kg, output_path: Path) -> Optional[Dict]:
    """
    NetworkX 그래프를 PyKEEN 형식으로 변환
    PyKEEN은 (head, relation, tail) 트리플 형식을 사용
    """
    try:
        from pykeen.datasets import Dataset
        from pykeen.triples import TriplesFactory
        
        triples = []
        relation_to_id = {}
        entity_to_id = {}
        
        # 엣지를 트리플로 변환
        for source, target, attrs in kg.edges(data=True):
            relation = attrs.get('type', 'related_to')
            
            # ID 매핑 생성
            if source not in entity_to_id:
                entity_to_id[source] = len(entity_to_id)
            if target not in entity_to_id:
                entity_to_id[target] = len(entity_to_id)
            if relation not in relation_to_id:
                relation_to_id[relation] = len(relation_to_id)
            
            triples.append((
                entity_to_id[source],
                relation_to_id[relation],
                entity_to_id[target]
            ))
        
        if not triples:
            return None
        
        # PyKEEN 형식으로 저장
        triples_data = {
            'triples': triples,
            'entity_to_id': {k: int(v) for k, v in entity_to_id.items()},
            'relation_to_id': {k: int(v) for k, v in relation_to_id.items()},
            'num_entities': len(entity_to_id),
            'num_relations': len(relation_to_id),
            'num_triples': len(triples)
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(triples_data, f, ensure_ascii=False, indent=2)
        
        return triples_data
    except ImportError:
        return None
    except Exception as e:
        print(f"PyKEEN 변환 실패: {e}")
        traceback.print_exc()
        return None


def analyze_kg_mapping(kg, table: Dict[str, Any]) -> Dict[str, Any]:
    """
    표 데이터가 KG의 어떤 부분으로 매핑되는지 상세 분석
    """
    mapping_analysis = {
        'entity_distribution': defaultdict(int),
        'relation_distribution': defaultdict(int),
        'header_hierarchy_depth': 0,
        'value_to_header_mappings': [],
        'cell_to_value_mappings': [],
        'structure_entities': {
            'tables': 0,
            'rows': 0,
            'columns': 0,
            'cells': 0
        },
        'content_entities': {
            'headers': 0,
            'values': 0
        }
    }
    
    # 엔티티 타입별 분포
    for node_id, attrs in kg.nodes(data=True):
        entity_type = attrs.get('type', 'Unknown')
        mapping_analysis['entity_distribution'][entity_type] += 1
        
        if entity_type == 'Table':
            mapping_analysis['structure_entities']['tables'] += 1
        elif entity_type == 'Row':
            mapping_analysis['structure_entities']['rows'] += 1
        elif entity_type == 'Column':
            mapping_analysis['structure_entities']['columns'] += 1
        elif entity_type == 'Cell':
            mapping_analysis['structure_entities']['cells'] += 1
        elif entity_type == 'Header':
            mapping_analysis['content_entities']['headers'] += 1
            level = attrs.get('level', 0)
            mapping_analysis['header_hierarchy_depth'] = max(
                mapping_analysis['header_hierarchy_depth'], level + 1
            )
        elif entity_type == 'Value':
            mapping_analysis['content_entities']['values'] += 1
    
    # 관계 타입별 분포
    for source, target, attrs in kg.edges(data=True):
        relation_type = attrs.get('type', 'unknown')
        mapping_analysis['relation_distribution'][relation_type] += 1
    
    # Value -> Header 매핑 추적
    for node_id, attrs in kg.nodes(data=True):
        if attrs.get('type') == 'Value':
            value_text = attrs.get('text') or attrs.get('name', '')
            # belongs_to 관계 찾기
            for pred in kg.predecessors(node_id):
                pred_attrs = kg.nodes[pred]
                if pred_attrs.get('type') == 'Header':
                    mapping_analysis['value_to_header_mappings'].append({
                        'value': value_text[:50],
                        'header': pred_attrs.get('name', '')[:50],
                        'header_level': pred_attrs.get('level', 0)
                    })
                    break
    
    # Cell -> Value 매핑 추적
    for node_id, attrs in kg.nodes(data=True):
        if attrs.get('type') == 'Cell':
            cell_row = attrs.get('row')
            cell_col = attrs.get('col')
            # has_value 관계 찾기
            for succ in kg.successors(node_id):
                succ_attrs = kg.nodes[succ]
                if succ_attrs.get('type') == 'Value':
                    mapping_analysis['cell_to_value_mappings'].append({
                        'cell_position': f"({cell_row}, {cell_col})",
                        'value': succ_attrs.get('text') or succ_attrs.get('name', '')[:50]
                    })
                    break
    
    return mapping_analysis


def generate_sparql_queries(kg, table_id: str) -> List[Dict[str, str]]:
    """
    KG 구조를 기반으로 유용한 SPARQL 쿼리 예제 생성
    """
    queries = []
    
    # 모든 Value 엔티티 조회
    queries.append({
        'name': '모든 값 조회',
        'description': '표에서 추출된 모든 Value 엔티티를 조회',
        'sparql': f"""
PREFIX kg: <http://example.org/kg/>
SELECT ?value ?text WHERE {{
    ?value a kg:Value .
    ?value kg:text ?text .
}}
"""
    })
    
    # 헤더 계층 구조 조회
    queries.append({
        'name': '헤더 계층 구조',
        'description': '헤더의 계층 구조를 조회',
        'sparql': f"""
PREFIX kg: <http://example.org/kg/>
SELECT ?header ?level ?name WHERE {{
    ?header a kg:Header .
    ?header kg:level ?level .
    ?header kg:name ?name .
}}
ORDER BY ?level
"""
    })
    
    # Value와 Header의 관계
    queries.append({
        'name': '값-헤더 관계',
        'description': '각 값이 어떤 헤더에 속하는지 조회',
        'sparql': f"""
PREFIX kg: <http://example.org/kg/>
SELECT ?value ?header ?valueText ?headerName WHERE {{
    ?value a kg:Value .
    ?value kg:text ?valueText .
    ?value kg:belongs_to ?header .
    ?header kg:name ?headerName .
}}
"""
    })
    
    return queries


def run_advanced_experiments(
    tables: List[Dict[str, Any]],
    samples_per_type: int,
    output_dir: Path
) -> Dict[str, Any]:
    """
    최신 KG 라이브러리를 사용한 확장 실험 실행
    """
    # 동적 import로 순환 참조 방지
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_table_kg_experiments",
        Path(__file__).parent / "run_table_kg_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    compute_metrics = module.compute_metrics
    classify_table = module.classify_table
    TYPE_DESCRIPTIONS = module.TYPE_DESCRIPTIONS
    collect_preview_lines = module.collect_preview_lines
    
    kg_builder = TableToKnowledgeGraph()
    type_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    # 표 정규화 및 분류
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
    
    experiments = []
    rdf_dir = output_dir / "rdf_exports"
    pykeen_dir = output_dir / "pykeen_exports"
    rdf_dir.mkdir(parents=True, exist_ok=True)
    pykeen_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n총 {len(type_groups)}개 유형, {sum(len(v) for v in type_groups.values())}개 표 분석 시작...")
    
    for table_type, entries in sorted(type_groups.items()):
        print(f"\n[{table_type}] {len(entries)}개 표 중 {min(samples_per_type, len(entries))}개 샘플 처리 중...")
        
        for entry in entries[:samples_per_type]:
            table = entry['table']
            table_id = table['table_id']
            
            try:
                table_payload = {
                    'rows': table['rows'],
                    'row_count': table['row_count'],
                    'col_count': table['col_count'],
                    'caption': table.get('caption')
                }
                
                # KG 구축
                kg = kg_builder.convert_table_to_kg(table_payload, table_id)
                kg_stats = kg_builder.get_graph_statistics()
                
                # 매핑 분석
                mapping_analysis = analyze_kg_mapping(kg, table)
                
                # RDF 내보내기
                rdf_path = rdf_dir / f"{table_id}_{table_type}.ttl"
                rdf_exported = kg_builder.export_to_rdf(str(rdf_path))
                
                # PyKEEN 형식 내보내기
                pykeen_path = pykeen_dir / f"{table_id}_{table_type}.json"
                pykeen_data = export_to_pykeen_format(kg, pykeen_path)
                pykeen_available = pykeen_data is not None
                
                # SPARQL 쿼리 생성
                sparql_queries = generate_sparql_queries(kg, table_id)
                
                experiments.append({
                    'table_type': table_type,
                    'type_description': TYPE_DESCRIPTIONS.get(table_type, '기타'),
                    'table_id': table_id,
                    'table_index': table['table_index'],
                    'row_count': table['row_count'],
                    'col_count': table['col_count'],
                    'caption': table.get('caption'),
                    'preview': entry['preview'],
                    'metrics': entry['metrics'],
                    'kg_stats': kg_stats,
                    'mapping_analysis': mapping_analysis,
                    'rdf_path': str(rdf_path),
                    'rdf_exported': bool(rdf_exported),
                    'pykeen_path': str(pykeen_path) if pykeen_data else None,
                    'pykeen_exported': bool(pykeen_data),
                    'sparql_queries': sparql_queries
                })
                
                print(f"  ✓ {table_id}: {kg_stats['num_nodes']} 노드, {kg_stats['num_edges']} 엣지")
                
            except Exception as e:
                print(f"  ✗ {table_id} 실패: {e}")
                traceback.print_exc()
                continue
    
    # 유형별 요약
    type_summary = []
    for table_type, entries in sorted(type_groups.items()):
        type_summary.append({
            'type': table_type,
            'description': TYPE_DESCRIPTIONS.get(table_type, '기타'),
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
    
    # PyKEEN 사용 가능 여부 확인
    pykeen_available = False
    try:
        import pykeen
        pykeen_available = True
    except ImportError:
        pass
    
    return {
        'total_tables': len(tables),
        'type_summary': type_summary,
        'experiments': experiments,
        'libraries_used': {
            'networkx': True,
            'rdflib': True,
            'pykeen': pykeen_available,
            'sparql': True
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="최신 KG 라이브러리를 사용한 표 유형별 지식 그래프 매핑 확장 실험"
    )
    parser.add_argument(
        "--input",
        default="extracted_tables_hwp5_extractor_improved.json",
        help="표 JSON 파일 경로"
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=3,
        help="유형별로 변환할 표 샘플 수 (기본값: 3)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/kg_table_experiments_advanced",
        help="실험 결과 출력 디렉터리"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} 파일을 찾을 수 없습니다.")
    
    with open(input_path, "r", encoding="utf-8") as f:
        raw_tables = json.load(f)
    
    # 표 정규화
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_table_kg_experiments",
        Path(__file__).parent / "run_table_kg_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    normalize_tables = module.normalize_tables
    
    tables = normalize_tables(raw_tables, str(input_path))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("최신 KG 라이브러리를 사용한 확장 실험 시작")
    print("=" * 80)
    
    result = run_advanced_experiments(tables, args.samples_per_type, output_dir)
    report_path = output_dir / "advanced_kg_experiments.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            'source_file': str(input_path),
            **result
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("실험 완료")
    print("=" * 80)
    print(f"\n총 {result['total_tables']}개 표 분석")
    print("\n유형 분포:")
    for summary in result['type_summary']:
        print(f"  - {summary['type']}: {summary['count']}개")
    
    print("\n실험 결과 요약:")
    for exp in result['experiments']:
        kg_stats = exp['kg_stats']
        mapping = exp['mapping_analysis']
        print(f"\n  [{exp['table_type']}] {exp['table_id']}")
        print(f"    노드: {kg_stats['num_nodes']}, 엣지: {kg_stats['num_edges']}")
        print(f"    구조 엔티티: Table({mapping['structure_entities']['tables']}), "
              f"Row({mapping['structure_entities']['rows']}), "
              f"Column({mapping['structure_entities']['columns']}), "
              f"Cell({mapping['structure_entities']['cells']})")
        print(f"    내용 엔티티: Header({mapping['content_entities']['headers']}), "
              f"Value({mapping['content_entities']['values']})")
        print(f"    헤더 계층 깊이: {mapping['header_hierarchy_depth']}")
        print(f"    RDF 내보내기: {exp['rdf_exported']}")
        print(f"    PyKEEN 내보내기: {exp['pykeen_exported']}")
    
    print(f"\n상세 리포트: {report_path}")
    print(f"RDF 내보내기: {output_dir / 'rdf_exports'}")
    print(f"PyKEEN 내보내기: {output_dir / 'pykeen_exports'}")


if __name__ == "__main__":
    main()


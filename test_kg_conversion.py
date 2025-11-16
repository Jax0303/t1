#!/usr/bin/env python3
"""
지식 그래프 변환 테스트 스크립트
표 데이터를 지식 그래프로 변환하는 기능 테스트
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.kg.table_to_kg import TableToKnowledgeGraph


def test_kg_conversion():
    """지식 그래프 변환 테스트"""
    print("=" * 80)
    print("지식 그래프 변환 테스트")
    print("=" * 80)
    
    # 테스트용 표 데이터 로드
    json_file = "extracted_tables_hwp5_extractor_improved.json"
    if not Path(json_file).exists():
        print(f"오류: {json_file} 파일을 찾을 수 없습니다.")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    print(f"총 {len(tables_data)}개 표 발견")
    
    # 지식 그래프 변환기 초기화
    kg_converter = TableToKnowledgeGraph()
    
    # 처음 5개 표만 테스트
    for idx, table_data in enumerate(tables_data[:5], 1):
        print(f"\n[{idx}] 표 {idx-1} 변환 중...")
        print(f"  행 수: {table_data.get('row_count', 0)}")
        print(f"  열 수: {table_data.get('col_count', 0)}")
        
        try:
            # 지식 그래프로 변환
            kg = kg_converter.convert_table_to_kg(table_data, f"table_{idx-1}")
            
            # 통계 정보
            stats = kg_converter.get_graph_statistics()
            print(f"  노드 수: {stats['num_nodes']}")
            print(f"  엣지 수: {stats['num_edges']}")
            print(f"  엔티티 타입별 개수:")
            for entity_type, count in stats['entity_types'].items():
                if count > 0:
                    print(f"    - {entity_type}: {count}")
            
            # 엔티티 샘플 출력
            print(f"\n  엔티티 샘플:")
            for entity_type in ['Table', 'Header', 'Value']:
                entities = kg_converter.get_entities_by_type(entity_type)
                if entities:
                    print(f"    {entity_type}:")
                    for entity in entities[:3]:  # 최대 3개만
                        name = entity.get('name', entity.get('id', ''))
                        print(f"      - {name}")
            
            # 관계 샘플 출력
            print(f"\n  관계 샘플:")
            for rel_type in ['has_header', 'has_value', 'belongs_to']:
                relations = kg_converter.get_relations_by_type(rel_type)
                if relations:
                    print(f"    {rel_type}:")
                    for rel in relations[:3]:  # 최대 3개만
                        source_name = kg_converter.graph.nodes[rel['source']].get('name', rel['source'])
                        target_name = kg_converter.graph.nodes[rel['target']].get('name', rel['target'])
                        print(f"      - {source_name} -> {target_name}")
            
        except Exception as e:
            print(f"  오류: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    test_kg_conversion()


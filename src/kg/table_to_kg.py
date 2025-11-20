"""
복잡한 표를 지식 그래프로 변환하는 모듈
병합셀, 중첩헤더, 계층 구조를 처리하여 엔티티-관계 그래프 구축
"""
import re
from typing import List, Dict, Optional, Tuple, Set
import pandas as pd
import networkx as nx
from collections import defaultdict


class TableToKnowledgeGraph:
    """표를 지식 그래프로 변환하는 클래스"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_counter = 0
        self.relation_types = {
            'has_value': '값을 가짐',
            'belongs_to': '소속됨',
            'related_to': '관련됨',
            'is_part_of': '일부임',
            'has_header': '헤더를 가짐',
            'has_row': '행을 가짐',
            'has_column': '열을 가짐',
            'contains': '포함함',
            'references': '참조함'
        }
    
    def convert_table_to_kg(self, table_data: Dict, table_id: str) -> nx.MultiDiGraph:
        """
        표 데이터를 지식 그래프로 변환
        
        Args:
            table_data: 표 데이터 딕셔너리 (rows, row_count, col_count 등 포함)
            table_id: 표 식별자
            
        Returns:
            NetworkX 그래프 객체
        """
        self.graph = nx.MultiDiGraph()
        self.entity_counter = 0
        
        # 표 엔티티 생성
        table_entity = self._create_entity('Table', table_id, {'type': 'table'})
        
        rows = table_data.get('rows', [])
        if not rows:
            return self.graph
        
        # 병합셀 정보 추출
        cell_info = self._extract_cell_info(rows)
        
        # 헤더 계층 구조 분석
        header_hierarchy = self._analyze_header_hierarchy(rows, cell_info)
        
        # 표 구조를 그래프로 변환
        self._build_table_structure_graph(table_entity, rows, cell_info, header_hierarchy)
        
        # 데이터 셀을 엔티티로 변환
        self._convert_data_cells_to_entities(table_entity, rows, cell_info, header_hierarchy)
        
        return self.graph
    
    def _create_entity(self, entity_type: str, name: str, attributes: Dict = None) -> str:
        """엔티티 생성 및 그래프에 추가"""
        entity_id = f"{entity_type}_{self.entity_counter}"
        self.entity_counter += 1
        
        extra_attrs = {}
        if attributes:
            extra_attrs = {k: v for k, v in attributes.items() if k != 'type'}
            if 'type' in attributes:
                extra_attrs['entity_subtype'] = attributes['type']
        
        attrs = {
            'name': name,
            'type': entity_type,
            **extra_attrs
        }
        
        self.graph.add_node(entity_id, **attrs)
        return entity_id
    
    def _create_relation(self, source: str, target: str, relation_type: str, attributes: Dict = None):
        """관계 생성 및 그래프에 추가"""
        attrs = {
            'type': relation_type,
            'label': self.relation_types.get(relation_type, relation_type),
            **(attributes or {})
        }
        self.graph.add_edge(source, target, **attrs)
    
    def _extract_cell_info(self, rows: List[List[Dict]]) -> Dict[Tuple[int, int], Dict]:
        """
        셀 정보 추출 (병합셀, 위치 등)
        
        Returns:
            {(row, col): {row_span, col_span, text, ...}} 형태의 딕셔너리
        """
        cell_info = {}
        
        for row_idx, row in enumerate(rows):
            for cell in row:
                row_pos = cell.get('row', row_idx)
                col_pos = cell.get('col', 0)
                row_span = cell.get('row_span', 1)
                col_span = cell.get('col_span', 1)
                text = cell.get('text', '')
                
                if not text and cell.get('lines'):
                    text = '\n'.join(cell.get('lines', []))
                
                cell_info[(row_pos, col_pos)] = {
                    'row_span': row_span,
                    'col_span': col_span,
                    'text': text.strip(),
                    'is_merged': row_span > 1 or col_span > 1,
                    'cell': cell
                }
        
        return cell_info
    
    def _analyze_header_hierarchy(self, rows: List[List[Dict]], cell_info: Dict) -> Dict:
        """
        헤더 계층 구조 분석
        여러 행의 헤더를 분석하여 계층 구조 파악
        """
        if not rows:
            return {}
        
        header_hierarchy = {
            'header_rows': [],
            'data_start_row': 0,
            'column_headers': {},
            'row_headers': {}
        }
        
        # 헤더 행 식별 (일반적으로 상단 1-3행)
        max_header_rows = min(3, len(rows))
        header_rows = []
        
        for row_idx in range(max_header_rows):
            row = rows[row_idx]
            if row:
                # 헤더 행인지 판단 (모든 셀이 비어있지 않고, 숫자가 적음)
                is_header = self._is_header_row(row, cell_info)
                if is_header:
                    header_rows.append(row_idx)
        
        header_hierarchy['header_rows'] = header_rows
        header_hierarchy['data_start_row'] = max(header_rows) + 1 if header_rows else 1
        
        # 열 헤더 매핑 생성
        if header_rows:
            for col_idx in range(100):  # 최대 100개 열까지
                header_path = []
                for header_row_idx in header_rows:
                    # 해당 열의 셀 찾기
                    for cell in rows[header_row_idx]:
                        cell_col = cell.get('col', 0)
                        cell_col_span = cell.get('col_span', 1)
                        if cell_col <= col_idx < cell_col + cell_col_span:
                            text = cell.get('text', '')
                            if not text and cell.get('lines'):
                                text = '\n'.join(cell.get('lines', []))
                            if text.strip():
                                header_path.append(text.strip())
                            break
                
                if header_path:
                    header_hierarchy['column_headers'][col_idx] = ' > '.join(header_path)
        
        return header_hierarchy
    
    def _is_header_row(self, row: List[Dict], cell_info: Dict) -> bool:
        """행이 헤더 행인지 판단"""
        if not row:
            return False
        
        # 셀 텍스트 확인
        texts = []
        for cell in row:
            text = cell.get('text', '')
            if not text and cell.get('lines'):
                text = '\n'.join(cell.get('lines', []))
            texts.append(text.strip())
        
        # 모든 셀이 비어있으면 헤더 아님
        if not any(texts):
            return False
        
        # 숫자 비율 확인 (헤더는 숫자가 적음)
        numeric_count = sum(1 for t in texts if t and re.search(r'\d+', t))
        if len(texts) > 0:
            numeric_ratio = numeric_count / len(texts)
            # 숫자 비율이 50% 미만이면 헤더로 간주
            return numeric_ratio < 0.5
        
        return True
    
    def _build_table_structure_graph(self, table_entity: str, rows: List[List[Dict]], 
                                    cell_info: Dict, header_hierarchy: Dict):
        """표 구조를 그래프로 변환"""
        # 행과 열 엔티티 생성
        row_entities = {}
        col_entities = {}
        
        for row_idx in range(len(rows)):
            row_entity = self._create_entity('Row', f"row_{row_idx}", {'index': row_idx})
            row_entities[row_idx] = row_entity
            self._create_relation(table_entity, row_entity, 'has_row', {'row_index': row_idx})
        
        # 열 엔티티 생성 (최대 열 수 기준)
        max_cols = max((cell.get('col', 0) + cell.get('col_span', 1) 
                       for row in rows for cell in row), default=0)
        
        for col_idx in range(max_cols):
            col_header = header_hierarchy.get('column_headers', {}).get(col_idx, f"열{col_idx+1}")
            col_entity = self._create_entity('Column', f"col_{col_idx}", {
                'index': col_idx,
                'header': col_header
            })
            col_entities[col_idx] = col_entity
            self._create_relation(table_entity, col_entity, 'has_column', {'col_index': col_idx})
        
        # 헤더 계층 구조 추가
        if header_hierarchy.get('column_headers'):
            for col_idx, header_path in header_hierarchy['column_headers'].items():
                if col_idx in col_entities:
                    header_parts = header_path.split(' > ')
                    prev_header = None
                    for level, header_text in enumerate(header_parts):
                        header_entity = self._create_entity('Header', header_text, {
                            'level': level,
                            'column': col_idx
                        })
                        if prev_header:
                            self._create_relation(prev_header, header_entity, 'contains')
                        else:
                            self._create_relation(col_entities[col_idx], header_entity, 'has_header')
                        prev_header = header_entity
    
    def _convert_data_cells_to_entities(self, table_entity: str, rows: List[List[Dict]], 
                                       cell_info: Dict, header_hierarchy: Dict):
        """데이터 셀을 엔티티로 변환"""
        data_start_row = header_hierarchy.get('data_start_row', 1)
        
        for row_idx, row in enumerate(rows):
            if row_idx < data_start_row:
                continue  # 헤더 행은 건너뜀
            
            for cell in row:
                row_pos = cell.get('row', row_idx)
                col_pos = cell.get('col', 0)
                text = cell.get('text', '')
                
                if not text and cell.get('lines'):
                    text = '\n'.join(cell.get('lines', []))
                
                text = text.strip()
                if not text:
                    continue
                
                # 위치 정보와 값을 그대로 보존한 Row/Col 수준 셀 엔티티 생성
                cell_entity = self._create_entity('Cell', f"cell_{row_pos}_{col_pos}", {
                    'row': row_pos,
                    'col': col_pos,
                    'value': text
                })
                
                # 표와의 관계
                self._create_relation(table_entity, cell_entity, 'contains', {
                    'row': row_pos,
                    'col': col_pos
                })
                
                # 값 엔티티 생성 (텍스트가 의미있는 경우)
                if self._is_meaningful_value(text):
                    # 숫자/텍스트 등 의미 있는 값만 Value 엔티티로 별도 승격
                    value_entity = self._create_entity('Value', text, {'text': text})
                    self._create_relation(cell_entity, value_entity, 'has_value')
                    
                    # 헤더와의 관계
                    col_header = header_hierarchy.get('column_headers', {}).get(col_pos, '')
                    if col_header:
                        # 헤더 엔티티 찾기 또는 생성
                        header_parts = col_header.split(' > ')
                        if header_parts:
                            header_text = header_parts[-1]
                            # 기존 헤더 엔티티 찾기
                            for node_id, attrs in self.graph.nodes(data=True):
                                if attrs.get('type') == 'Header' and attrs.get('name') == header_text:
                                    # 값이 속한 최하위 헤더에 belongs_to 관계 부여
                                    self._create_relation(value_entity, node_id, 'belongs_to')
                                    break
    
    def _is_meaningful_value(self, text: str) -> bool:
        """텍스트가 의미있는 값인지 판단"""
        if not text or len(text.strip()) < 2:
            return False
        
        # 숫자만 있거나 특수문자만 있으면 제외
        if re.match(r'^[\d\s\-\.]+$', text):
            return len(text.strip()) > 0
        
        # 한글, 영문, 숫자가 포함되어 있으면 의미있음
        return bool(re.search(r'[가-힣a-zA-Z0-9]', text))
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict]:
        """특정 타입의 엔티티 조회"""
        entities = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == entity_type:
                entities.append({
                    'id': node_id,
                    **attrs
                })
        return entities
    
    def get_relations_by_type(self, relation_type: str) -> List[Dict]:
        """특정 타입의 관계 조회"""
        relations = []
        for source, target, attrs in self.graph.edges(data=True):
            if attrs.get('type') == relation_type:
                relations.append({
                    'source': source,
                    'target': target,
                    **attrs
                })
        return relations
    
    def export_to_rdf(self, output_path: str):
        """RDF 형식으로 내보내기"""
        try:
            from rdflib import Graph, URIRef, Literal, Namespace, RDF
            from rdflib.namespace import RDFS
            
            rdf_graph = Graph()
            ns = Namespace("http://example.org/kg/")
            
            # 노드를 RDF로 변환
            for node_id, attrs in self.graph.nodes(data=True):
                subject = URIRef(ns[node_id])
                rdf_graph.add((subject, RDF.type, URIRef(ns[attrs.get('type', 'Entity')])))
                
                for key, value in attrs.items():
                    if key != 'type':
                        rdf_graph.add((subject, URIRef(ns[key]), Literal(value)))
            
            # 엣지를 RDF로 변환
            for source, target, attrs in self.graph.edges(data=True):
                subject = URIRef(ns[source])
                predicate = URIRef(ns[attrs.get('type', 'related_to')])
                obj = URIRef(ns[target])
                rdf_graph.add((subject, predicate, obj))
            
            rdf_graph.serialize(destination=output_path, format='turtle')
            return True
        except ImportError:
            print("RDFLib이 설치되지 않았습니다. RDF 내보내기를 건너뜁니다.")
            return False
    
    def get_graph_statistics(self) -> Dict:
        """그래프 통계 정보"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'entity_types': {
                entity_type: len(self.get_entities_by_type(entity_type))
                for entity_type in ['Table', 'Row', 'Column', 'Cell', 'Header', 'Value']
            }
        }

"""
데이터셋 로더 및 전처리 모듈
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


class DatasetLoader:
    """데이터셋 로더"""
    
    def __init__(self, base_path: str = "data/raw"):
        self.base_path = Path(base_path)
    
    def load_qa_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        질의응답 데이터셋 로드 (149.표 정보 질의 응답 데이터)
        
        Returns:
            {
                'training': {'source': [...], 'label': [...]},
                'validation': {'source': [...], 'label': [...]}
            }
        """
        dataset_dir = self.base_path / dataset_path
        
        result = {
            'training': {'source': [], 'label': []},
            'validation': {'source': [], 'label': []}
        }
        
        # Training 데이터
        training_source_dir = self.base_path / "dataset1" / "training" / "source"
        training_label_dir = self.base_path / "dataset1" / "training" / "label"
        
        if training_source_dir.exists():
            result['training']['source'] = self._load_json_files(training_source_dir)
        if training_label_dir.exists():
            result['training']['label'] = self._load_json_files(training_label_dir)
        
        # Validation 데이터
        validation_source_dir = self.base_path / "dataset1" / "validation" / "source"
        validation_label_dir = self.base_path / "dataset1" / "validation" / "label"
        
        if validation_source_dir.exists():
            result['validation']['source'] = self._load_json_files(validation_source_dir)
        if validation_label_dir.exists():
            result['validation']['label'] = self._load_json_files(validation_label_dir)
        
        return result
    
    def load_multi_format_dataset(self, dataset_path: str) -> Dict[str, List[str]]:
        """
        다중 형식 데이터셋 로드 (개정 표준취업규칙)
        
        Returns:
            {
                'hwp': [파일경로들],
                'hwpx': [파일경로들],
                'pdf': [파일경로들]
            }
        """
        dataset_dir = self.base_path / dataset_path
        
        result = {
            'hwp': [],
            'hwpx': [],
            'pdf': []
        }
        
        if not dataset_dir.exists():
            return result
        
        # 각 형식의 파일 찾기
        for ext in ['hwp', 'hwpx', 'pdf']:
            files = list(dataset_dir.glob(f"*.{ext}"))
            result[ext] = [str(f) for f in files]
        
        return result
    
    def _load_json_files(self, directory: Path) -> List[Dict]:
        """디렉토리에서 모든 JSON 파일 로드"""
        json_files = list(directory.glob("*.json"))
        all_data = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 데이터 구조에 따라 처리
                    if isinstance(data, dict):
                        if 'data' in data:
                            all_data.extend(data['data'])
                        else:
                            all_data.append(data)
                    elif isinstance(data, list):
                        all_data.extend(data)
            
            except Exception as e:
                print(f"JSON 파일 로드 오류 ({json_file}): {e}")
        
        return all_data
    
    def extract_qa_pairs(self, qa_data: List[Dict]) -> List[Dict]:
        """
        질의응답 데이터에서 질문-답변 쌍 추출
        
        Args:
            qa_data: 질의응답 데이터 리스트
            
        Returns:
            질문-답변 쌍 리스트
        """
        qa_pairs = []
        
        for item in qa_data:
            # 다양한 데이터 구조 지원
            if 'question' in item and 'answer' in item:
                qa_pairs.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'relevant_tables': item.get('relevant_tables', []),
                    'doc_id': item.get('doc_id', ''),
                    'metadata': item.get('metadata', {})
                })
            elif 'questions' in item:
                # 중첩된 구조
                for q in item['questions']:
                    qa_pairs.append({
                        'question': q.get('question', ''),
                        'answer': q.get('answer', ''),
                        'relevant_tables': q.get('relevant_tables', []),
                        'doc_id': item.get('doc_id', ''),
                        'metadata': item.get('metadata', {})
                    })
            elif 'paragraphs' in item:
                # paragraphs에서 질문-답변 추출
                for para in item.get('paragraphs', []):
                    if isinstance(para, dict):
                        # qas 키가 있는 경우 (질의응답 데이터셋 형식)
                        if 'qas' in para:
                            for qa in para.get('qas', []):
                                if isinstance(qa, dict):
                                    question = qa.get('question') or qa.get('질문')
                                    answer = qa.get('answer') or qa.get('답변')
                                    
                                    # answer 처리 (딕셔너리, 리스트, 문자열)
                                    if isinstance(answer, dict):
                                        answer = answer.get('text', '')
                                    elif isinstance(answer, list) and len(answer) > 0:
                                        answer = answer[0].get('text', '') if isinstance(answer[0], dict) else str(answer[0])
                                    else:
                                        answer = str(answer) if answer else ''
                                    
                                    if question and answer:
                                        qa_pairs.append({
                                            'question': question,
                                            'answer': answer,
                                            'relevant_tables': para.get('table_title', ''),
                                            'doc_id': item.get('doc_id', ''),
                                            'context_id': para.get('context_id', ''),
                                            'table_title': para.get('table_title', ''),
                                            'metadata': qa.get('metadata', {})
                                        })
                        else:
                            # 직접 question/answer 키가 있는 경우
                            question = para.get('question') or para.get('query') or para.get('질문')
                            answer = para.get('answer') or para.get('답변') or para.get('response')
                            
                            if question and answer:
                                qa_pairs.append({
                                    'question': question,
                                    'answer': answer,
                                    'relevant_tables': para.get('relevant_tables', []) or para.get('tables', []),
                                    'doc_id': item.get('doc_id', ''),
                                    'metadata': para.get('metadata', {})
                                })
        
        return qa_pairs
    
    def extract_table_labels(self, label_data: List[Dict]) -> Dict[str, Any]:
        """
        라벨링 데이터에서 표 정보 추출
        
        Args:
            label_data: 라벨링 데이터 리스트
            
        Returns:
            표 라벨 정보 딕셔너리
        """
        table_labels = {}
        
        for item in label_data:
            doc_id = item.get('doc_id', '')
            tables = []
            
            # 표 정보 추출 (다양한 구조 지원)
            if 'tables' in item:
                tables = item['tables'] if isinstance(item['tables'], list) else [item['tables']]
            elif 'table_info' in item:
                tables = item['table_info'] if isinstance(item['table_info'], list) else [item['table_info']]
            elif 'annotations' in item:
                # annotations에서 표 정보 찾기
                for ann in item['annotations']:
                    if ann.get('type') == 'table':
                        tables.append(ann)
            elif 'paragraphs' in item:
                # paragraphs에서 표 정보 찾기
                for para in item.get('paragraphs', []):
                    if isinstance(para, dict):
                        # table_title과 context가 있는 경우 (질의응답 데이터셋 형식)
                        if 'table_title' in para and 'context' in para:
                            # context에서 HTML 테이블 추출 가능
                            tables.append({
                                'table_title': para.get('table_title', ''),
                                'context': para.get('context', ''),
                                'context_id': para.get('context_id', '')
                            })
                        # 표 관련 키 찾기
                        elif 'table' in para or 'tables' in para:
                            table_data = para.get('table') or para.get('tables')
                            if table_data:
                                if isinstance(table_data, list):
                                    tables.extend(table_data)
                                else:
                                    tables.append(table_data)
                        # type이 'table'인 경우
                        elif para.get('type') == 'table' or para.get('paragraph_type') == 'table':
                            tables.append(para)
            
            if tables:
                table_labels[doc_id] = tables
        
        return table_labels
    
    def match_files_by_content(self, multi_format_data: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        같은 내용의 파일들을 매칭 (파일명 기반)
        
        Args:
            multi_format_data: 다중 형식 데이터
            
        Returns:
            매칭된 파일 그룹 리스트
        """
        matched_groups = []
        
        # 파일명에서 기본 이름 추출
        base_names = {}
        
        for ext, files in multi_format_data.items():
            for file_path in files:
                file_name = Path(file_path).stem
                # 공통 패턴 찾기 (예: "개정 표준취업규칙")
                base_name = self._extract_base_name(file_name)
                
                if base_name not in base_names:
                    base_names[base_name] = {}
                
                base_names[base_name][ext] = file_path
        
        # 매칭된 그룹 생성
        for base_name, files in base_names.items():
            if len(files) > 1:  # 최소 2개 형식 이상
                matched_groups.append({
                    'base_name': base_name,
                    'files': files
                })
        
        return matched_groups
    
    def _extract_base_name(self, filename: str) -> str:
        """파일명에서 기본 이름 추출"""
        # 특수 문자 제거 및 공통 패턴 찾기
        import re
        
        # 괄호 내용 제거
        base = re.sub(r'\([^)]*\)', '', filename)
        base = re.sub(r'\[[^\]]*\]', '', base)
        
        # 확장자 관련 키워드 제거
        base = base.replace('hwpx', '').replace('hwp', '').replace('pdf', '')
        base = base.replace('+', ' ').replace('_', ' ')
        
        # 공백 정리
        base = ' '.join(base.split())
        
        return base.strip()


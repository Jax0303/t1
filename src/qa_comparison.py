"""
HWP, HWPX, PDF 추출 결과를 기반으로 QA 성능 비교
"""
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from src.extractors import HWPXTableExtractor, PDFTableExtractor, HWPTableExtractor
from src.rag import TableRAGSystem
from src.evaluation import RAGMetrics


class QAComparisonExperiment:
    """QA 성능 비교 실험"""
    
    def __init__(self, config_path: str = "config.yaml", results_path: str = "results_multi_format.json"):
        """설정 및 추출 결과 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        with open(results_path, 'r', encoding='utf-8') as f:
            self.extraction_results = json.load(f)['multi_format']['extraction_results']
        
        # OpenAI 우선 사용, 없으면 Gemini 사용
        if 'openai' in self.config.get('api', {}) and self.config['api']['openai'].get('api_key'):
            self.api_key = self.config['api']['openai']['api_key']
            self.model_name = self.config['api']['openai']['model']
            self.provider = 'openai'
        elif 'gemini' in self.config.get('api', {}):
            self.api_key = self.config['api']['gemini']['api_key']
            self.model_name = self.config['api']['gemini']['model']
            self.provider = 'gemini'
        else:
            raise ValueError("API 설정을 찾을 수 없습니다. config.yaml에 openai 또는 gemini 설정이 필요합니다.")
        
        self.qa_results = {}
    
    def generate_sample_queries(self, num_queries: int = 30) -> List[Dict]:
        """
        추출된 표 데이터를 기반으로 샘플 쿼리 생성
        
        Args:
            num_queries: 생성할 쿼리 개수
            
        Returns:
            쿼리 리스트
        """
        queries = []
        
        # 모든 형식의 표 데이터에서 샘플 추출
        all_tables = []
        
        for format_name, result in self.extraction_results.items():
            if result['total_tables'] > 0:
                for file_key, file_data in result['files'].items():
                    for table_info in file_data.get('tables', []):
                        if 'dataframe' in table_info:
                            df_data = table_info['dataframe']
                            all_tables.append({
                                'format': format_name,
                                'table_id': table_info.get('table_id', ''),
                                'data': df_data
                            })
        
        if not all_tables:
            print("경고: 추출된 표가 없습니다.")
            return []
        
        # 표 데이터를 기반으로 쿼리 생성
        query_templates = [
            "표에서 {column}이 {value}인 행의 정보를 알려주세요",
            "{column}의 {value}에 해당하는 내용은 무엇인가요?",
            "표에서 {column} 열의 모든 값을 나열해주세요",
            "{value}와 관련된 정보를 표에서 찾아주세요",
            "표의 총 행 수는 몇 개인가요?",
            "표에서 {column}이 가장 큰 값은 무엇인가요?",
            "표에서 {column}이 가장 작은 값은 무엇인가요?",
            "{column} 열에 어떤 값들이 있는지 알려주세요",
            "표의 첫 번째 행 정보를 알려주세요",
            "표의 마지막 행 정보를 알려주세요",
            "표에서 {value}를 포함하는 모든 행을 찾아주세요",
            "표의 헤더(열 이름)는 무엇인가요?",
            "{column} 열의 평균값은 얼마인가요?",
            "표에서 {column}이 {value}인 경우 다른 열의 값은 무엇인가요?",
            "표의 구조를 설명해주세요",
            "{column} 열에서 고유한 값은 몇 개인가요?",
            "표에서 특정 조건을 만족하는 행을 찾아주세요",
            "{value}가 표에 존재하는지 확인해주세요",
            "표의 데이터 타입은 무엇인가요?",
            "표에서 {column}을 기준으로 정렬하면 어떻게 되나요?",
            "{column} 열의 합계는 얼마인가요?",
            "표에서 빈 셀이 있나요?",
            "{column} 열의 최빈값은 무엇인가요?",
            "표의 전체 내용을 요약해주세요",
            "{value}와 관련된 표의 정보를 모두 알려주세요",
            "표에서 숫자 데이터의 범위는 어떻게 되나요?",
            "{column} 열의 분포를 설명해주세요",
            "표에서 특정 패턴을 찾아주세요",
            "표의 데이터 품질을 평가해주세요",
            "표에서 가장 중요한 정보는 무엇인가요?"
        ]
        
        # 표 데이터에서 실제 값 추출하여 쿼리 생성
        generated_queries = []
        
        for i in range(min(num_queries, len(query_templates))):
            table = all_tables[i % len(all_tables)]
            df_data = table['data']
            
            # DataFrame으로 변환
            if isinstance(df_data, list):
                try:
                    df = pd.DataFrame(df_data)
                except:
                    continue
            elif isinstance(df_data, dict):
                try:
                    df = pd.DataFrame([df_data])
                except:
                    continue
            else:
                continue
            
            if df.empty or len(df.columns) == 0:
                continue
            
            # 실제 값 추출
            columns = list(df.columns)[:3]  # 처음 3개 열만 사용
            sample_values = []
            
            for col in columns:
                try:
                    non_null_values = df[col].dropna().astype(str).head(3).tolist()
                    sample_values.extend(non_null_values)
                except:
                    pass
            
            if not sample_values:
                continue
            
            # 쿼리 생성
            template = query_templates[i % len(query_templates)]
            
            # 템플릿 변수 채우기
            query = template
            if '{column}' in template and columns:
                query = query.replace('{column}', str(columns[0]))
            if '{value}' in template and sample_values:
                query = query.replace('{value}', str(sample_values[0]))
            
            generated_queries.append({
                'query_id': f"query_{i+1:03d}",
                'question': query,
                'format': table['format'],
                'table_id': table['table_id'],
                'expected_info': {
                    'columns': columns,
                    'sample_values': sample_values[:3]
                }
            })
        
        return generated_queries
    
    def run_qa_comparison(self, queries: List[Dict]):
        """
        각 형식별로 RAG 시스템 구축 및 QA 성능 비교
        
        Args:
            queries: 테스트 쿼리 리스트
        """
        print("=" * 60)
        print("QA 성능 비교 실험 시작")
        print("=" * 60)
        
        # 각 형식별로 RAG 시스템 구축 및 테스트
        formats_to_test = ['hwpx', 'pdf_pdfplumber', 'pdf_camelot']
        
        for format_name in formats_to_test:
            if format_name not in self.extraction_results:
                continue
            
            result = self.extraction_results[format_name]
            if result['total_tables'] == 0:
                print(f"\n[{format_name}] 표가 없어 건너뜁니다.")
                continue
            
            print(f"\n[{format_name.upper()}]")
            print(f"  표 개수: {result['total_tables']}개")
            
            # 표 데이터 수집
            tables = []
            for file_key, file_data in result['files'].items():
                for table_info in file_data.get('tables', []):
                    if 'dataframe' in table_info:
                        df_data = table_info['dataframe']
                        # DataFrame으로 변환
                        if isinstance(df_data, list):
                            try:
                                df = pd.DataFrame(df_data)
                                tables.append({
                                    'table_id': table_info.get('table_id', ''),
                                    'dataframe': df,
                                    'source_file': table_info.get('source_file', ''),
                                    'extraction_method': table_info.get('extraction_method', '')
                                })
                            except:
                                continue
            
            if not tables:
                print(f"  경고: 유효한 표 데이터가 없습니다.")
                continue
            
            # RAG 시스템 구축
            print(f"  RAG 시스템 구축 중... (표 {len(tables)}개)")
            try:
                # API 키가 유효한지 먼저 확인
                try:
                    rag_system = TableRAGSystem(self.api_key, self.model_name, provider=self.provider)
                    rag_system.build_knowledge_base(tables)
                except Exception as api_error:
                    error_msg = str(api_error)
                    if "API key" in error_msg or "403" in error_msg or "PermissionDenied" in error_msg or "leaked" in error_msg.lower():
                        print(f"  경고: API 키 문제 ({error_msg[:100]}).")
                        print(f"  LLM 사용 불가. 간단한 키워드 검색 방식으로 진행합니다.")
                        # 간단한 키워드 기반 검색으로 대체
                        qa_metrics = self._evaluate_qa_simple(tables, queries, format_name)
                        self.qa_results[format_name] = {
                            'metrics': qa_metrics,
                            'table_count': len(tables),
                            'method': 'keyword_search'  # LLM 사용 안 함 표시
                        }
                        continue
                    else:
                        raise
                
                # QA 테스트
                print(f"  QA 테스트 실행 중...")
                qa_metrics = self._evaluate_qa(rag_system, queries, format_name)
                
                self.qa_results[format_name] = {
                    'metrics': qa_metrics,
                    'table_count': len(tables),
                    'method': 'rag_llm'  # LLM 사용 표시
                }
                
                # RAG 시스템 정리
                if rag_system:
                    rag_system.clear_knowledge_base()
                
            except Exception as e:
                print(f"  오류: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _evaluate_qa(self, rag_system: TableRAGSystem, queries: List[Dict], format_name: str) -> Dict:
        """QA 성능 평가"""
        metrics = {
            'em_scores': [],
            'hit_at_1': [],
            'hit_at_3': [],
            'hit_at_5': [],
            'bleu_scores': [],
            'response_times': [],
            'detailed_results': []
        }
        
        # 해당 형식과 관련된 쿼리만 필터링
        # format 필드가 없거나 해당 형식과 일치하는 쿼리 사용
        relevant_queries = []
        for q in queries:
            q_format = q.get('format', '')
            # format 필드가 없거나, 해당 형식과 일치하거나, 빈 문자열인 경우 포함
            if not q_format or q_format == format_name or q_format == '':
                relevant_queries.append(q)
        
        # 필터링된 쿼리가 없으면 모든 쿼리 사용 (최대 30개)
        if not relevant_queries:
            relevant_queries = queries[:30]
        
        print(f"    평가할 쿼리 수: {len(relevant_queries)}개")
        
        error_count = 0
        for query_info in tqdm(relevant_queries, desc=f"    {format_name} 평가"):
            question = query_info.get('question', '') if isinstance(query_info, dict) else str(query_info)
            
            if not question:
                print(f"      경고: 빈 질문 건너뜀")
                continue
            
            try:
                import time
                start_time = time.time()
                
                # 할당량 문제 대비 딜레이 (무료 티어는 분당 제한이 있음)
                time.sleep(3)  # 요청 간 3초 대기
                
                result = rag_system.query(question)
                
                response_time = time.time() - start_time
                metrics['response_times'].append(response_time)
                
                answer = result.get('answer', '')
                retrieved_docs = result.get('source_documents', [])
                
                if not answer:
                    print(f"      경고: 빈 답변 반환됨")
                    metrics['em_scores'].append(0.0)
                    metrics['hit_at_1'].append(0.0)
                    metrics['hit_at_3'].append(0.0)
                    metrics['hit_at_5'].append(0.0)
                    metrics['bleu_scores'].append(0.0)
                    continue
                
                # 간단한 정답 생성 (실제로는 더 정교한 평가 필요)
                expected_answer = self._generate_expected_answer(query_info, retrieved_docs)
                
                # EM Score
                em = RAGMetrics.calculate_em_score(answer, expected_answer)
                metrics['em_scores'].append(em)
                
                # Hit@K
                relevant_table_ids = [query_info.get('table_id', '')] if query_info.get('table_id') else []
                hit1 = RAGMetrics.calculate_hit_at_k(retrieved_docs, relevant_table_ids, 1)
                hit3 = RAGMetrics.calculate_hit_at_k(retrieved_docs, relevant_table_ids, 3)
                hit5 = RAGMetrics.calculate_hit_at_k(retrieved_docs, relevant_table_ids, 5)
                
                metrics['hit_at_1'].append(hit1)
                metrics['hit_at_3'].append(hit3)
                metrics['hit_at_5'].append(hit5)
                
                # BLEU Score
                bleu = RAGMetrics.calculate_bleu_score(answer, expected_answer)
                metrics['bleu_scores'].append(bleu)
                
                # 상세 결과 저장
                metrics['detailed_results'].append({
                    'query': question,
                    'answer': answer,
                    'expected_answer': expected_answer,
                    'em_score': em,
                    'hit_at_1': hit1,
                    'hit_at_3': hit3,
                    'hit_at_5': hit5,
                    'bleu_score': bleu,
                    'response_time': response_time
                })
            
            except Exception as e:
                error_count += 1
                print(f"      쿼리 처리 오류 ({error_count}번째): {question[:50]}... - {str(e)[:100]}")
                import traceback
                if error_count <= 3:  # 처음 3개 오류만 상세 출력
                    traceback.print_exc()
                metrics['em_scores'].append(0.0)
                metrics['hit_at_1'].append(0.0)
                metrics['hit_at_3'].append(0.0)
                metrics['hit_at_5'].append(0.0)
                metrics['bleu_scores'].append(0.0)
                metrics['response_times'].append(0.0)
        
        if error_count > 0:
            print(f"    총 {error_count}개 쿼리 처리 중 오류 발생")
        
        # 평균 계산
        successful_queries = len(metrics['em_scores']) - error_count
        final_metrics = {
            'em_score': sum(metrics['em_scores']) / len(metrics['em_scores']) if metrics['em_scores'] else 0.0,
            'hit_at_1': sum(metrics['hit_at_1']) / len(metrics['hit_at_1']) if metrics['hit_at_1'] else 0.0,
            'hit_at_3': sum(metrics['hit_at_3']) / len(metrics['hit_at_3']) if metrics['hit_at_3'] else 0.0,
            'hit_at_5': sum(metrics['hit_at_5']) / len(metrics['hit_at_5']) if metrics['hit_at_5'] else 0.0,
            'bleu_score': sum(metrics['bleu_scores']) / len(metrics['bleu_scores']) if metrics['bleu_scores'] else 0.0,
            'avg_response_time': sum(metrics['response_times']) / len(metrics['response_times']) if metrics['response_times'] else 0.0,
            'total_queries': len(relevant_queries),
            'successful_queries': successful_queries,
            'failed_queries': error_count
        }
        
        return final_metrics
    
    def _evaluate_qa_simple(self, tables: List[Dict], queries: List[Dict], format_name: str) -> Dict:
        """간단한 키워드 기반 QA 평가 (API 키 없이)"""
        metrics = {
            'em_scores': [],
            'hit_at_1': [],
            'hit_at_3': [],
            'hit_at_5': [],
            'bleu_scores': [],
            'response_times': [],
            'detailed_results': []
        }
        
        relevant_queries = queries[:30]
        
        # 모든 표의 텍스트 수집
        all_texts = []
        for table_info in tables:
            df = table_info['dataframe']
            table_text = df.astype(str).values.flatten().tolist()
            all_texts.extend(table_text)
        all_text = " ".join(all_texts).lower()
        
        for query_info in tqdm(relevant_queries, desc=f"    {format_name} (간단 검색)"):
            question = query_info.get('question', '') if isinstance(query_info, dict) else str(query_info)
            
            import time
            start_time = time.time()
            
            # 키워드 추출
            keywords = [w for w in question.split() if len(w) > 2]
            
            # 간단한 키워드 매칭
            matches = sum(1 for kw in keywords if kw.lower() in all_text)
            match_ratio = matches / len(keywords) if keywords else 0.0
            
            # 답변 생성 (간단)
            if match_ratio > 0.3:
                answer = f"표에서 관련 정보를 찾았습니다. 키워드 매칭률: {match_ratio:.2f}"
            else:
                answer = "표에서 관련 정보를 찾지 못했습니다."
            
            response_time = time.time() - start_time
            
            # 메트릭 계산 (간단)
            em = 1.0 if match_ratio > 0.5 else 0.0
            hit1 = 1.0 if match_ratio > 0.3 else 0.0
            hit3 = hit1
            hit5 = hit1
            bleu = match_ratio
            
            metrics['em_scores'].append(em)
            metrics['hit_at_1'].append(hit1)
            metrics['hit_at_3'].append(hit3)
            metrics['hit_at_5'].append(hit5)
            metrics['bleu_scores'].append(bleu)
            metrics['response_times'].append(response_time)
            
            metrics['detailed_results'].append({
                'query': question,
                'answer': answer,
                'match_ratio': match_ratio,
                'em_score': em,
                'hit_at_1': hit1,
                'response_time': response_time
            })
        
        # 평균 계산
        return {
            'em_score': sum(metrics['em_scores']) / len(metrics['em_scores']) if metrics['em_scores'] else 0.0,
            'hit_at_1': sum(metrics['hit_at_1']) / len(metrics['hit_at_1']) if metrics['hit_at_1'] else 0.0,
            'hit_at_3': sum(metrics['hit_at_3']) / len(metrics['hit_at_3']) if metrics['hit_at_3'] else 0.0,
            'hit_at_5': sum(metrics['hit_at_5']) / len(metrics['hit_at_5']) if metrics['hit_at_5'] else 0.0,
            'bleu_score': sum(metrics['bleu_scores']) / len(metrics['bleu_scores']) if metrics['bleu_scores'] else 0.0,
            'avg_response_time': sum(metrics['response_times']) / len(metrics['response_times']) if metrics['response_times'] else 0.0,
            'total_queries': len(relevant_queries)
        }
    
    def _generate_expected_answer(self, query_info: Dict, retrieved_docs: List) -> str:
        """예상 답변 생성 (간단한 버전)"""
        # 실제로는 더 정교한 답변 생성 필요
        expected_info = query_info.get('expected_info', {})
        columns = expected_info.get('columns', [])
        sample_values = expected_info.get('sample_values', [])
        
        if columns:
            return f"표의 {columns[0]} 열 정보"
        elif sample_values:
            return f"{sample_values[0]} 관련 정보"
        else:
            return "표 정보"
    
    def print_comparison_report(self):
        """비교 리포트 출력"""
        print("\n" + "=" * 60)
        print("QA 성능 비교 결과")
        print("=" * 60)
        
        for format_name, result in self.qa_results.items():
            metrics = result['metrics']
            print(f"\n[{format_name.upper()}]")
            print(f"  테스트 쿼리 수: {metrics['total_queries']}개")
            if 'successful_queries' in metrics:
                print(f"  성공한 쿼리: {metrics['successful_queries']}개")
                print(f"  실패한 쿼리: {metrics.get('failed_queries', 0)}개")
            print(f"  EM Score: {metrics['em_score']:.4f}")
            print(f"  Hit@1: {metrics['hit_at_1']:.4f}")
            print(f"  Hit@3: {metrics['hit_at_3']:.4f}")
            print(f"  Hit@5: {metrics['hit_at_5']:.4f}")
            print(f"  BLEU Score: {metrics['bleu_score']:.4f}")
            if metrics['avg_response_time'] > 0:
                print(f"  평균 응답 시간: {metrics['avg_response_time']:.2f}초")
            else:
                print(f"  평균 응답 시간: 측정 불가 (모든 쿼리 실패)")
        
        # 비교 분석
        if len(self.qa_results) > 1:
            print("\n[비교 분석]")
            formats = list(self.qa_results.keys())
            
            # EM Score 비교
            em_scores = {f: self.qa_results[f]['metrics']['em_score'] for f in formats}
            best_em = max(em_scores.items(), key=lambda x: x[1])
            print(f"  최고 EM Score: {best_em[0]} ({best_em[1]:.4f})")
            
            # Hit@K 비교
            hit_at_1_scores = {f: self.qa_results[f]['metrics']['hit_at_1'] for f in formats}
            best_hit1 = max(hit_at_1_scores.items(), key=lambda x: x[1])
            print(f"  최고 Hit@1: {best_hit1[0]} ({best_hit1[1]:.4f})")
            
            # 응답 시간 비교
            response_times = {f: self.qa_results[f]['metrics']['avg_response_time'] for f in formats}
            fastest = min(response_times.items(), key=lambda x: x[1])
            print(f"  가장 빠른 응답: {fastest[0]} ({fastest[1]:.2f}초)")
    
    def save_results(self, output_path: str = "results_qa_comparison.json"):
        """결과 저장"""
        results_serializable = self._make_serializable({
            'qa_results': self.qa_results,
            'extraction_results': self.extraction_results
        })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과가 저장되었습니다: {output_path}")
    
    def _make_serializable(self, obj):
        """객체를 JSON 직렬화 가능하도록 변환"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


if __name__ == "__main__":
    experiment = QAComparisonExperiment()
    
    # 샘플 쿼리 생성
    print("=" * 60)
    print("샘플 쿼리 생성")
    print("=" * 60)
    queries = experiment.generate_sample_queries(30)
    print(f"\n생성된 쿼리: {len(queries)}개")
    
    # 쿼리 저장
    with open('sample_queries.json', 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print("쿼리가 저장되었습니다: sample_queries.json")
    
    # QA 비교 실험 실행
    experiment.run_qa_comparison(queries)
    
    # 리포트 출력
    experiment.print_comparison_report()
    
    # 결과 저장
    experiment.save_results()


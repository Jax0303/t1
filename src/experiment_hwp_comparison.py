"""
HWP 직접 파싱 vs HWPX 변환 기반 비교 실험
동일한 문서에 대해 두 방법을 적용하고 RAG 성능을 비교
"""
import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
import time

from src.extractors import HWPTableExtractor, HWPXTableExtractor
from src.rag import TableRAGSystem
from src.evaluation import RAGMetrics, TableExtractionMetrics


class HWPComparisonExperiment:
    """HWP 직접 파싱 vs HWPX 변환 비교 실험"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """설정 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # API 설정 (우선순위: gemini > ollama > openai)
        if 'gemini' in self.config.get('api', {}) and self.config['api']['gemini'].get('api_key'):
            self.api_key = self.config['api']['gemini']['api_key']
            self.model_name = self.config['api']['gemini']['model']
            self.provider = 'gemini'
        elif 'ollama' in self.config.get('api', {}):
            self.api_key = self.config['api']['ollama'].get('api_key', '')
            self.model_name = self.config['api']['ollama']['model']
            self.provider = 'ollama'
            self.ollama_base_url = self.config['api']['ollama'].get('base_url', 'http://localhost:11434')
        elif 'openai' in self.config.get('api', {}) and self.config['api']['openai'].get('api_key'):
            self.api_key = self.config['api']['openai']['api_key']
            self.model_name = self.config['api']['openai']['model']
            self.provider = 'openai'
        else:
            raise ValueError("API 설정을 찾을 수 없습니다. config.yaml에 gemini, ollama 또는 openai 설정이 필요합니다.")
        
        self.results = {}
    
    def run_comparison(self, hwp_files: List[str], queries: List[Dict], 
                      hwpx_files: Optional[List[str]] = None):
        """
        HWP 직접 파싱 vs HWPX 변환 비교 실험 실행
        
        Args:
            hwp_files: HWP 파일 경로 리스트
            queries: 테스트 쿼리 리스트
            hwpx_files: HWPX 파일 경로 리스트 (없으면 자동으로 찾음)
        """
        print("=" * 70)
        print("HWP 직접 파싱 vs HWPX 변환 비교 실험")
        print("=" * 70)
        
        # HWPX 파일 자동 찾기
        if hwpx_files is None:
            hwpx_files = []
            for hwp_file in hwp_files:
                hwp_path = Path(hwp_file)
                hwpx_path = hwp_path.with_suffix('.hwpx')
                if hwpx_path.exists():
                    hwpx_files.append(str(hwpx_path))
        
        if not hwpx_files:
            print("경고: HWPX 파일을 찾을 수 없습니다. HWPX 변환 방법은 건너뜁니다.")
            hwpx_files = []
        
        # 각 방법별로 표 추출
        print("\n[1단계] 표 추출")
        print("-" * 70)
        
        # 방법 1: HWP 직접 파싱
        print("\n[방법 1] HWP 직접 파싱")
        hwp_direct_extractor = HWPTableExtractor(use_direct_parsing=True)
        hwp_direct_tables = []
        hwp_direct_extraction_times = []
        
        for hwp_file in tqdm(hwp_files, desc="  HWP 직접 파싱"):
            start_time = time.time()
            tables = hwp_direct_extractor.extract_tables(hwp_file, method='direct')
            extraction_time = time.time() - start_time
            
            hwp_direct_tables.extend(tables)
            hwp_direct_extraction_times.append(extraction_time)
        
        print(f"  추출된 표: {len(hwp_direct_tables)}개")
        print(f"  평균 추출 시간: {sum(hwp_direct_extraction_times) / len(hwp_direct_extraction_times):.2f}초")
        
        # 방법 2: HWPX 변환 기반
        print("\n[방법 2] HWPX 변환 기반 (베이스라인)")
        hwpx_extractor = HWPXTableExtractor()
        hwpx_tables = []
        hwpx_extraction_times = []
        
        for hwpx_file in tqdm(hwpx_files, desc="  HWPX 파싱"):
            start_time = time.time()
            tables = hwpx_extractor.extract_tables(hwpx_file)
            extraction_time = time.time() - start_time
            
            hwpx_tables.extend(tables)
            hwpx_extraction_times.append(extraction_time)
        
        print(f"  추출된 표: {len(hwpx_tables)}개")
        if hwpx_extraction_times:
            print(f"  평균 추출 시간: {sum(hwpx_extraction_times) / len(hwpx_extraction_times):.2f}초")
        
        # 방법 3: HWP -> HWPX 변환 후 추출 (추가 베이스라인)
        print("\n[방법 3] HWP -> HWPX 변환 후 추출")
        hwp_via_hwpx_extractor = HWPTableExtractor(use_direct_parsing=False)
        hwp_via_hwpx_tables = []
        hwp_via_hwpx_extraction_times = []
        
        for hwp_file in tqdm(hwp_files, desc="  HWP->HWPX 변환"):
            start_time = time.time()
            tables = hwp_via_hwpx_extractor.extract_tables(hwp_file, method='hwpx')
            extraction_time = time.time() - start_time
            
            hwp_via_hwpx_tables.extend(tables)
            hwp_via_hwpx_extraction_times.append(extraction_time)
        
        print(f"  추출된 표: {len(hwp_via_hwpx_tables)}개")
        if hwp_via_hwpx_extraction_times:
            print(f"  평균 추출 시간: {sum(hwp_via_hwpx_extraction_times) / len(hwp_via_hwpx_extraction_times):.2f}초")
        
        # 표 추출 성능 비교
        print("\n[표 추출 성능 비교]")
        print("-" * 70)
        self._compare_extraction_performance(
            hwp_direct_tables, hwpx_tables, hwp_via_hwpx_tables,
            hwp_direct_extraction_times, hwpx_extraction_times, hwp_via_hwpx_extraction_times
        )
        
        # RAG 시스템 구축 및 QA 평가
        print("\n[2단계] RAG 시스템 구축 및 QA 평가")
        print("-" * 70)
        
        # 방법별 RAG 평가
        methods = {
            'hwp_direct': {
                'name': 'HWP 직접 파싱',
                'tables': hwp_direct_tables,
                'extraction_times': hwp_direct_extraction_times
            },
            'hwpx': {
                'name': 'HWPX 직접 파싱',
                'tables': hwpx_tables,
                'extraction_times': hwpx_extraction_times
            },
            'hwp_via_hwpx': {
                'name': 'HWP->HWPX 변환',
                'tables': hwp_via_hwpx_tables,
                'extraction_times': hwp_via_hwpx_extraction_times
            }
        }
        
        qa_results = {}
        
        for method_key, method_data in methods.items():
            if not method_data['tables']:
                print(f"\n[{method_data['name']}] 표가 없어 건너뜁니다.")
                continue
            
            print(f"\n[{method_data['name']}]")
            print(f"  표 개수: {len(method_data['tables'])}개")
            
            try:
                # RAG 시스템 구축
                print("  RAG 시스템 구축 중...")
                ollama_base_url = getattr(self, 'ollama_base_url', 'http://localhost:11434')
                
                # API 한도 초과 시 ollama로 전환
                current_provider = self.provider
                try:
                    rag_system = TableRAGSystem(
                        self.api_key, 
                        self.model_name, 
                        provider=self.provider,
                        ollama_base_url=ollama_base_url
                    )
                    rag_system.build_knowledge_base(method_data['tables'])
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    if "quota" in error_str or "429" in error_str or "limit" in error_str or "403" in error_str:
                        print(f"  경고: {self.provider} API 한도 초과. Ollama로 전환합니다...")
                        # ollama로 전환
                        rag_system = TableRAGSystem(
                            "",  # API 키 불필요
                            self.config.get('api', {}).get('ollama', {}).get('model', 'llama3.2'),
                            provider='ollama',
                            ollama_base_url=ollama_base_url
                        )
                        rag_system.build_knowledge_base(method_data['tables'])
                        current_provider = 'ollama'
                    else:
                        raise
                
                # QA 평가
                print("  QA 평가 실행 중...")
                qa_metrics = self._evaluate_qa(rag_system, queries, method_key)
                qa_results[method_key] = qa_metrics
                
                # RAG 시스템 정리
                rag_system.clear_knowledge_base()
                
            except Exception as e:
                print(f"  오류: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 결과 저장
        self.results = {
            'extraction': {
                'hwp_direct': {
                    'table_count': len(hwp_direct_tables),
                    'avg_extraction_time': sum(hwp_direct_extraction_times) / len(hwp_direct_extraction_times) if hwp_direct_extraction_times else 0.0
                },
                'hwpx': {
                    'table_count': len(hwpx_tables),
                    'avg_extraction_time': sum(hwpx_extraction_times) / len(hwpx_extraction_times) if hwpx_extraction_times else 0.0
                },
                'hwp_via_hwpx': {
                    'table_count': len(hwp_via_hwpx_tables),
                    'avg_extraction_time': sum(hwp_via_hwpx_extraction_times) / len(hwp_via_hwpx_extraction_times) if hwp_via_hwpx_extraction_times else 0.0
                }
            },
            'qa': qa_results
        }
        
        return self.results
    
    def _compare_extraction_performance(self, hwp_direct_tables, hwpx_tables, 
                                       hwp_via_hwpx_tables, hwp_direct_times, 
                                       hwpx_times, hwp_via_hwpx_times):
        """표 추출 성능 비교"""
        
        # 표 개수 비교
        print(f"표 개수:")
        print(f"  HWP 직접 파싱: {len(hwp_direct_tables)}개")
        print(f"  HWPX 직접 파싱: {len(hwpx_tables)}개")
        print(f"  HWP->HWPX 변환: {len(hwp_via_hwpx_tables)}개")
        
        # 추출 시간 비교
        if hwp_direct_times:
            avg_direct = sum(hwp_direct_times) / len(hwp_direct_times)
            print(f"\n평균 추출 시간:")
            print(f"  HWP 직접 파싱: {avg_direct:.2f}초")
            
            if hwpx_times:
                avg_hwpx = sum(hwpx_times) / len(hwpx_times)
                print(f"  HWPX 직접 파싱: {avg_hwpx:.2f}초")
                if avg_hwpx > 0:
                    speedup = (avg_hwpx - avg_direct) / avg_hwpx * 100
                    print(f"  속도 개선: {speedup:+.1f}%")
            
            if hwp_via_hwpx_times:
                avg_via_hwpx = sum(hwp_via_hwpx_times) / len(hwp_via_hwpx_times)
                print(f"  HWP->HWPX 변환: {avg_via_hwpx:.2f}초")
                if avg_via_hwpx > 0:
                    speedup = (avg_via_hwpx - avg_direct) / avg_via_hwpx * 100
                    print(f"  속도 개선: {speedup:+.1f}%")
        
        # 표 내용 비교 (F1-score)
        if hwp_direct_tables and hwpx_tables:
            print(f"\n표 내용 유사도 (F1-score):")
            # 같은 파일에서 추출된 표끼리 비교
            direct_dfs = [t['dataframe'] for t in hwp_direct_tables]
            hwpx_dfs = [t['dataframe'] for t in hwpx_tables]
            
            min_len = min(len(direct_dfs), len(hwpx_dfs))
            if min_len > 0:
                f1_scores = []
                for i in range(min_len):
                    f1 = TableExtractionMetrics._dataframe_similarity(direct_dfs[i], hwpx_dfs[i])
                    f1_scores.append(f1)
                
                avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                print(f"  평균 F1-score: {avg_f1:.4f}")
    
    def _evaluate_qa(self, rag_system: TableRAGSystem, queries: List[Dict], 
                    method_name: str) -> Dict:
        """QA 성능 평가"""
        metrics = {
            'em_scores': [],
            'hit_at_1': [],
            'hit_at_3': [],
            'hit_at_5': [],
            'f1_scores': [],
            'bleu_scores': [],
            'response_times': [],
            'detailed_results': []
        }
        
        print(f"    평가할 쿼리 수: {len(queries)}개")
        
        error_count = 0
        for query_info in tqdm(queries, desc=f"    {method_name} 평가"):
            question = query_info.get('question', '') if isinstance(query_info, dict) else str(query_info)
            
            if not question:
                continue
            
            try:
                start_time = time.time()
                
                # API 할당량 문제 대비 딜레이
                time.sleep(2)
                
                result = rag_system.query(question)
                
                response_time = time.time() - start_time
                metrics['response_times'].append(response_time)
                
                answer = result.get('answer', '')
                retrieved_docs = result.get('source_documents', [])
                
                if not answer:
                    metrics['em_scores'].append(0.0)
                    metrics['hit_at_1'].append(0.0)
                    metrics['hit_at_3'].append(0.0)
                    metrics['hit_at_5'].append(0.0)
                    metrics['f1_scores'].append(0.0)
                    metrics['bleu_scores'].append(0.0)
                    continue
                
                # 정답 생성
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
                
                # F1 Score (답변과 정답의 토큰 기반 F1)
                f1 = self._calculate_answer_f1(answer, expected_answer)
                metrics['f1_scores'].append(f1)
                
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
                    'f1_score': f1,
                    'bleu_score': bleu,
                    'response_time': response_time
                })
            
            except Exception as e:
                error_count += 1
                if error_count <= 3:
                    print(f"      쿼리 처리 오류: {str(e)[:100]}")
                metrics['em_scores'].append(0.0)
                metrics['hit_at_1'].append(0.0)
                metrics['hit_at_3'].append(0.0)
                metrics['hit_at_5'].append(0.0)
                metrics['f1_scores'].append(0.0)
                metrics['bleu_scores'].append(0.0)
                metrics['response_times'].append(0.0)
        
        if error_count > 0:
            print(f"    총 {error_count}개 쿼리 처리 중 오류 발생")
        
        # 평균 계산
        final_metrics = {
            'em_score': sum(metrics['em_scores']) / len(metrics['em_scores']) if metrics['em_scores'] else 0.0,
            'hit_at_1': sum(metrics['hit_at_1']) / len(metrics['hit_at_1']) if metrics['hit_at_1'] else 0.0,
            'hit_at_3': sum(metrics['hit_at_3']) / len(metrics['hit_at_3']) if metrics['hit_at_3'] else 0.0,
            'hit_at_5': sum(metrics['hit_at_5']) / len(metrics['hit_at_5']) if metrics['hit_at_5'] else 0.0,
            'f1_score': sum(metrics['f1_scores']) / len(metrics['f1_scores']) if metrics['f1_scores'] else 0.0,
            'bleu_score': sum(metrics['bleu_scores']) / len(metrics['bleu_scores']) if metrics['bleu_scores'] else 0.0,
            'avg_response_time': sum(metrics['response_times']) / len(metrics['response_times']) if metrics['response_times'] else 0.0,
            'total_queries': len(queries),
            'successful_queries': len(queries) - error_count,
            'failed_queries': error_count
        }
        
        return final_metrics
    
    def _calculate_answer_f1(self, predicted: str, ground_truth: str) -> float:
        """답변의 F1-score 계산 (토큰 기반)"""
        pred_tokens = set(RAGMetrics._tokenize(predicted))
        gt_tokens = set(RAGMetrics._tokenize(ground_truth))
        
        if len(gt_tokens) == 0:
            return 0.0
        
        if len(pred_tokens) == 0:
            return 0.0
        
        # Precision
        matches = len(pred_tokens & gt_tokens)
        precision = matches / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        
        # Recall
        recall = matches / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
        
        # F1
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _generate_expected_answer(self, query_info: Dict, retrieved_docs: List) -> str:
        """예상 답변 생성"""
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
        print("\n" + "=" * 70)
        print("HWP 직접 파싱 vs HWPX 변환 비교 결과")
        print("=" * 70)
        
        if 'qa' not in self.results:
            print("QA 결과가 없습니다.")
            return
        
        qa_results = self.results['qa']
        
        # 각 방법별 성능 출력
        method_names = {
            'hwp_direct': 'HWP 직접 파싱',
            'hwpx': 'HWPX 직접 파싱',
            'hwp_via_hwpx': 'HWP->HWPX 변환'
        }
        
        for method_key, method_name in method_names.items():
            if method_key not in qa_results:
                continue
            
            metrics = qa_results[method_key]
            print(f"\n[{method_name}]")
            print(f"  테스트 쿼리 수: {metrics['total_queries']}개")
            print(f"  성공한 쿼리: {metrics['successful_queries']}개")
            print(f"  실패한 쿼리: {metrics.get('failed_queries', 0)}개")
            print(f"  EM Score: {metrics['em_score']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Hit@1: {metrics['hit_at_1']:.4f}")
            print(f"  Hit@3: {metrics['hit_at_3']:.4f}")
            print(f"  Hit@5: {metrics['hit_at_5']:.4f}")
            print(f"  BLEU Score: {metrics['bleu_score']:.4f}")
            if metrics['avg_response_time'] > 0:
                print(f"  평균 응답 시간: {metrics['avg_response_time']:.2f}초")
        
        # 비교 분석
        if len(qa_results) > 1:
            print("\n[성능 개선 분석]")
            print("-" * 70)
            
            baseline_key = 'hwpx' if 'hwpx' in qa_results else 'hwp_via_hwpx'
            if baseline_key not in qa_results:
                baseline_key = list(qa_results.keys())[0]
            
            baseline_metrics = qa_results[baseline_key]
            baseline_name = method_names.get(baseline_key, baseline_key)
            
            print(f"\n베이스라인: {baseline_name}")
            
            for method_key, method_name in method_names.items():
                if method_key == baseline_key or method_key not in qa_results:
                    continue
                
                metrics = qa_results[method_key]
                print(f"\n[{method_name}] vs [{baseline_name}]")
                
                # EM Score 개선
                em_improvement = metrics['em_score'] - baseline_metrics['em_score']
                em_improvement_pct = (em_improvement / baseline_metrics['em_score'] * 100) if baseline_metrics['em_score'] > 0 else 0
                print(f"  EM Score: {metrics['em_score']:.4f} ({em_improvement:+.4f}, {em_improvement_pct:+.1f}%)")
                
                # F1 Score 개선
                f1_improvement = metrics['f1_score'] - baseline_metrics['f1_score']
                f1_improvement_pct = (f1_improvement / baseline_metrics['f1_score'] * 100) if baseline_metrics['f1_score'] > 0 else 0
                print(f"  F1 Score: {metrics['f1_score']:.4f} ({f1_improvement:+.4f}, {f1_improvement_pct:+.1f}%)")
                
                # Hit@K 개선
                hit1_improvement = metrics['hit_at_1'] - baseline_metrics['hit_at_1']
                hit1_improvement_pct = (hit1_improvement / baseline_metrics['hit_at_1'] * 100) if baseline_metrics['hit_at_1'] > 0 else 0
                print(f"  Hit@1: {metrics['hit_at_1']:.4f} ({hit1_improvement:+.4f}, {hit1_improvement_pct:+.1f}%)")
                
                hit3_improvement = metrics['hit_at_3'] - baseline_metrics['hit_at_3']
                hit3_improvement_pct = (hit3_improvement / baseline_metrics['hit_at_3'] * 100) if baseline_metrics['hit_at_3'] > 0 else 0
                print(f"  Hit@3: {metrics['hit_at_3']:.4f} ({hit3_improvement:+.4f}, {hit3_improvement_pct:+.1f}%)")
                
                hit5_improvement = metrics['hit_at_5'] - baseline_metrics['hit_at_5']
                hit5_improvement_pct = (hit5_improvement / baseline_metrics['hit_at_5'] * 100) if baseline_metrics['hit_at_5'] > 0 else 0
                print(f"  Hit@5: {metrics['hit_at_5']:.4f} ({hit5_improvement:+.4f}, {hit5_improvement_pct:+.1f}%)")
    
    def save_results(self, output_path: str = "results_hwp_comparison.json"):
        """결과 저장"""
        results_serializable = self._make_serializable(self.results)
        
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


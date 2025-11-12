"""
실험 실행 스크립트
HWPX vs PDF 표 추출 및 RAG 성능 비교
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from src.extractors import HWPXTableExtractor, PDFTableExtractor
from src.rag import TableRAGSystem
from src.evaluation import TableExtractionMetrics, RAGMetrics


class ExperimentRunner:
    """실험 실행 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.api_key = self.config['api']['gemini']['api_key']
        self.model_name = self.config['api']['gemini']['model']
        
        # 목표 메트릭
        self.targets = self.config['target_metrics']
        
        # 결과 저장
        self.results = {
            'hwpx': {},
            'pdf': {}
        }
    
    def run_table_extraction_experiment(self):
        """표 추출 실험 실행"""
        print("=" * 60)
        print("표 추출 실험 시작")
        print("=" * 60)
        
        hwpx_path = self.config['dataset']['hwpx_path']
        pdf_path = self.config['dataset']['pdf_path']
        gt_path = self.config['dataset']['ground_truth_path']
        
        # HWPX 추출
        print("\n[HWPX 표 추출]")
        hwpx_extractor = HWPXTableExtractor()
        hwpx_results = self._extract_tables_from_directory(
            hwpx_path, hwpx_extractor, 'hwpx'
        )
        
        # PDF 추출 (베이스라인)
        print("\n[PDF 표 추출 - 베이스라인]")
        pdf_results = {}
        
        for baseline in self.config['baselines']['pdf']:
            if baseline['enabled']:
                method = baseline['name']
                print(f"\n  방법: {method}")
                pdf_extractor = PDFTableExtractor(method=method)
                pdf_results[method] = self._extract_tables_from_directory(
                    pdf_path, pdf_extractor, f'pdf_{method}'
                )
        
        # Ground Truth 로드
        ground_truth = self._load_ground_truth(gt_path)
        
        # 평가
        print("\n[표 추출 평가]")
        hwpx_metrics = self._evaluate_extraction(
            hwpx_results, ground_truth, 'HWPX'
        )
        
        pdf_metrics_dict = {}
        for method, results in pdf_results.items():
            pdf_metrics_dict[method] = self._evaluate_extraction(
                results, ground_truth, f'PDF-{method}'
            )
        
        # 결과 저장
        self.results['hwpx']['extraction'] = {
            'results': hwpx_results,
            'metrics': hwpx_metrics
        }
        
        self.results['pdf']['extraction'] = {
            'results': pdf_results,
            'metrics': pdf_metrics_dict
        }
        
        # 비교 리포트
        self._print_extraction_comparison(hwpx_metrics, pdf_metrics_dict)
        
        return hwpx_results, pdf_results, ground_truth
    
    def _extract_tables_from_directory(self, directory: str, extractor, method_name: str) -> Dict:
        """디렉토리에서 표 추출"""
        results = {
            'tables': [],
            'files': [],
            'extraction_times': []
        }
        
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"  경고: 디렉토리가 존재하지 않습니다: {directory}")
            return results
        
        files = list(dir_path.glob("*.hwpx")) if method_name.startswith('hwpx') else list(dir_path.glob("*.pdf"))
        
        if not files:
            print(f"  경고: 파일을 찾을 수 없습니다: {directory}")
            return results
        
        for file_path in tqdm(files, desc=f"  {method_name} 추출"):
            try:
                tables = extractor.extract_tables(str(file_path))
                results['tables'].extend(tables)
                results['files'].append(str(file_path))
                
                if tables:
                    avg_time = sum(t.get('extraction_time', 0) for t in tables) / len(tables)
                    results['extraction_times'].append(avg_time)
            
            except Exception as e:
                print(f"  오류 ({file_path}): {e}")
        
        return results
    
    def _load_ground_truth(self, gt_path: str) -> Dict:
        """Ground Truth 로드"""
        gt_dict = {}
        
        gt_dir = Path(gt_path)
        if not gt_dir.exists():
            print(f"  경고: Ground Truth 디렉토리가 없습니다: {gt_path}")
            return gt_dict
        
        # JSON 형식의 Ground Truth 파일들
        for gt_file in gt_dir.glob("*.json"):
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    gt_dict[gt_file.stem] = data
            except Exception as e:
                print(f"  Ground Truth 로드 오류 ({gt_file}): {e}")
        
        return gt_dict
    
    def _evaluate_extraction(self, results: Dict, ground_truth: Dict, method_name: str) -> Dict:
        """표 추출 결과 평가"""
        metrics = {
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'extraction_time': {},
            'table_count': len(results['tables'])
        }
        
        if not results['tables']:
            print(f"  {method_name}: 추출된 표가 없습니다.")
            return metrics
        
        # F1-score 계산 (Ground Truth가 있는 경우)
        if ground_truth:
            # 간단한 평가: 표 개수 비교
            pred_count = len(results['tables'])
            gt_count = sum(len(gt.get('tables', [])) for gt in ground_truth.values())
            
            if gt_count > 0:
                precision = min(pred_count / gt_count, 1.0) if pred_count > 0 else 0.0
                recall = min(pred_count / gt_count, 1.0) if gt_count > 0 else 0.0
                
                if precision + recall > 0:
                    metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
                metrics['precision'] = precision
                metrics['recall'] = recall
        
        # 추출 시간 통계
        if results['extraction_times']:
            metrics['extraction_time'] = TableExtractionMetrics.calculate_extraction_time(
                results['extraction_times']
            )
        
        print(f"  {method_name}:")
        print(f"    표 개수: {metrics['table_count']}")
        print(f"    F1-score: {metrics['f1_score']:.4f}")
        print(f"    평균 추출 시간: {metrics['extraction_time'].get('mean', 0):.4f}초")
        
        return metrics
    
    def _print_extraction_comparison(self, hwpx_metrics: Dict, pdf_metrics_dict: Dict):
        """추출 결과 비교 리포트"""
        print("\n" + "=" * 60)
        print("표 추출 성능 비교")
        print("=" * 60)
        
        hwpx_f1 = hwpx_metrics['f1_score']
        hwpx_time = hwpx_metrics['extraction_time'].get('mean', 0)
        
        print(f"\n[HWPX]")
        print(f"  F1-score: {hwpx_f1:.4f}")
        print(f"  평균 추출 시간: {hwpx_time:.4f}초")
        
        print(f"\n[PDF 베이스라인]")
        best_pdf_f1 = 0.0
        best_pdf_method = ""
        best_pdf_time = float('inf')
        
        for method, metrics in pdf_metrics_dict.items():
            pdf_f1 = metrics['f1_score']
            pdf_time = metrics['extraction_time'].get('mean', 0)
            
            print(f"  {method}:")
            print(f"    F1-score: {pdf_f1:.4f}")
            print(f"    평균 추출 시간: {pdf_time:.4f}초")
            
            if pdf_f1 > best_pdf_f1:
                best_pdf_f1 = pdf_f1
                best_pdf_method = method
            
            if pdf_time < best_pdf_time and pdf_time > 0:
                best_pdf_time = pdf_time
        
        # 개선율 계산
        if best_pdf_f1 > 0:
            f1_improvement = ((hwpx_f1 - best_pdf_f1) / best_pdf_f1) * 100
            print(f"\n[개선율]")
            print(f"  F1-score: {f1_improvement:+.2f}%")
            
            if best_pdf_time > 0:
                speed_improvement = ((best_pdf_time - hwpx_time) / best_pdf_time) * 100
                print(f"  처리 속도: {speed_improvement:+.2f}%")
        
        # 목표 달성 여부
        target_f1_improvement = self.targets['table_extraction']['f1_score_improvement'] * 100
        target_speed_improvement = self.targets['table_extraction']['speed_improvement'] * 100
        
        print(f"\n[목표 달성]")
        print(f"  F1-score 목표: {target_f1_improvement:.1f}% 향상")
        print(f"    달성: {'✓' if f1_improvement >= target_f1_improvement else '✗'}")
        print(f"  처리 속도 목표: {target_speed_improvement:.1f}% 향상")
        print(f"    달성: {'✓' if speed_improvement >= target_speed_improvement else '✗'}")
    
    def run_rag_experiment(self, hwpx_results: Dict, pdf_results: Dict, ground_truth: Dict):
        """RAG 실험 실행"""
        print("\n" + "=" * 60)
        print("RAG 성능 실험 시작")
        print("=" * 60)
        
        # 질문 로드
        questions_path = self.config['dataset']['questions_path']
        questions = self._load_questions(questions_path)
        
        if not questions:
            print("  경고: 질문 파일을 찾을 수 없습니다.")
            return
        
        # HWPX 기반 RAG
        print("\n[HWPX 기반 RAG 시스템]")
        hwpx_rag = TableRAGSystem(self.api_key, self.model_name)
        hwpx_rag.build_knowledge_base(hwpx_results['tables'])
        hwpx_rag_metrics = self._evaluate_rag(hwpx_rag, questions, ground_truth, 'HWPX')
        
        # PDF 기반 RAG (최고 성능 베이스라인 사용)
        print("\n[PDF 기반 RAG 시스템 - 베이스라인]")
        pdf_rag_metrics_dict = {}
        
        for method, results in pdf_results.items():
            if results['tables']:
                print(f"\n  방법: {method}")
                pdf_rag = TableRAGSystem(self.api_key, self.model_name)
                pdf_rag.build_knowledge_base(results['tables'])
                pdf_rag_metrics_dict[method] = self._evaluate_rag(
                    pdf_rag, questions, ground_truth, f'PDF-{method}'
                )
                pdf_rag.clear_knowledge_base()
        
        hwpx_rag.clear_knowledge_base()
        
        # 결과 저장
        self.results['hwpx']['rag'] = hwpx_rag_metrics
        self.results['pdf']['rag'] = pdf_rag_metrics_dict
        
        # 비교 리포트
        self._print_rag_comparison(hwpx_rag_metrics, pdf_rag_metrics_dict)
    
    def _load_questions(self, questions_path: str) -> List[Dict]:
        """질문 파일 로드"""
        q_path = Path(questions_path)
        
        if not q_path.exists():
            print(f"  경고: 질문 파일이 없습니다: {questions_path}")
            return []
        
        try:
            with open(q_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'questions' in data:
                    return data['questions']
                else:
                    return []
        except Exception as e:
            print(f"  질문 파일 로드 오류: {e}")
            return []
    
    def _evaluate_rag(self, rag_system: TableRAGSystem, questions: List[Dict], 
                     ground_truth: Dict, method_name: str) -> Dict:
        """RAG 시스템 평가"""
        metrics = {
            'em_scores': [],
            'hit_at_1': [],
            'hit_at_3': [],
            'hit_at_5': [],
            'bleu_scores': []
        }
        
        for q_data in tqdm(questions, desc=f"  {method_name} 평가"):
            question = q_data.get('question', '')
            expected_answer = q_data.get('answer', '')
            relevant_table_ids = q_data.get('relevant_tables', [])
            
            if not question:
                continue
            
            try:
                result = rag_system.query(question)
                predicted_answer = result['answer']
                retrieved_docs = result.get('source_documents', [])
                
                # EM Score
                em = RAGMetrics.calculate_em_score(predicted_answer, expected_answer)
                metrics['em_scores'].append(em)
                
                # Hit@K
                hit1 = RAGMetrics.calculate_hit_at_k(retrieved_docs, relevant_table_ids, 1)
                hit3 = RAGMetrics.calculate_hit_at_k(retrieved_docs, relevant_table_ids, 3)
                hit5 = RAGMetrics.calculate_hit_at_k(retrieved_docs, relevant_table_ids, 5)
                
                metrics['hit_at_1'].append(hit1)
                metrics['hit_at_3'].append(hit3)
                metrics['hit_at_5'].append(hit5)
                
                # BLEU Score
                bleu = RAGMetrics.calculate_bleu_score(predicted_answer, expected_answer)
                metrics['bleu_scores'].append(bleu)
            
            except Exception as e:
                print(f"    질문 처리 오류: {e}")
                metrics['em_scores'].append(0.0)
                metrics['hit_at_1'].append(0.0)
                metrics['hit_at_3'].append(0.0)
                metrics['hit_at_5'].append(0.0)
                metrics['bleu_scores'].append(0.0)
        
        # 평균 계산
        final_metrics = {
            'em_score': sum(metrics['em_scores']) / len(metrics['em_scores']) if metrics['em_scores'] else 0.0,
            'hit_at_1': sum(metrics['hit_at_1']) / len(metrics['hit_at_1']) if metrics['hit_at_1'] else 0.0,
            'hit_at_3': sum(metrics['hit_at_3']) / len(metrics['hit_at_3']) if metrics['hit_at_3'] else 0.0,
            'hit_at_5': sum(metrics['hit_at_5']) / len(metrics['hit_at_5']) if metrics['hit_at_5'] else 0.0,
            'bleu_score': sum(metrics['bleu_scores']) / len(metrics['bleu_scores']) if metrics['bleu_scores'] else 0.0
        }
        
        print(f"  {method_name} 결과:")
        print(f"    EM Score: {final_metrics['em_score']:.4f}")
        print(f"    Hit@1: {final_metrics['hit_at_1']:.4f}")
        print(f"    Hit@3: {final_metrics['hit_at_3']:.4f}")
        print(f"    Hit@5: {final_metrics['hit_at_5']:.4f}")
        print(f"    BLEU Score: {final_metrics['bleu_score']:.4f}")
        
        return final_metrics
    
    def _print_rag_comparison(self, hwpx_metrics: Dict, pdf_metrics_dict: Dict):
        """RAG 결과 비교 리포트"""
        print("\n" + "=" * 60)
        print("RAG 성능 비교")
        print("=" * 60)
        
        print(f"\n[HWPX 기반 RAG]")
        print(f"  EM Score: {hwpx_metrics['em_score']:.4f}")
        print(f"  Hit@1: {hwpx_metrics['hit_at_1']:.4f}")
        print(f"  Hit@3: {hwpx_metrics['hit_at_3']:.4f}")
        print(f"  Hit@5: {hwpx_metrics['hit_at_5']:.4f}")
        
        print(f"\n[PDF 기반 RAG - 베이스라인]")
        best_pdf_em = 0.0
        best_pdf_method = ""
        
        for method, metrics in pdf_metrics_dict.items():
            print(f"  {method}:")
            print(f"    EM Score: {metrics['em_score']:.4f}")
            print(f"    Hit@1: {metrics['hit_at_1']:.4f}")
            print(f"    Hit@3: {metrics['hit_at_3']:.4f}")
            print(f"    Hit@5: {metrics['hit_at_5']:.4f}")
            
            if metrics['em_score'] > best_pdf_em:
                best_pdf_em = metrics['em_score']
                best_pdf_method = method
        
        # 개선율 계산
        if best_pdf_em > 0:
            em_improvement = ((hwpx_metrics['em_score'] - best_pdf_em) / best_pdf_em) * 100
            print(f"\n[개선율]")
            print(f"  EM Score: {em_improvement:+.2f}%")
            
            # 목표 달성 여부
            target_em_improvement = self.targets['rag_performance']['em_score_improvement'] * 100
            print(f"\n[목표 달성]")
            print(f"  EM Score 목표: {target_em_improvement:.1f}% 향상")
            print(f"    달성: {'✓' if em_improvement >= target_em_improvement else '✗'}")
    
    def save_results(self, output_path: str = "results.json"):
        """결과 저장"""
        # JSON 직렬화 가능하도록 변환
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
    
    def run_full_experiment(self):
        """전체 실험 실행"""
        print("=" * 60)
        print("HWPX vs PDF Table Extraction RAG 실험")
        print("=" * 60)
        
        # 1. 표 추출 실험
        hwpx_results, pdf_results, ground_truth = self.run_table_extraction_experiment()
        
        # 2. RAG 실험
        self.run_rag_experiment(hwpx_results, pdf_results, ground_truth)
        
        # 3. 결과 저장
        self.save_results()
        
        print("\n" + "=" * 60)
        print("실험 완료!")
        print("=" * 60)


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_full_experiment()


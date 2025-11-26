#!/usr/bin/env python3
"""
모의 실험 실행 스크립트 (API 키 없이 실험 구조 테스트)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import EvaluationFramework, QA, QAMetrics
from src.generation.prompting import Answer, CellRef
from typing import List, Dict

def run_mock_experiment(dataset_path: str, output_dir: str = "experiments/mock_test"):
    """모의 실험 실행 (API 호출 없이)"""
    
    # 데이터셋 로드
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    qa_pairs = dataset.get('qa_pairs', [])
    tables = dataset.get('tables', {})
    
    print(f"데이터셋 로드 완료:")
    print(f"  QA 쌍: {len(qa_pairs)}개")
    print(f"  표: {len(tables)}개")
    
    # Ground truth 생성
    ground_truth = []
    for qa_data in qa_pairs:
        cell_refs = [
            CellRef(
                row=cell.get('row', 0),
                col=cell.get('col', 0),
                format=cell.get('format', 'tuple'),
                original_text=cell.get('original_text', ''),
            )
            for cell in qa_data.get('required_cells', [])
        ]
        
        answer_text = str(qa_data.get('answer', ''))
        if isinstance(qa_data.get('answer'), list):
            answer_text = ', '.join(str(a) for a in qa_data['answer'])
        
        qa = QA(
            question=qa_data['question'],
            answer=answer_text,
            gold_cells=cell_refs,
            query_type=qa_data.get('question_type', 'factual'),
            metadata=qa_data.get('metadata', {}),
        )
        ground_truth.append(qa)
    
    # 모의 예측 생성 (실제 API 호출 대신)
    predictions = []
    for i, qa_data in enumerate(qa_pairs):
        # 간단한 모의 답변 생성
        answer_text = str(qa_data.get('answer', ''))
        if isinstance(qa_data.get('answer'), list):
            answer_text = ', '.join(str(a) for a in qa_data['answer'])
        
        # 약간의 변형을 주어 완벽하지 않은 답변 시뮬레이션
        if i % 3 == 0:  # 일부는 정확한 답변
            mock_answer = answer_text
        elif i % 3 == 1:  # 일부는 부분적으로 정확
            mock_answer = answer_text + " (추가 정보)"
        else:  # 일부는 부정확
            mock_answer = "답변을 찾을 수 없습니다."
        
        # 셀 참조 생성
        cited_cells = []
        for cell in qa_data.get('required_cells', [])[:2]:  # 일부만 인용
            cited_cells.append(
                CellRef(
                    row=cell.get('row', 0),
                    col=cell.get('col', 0),
                    format=cell.get('format', 'tuple'),
                    original_text=cell.get('original_text', ''),
                )
            )
        
        answer = Answer(
            answer_text=mock_answer,
            model="mock_model",
            cited_cells=cited_cells,
        )
        predictions.append(answer)
    
    print(f"\n모의 예측 생성 완료: {len(predictions)}개")
    
    # 평가 실행
    print("\n평가 실행 중...")
    evaluator = EvaluationFramework()
    
    results = evaluator.evaluate_on_dataset(
        predictions=predictions,
        ground_truth=ground_truth,
        contexts=None,  # 모의 실험이므로 컨텍스트 없음
    )
    
    # 결과 출력
    print("\n" + "="*60)
    print("실험 결과")
    print("="*60)
    
    overall = results['overall_metrics']
    print(f"\n전체 메트릭:")
    print(f"  Exact Match: {overall['exact_match']:.4f}")
    print(f"  F1 Score: {overall['f1']:.4f}")
    print(f"  Structure Recall: {overall['structure_recall']:.4f}")
    
    # 쿼리 타입별 분석
    if results.get('query_type_breakdown'):
        print(f"\n쿼리 타입별 분석:")
        for qtype, stats in results['query_type_breakdown'].items():
            print(f"  {qtype}:")
            print(f"    개수: {stats['count']}")
            print(f"    EM: {stats['exact_match']:.4f}")
            print(f"    F1: {stats['f1']:.4f}")
    
    # 오류 분석
    error_analysis = results.get('error_analysis', [])
    if error_analysis:
        print(f"\n오류 분석 (상위 5개):")
        for i, error in enumerate(error_analysis[:5]):
            print(f"  {i+1}. 질문: {error['question'][:50]}...")
            print(f"     예측: {error['predicted'][:50]}...")
            print(f"     정답: {error['gold'][:50]}...")
            print(f"     EM: {error['exact_match']:.4f}, F1: {error['f1']:.4f}")
    
    # 결과 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n결과 저장: {results_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="모의 실험 실행")
    parser.add_argument("--dataset", type=str, default="data/hitab_experiment_dataset.json",
                       help="데이터셋 경로")
    parser.add_argument("--output-dir", type=str, default="experiments/mock_test",
                       help="출력 디렉토리")
    
    args = parser.parse_args()
    
    run_mock_experiment(args.dataset, args.output_dir)


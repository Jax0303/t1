#!/usr/bin/env python3
"""
QA 성능 비교 실험 실행 스크립트
"""
from src.qa_comparison import QAComparisonExperiment

if __name__ == "__main__":
    experiment = QAComparisonExperiment()
    
    # 샘플 쿼리 생성
    print("=" * 60)
    print("샘플 쿼리 생성")
    print("=" * 60)
    queries = experiment.generate_sample_queries(30)
    print(f"\n생성된 쿼리: {len(queries)}개")
    
    # 쿼리 로드 (이미 있으면 사용)
    import json
    from pathlib import Path
    
    query_file = Path('sample_queries.json')
    if query_file.exists():
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        print(f"기존 쿼리 파일 로드: {len(queries)}개")
    else:
        with open('sample_queries.json', 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        print(f"쿼리 저장: {len(queries)}개")
    
    # QA 비교 실험 실행
    experiment.run_qa_comparison(queries)
    
    # 리포트 출력
    experiment.print_comparison_report()
    
    # 결과 저장
    experiment.save_results()
    
    print("\n" + "=" * 60)
    print("QA 비교 실험 완료!")
    print("=" * 60)


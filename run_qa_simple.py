#!/usr/bin/env python3
"""
간단한 QA 비교 실험 (API 키 없이도 실행 가능)
"""
import json
import pandas as pd
from src.qa_comparison import QAComparisonExperiment

# 쿼리 생성만 먼저 실행
experiment = QAComparisonExperiment()
queries = experiment.generate_sample_queries(30)

print(f"생성된 쿼리: {len(queries)}개")
with open('sample_queries.json', 'w', encoding='utf-8') as f:
    json.dump(queries, f, ensure_ascii=False, indent=2)

print("\n생성된 쿼리 샘플:")
for i, q in enumerate(queries[:10]):
    print(f"{i+1:2d}. {q['question']}")

print(f"\n총 {len(queries)}개 쿼리가 생성되었습니다.")
print("API 키 문제 해결 후 run_qa_comparison.py를 실행하세요.")


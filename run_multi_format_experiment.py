#!/usr/bin/env python3
"""
다중 형식 비교 실험 실행 스크립트
"""
from src.experiment_multi_format import MultiFormatComparisonExperiment

if __name__ == "__main__":
    experiment = MultiFormatComparisonExperiment()
    
    # 다중 형식 비교 실험
    print("=" * 60)
    print("다중 형식 비교 실험")
    print("=" * 60)
    experiment.run_multi_format_comparison("dataset2")
    
    # 결과 저장
    experiment.save_results("results_multi_format.json")
    
    print("\n" + "=" * 60)
    print("실험 완료!")
    print("=" * 60)


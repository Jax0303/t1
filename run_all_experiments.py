#!/usr/bin/env python3
"""
전체 실험 실행 스크립트
1. 다중 형식 비교 실험
2. 라벨링 데이터 비교 실험
"""
from src.experiment_multi_format import MultiFormatComparisonExperiment
from src.experiment_with_labels import LabelComparisonExperiment

if __name__ == "__main__":
    print("=" * 60)
    print("전체 실험 실행")
    print("=" * 60)
    
    # 1. 다중 형식 비교 실험
    print("\n" + "=" * 60)
    print("1. 다중 형식 비교 실험")
    print("=" * 60)
    
    multi_format_exp = MultiFormatComparisonExperiment()
    multi_format_exp.run_multi_format_comparison("dataset2")
    multi_format_exp.save_results("results_multi_format.json")
    
    # 2. 라벨링 데이터 비교 실험
    print("\n" + "=" * 60)
    print("2. 라벨링 데이터 비교 실험")
    print("=" * 60)
    
    label_exp = LabelComparisonExperiment()
    label_exp.run_label_comparison(
        "dataset1/149.표 정보 질의 응답 데이터",
        "dataset2"
    )
    label_exp.save_results("results_label_comparison.json")
    
    print("\n" + "=" * 60)
    print("모든 실험 완료!")
    print("=" * 60)
    print("\n결과 파일:")
    print("  - results_multi_format.json")
    print("  - results_label_comparison.json")


#!/usr/bin/env python3
"""
HWP 직접 파싱 vs HWPX 변환 비교 실험 실행 스크립트
"""
import json
from pathlib import Path
from src.experiment_hwp_comparison import HWPComparisonExperiment
from src.data.dataset_loader import DatasetLoader


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("HWP 직접 파싱 vs HWPX 변환 비교 실험")
    print("=" * 70)
    
    # 데이터셋 로더 초기화
    loader = DatasetLoader()
    
    # HWP 파일 찾기
    hwp_files = []
    hwpx_files = []
    
    # dataset2에서 HWP/HWPX 파일 찾기
    dataset2_path = Path("data/raw/dataset2")
    if dataset2_path.exists():
        for file_path in dataset2_path.glob("*.hwp"):
            hwp_files.append(str(file_path))
        
        for file_path in dataset2_path.glob("*.hwpx"):
            hwpx_files.append(str(file_path))
    
    if not hwp_files:
        print("경고: HWP 파일을 찾을 수 없습니다.")
        print("data/raw/dataset2/ 디렉토리에 HWP 파일이 있는지 확인하세요.")
        return
    
    print(f"\n발견된 파일:")
    print(f"  HWP 파일: {len(hwp_files)}개")
    print(f"  HWPX 파일: {len(hwpx_files)}개")
    
    # 쿼리 로드 또는 생성
    query_file = Path('sample_queries.json')
    if query_file.exists():
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        print(f"\n쿼리 파일 로드: {len(queries)}개")
    else:
        print("\n쿼리 파일이 없습니다. 기본 쿼리를 생성합니다.")
        queries = [
            {
                'question': '표의 첫 번째 행 정보를 알려주세요',
                'table_id': '',
                'expected_info': {}
            },
            {
                'question': '표의 헤더(열 이름)는 무엇인가요?',
                'table_id': '',
                'expected_info': {}
            },
            {
                'question': '표의 총 행 수는 몇 개인가요?',
                'table_id': '',
                'expected_info': {}
            }
        ]
        print(f"생성된 기본 쿼리: {len(queries)}개")
    
    # 실험 실행
    experiment = HWPComparisonExperiment()
    
    results = experiment.run_comparison(
        hwp_files=hwp_files,
        queries=queries,
        hwpx_files=hwpx_files if hwpx_files else None
    )
    
    # 리포트 출력
    experiment.print_comparison_report()
    
    # 결과 저장
    experiment.save_results()
    
    print("\n" + "=" * 70)
    print("실험 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()


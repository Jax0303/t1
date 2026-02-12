"""
Complete Analysis Pipeline
실험 실행 → 시각화 → 리포트 생성 전체 파이프라인
"""
import sys
from pathlib import Path
import argparse
import json

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.run_experiment import ExperimentRunner
from visualization.error_heatmap import ErrorHeatmapGenerator, PerformanceChartGenerator
from visualization.report_generator import generate_report_from_experiment


def run_complete_analysis(
    datasets=['synthetic'],
    max_samples=50,
    results_dir='results/experiments'
):
    """
    전체 분석 파이프라인 실행
    
    Args:
        datasets: 데이터셋 리스트
        max_samples: 최대 샘플 수
        results_dir: 결과 디렉토리
    """
    print("="*80)
    print("COMPREHENSIVE TABLE PARSING EXPERIMENT")
    print("="*80)
    
    # 1. 실험 실행
    print("\n[Step 1/3] Running Experiment...")
    config = {'results_dir': results_dir}
    runner = ExperimentRunner(config)
    runner.run_experiment(datasets=datasets, max_samples_per_dataset=max_samples)
    
    experiment_dir = runner.experiment_dir
    
    # 2. 시각화 생성
    print("\n[Step 2/3] Generating Visualizations...")
    
    # 결과 데이터 로드
    summary_file = experiment_dir / 'summary_statistics.json'
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_stats = json.load(f)
        
        viz_dir = experiment_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 히트맵 생성
        heatmap_gen = ErrorHeatmapGenerator(summary_stats)
        heatmap_gen.generate_all_heatmaps(str(viz_dir))
        
        # 성능 차트 생성
        chart_gen = PerformanceChartGenerator(summary_stats)
        chart_gen.generate_all_charts(str(viz_dir))
    
    # 3. HTML 리포트 생성
    print("\n[Step 3/3] Generating Report...")
    report_path = generate_report_from_experiment(str(experiment_dir))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Results Directory: {experiment_dir}")
    print(f"📊 HTML Report: {report_path}")
    print(f"📈 Visualizations: {experiment_dir / 'visualizations'}")
    print(f"📋 Raw Data: {experiment_dir / 'all_results.json'}")
    print("\n" + "="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run complete table parsing analysis pipeline'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['synthetic'],
        help='Datasets to use (synthetic, pubtabnet, etc.)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=30,
        help='Maximum samples per dataset'
    )
    parser.add_argument(
        '--results-dir',
        default='results/experiments',
        help='Results output directory'
    )
    
    args = parser.parse_args()
    
    run_complete_analysis(
        datasets=args.datasets,
        max_samples=args.max_samples,
        results_dir=args.results_dir
    )

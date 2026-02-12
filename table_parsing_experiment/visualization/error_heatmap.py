"""
Visualization: Error Heatmap
오류 유형별 분포 히트맵 생성
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


class ErrorHeatmapGenerator:
    """오류 분포 히트맵 생성기"""
    
    def __init__(self, results_data: Dict[str, Any]):
        """
        Args:
            results_data: 실험 결과 요약 데이터
        """
        self.results_data = results_data
        self.output_dir = None
    
    def generate_table_type_error_heatmap(
        self,
        output_path: str,
        figsize: tuple = (12, 8)
    ):
        """
        표 유형 × 오류 타입 히트맵
        
        Args:
            output_path: 출력 파일 경로
            figsize: 그림 크기
        """
        # 데이터 추출
        by_table_type = self.results_data.get('by_table_type', {})
        
        if not by_table_type:
            print("No table type data available for heatmap")
            return
        
        # 모든 오류 유형 수집
        all_error_types = set()
        for data in by_table_type.values():
            all_error_types.update(data.get('errors', {}).keys())
        
        all_error_types = sorted(list(all_error_types))
        table_types = sorted(by_table_type.keys())
        
        # 히트맵 데이터 생성
        heatmap_data = np.zeros((len(table_types), len(all_error_types)))
        
        for i, table_type in enumerate(table_types):
            errors = by_table_type[table_type].get('errors', {})
            total_samples = by_table_type[table_type].get('count', 1)
            
            for j, error_type in enumerate(all_error_types):
                # 정규화: 전체 샘플 수 대비 비율
                error_count = errors.get(error_type, 0)
                heatmap_data[i, j] = (error_count / total_samples * 100) if total_samples > 0 else 0
        
        # 히트맵 그리기
        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_data,
            xticklabels=all_error_types,
            yticklabels=table_types,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Error Rate (%)'}
        )
        
        plt.title('Error Distribution by Table Type', fontsize=16, fontweight='bold')
        plt.xlabel('Error Type', fontsize=12)
        plt.ylabel('Table Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 저장
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved table type error heatmap to {output_path}")
        plt.close()
    
    def generate_model_error_heatmap(
        self,
        output_path: str,
        figsize: tuple = (14, 6)
    ):
        """
        모델 × 오류 타입 히트맵
        
        Args:
            output_path: 출력 파일 경로
            figsize: 그림 크기
        """
        by_model = self.results_data.get('by_model', {})
        
        if not by_model:
            print("No model data available for heatmap")
            return
        
        # 모든 오류 유형 수집
        all_error_types = set()
        for data in by_model.values():
            all_error_types.update(data.get('errors', {}).keys())
        
        all_error_types = sorted(list(all_error_types))
        models = sorted(by_model.keys())
        
        # 히트맵 데이터 생성
        heatmap_data = np.zeros((len(models), len(all_error_types)))
        
        for i, model in enumerate(models):
            errors = by_model[model].get('errors', {})
            total_samples = by_model[model].get('count', 1)
            
            for j, error_type in enumerate(all_error_types):
                error_count = errors.get(error_type, 0)
                heatmap_data[i, j] = (error_count / total_samples * 100) if total_samples > 0 else 0
        
        # 히트맵 그리기
        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_data,
            xticklabels=all_error_types,
            yticklabels=models,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Error Rate (%)'}
        )
        
        plt.title('Error Distribution by Model', fontsize=16, fontweight='bold')
        plt.xlabel('Error Type', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved model error heatmap to {output_path}")
        plt.close()
    
    def generate_all_heatmaps(self, output_dir: str):
        """모든 히트맵 생성"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.generate_table_type_error_heatmap(
            str(output_path / 'error_by_table_type.png')
        )
        
        self.generate_model_error_heatmap(
            str(output_path / 'error_by_model.png')
        )


class PerformanceChartGenerator:
    """성능 비교 차트 생성기"""
    
    def __init__(self, results_data: Dict[str, Any]):
        self.results_data = results_data
    
    def generate_model_comparison_chart(
        self,
        output_path: str,
        metric: str = 'teds',
        figsize: tuple = (12, 6)
    ):
        """
        모델별 성능 비교 막대 그래프
        
        Args:
            output_path: 출력 경로
            metric: 비교할 메트릭 (teds, s_teds, etc.)
            figsize: 그림 크기
        """
        by_model = self.results_data.get('by_model', {})
        
        if not by_model:
            print("No model data available")
            return
        
        # 데이터 추출
        models = []
        scores = []
        
        for model, data in by_model.items():
            metrics_avg = data.get('metrics_avg', {})
            if metric in metrics_avg:
                models.append(model)
                scores.append(metrics_avg[metric])
        
        if not models:
            print(f"No data for metric {metric}")
            return
        
        # 막대 그래프
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(models)), scores, color='steelblue', alpha=0.8)
        
        # 값 표시
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(
                i, score + 0.02,
                f'{score:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.title(f'Model Comparison: {metric.upper()}', fontsize=16, fontweight='bold')
        plt.xticks(range(len(models)), models, rotation=15, ha='right')
        plt.ylim(0, max(scores) * 1.1 if scores else 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved model comparison chart to {output_path}")
        plt.close()
    
    def generate_table_type_performance_chart(
        self,
        output_path: str,
        metric: str = 's_teds',
        figsize: tuple = (10, 6)
    ):
        """표 유형별 성능 비교"""
        by_table_type = self.results_data.get('by_table_type', {})
        
        if not by_table_type:
            print("No table type data available")
            return
        
        table_types = []
        scores = []
        
        for table_type, data in by_table_type.items():
            metrics_avg = data.get('metrics_avg', {})
            if metric in metrics_avg:
                table_types.append(table_type)
                scores.append(metrics_avg[metric])
        
        if not table_types:
            print(f"No data for metric {metric}")
            return
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(table_types)), scores, color='coral', alpha=0.8)
        
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(
                i, score + 0.02,
                f'{score:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.xlabel('Table Type', fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.title(f'Performance by Table Type: {metric.upper()}', fontsize=16, fontweight='bold')
        plt.xticks(range(len(table_types)), table_types, rotation=30, ha='right')
        plt.ylim(0, max(scores) * 1.1 if scores else 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved table type performance chart to {output_path}")
        plt.close()
    
    def generate_all_charts(self, output_dir: str):
        """모든 차트 생성"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 여러 메트릭에 대해 차트 생성
        for metric in ['teds', 's_teds', 'td_f1', 'car_similarity']:
            self.generate_model_comparison_chart(
                str(output_path / f'model_comparison_{metric}.png'),
                metric=metric
            )
            
            self.generate_table_type_performance_chart(
                str(output_path / f'table_type_performance_{metric}.png'),
                metric=metric
            )

"""
Comprehensive Experiment Runner
다중 모델, 다중 데이터셋으로 대규모 실험 수행
"""
import sys
from pathlib import Path
import argparse
import json
from typing import Dict, List, Any
from tqdm import tqdm
import time
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import MODEL_CONFIG, METRICS_CONFIG, DATASET_CONFIG
from experiments.pipeline import TableParsingPipeline
from data.dataset_loader import get_dataset_loader, DatasetSample
from data.table_classifier import TableClassifier
from evaluation import ErrorAnalyzer


class ExperimentRunner:
    """종합 실험 실행 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 실험 설정
        """
        self.config = config
        self.results_dir = Path(config.get('results_dir', 'results/experiments'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 ID (타임스탬프)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 파이프라인들 (모델 조합별)
        self.pipelines = {}
        
        # 표 분류기
        self.table_classifier = TableClassifier()
        
        # 결과 저장
        self.all_results = []
        self.summary_stats = {
            'by_table_type': {},
            'by_model': {},
            'by_dataset': {}
        }
    
    def _init_pipelines(self):
        """다양한 모델 조합으로 파이프라인 초기화"""
        print("Initializing pipelines with different model combinations...")
        
        # OCR 엔진 목록
        ocr_engines = []
        for ocr_name, ocr_cfg in MODEL_CONFIG['ocr_engines'].items():
            if ocr_cfg.get('enabled', True):
                ocr_engines.append((ocr_name, ocr_cfg))
        
        # 각 OCR 엔진별로 파이프라인 생성
        for ocr_name, ocr_cfg in ocr_engines:
            pipeline_config = {
                'ocr': {'type': ocr_name, **ocr_cfg},
                'table_detection': {'type': 'huggingface', **MODEL_CONFIG['table_detection']},
                'table_structure': {'type': 'huggingface', **MODEL_CONFIG['table_structure']},
                'metrics': METRICS_CONFIG
            }
            
            pipeline_id = f"td_hf_tsr_hf_ocr_{ocr_name}"
            
            try:
                self.pipelines[pipeline_id] = TableParsingPipeline(pipeline_config)
                print(f"  ✓ Initialized pipeline: {pipeline_id}")
            except Exception as e:
                print(f"  ✗ Failed to initialize {pipeline_id}: {e}")
        
        if not self.pipelines:
            raise RuntimeError("No pipelines were successfully initialized!")
    
    def run_experiment(
        self,
        datasets: List[str],
        max_samples_per_dataset: int = 100
    ):
        """
        전체 실험 실행
        
        Args:
            datasets: 데이터셋 이름 리스트
            max_samples_per_dataset: 각 데이터셋당 최대 샘플 수
        """
        print("="*60)
        print(f"Starting Experiment: {self.experiment_id}")
        print("="*60)
        
        # 파이프라인 초기화
        self._init_pipelines()
        
        # 각 데이터셋별로 실험
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")
            
            # 데이터셋 로드
            samples = self._load_dataset(dataset_name, max_samples_per_dataset)
            
            if not samples:
                print(f"No samples loaded from {dataset_name}, skipping...")
                continue
            
            # 각 샘플에 대해 실험
            for sample_idx, sample in enumerate(tqdm(samples, desc=f"Processing {dataset_name}")):
                self._process_sample(sample, dataset_name, sample_idx)
        
        # 결과 분석 및 저장
        self._analyze_and_save_results()
        
        print("\n" + "="*60)
        print(f"Experiment Completed: {self.experiment_id}")
        print(f"Results saved to: {self.experiment_dir}")
        print("="*60)
    
    def _load_dataset(self, dataset_name: str, max_samples: int) -> List[DatasetSample]:
        """데이터셋 로드"""
        try:
            if dataset_name == 'synthetic':
                loader = get_dataset_loader('synthetic', str(self.results_dir / 'synthetic_data'))
                # 합성 데이터 생성
                table_types = DATASET_CONFIG.get('table_types', ['fully_bordered', 'unbordered'])
                samples = loader.generate_samples(num_samples=max_samples, table_types=table_types)
            elif dataset_name == 'pubtabnet':
                # PubTabNet은 데이터가 있다고 가정
                data_dir = self.config.get('pubtabnet_dir', '/data/pubtabnet')
                loader = get_dataset_loader('pubtabnet', data_dir)
                samples = loader.load_samples(split='val', max_samples=max_samples)
            else:
                print(f"Dataset {dataset_name} not yet implemented")
                return []
            
            return samples
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []
    
    def _process_sample(self, sample: DatasetSample, dataset_name: str, sample_idx: int):
        """단일 샘플 처리 (모든 파이프라인으로)"""
        # 이미지 로드
        try:
            image = sample.load_image()
        except Exception as e:
            print(f"Error loading image {sample.sample_id}: {e}")
            return
        
        # 표 유형 분류
        table_classification = self.table_classifier.classify_table(image)
        table_type = table_classification.get('type', ['unknown'])[0]
        
        # 각 파이프라인으로 실행
        for pipeline_id, pipeline in self.pipelines.items():
            try:
                start_time = time.time()
                
                # 파싱
                parse_results = pipeline.parse_image(image, return_intermediates=False)
                
                # 평가 (ground truth가 있으면)
                if sample.ground_truth and sample.ground_truth.get('tables'):
                    eval_results = pipeline.evaluate(
                        parse_results,
                        sample.ground_truth,
                        sample.sample_id
                    )
                else:
                    eval_results = {'sample_id': sample.sample_id, 'metrics': {}, 'errors': []}
                
                execution_time = time.time() - start_time
                
                # 결과 저장
                result = {
                    'experiment_id': self.experiment_id,
                    'sample_id': sample.sample_id,
                    'dataset': dataset_name,
                    'pipeline': pipeline_id,
                    'table_type': table_type,
                    'table_classification': table_classification,
                    'parse_success': parse_results.get('success', False),
                    'num_tables_detected': len(parse_results.get('tables', [])),
                    'metrics': eval_results.get('metrics', {}),
                    'errors': eval_results.get('errors', []),
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.all_results.append(result)
                
            except Exception as e:
                print(f"Error processing {sample.sample_id} with {pipeline_id}: {e}")
                # 오류도 기록
                self.all_results.append({
                    'experiment_id': self.experiment_id,
                    'sample_id': sample.sample_id,
                    'dataset': dataset_name,
                    'pipeline': pipeline_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
    
    def _analyze_and_save_results(self):
        """결과 분석 및 저장"""
        print("\nAnalyzing results...")
        
        # 전체 결과 저장
        results_file = self.experiment_dir / 'all_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        print(f"Saved all results to {results_file}")
        
        # 표 유형별 집계
        self._aggregate_by_table_type()
        
        # 모델별 집계
        self._aggregate_by_model()
        
        # 데이터셋별 집계
        self._aggregate_by_dataset()
        
        # 요약 통계 저장
        summary_file = self.experiment_dir / 'summary_statistics.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary_stats, f, indent=2, ensure_ascii=False)
        print(f"Saved summary statistics to {summary_file}")
    
    def _aggregate_by_table_type(self):
        """표 유형별 성능 집계"""
        from collections import defaultdict
        
        by_type = defaultdict(lambda: {
            'count': 0,
            'metrics': defaultdict(list),
            'errors': defaultdict(int)
        })
        
        for result in self.all_results:
            if 'error' in result:
                continue
            
            table_type = result.get('table_type', 'unknown')
            by_type[table_type]['count'] += 1
            
            # 메트릭 수집
            for metric_name, metric_value in result.get('metrics', {}).items():
                if isinstance(metric_value, (int, float)):
                    by_type[table_type]['metrics'][metric_name].append(metric_value)
            
            # 오류 수집
            for error in result.get('errors', []):
                error_type = error.get('error_type', 'unknown')
                by_type[table_type]['errors'][error_type] += 1
        
        # 평균 계산
        for table_type, data in by_type.items():
            data['metrics_avg'] = {
                metric: sum(values) / len(values) if values else 0
                for metric, values in data['metrics'].items()
            }
        
        self.summary_stats['by_table_type'] = dict(by_type)
    
    def _aggregate_by_model(self):
        """모델별 성능 집계"""
        from collections import defaultdict
        
        by_model = defaultdict(lambda: {
            'count': 0,
            'metrics': defaultdict(list),
            'errors': defaultdict(int)
        })
        
        for result in self.all_results:
            if 'error' in result:
                continue
            
            pipeline = result.get('pipeline', 'unknown')
            by_model[pipeline]['count'] += 1
            
            for metric_name, metric_value in result.get('metrics', {}).items():
                if isinstance(metric_value, (int, float)):
                    by_model[pipeline]['metrics'][metric_name].append(metric_value)
            
            for error in result.get('errors', []):
                error_type = error.get('error_type', 'unknown')
                by_model[pipeline]['errors'][error_type] += 1
        
        # 평균 계산
        for pipeline, data in by_model.items():
            data['metrics_avg'] = {
                metric: sum(values) / len(values) if values else 0
                for metric, values in data['metrics'].items()
            }
        
        self.summary_stats['by_model'] = dict(by_model)
    
    def _aggregate_by_dataset(self):
        """데이터셋별 성능 집계"""
        from collections import defaultdict
        
        by_dataset = defaultdict(lambda: {'count': 0, 'metrics': defaultdict(list)})
        
        for result in self.all_results:
            if 'error' in result:
                continue
            
            dataset = result.get('dataset', 'unknown')
            by_dataset[dataset]['count'] += 1
            
            for metric_name, metric_value in result.get('metrics', {}).items():
                if isinstance(metric_value, (int, float)):
                    by_dataset[dataset]['metrics'][metric_name].append(metric_value)
        
        for dataset, data in by_dataset.items():
            data['metrics_avg'] = {
                metric: sum(values) / len(values) if values else 0
                for metric, values in data['metrics'].items()
            }
        
        self.summary_stats['by_dataset'] = dict(by_dataset)


def main():
    """CLI 메인 함수"""
    parser = argparse.ArgumentParser(description='Run table parsing experiments')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['synthetic'],
        help='Datasets to use (synthetic, pubtabnet, etc.)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50,
        help='Maximum samples per dataset'
    )
    parser.add_argument(
        '--results-dir',
        default='results/experiments',
        help='Results output directory'
    )
    parser.add_argument(
        '--pubtabnet-dir',
        default='/data/pubtabnet',
        help='PubTabNet data directory'
    )
    
    args = parser.parse_args()
    
    # 실험 설정
    config = {
        'results_dir': args.results_dir,
        'pubtabnet_dir': args.pubtabnet_dir
    }
    
    # 실험 실행
    runner = ExperimentRunner(config)
    runner.run_experiment(
        datasets=args.datasets,
        max_samples_per_dataset=args.max_samples
    )


if __name__ == '__main__':
    main()

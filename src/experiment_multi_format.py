"""
다중 형식 비교 실험 스크립트
같은 내용의 여러 형식 파일들을 파싱하여 결과 비교
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from src.extractors import HWPXTableExtractor, PDFTableExtractor, HWPTableExtractor
from src.data import DatasetLoader
from src.evaluation import TableExtractionMetrics


class MultiFormatComparisonExperiment:
    """다중 형식 비교 실험"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.loader = DatasetLoader()
        self.results = {}
    
    def run_multi_format_comparison(self, dataset_path: str = "dataset2"):
        """
        다중 형식 비교 실험 실행
        
        Args:
            dataset_path: 데이터셋 경로 (기본: dataset2)
        """
        print("=" * 60)
        print("다중 형식 비교 실험 시작")
        print("=" * 60)
        
        # 데이터셋 로드
        print("\n[데이터셋 로드]")
        multi_format_data = self.loader.load_multi_format_dataset(dataset_path)
        
        print(f"  HWP 파일: {len(multi_format_data['hwp'])}개")
        print(f"  HWPX 파일: {len(multi_format_data['hwpx'])}개")
        print(f"  PDF 파일: {len(multi_format_data['pdf'])}개")
        
        # 파일 매칭
        matched_groups = self.loader.match_files_by_content(multi_format_data)
        print(f"\n  매칭된 파일 그룹: {len(matched_groups)}개")
        
        # 각 형식별 추출
        print("\n[표 추출]")
        extraction_results = {}
        
        # HWPX 추출
        if multi_format_data['hwpx']:
            print("\n  [HWPX 형식]")
            hwpx_extractor = HWPXTableExtractor()
            extraction_results['hwpx'] = self._extract_from_files(
                multi_format_data['hwpx'], hwpx_extractor, 'hwpx'
            )
        
        # PDF 추출 (여러 방법)
        if multi_format_data['pdf']:
            print("\n  [PDF 형식]")
            for baseline in self.config['baselines']['pdf']:
                if baseline['enabled']:
                    method = baseline['name']
                    print(f"    방법: {method}")
                    pdf_extractor = PDFTableExtractor(method=method)
                    extraction_results[f'pdf_{method}'] = self._extract_from_files(
                        multi_format_data['pdf'], pdf_extractor, f'pdf_{method}'
                    )
        
        # HWP 추출
        if multi_format_data['hwp']:
            print("\n  [HWP 형식]")
            hwp_extractor = HWPTableExtractor()
            extraction_results['hwp'] = self._extract_from_files(
                multi_format_data['hwp'], hwp_extractor, 'hwp'
            )
        
        # 결과 비교
        print("\n[결과 비교]")
        comparison_results = self._compare_extractions(extraction_results, matched_groups)
        
        # 결과 저장
        self.results['multi_format'] = {
            'extraction_results': extraction_results,
            'comparison': comparison_results,
            'matched_groups': matched_groups
        }
        
        # 리포트 출력
        self._print_comparison_report(comparison_results)
        
        return extraction_results, comparison_results
    
    def _extract_from_files(self, file_paths: List[str], extractor, method_name: str) -> Dict:
        """파일 리스트에서 표 추출"""
        results = {
            'files': {},
            'total_tables': 0,
            'extraction_times': []
        }
        
        for file_path in tqdm(file_paths, desc=f"    {method_name}"):
            try:
                tables = extractor.extract_tables(file_path)
                
                file_key = Path(file_path).stem
                results['files'][file_key] = {
                    'tables': tables,
                    'table_count': len(tables),
                    'file_path': file_path
                }
                
                results['total_tables'] += len(tables)
                
                if tables:
                    avg_time = sum(t.get('extraction_time', 0) for t in tables) / len(tables)
                    results['extraction_times'].append(avg_time)
            
            except Exception as e:
                print(f"      오류 ({file_path}): {e}")
                results['files'][Path(file_path).stem] = {
                    'tables': [],
                    'table_count': 0,
                    'error': str(e)
                }
        
        if results['extraction_times']:
            results['avg_time'] = sum(results['extraction_times']) / len(results['extraction_times'])
        else:
            results['avg_time'] = 0.0
        
        return results
    
    def _compare_extractions(self, extraction_results: Dict, matched_groups: List[Dict]) -> Dict:
        """추출 결과 비교"""
        comparison = {
            'by_format': {},
            'by_file_group': []
        }
        
        # 형식별 통계
        for format_name, results in extraction_results.items():
            comparison['by_format'][format_name] = {
                'total_tables': results['total_tables'],
                'avg_time': results.get('avg_time', 0.0),
                'file_count': len(results['files'])
            }
        
        # 파일 그룹별 비교
        for group in matched_groups:
            base_name = group['base_name']
            files = group['files']
            
            group_comparison = {
                'base_name': base_name,
                'formats': {}
            }
            
            for ext, file_path in files.items():
                file_key = Path(file_path).stem
                
                # 해당 형식의 추출 결과 찾기
                format_key = ext if ext == 'hwpx' else f'pdf_{ext}' if ext == 'pdf' else ext
                
                if format_key in extraction_results:
                    file_results = extraction_results[format_key]['files'].get(file_key, {})
                    group_comparison['formats'][ext] = {
                        'table_count': file_results.get('table_count', 0),
                        'tables': file_results.get('tables', [])
                    }
            
            comparison['by_file_group'].append(group_comparison)
        
        return comparison
    
    def _print_comparison_report(self, comparison_results: Dict):
        """비교 리포트 출력"""
        print("\n" + "=" * 60)
        print("형식별 추출 결과 비교")
        print("=" * 60)
        
        print("\n[형식별 통계]")
        for format_name, stats in comparison_results['by_format'].items():
            print(f"\n  {format_name}:")
            print(f"    총 표 개수: {stats['total_tables']}")
            print(f"    평균 처리 시간: {stats['avg_time']:.4f}초")
            print(f"    처리 파일 수: {stats['file_count']}")
        
        print("\n[파일 그룹별 비교]")
        for group in comparison_results['by_file_group']:
            print(f"\n  파일: {group['base_name']}")
            for ext, data in group['formats'].items():
                print(f"    {ext.upper()}: {data['table_count']}개 표 추출")
    
    def compare_with_labels(self, extraction_results: Dict, label_data: Dict) -> Dict:
        """
        추출 결과와 라벨링 데이터 비교
        
        Args:
            extraction_results: 추출 결과
            label_data: 라벨링 데이터 (doc_id -> tables)
            
        Returns:
            비교 결과
        """
        comparison = {
            'precision': {},
            'recall': {},
            'f1_score': {}
        }
        
        # 각 형식별로 라벨과 비교
        for format_name, results in extraction_results.items():
            precisions = []
            recalls = []
            f1_scores = []
            
            for file_key, file_results in results['files'].items():
                # 파일명에서 doc_id 추출 시도
                doc_id = self._extract_doc_id(file_key)
                
                if doc_id in label_data:
                    predicted_tables = [t['dataframe'] for t in file_results.get('tables', [])]
                    ground_truth_tables = self._convert_labels_to_dataframes(label_data[doc_id])
                    
                    if predicted_tables and ground_truth_tables:
                        f1 = TableExtractionMetrics.calculate_f1_score(
                            predicted_tables, ground_truth_tables
                        )
                        f1_scores.append(f1)
            
            if f1_scores:
                comparison['f1_score'][format_name] = sum(f1_scores) / len(f1_scores)
                comparison['precision'][format_name] = sum(precisions) / len(precisions) if precisions else 0.0
                comparison['recall'][format_name] = sum(recalls) / len(recalls) if recalls else 0.0
        
        return comparison
    
    def _extract_doc_id(self, filename: str) -> str:
        """파일명에서 doc_id 추출"""
        # 파일명에서 숫자 ID 추출 시도
        import re
        match = re.search(r'(\d+)', filename)
        return match.group(1) if match else filename
    
    def _convert_labels_to_dataframes(self, label_tables: List[Dict]) -> List[pd.DataFrame]:
        """라벨 표 데이터를 DataFrame으로 변환"""
        dataframes = []
        
        for table_label in label_tables:
            if 'data' in table_label:
                df = pd.DataFrame(table_label['data'])
                dataframes.append(df)
            elif 'rows' in table_label:
                df = pd.DataFrame(table_label['rows'])
                dataframes.append(df)
        
        return dataframes
    
    def save_results(self, output_path: str = "results_multi_format.json"):
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


if __name__ == "__main__":
    experiment = MultiFormatComparisonExperiment()
    experiment.run_multi_format_comparison("dataset2")
    experiment.save_results()


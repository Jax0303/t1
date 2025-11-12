"""
라벨링 데이터와 추출 결과 비교 실험
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from src.extractors import HWPXTableExtractor, PDFTableExtractor
from src.data import DatasetLoader
from src.evaluation import TableExtractionMetrics


class LabelComparisonExperiment:
    """라벨링 데이터와 추출 결과 비교 실험"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.loader = DatasetLoader()
        self.results = {}
    
    def run_label_comparison(self, qa_dataset_path: str, multi_format_dataset_path: str = "dataset2"):
        """
        라벨링 데이터와 추출 결과 비교 실험
        
        Args:
            qa_dataset_path: 질의응답 데이터셋 경로
            multi_format_dataset_path: 다중 형식 데이터셋 경로
        """
        print("=" * 60)
        print("라벨링 데이터 비교 실험 시작")
        print("=" * 60)
        
        # 질의응답 데이터셋 로드
        print("\n[질의응답 데이터셋 로드]")
        qa_dataset = self.loader.load_qa_dataset(qa_dataset_path)
        
        # 표 라벨 추출
        all_labels = {}
        for split in ['training', 'validation']:
            labels = self.loader.extract_table_labels(qa_dataset[split]['label'])
            all_labels.update(labels)
        
        print(f"  총 표 라벨: {len(all_labels)}개 문서")
        
        # 다중 형식 데이터셋 로드
        print("\n[다중 형식 데이터셋 로드]")
        multi_format_data = self.loader.load_multi_format_dataset(multi_format_dataset_path)
        
        # 각 형식별 표 추출
        print("\n[표 추출]")
        extraction_results = {}
        
        # HWPX 추출
        if multi_format_data['hwpx']:
            print("\n  [HWPX 형식]")
            hwpx_extractor = HWPXTableExtractor()
            extraction_results['hwpx'] = self._extract_from_files(
                multi_format_data['hwpx'], hwpx_extractor, 'hwpx'
            )
        
        # PDF 추출
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
        
        # 라벨과 비교
        print("\n[라벨링 데이터와 비교]")
        comparison_results = self._compare_with_labels(extraction_results, all_labels)
        
        # 결과 저장
        self.results['label_comparison'] = {
            'extraction_results': extraction_results,
            'comparison': comparison_results,
            'label_count': len(all_labels)
        }
        
        # 리포트 출력
        self._print_label_comparison_report(comparison_results)
        
        return comparison_results
    
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
    
    def _compare_with_labels(self, extraction_results: Dict, label_data: Dict) -> Dict:
        """추출 결과와 라벨링 데이터 비교"""
        comparison = {
            'by_format': {},
            'detailed': []
        }
        
        # 각 형식별로 비교
        for format_name, results in extraction_results.items():
            format_comparison = {
                'format': format_name,
                'f1_scores': [],
                'precisions': [],
                'recalls': [],
                'matched_files': 0,
                'total_files': len(results['files'])
            }
            
            for file_key, file_results in results['files'].items():
                # 파일명에서 doc_id 매칭 시도
                doc_id = self._match_doc_id(file_key, label_data.keys())
                
                if doc_id:
                    predicted_tables = [t['dataframe'] for t in file_results.get('tables', [])]
                    ground_truth_tables = self._convert_labels_to_dataframes(label_data[doc_id])
                    
                    if ground_truth_tables:
                        format_comparison['matched_files'] += 1
                        
                        # F1-score 계산
                        f1 = TableExtractionMetrics.calculate_f1_score(
                            predicted_tables, ground_truth_tables
                        )
                        format_comparison['f1_scores'].append(f1)
                        
                        # 상세 비교 정보 저장
                        comparison['detailed'].append({
                            'format': format_name,
                            'file': file_key,
                            'doc_id': doc_id,
                            'predicted_count': len(predicted_tables),
                            'ground_truth_count': len(ground_truth_tables),
                            'f1_score': f1
                        })
            
            # 평균 계산
            if format_comparison['f1_scores']:
                format_comparison['avg_f1'] = sum(format_comparison['f1_scores']) / len(format_comparison['f1_scores'])
            else:
                format_comparison['avg_f1'] = 0.0
            
            comparison['by_format'][format_name] = format_comparison
        
        return comparison
    
    def _match_doc_id(self, filename: str, label_keys: List[str]) -> str:
        """파일명과 라벨 키 매칭"""
        import re
        
        # 파일명에서 숫자 추출
        numbers = re.findall(r'\d+', filename)
        
        # 라벨 키와 매칭 시도
        for label_key in label_keys:
            # 정확히 일치
            if filename == label_key or filename in label_key or label_key in filename:
                return label_key
            
            # 숫자로 매칭
            label_numbers = re.findall(r'\d+', label_key)
            if numbers and label_numbers and numbers[0] == label_numbers[0]:
                return label_key
        
        return None
    
    def _convert_labels_to_dataframes(self, label_tables: List[Dict]) -> List[pd.DataFrame]:
        """라벨 표 데이터를 DataFrame으로 변환"""
        dataframes = []
        
        for table_label in label_tables:
            try:
                if isinstance(table_label, dict):
                    if 'data' in table_label:
                        # 2D 리스트 데이터
                        data = table_label['data']
                        if data and isinstance(data[0], list):
                            df = pd.DataFrame(data[1:], columns=data[0] if len(data) > 0 else None)
                            dataframes.append(df)
                    elif 'rows' in table_label:
                        df = pd.DataFrame(table_label['rows'])
                        dataframes.append(df)
                    elif 'cells' in table_label:
                        # 셀 기반 데이터
                        cells = table_label['cells']
                        if cells:
                            # 행과 열로 재구성
                            max_row = max(c.get('row', 0) for c in cells)
                            max_col = max(c.get('col', 0) for c in cells)
                            
                            table_data = [[''] * (max_col + 1) for _ in range(max_row + 1)]
                            for cell in cells:
                                row = cell.get('row', 0)
                                col = cell.get('col', 0)
                                value = cell.get('value', '')
                                table_data[row][col] = value
                            
                            df = pd.DataFrame(table_data)
                            dataframes.append(df)
            except Exception as e:
                print(f"라벨 변환 오류: {e}")
                continue
        
        return dataframes
    
    def _print_label_comparison_report(self, comparison_results: Dict):
        """라벨 비교 리포트 출력"""
        print("\n" + "=" * 60)
        print("라벨링 데이터 비교 결과")
        print("=" * 60)
        
        print("\n[형식별 성능]")
        for format_name, stats in comparison_results['by_format'].items():
            print(f"\n  {format_name}:")
            print(f"    매칭된 파일: {stats['matched_files']}/{stats['total_files']}")
            if stats['f1_scores']:
                print(f"    평균 F1-score: {stats['avg_f1']:.4f}")
                print(f"    최고 F1-score: {max(stats['f1_scores']):.4f}")
                print(f"    최저 F1-score: {min(stats['f1_scores']):.4f}")
        
        print("\n[상세 비교]")
        if comparison_results['detailed']:
            df = pd.DataFrame(comparison_results['detailed'])
            print(df.to_string(index=False))
    
    def save_results(self, output_path: str = "results_label_comparison.json"):
        """결과 저장"""
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
    experiment = LabelComparisonExperiment()
    experiment.run_label_comparison(
        "dataset1/149.표 정보 질의 응답 데이터",
        "dataset2"
    )
    experiment.save_results()


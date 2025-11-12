#!/usr/bin/env python3
"""
추출한 HWP 표 데이터를 사용하여 RAG 파이프라인 실행
"""
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import yaml

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.rag.rag_system import TableRAGSystem


def convert_hwp5_extractor_json_to_rag_format(json_file: str) -> List[Dict]:
    """
    hwp5-table-extractor로 추출한 JSON을 RAG 시스템 형식으로 변환
    
    Args:
        json_file: JSON 파일 경로
        
    Returns:
        RAG 시스템에 맞는 표 정보 리스트
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tables = []
    
    for idx, table_data in enumerate(data):
        try:
            # 행 데이터를 DataFrame으로 변환
            rows = []
            for row_data in table_data.get('rows', []):
                row_values = []
                for cell in row_data:
                    # 셀의 텍스트를 가져옴
                    cell_text = cell.get('text', '')
                    if not cell_text and cell.get('lines'):
                        cell_text = '\n'.join(cell.get('lines', []))
                    row_values.append(cell_text)
                
                if row_values:
                    rows.append(row_values)
            
            if not rows:
                continue
            
            # 열 개수 맞추기 (가장 긴 행 기준)
            max_cols = max(len(row) for row in rows) if rows else 0
            if max_cols == 0:
                continue
            
            for row in rows:
                while len(row) < max_cols:
                    row.append("")
            
            # 첫 번째 행을 헤더로 사용하거나, 열 번호로 생성
            if rows:
                headers = rows[0] if len(rows) > 1 else [f"열{i+1}" for i in range(max_cols)]
                data_rows = rows[1:] if len(rows) > 1 else []
                
                # 헤더가 비어있으면 열 번호로 생성
                if not headers or all(not str(h).strip() for h in headers):
                    headers = [f"열{i+1}" for i in range(max_cols)]
                
                # DataFrame 생성
                if data_rows:
                    df = pd.DataFrame(data_rows, columns=headers[:max_cols])
                else:
                    df = pd.DataFrame(columns=headers[:max_cols])
                
                # RAG 시스템 형식으로 변환
                table_info = {
                    'table_id': f"hwp5_table_{idx}",
                    'dataframe': df,
                    'source_file': json_file,
                    'extraction_method': 'hwp5-table-extractor',
                    'row_count': table_data.get('row_count', len(df)),
                    'col_count': table_data.get('col_count', len(df.columns))
                }
                
                tables.append(table_info)
        except Exception as e:
            print(f"  경고: 표 {idx} 변환 실패: {e}")
            continue
    
    return tables


def load_config() -> Dict:
    """설정 파일 로드"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def main():
    """메인 함수"""
    print("=" * 80)
    print("HWP 표 데이터 기반 RAG 파이프라인 실행")
    print("=" * 80)
    
    # 설정 로드
    config = load_config()
    
    # API 키 확인 (Ollama 우선, 없으면 Gemini)
    api_key = None
    provider = "ollama"
    model_name = "llama3.2"
    
    # Ollama 확인
    import subprocess
    try:
        result = subprocess.run(['curl', '-s', 'http://localhost:11434/api/tags'], 
                              capture_output=True, timeout=2)
        if result.returncode == 0:
            print("Ollama가 실행 중입니다. Ollama를 사용합니다.")
            provider = "ollama"
            model_name = "llama3.2"
            api_key = ""  # Ollama는 API 키 불필요
        else:
            raise Exception("Ollama 연결 실패")
    except:
        # Ollama가 없으면 Gemini 사용
        print("Ollama를 찾을 수 없습니다. Gemini를 사용합니다.")
        provider = "gemini"
        model_name = "gemini-2.0-flash-exp"
        
        if config.get('api', {}).get('gemini', {}).get('api_key'):
            api_key = config['api']['gemini']['api_key']
            model_name = config['api']['gemini'].get('model', model_name)
        elif os.environ.get('GEMINI_API_KEY'):
            api_key = os.environ.get('GEMINI_API_KEY')
        
        if not api_key:
            print("\n오류: Gemini API 키가 필요합니다.")
            print("config.yaml 파일에 api.gemini.api_key를 설정하거나")
            print("환경변수 GEMINI_API_KEY를 설정하세요.")
            print("\n또는 Ollama를 설치하고 실행하세요:")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
            print("  ollama pull llama3.2")
            return
    
    # 추출한 표 데이터 로드
    json_file = "extracted_tables_hwp5_extractor_improved.json"
    if not Path(json_file).exists():
        print(f"\n오류: {json_file} 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n표 데이터 로드 중: {json_file}")
    tables = convert_hwp5_extractor_json_to_rag_format(json_file)
    print(f"총 {len(tables)}개 표 변환 완료")
    
    if not tables:
        print("\n오류: 변환된 표가 없습니다.")
        return
    
    # RAG 시스템 초기화
    print(f"\nRAG 시스템 초기화 중...")
    print(f"  Provider: {provider}")
    print(f"  Model: {model_name}")
    
    try:
        rag_system = TableRAGSystem(
            api_key=api_key,
            model_name=model_name,
            provider=provider
        )
    except Exception as e:
        print(f"\n오류: RAG 시스템 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 지식 베이스 구축
    print("\n지식 베이스 구축 중...")
    try:
        rag_system.build_knowledge_base(tables)
        print("지식 베이스 구축 완료!")
    except Exception as e:
        print(f"\n오류: 지식 베이스 구축 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 예제 쿼리 실행
    print("\n" + "=" * 80)
    print("예제 쿼리 실행")
    print("=" * 80)
    
    # 샘플 쿼리 로드
    sample_queries_file = Path("sample_queries.json")
    queries = []
    
    if sample_queries_file.exists():
        with open(sample_queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
            queries = [q.get('question', '') for q in queries_data[:10]]  # 처음 10개만
    else:
        # 기본 쿼리
        queries = [
            "표의 총 행 수는 몇 개인가요?",
            "표에서 취업규칙과 관련된 내용을 찾아주세요",
            "표에서 근로조건에 대한 정보를 알려주세요",
            "표에서 필수 항목을 찾아주세요",
            "표에서 인사위원회 관련 정보를 찾아주세요",
            "표에서 휴직 관련 내용을 알려주세요",
            "표에서 제13조 관련 내용을 찾아주세요",
            "표에서 주40시간제 관련 내용을 찾아주세요",
            "표에서 근로기준법 관련 내용을 찾아주세요",
            "표의 전체 내용을 요약해주세요"
        ]
    
    # 쿼리 실행
    results = []
    for idx, query in enumerate(queries, 1):
        print(f"\n[쿼리 {idx}/{len(queries)}]")
        print(f"질문: {query}")
        print("-" * 80)
        
        try:
            result = rag_system.query(query)
            answer = result.get('answer', '')
            sources = result.get('source_documents', [])
            
            print(f"답변: {answer}")
            print(f"\n참고 문서: {len(sources)}개")
            for i, source in enumerate(sources[:3], 1):  # 상위 3개만 표시
                print(f"  {i}. {source.page_content[:200]}...")
            
            results.append({
                'query': query,
                'answer': answer,
                'sources_count': len(sources)
            })
        except Exception as e:
            print(f"오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 결과 저장
    output_file = Path("rag_results_hwp_tables.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print(f"결과가 {output_file}에 저장되었습니다.")
    print("=" * 80)


if __name__ == '__main__':
    import os
    main()


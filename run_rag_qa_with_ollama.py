#!/usr/bin/env python3
"""
Ollama를 사용하여 RAG QA 테스트 실행
Ollama가 설치되어 있지 않으면 설치 안내를 제공
"""
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import yaml
import subprocess
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.rag.rag_system import TableRAGSystem


def check_ollama_installed() -> bool:
    """Ollama가 설치되어 있고 실행 중인지 확인"""
    try:
        result = subprocess.run(
            ['curl', '-s', 'http://localhost:11434/api/tags'],
            capture_output=True,
            timeout=3
        )
        return result.returncode == 0
    except:
        return False


def check_ollama_model(model_name: str = "llama3.2") -> bool:
    """특정 모델이 다운로드되어 있는지 확인"""
    try:
        result = subprocess.run(
            ['curl', '-s', 'http://localhost:11434/api/tags'],
            capture_output=True,
            timeout=3
        )
        if result.returncode == 0:
            models = json.loads(result.stdout.decode('utf-8'))
            model_names = [m.get('name', '') for m in models.get('models', [])]
            return any(model_name in name for name in model_names)
        return False
    except:
        return False


def convert_hwp5_extractor_json_to_rag_format(json_file: str) -> List[Dict]:
    """hwp5-table-extractor로 추출한 JSON을 RAG 시스템 형식으로 변환"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tables = []
    
    for idx, table_data in enumerate(data):
        try:
            rows = []
            for row_data in table_data.get('rows', []):
                row_values = []
                for cell in row_data:
                    cell_text = cell.get('text', '')
                    if not cell_text and cell.get('lines'):
                        cell_text = '\n'.join(cell.get('lines', []))
                    row_values.append(cell_text)
                
                if row_values:
                    rows.append(row_values)
            
            if not rows:
                continue
            
            max_cols = max(len(row) for row in rows) if rows else 0
            if max_cols == 0:
                continue
            
            for row in rows:
                while len(row) < max_cols:
                    row.append("")
            
            if rows:
                headers = rows[0] if len(rows) > 1 else [f"열{i+1}" for i in range(max_cols)]
                data_rows = rows[1:] if len(rows) > 1 else []
                
                if not headers or all(not str(h).strip() for h in headers):
                    headers = [f"열{i+1}" for i in range(max_cols)]
                
                if data_rows:
                    df = pd.DataFrame(data_rows, columns=headers[:max_cols])
                else:
                    df = pd.DataFrame(columns=headers[:max_cols])
                
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


def main():
    print("=" * 80)
    print("Ollama를 사용한 RAG QA 테스트")
    print("=" * 80)
    
    # Ollama 확인
    print("\nOllama 상태 확인 중...")
    if not check_ollama_installed():
        print("\n❌ Ollama가 실행되고 있지 않습니다.")
        print("\n설치 방법:")
        print("1. 공식 설치 스크립트 실행:")
        print("   curl -fsSL https://ollama.com/install.sh | sh")
        print("\n2. 또는 수동 설치:")
        print("   - https://ollama.com/download 에서 다운로드")
        print("   - 설치 후 'ollama serve' 실행")
        print("\n3. 서비스 시작:")
        print("   ollama serve")
        print("\n4. 모델 다운로드:")
        print("   ollama pull llama3.2")
        print("   또는")
        print("   ollama pull qwen2.5  # 한국어 지원")
        return
    
    print("✓ Ollama가 실행 중입니다.")
    
    # 모델 확인
    model_name = "llama3.2"
    print(f"\n모델 확인 중: {model_name}")
    if not check_ollama_model(model_name):
        print(f"\n⚠ 모델 {model_name}이 다운로드되어 있지 않습니다.")
        print(f"\n다운로드 중... (시간이 걸릴 수 있습니다)")
        try:
            result = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True,
                timeout=300  # 5분 타임아웃
            )
            if result.returncode == 0:
                print(f"✓ 모델 {model_name} 다운로드 완료")
            else:
                print(f"❌ 모델 다운로드 실패: {result.stderr.decode('utf-8')}")
                print("\n수동으로 다운로드하세요:")
                print(f"  ollama pull {model_name}")
                return
        except subprocess.TimeoutExpired:
            print("❌ 모델 다운로드 시간 초과")
            print("\n수동으로 다운로드하세요:")
            print(f"  ollama pull {model_name}")
            return
        except FileNotFoundError:
            print("❌ ollama 명령어를 찾을 수 없습니다.")
            print("PATH에 ollama가 포함되어 있는지 확인하세요.")
            return
    else:
        print(f"✓ 모델 {model_name}이 설치되어 있습니다.")
    
    # 표 데이터 로드
    json_file = "extracted_tables_hwp5_extractor_improved.json"
    if not Path(json_file).exists():
        print(f"\n❌ 오류: {json_file} 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n표 데이터 로드 중: {json_file}")
    tables = convert_hwp5_extractor_json_to_rag_format(json_file)
    print(f"총 {len(tables)}개 표 변환 완료")
    
    if not tables:
        print("\n❌ 오류: 변환된 표가 없습니다.")
        return
    
    # RAG 시스템 초기화
    print(f"\nRAG 시스템 초기화 중...")
    print(f"  Provider: ollama")
    print(f"  Model: {model_name}")
    
    try:
        rag_system = TableRAGSystem(
            api_key="",  # Ollama는 API 키 불필요
            model_name=model_name,
            provider="ollama",
            ollama_base_url="http://localhost:11434"
        )
    except Exception as e:
        print(f"\n❌ 오류: RAG 시스템 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 지식 베이스 구축
    print("\n지식 베이스 구축 중...")
    try:
        rag_system.build_knowledge_base(tables)
        print("✓ 지식 베이스 구축 완료!")
    except Exception as e:
        print(f"\n❌ 오류: 지식 베이스 구축 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 예제 쿼리 실행
    print("\n" + "=" * 80)
    print("QA 테스트 시작")
    print("=" * 80)
    
    # 샘플 쿼리
    queries = [
        "표에서 취업규칙과 관련된 내용을 찾아주세요",
        "표에서 근로조건에 대한 정보를 알려주세요",
        "표에서 필수 항목을 찾아주세요",
        "표에서 인사위원회 관련 정보를 찾아주세요",
        "표에서 휴직 관련 내용을 알려주세요",
        "표에서 제13조 관련 내용을 찾아주세요",
        "표에서 주40시간제 관련 내용을 찾아주세요",
        "표에서 근로기준법 관련 내용을 찾아주세요"
    ]
    
    results = []
    for idx, query in enumerate(queries, 1):
        print(f"\n[쿼리 {idx}/{len(queries)}]")
        print(f"질문: {query}")
        print("-" * 80)
        
        try:
            start_time = time.time()
            result = rag_system.query(query)
            elapsed_time = time.time() - start_time
            
            answer = result.get('answer', '')
            sources = result.get('source_documents', [])
            
            print(f"답변: {answer}")
            print(f"\n참고 문서: {len(sources)}개")
            for i, source in enumerate(sources[:2], 1):  # 상위 2개만 표시
                content_preview = source.page_content[:150].replace('\n', ' ')
                print(f"  {i}. {content_preview}...")
            
            print(f"\n처리 시간: {elapsed_time:.2f}초")
            
            results.append({
                'query': query,
                'answer': answer,
                'sources_count': len(sources),
                'elapsed_time': elapsed_time
            })
        except Exception as e:
            print(f"❌ 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 결과 저장
    output_file = Path("rag_results_ollama.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✓ 결과가 {output_file}에 저장되었습니다.")
    print("=" * 80)
    
    # 통계 출력
    if results:
        avg_time = sum(r['elapsed_time'] for r in results) / len(results)
        print(f"\n평균 처리 시간: {avg_time:.2f}초")
        print(f"총 쿼리 수: {len(results)}개")


if __name__ == '__main__':
    main()


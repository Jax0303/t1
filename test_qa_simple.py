#!/usr/bin/env python3
"""
간단한 QA 테스트 (LLM 없이 검색 기능만 테스트)
"""
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings


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
            continue
    
    return tables


def table_to_text(df: pd.DataFrame, table_id: str) -> str:
    """DataFrame을 구조화된 텍스트로 변환"""
    text_parts = [f"표 ID: {table_id}"]
    text_parts.append(f"행 수: {len(df)}, 열 수: {len(df.columns)}")
    
    headers = " | ".join(str(col) for col in df.columns)
    text_parts.append(f"헤더: {headers}")
    text_parts.append("데이터:")
    
    for idx, row in df.iterrows():
        row_data = " | ".join(str(val) for val in row.values)
        text_parts.append(f"행 {idx}: {row_data}")
    
    return "\n".join(text_parts)


def main():
    print("=" * 80)
    print("간단한 QA 테스트 (검색 기능)")
    print("=" * 80)
    
    # 표 데이터 로드
    json_file = "extracted_tables_hwp5_extractor_improved.json"
    if not Path(json_file).exists():
        print(f"\n❌ 오류: {json_file} 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n표 데이터 로드 중: {json_file}")
    tables = convert_hwp5_extractor_json_to_rag_format(json_file)
    print(f"✓ 총 {len(tables)}개 표 변환 완료")
    
    if not tables:
        print("\n❌ 오류: 변환된 표가 없습니다.")
        return
    
    # 문서 생성
    print("\n문서 생성 중...")
    documents = []
    for table_info in tables[:20]:  # 처음 20개만 테스트
        df = table_info['dataframe']
        table_id = table_info['table_id']
        table_text = table_to_text(df, table_id)
        
        doc = Document(
            page_content=table_text,
            metadata={
                'table_id': table_id,
                'source_file': json_file,
                'extraction_method': 'hwp5-table-extractor'
            }
        )
        documents.append(doc)
    
    print(f"✓ {len(documents)}개 문서 생성 완료")
    
    # 벡터 스토어 생성
    print("\n벡터 스토어 생성 중...")
    embeddings = FakeEmbeddings(size=384)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )
    print("✓ 지식 베이스 구축 완료!")
    
    # QA 테스트
    print("\n" + "=" * 80)
    print("QA 테스트 시작")
    print("=" * 80)
    
    test_queries = [
        "취업규칙",
        "근로조건",
        "인사위원회",
        "휴직",
        "제13조",
        "주40시간제",
        "근로기준법",
        "필수 항목"
    ]
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = []
    
    for query in test_queries:
        print(f"\n[질문] {query}")
        print("-" * 80)
        
        try:
            docs = retriever.invoke(query)
            print(f"✓ 검색 결과: {len(docs)}개 문서 발견")
            
            for i, doc in enumerate(docs, 1):
                table_id = doc.metadata.get('table_id', 'N/A')
                content = doc.page_content[:200].replace('\n', ' ')
                print(f"\n  결과 {i}:")
                print(f"    표 ID: {table_id}")
                print(f"    내용: {content}...")
            
            results.append({
                'query': query,
                'results_count': len(docs),
                'top_result': docs[0].metadata.get('table_id', 'N/A') if docs else None
            })
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    # 결과 저장
    output_file = Path("qa_test_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✓ 결과가 {output_file}에 저장되었습니다.")
    print("=" * 80)
    
    # 통계
    print(f"\n테스트 통계:")
    print(f"  총 쿼리 수: {len(results)}개")
    print(f"  성공한 쿼리: {sum(1 for r in results if r.get('results_count', 0) > 0)}개")
    print(f"  평균 검색 결과 수: {sum(r.get('results_count', 0) for r in results) / len(results):.1f}개")
    
    print("\n" + "=" * 80)
    print("참고: LLM을 사용한 답변 생성을 위해서는 Ollama 설치가 필요합니다.")
    print("설치 방법: curl -fsSL https://ollama.com/install.sh | sh")
    print("설치 후: python run_rag_qa_with_ollama.py 실행")
    print("=" * 80)


if __name__ == '__main__':
    main()


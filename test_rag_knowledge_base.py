#!/usr/bin/env python3
"""
RAG 지식 베이스 구축 및 간단한 검색 테스트 (LLM 없이)
"""
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.rag.rag_system import TableRAGSystem


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
    print("RAG 지식 베이스 구축 및 검색 테스트")
    print("=" * 80)
    
    # 표 데이터 로드
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
    
    # 표 통계 출력
    print("\n표 통계:")
    print(f"  총 표 개수: {len(tables)}")
    total_rows = sum(t['row_count'] for t in tables)
    total_cols = sum(t['col_count'] for t in tables)
    print(f"  총 행 수: {total_rows}")
    print(f"  총 열 수: {total_cols}")
    print(f"  평균 행 수: {total_rows/len(tables):.1f}")
    print(f"  평균 열 수: {total_cols/len(tables):.1f}")
    
    # RAG 시스템 초기화 (FakeEmbeddings 사용 - 빠른 테스트용)
    print("\nRAG 시스템 초기화 중... (FakeEmbeddings 사용)")
    
    try:
        # FakeEmbeddings로 초기화 (실제 임베딩 없이 테스트)
        from langchain_core.embeddings import FakeEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        
        embeddings = FakeEmbeddings(size=384)
        
        # 문서 생성
        documents = []
        for table_info in tables[:10]:  # 처음 10개만 테스트
            df = table_info['dataframe']
            table_id = table_info['table_id']
            
            # 표를 텍스트로 변환
            text_parts = [f"표 ID: {table_id}"]
            text_parts.append(f"행 수: {len(df)}, 열 수: {len(df.columns)}")
            
            headers = " | ".join(str(col) for col in df.columns)
            text_parts.append(f"헤더: {headers}")
            text_parts.append("데이터:")
            
            for idx, row in df.head(5).iterrows():  # 처음 5행만
                row_data = " | ".join(str(val) for val in row.values)
                text_parts.append(f"행 {idx}: {row_data}")
            
            doc = Document(
                page_content="\n".join(text_parts),
                metadata={
                    'table_id': table_id,
                    'source_file': json_file,
                    'extraction_method': 'hwp5-table-extractor'
                }
            )
            documents.append(doc)
        
        print(f"  {len(documents)}개 문서 생성 완료")
        
        # 벡터 스토어 생성
        print("벡터 스토어 생성 중...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        print("지식 베이스 구축 완료!")
        
        # 검색 테스트
        print("\n" + "=" * 80)
        print("검색 테스트 (LLM 없이)")
        print("=" * 80)
        
        test_queries = [
            "취업규칙",
            "근로조건",
            "인사위원회",
            "휴직",
            "제13조"
        ]
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        for query in test_queries:
            print(f"\n질문: {query}")
            print("-" * 80)
            
            results = retriever.invoke(query)
            print(f"검색 결과: {len(results)}개")
            
            for i, doc in enumerate(results, 1):
                print(f"\n  결과 {i}:")
                print(f"    표 ID: {doc.metadata.get('table_id', 'N/A')}")
                content = doc.page_content[:200]
                print(f"    내용: {content}...")
        
        print("\n" + "=" * 80)
        print("테스트 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


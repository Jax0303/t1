#!/usr/bin/env python3
"""
지식 그래프 기반 RAG 파이프라인 실행 스크립트
복잡한 표를 지식 그래프로 변환하고 QA 수행
"""
import sys
import json
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from src.kg.kg_rag_system import KnowledgeGraphRAGSystem
from src.utils import convert_hwp5_extractor_json_to_rag_format, load_config


def main():
    """메인 함수"""
    print("=" * 80)
    print("지식 그래프 기반 RAG 파이프라인")
    print("=" * 80)
    
    # 설정 로드
    config = load_config()
    
    # API 설정 (다양한 형식 지원)
    api_config = config.get('api', {})
    if isinstance(api_config, dict) and 'provider' in api_config:
        # 새로운 형식: {provider: 'ollama', model: '...', api_key: '...'}
        provider = api_config.get('provider', 'ollama')
        model_name = api_config.get('model', 'llama3.2')
        api_key = api_config.get('api_key', 'dummy')
    else:
        # 기존 형식: {ollama: {...}, gemini: {...}, ...}
        if 'ollama' in api_config:
            provider = 'ollama'
            model_name = api_config['ollama'].get('model', 'llama3.2')
            api_key = api_config['ollama'].get('api_key', 'dummy')
        elif 'gemini' in api_config:
            provider = 'gemini'
            model_name = api_config['gemini'].get('model', 'gemini-2.0-flash-exp')
            api_key = api_config['gemini'].get('api_key', '')
        elif 'openai' in api_config:
            provider = 'openai'
            model_name = api_config['openai'].get('model', 'gpt-4o-mini')
            api_key = api_config['openai'].get('api_key', '')
        else:
            provider = 'ollama'
            model_name = 'llama3.2'
            api_key = 'dummy'
    
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
    
    # 지식 그래프 기반 RAG 시스템 초기화
    print(f"\n지식 그래프 기반 RAG 시스템 초기화 중...")
    print(f"  Provider: {provider}")
    print(f"  Model: {model_name}")
    print(f"  한국어 임베딩: 사용")
    
    try:
        rag_system = KnowledgeGraphRAGSystem(
            api_key=api_key,
            model_name=model_name,
            provider=provider,
            use_korean_embedding=True
        )
    except Exception as e:
        print(f"\n오류: RAG 시스템 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 지식 베이스 구축
    print("\n지식 베이스 구축 중...")
    print("  - 표를 지식 그래프로 변환")
    print("  - 벡터 스토어 생성")
    try:
        rag_system.build_knowledge_base(tables)
        print("지식 베이스 구축 완료!")
        
        # 통계 정보 출력
        kg_stats = rag_system.get_kg_statistics()
        print(f"\n지식 그래프 통계:")
        print(f"  총 표 수: {kg_stats['total_tables']}")
        print(f"  총 노드 수: {kg_stats['total_nodes']}")
        print(f"  총 엣지 수: {kg_stats['total_edges']}")
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
            queries = json.load(f)
    else:
        # 기본 쿼리
        queries = [
            "취업규칙의 목적은 무엇인가요?",
            "근로시간은 어떻게 규정되어 있나요?",
            "휴가 제도에 대해 설명해주세요."
        ]
    
    # 각 쿼리 실행
    for i, query in enumerate(queries[:5], 1):  # 최대 5개만
        print(f"\n[{i}] 질문: {query}")
        print("-" * 80)
        
        try:
            result = rag_system.query_with_kg(query, use_kg=True)
            print(f"답변: {result['answer']}")
            
            if result.get('kg_context'):
                print(f"\n지식 그래프 컨텍스트:")
                print(result['kg_context'])
            
            print(f"\n관련 문서 수: {len(result['source_documents'])}")
        except Exception as e:
            print(f"오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 대화형 모드
    print("\n" + "=" * 80)
    print("대화형 모드 (종료하려면 'quit' 또는 'exit' 입력)")
    print("=" * 80)
    
    while True:
        try:
            question = input("\n질문: ").strip()
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', '종료']:
                break
            
            result = rag_system.query_with_kg(question, use_kg=True)
            print(f"\n답변: {result['answer']}")
            
            if result.get('kg_context'):
                print(f"\n[지식 그래프 정보]")
                print(result['kg_context'])
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"오류: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()


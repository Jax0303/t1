# 지식 그래프 기반 RAG 파이프라인 설치 및 사용 가이드

## 개요

이 프로젝트는 복잡한 한국어 문서(특히 표가 많은 문서)에서 표 데이터를 정확히 추출하고, 지식 그래프로 구축한 뒤, QA가 가능한 RAG 파이프라인을 제공합니다.

## 주요 기능

1. **복잡한 표 추출**: 병합셀, 중첩헤더, 계층 구조를 가진 표 처리
2. **지식 그래프 구축**: 표 데이터를 엔티티-관계 그래프로 변환
3. **한국어 최적화**: 한국어 임베딩 모델 사용 (ko-sroberta-multitask)
4. **QA 시스템**: 지식 그래프와 벡터 검색을 결합한 질의응답

## 설치 방법

### 1. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 한국어 임베딩 모델 다운로드

프로그램 실행 시 자동으로 다운로드되지만, 수동으로 다운로드하려면:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
```

### 3. LLM 설정

#### 옵션 1: Ollama 사용 (로컬, 추천)

```bash
# Ollama 설치 (이미 설치되어 있다면 생략)
curl -fsSL https://ollama.com/install.sh | sh

# 모델 다운로드
ollama pull llama3.2
# 또는
ollama pull gemma2:2b
```

`config.yaml` 설정:
```yaml
api:
  provider: ollama
  model: llama3.2
  api_key: dummy
```

#### 옵션 2: OpenAI 사용

`config.yaml` 설정:
```yaml
api:
  provider: openai
  model: gpt-4o-mini
  api_key: your-openai-api-key
```

#### 옵션 3: Google Gemini 사용

`config.yaml` 설정:
```yaml
api:
  provider: gemini
  model: gemini-2.0-flash-exp
  api_key: your-gemini-api-key
```

## 사용 방법

### 1. 표 데이터 추출

이미 `extracted_tables_hwp5_extractor_improved.json` 파일이 있다면 다음 단계로 진행합니다.

표 추출이 필요하다면:
```bash
python test_hwp5_table_extractor_improved.py
```

### 2. 지식 그래프 기반 RAG 파이프라인 실행

```bash
python run_kg_rag_pipeline.py
```

### 3. 프로그래밍 방식 사용

```python
from src.kg.kg_rag_system import KnowledgeGraphRAGSystem
import json

# 표 데이터 로드
with open('extracted_tables_hwp5_extractor_improved.json', 'r', encoding='utf-8') as f:
    tables_data = json.load(f)

# 표를 RAG 형식으로 변환 (convert_hwp5_extractor_json_to_rag_format 함수 사용)

# RAG 시스템 초기화
rag_system = KnowledgeGraphRAGSystem(
    api_key='your-api-key',
    model_name='llama3.2',
    provider='ollama',
    use_korean_embedding=True
)

# 지식 베이스 구축
rag_system.build_knowledge_base(tables)

# 질의응답
result = rag_system.query_with_kg("취업규칙의 목적은 무엇인가요?", use_kg=True)
print(result['answer'])
```

## 프로젝트 구조

```
t1/
├── src/
│   ├── kg/
│   │   ├── __init__.py
│   │   ├── table_to_kg.py          # 표를 지식 그래프로 변환
│   │   └── kg_rag_system.py        # 지식 그래프 기반 RAG 시스템
│   ├── extractors/                 # 표 추출기
│   └── rag/                        # 기본 RAG 시스템
├── run_kg_rag_pipeline.py          # 메인 실행 스크립트
├── requirements.txt                # 필수 라이브러리
└── config.yaml                     # 설정 파일
```

## 지식 그래프 구조

표 데이터는 다음과 같은 엔티티 타입으로 변환됩니다:

- **Table**: 표 전체
- **Row**: 행
- **Column**: 열
- **Header**: 헤더 (계층 구조 지원)
- **Cell**: 셀
- **Value**: 값

관계 타입:
- `has_row`, `has_column`: 표 구조 관계
- `has_header`: 헤더 관계
- `contains`: 포함 관계
- `has_value`: 값 관계
- `belongs_to`: 소속 관계
- `related_to`: 관련 관계

## 성능 최적화

1. **한국어 임베딩**: `jhgan/ko-sroberta-multitask` 모델 사용으로 한국어 텍스트 검색 성능 향상
2. **지식 그래프 캐싱**: 한 번 구축한 지식 그래프는 메모리에 저장되어 재사용
3. **청킹 전략**: 큰 표는 적절한 크기로 분할하여 처리

## 문제 해결

### 한국어 임베딩 모델 다운로드 실패

```python
# 수동 다운로드
from sentence_transformers import SentenceTransformer
import os

os.environ['HF_HOME'] = '/path/to/cache'
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
```

### Ollama 연결 오류

```bash
# Ollama 서비스 확인
ollama serve

# 다른 포트 사용 시
export OLLAMA_BASE_URL=http://localhost:11434
```

### 메모리 부족

큰 데이터셋의 경우:
- `chunk_size`를 줄이기
- 배치 처리로 표를 나누어 처리
- GPU가 있다면 GPU 사용

## 예제 질문

- "취업규칙의 목적은 무엇인가요?"
- "근로시간은 어떻게 규정되어 있나요?"
- "휴가 제도에 대해 설명해주세요."
- "표에서 특정 정보를 찾아주세요."

## 참고 자료

- [NetworkX 문서](https://networkx.org/)
- [LangChain 문서](https://python.langchain.com/)
- [Sentence Transformers 문서](https://www.sbert.net/)
- [Ollama 문서](https://ollama.com/)


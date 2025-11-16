# 지식 그래프 기반 RAG 파이프라인 빠른 시작 가이드

## 1단계: 라이브러리 설치

```bash
pip install -r requirements.txt
```

## 2단계: Ollama 설정 (로컬 LLM 사용 시)

```bash
# Ollama 설치 (이미 설치되어 있다면 생략)
curl -fsSL https://ollama.com/install.sh | sh

# 모델 다운로드
ollama pull llama3.2
```

## 3단계: 설정 파일 확인

`config.yaml` 파일이 있는지 확인하고, 없으면 생성:

```yaml
api:
  provider: ollama  # 또는 openai, gemini
  model: llama3.2
  api_key: dummy  # Ollama는 dummy, 다른 서비스는 실제 API 키
```

## 4단계: 실행

```bash
python run_kg_rag_pipeline.py
```

## 주요 기능 테스트

### 지식 그래프 변환 테스트

```bash
python test_kg_conversion.py
```

## 사용 예제

### 프로그래밍 방식 사용

```python
from src.kg.kg_rag_system import KnowledgeGraphRAGSystem
from run_kg_rag_pipeline import convert_hwp5_extractor_json_to_rag_format

# 표 데이터 로드 및 변환
tables = convert_hwp5_extractor_json_to_rag_format(
    'extracted_tables_hwp5_extractor_improved.json'
)

# RAG 시스템 초기화
rag = KnowledgeGraphRAGSystem(
    api_key='dummy',
    model_name='llama3.2',
    provider='ollama',
    use_korean_embedding=True
)

# 지식 베이스 구축
rag.build_knowledge_base(tables)

# 질의응답
result = rag.query_with_kg("취업규칙의 목적은 무엇인가요?")
print(result['answer'])
```

## 문제 해결

### 한국어 임베딩 모델 다운로드가 느린 경우

프로그램이 자동으로 다운로드하지만, 수동으로 미리 다운로드할 수 있습니다:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
```

### Ollama 연결 오류

```bash
# Ollama 서비스 시작
ollama serve

# 다른 터미널에서 확인
curl http://localhost:11434/api/tags
```

## 다음 단계

- `KG_RAG_SETUP.md`에서 상세한 설정 방법 확인
- 복잡한 표 추출 방법은 `HWP_PARSING_IMPROVEMENT_PLAN.md` 참고
- RAG 시스템 개선은 `src/rag/` 디렉토리 확인


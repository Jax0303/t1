# HierTable-RAG

Hierarchical Table-based Retrieval Augmented Generation (HierTable-RAG)는 계층적 테이블 데이터를 추출, 구조화, 쿼리하기 위한 연구 프로젝트입니다.

## 프로젝트 개요

HierTable-RAG는 PDF 문서에서 테이블을 추출하고, 계층적 관계를 감지하며, RAG(Retrieval Augmented Generation) 기법을 사용하여 자연어 쿼리에 대한 정확한 답변을 생성하는 시스템입니다.

### 주요 기능

- **다양한 문서 형식 지원**: PDF, HWP 등에서 테이블 추출
- **계층적 구조 감지**: 테이블 간 부모-자식 관계 자동 감지
- **효율적인 벡터 검색**: FAISS를 활용한 고속 유사도 검색
- **LLM 기반 응답 생성**: LangChain과 OpenAI를 활용한 컨텍스트 인식 응답 생성

## 프로젝트 구조

```
HierTable-RAG/
├── data/              # 데이터 파일 (원본 및 처리된 데이터)
├── src/               # 소스 코드
│   └── hiertable_rag/
│       ├── core/      # 핵심 RAG 파이프라인
│       ├── extractors/# 테이블 추출 모듈
│       ├── processors/# 테이블 처리 및 구조화 모듈
│       ├── retrievers/# 벡터 검색 모듈
│       └── generators/# LLM 응답 생성 모듈
├── notebooks/         # Jupyter 노트북 (실험 및 분석)
├── tests/             # 테스트 코드
├── configs/           # 설정 파일
├── experiments/       # 실험 결과 및 로그
├── requirements.txt   # Python 의존성
├── config.yaml        # 설정 파일 (템플릿에서 생성)
└── README.md          # 프로젝트 문서
```

## 설치 및 설정

### 요구사항

- Python 3.10 이상
- pip 또는 conda

### 설치 방법

1. 저장소 클론:
```bash
git clone <repository-url>
cd HierTable-RAG
```

2. 가상 환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. 설정 파일 생성:
```bash
cp configs/config.yaml.template config.yaml
```

5. `config.yaml` 파일을 열어 API 키 및 모델 경로를 설정하세요.

## 사용 방법

### 기본 사용 예제

```python
from pathlib import Path
from hiertable_rag import HierTableRAG

# RAG 시스템 초기화
rag = HierTableRAG(
    config_path=Path("config.yaml"),
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# 문서에서 테이블 추출
tables = rag.extract_tables(Path("data/sample.pdf"))

# 테이블 처리 및 계층 구조 감지
processed_tables = rag.process_tables(tables)

# 검색 인덱스 구축
rag.build_index(processed_tables)

# 쿼리 실행
results = rag.query("2023년 매출은 얼마인가요?", top_k=5)

# 응답 생성
response = rag.generate_response(
    query="2023년 매출은 얼마인가요?",
    retrieved_context=results
)
print(response)
```

## 개발 가이드

### 코드 스타일

- Python 3.10+ 기능 활용 (타입 힌트, 구조적 패턴 매칭 등)
- PEP 8 스타일 가이드 준수
- 모든 공개 함수/클래스에 타입 힌트 포함
- docstring 작성 (Google 스타일)

### 테스트 실행

```bash
pytest tests/
```

코드 커버리지 포함:
```bash
pytest tests/ --cov=src/hiertable_rag --cov-report=html
```

### 타입 체크

```bash
mypy src/
```

## 실험 및 재현성

실험은 `experiments/` 디렉토리에서 관리됩니다. 각 실험은 다음을 포함해야 합니다:

- 실험 설정 파일
- 실행 스크립트
- 결과 로그 및 메트릭
- 재현을 위한 시드 값

## 라이선스

[라이선스 정보를 여기에 추가하세요]

## 기여

기여를 환영합니다! 이슈를 먼저 생성하거나 Pull Request를 제출해주세요.

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.

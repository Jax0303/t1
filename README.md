# 지식 그래프 기반 RAG 파이프라인

복잡한 한국어 문서(특히 표가 많은 문서)에서 표 데이터를 정확히 추출하고, 지식 그래프로 구축한 뒤, QA가 가능한 RAG 파이프라인입니다.

## 주요 기능

1. **복잡한 표 추출**: 병합셀, 중첩헤더, 계층 구조를 가진 표 처리
2. **지식 그래프 구축**: 표 데이터를 엔티티-관계 그래프로 변환
3. **한국어 최적화**: 한국어 임베딩 모델 사용 (ko-sroberta-multitask)
4. **QA 시스템**: 지식 그래프와 벡터 검색을 결합한 질의응답

## 빠른 시작

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 설정

`config.yaml` 파일 생성:

```yaml
api:
  provider: ollama  # 또는 openai, gemini
  model: llama3.2
  api_key: dummy  # Ollama는 dummy, 다른 서비스는 실제 API 키
```

### 3. 실행

```bash
python run_kg_rag_pipeline.py
```

## 프로젝트 구조

```
t1/
├── run_kg_rag_pipeline.py      # 메인 실행 스크립트
├── test_kg_conversion.py       # 지식 그래프 변환 테스트
├── test_hwp5_table_extractor_improved.py  # 표 추출 테스트
├── config.yaml                 # 설정 파일
├── requirements.txt            # 필수 라이브러리
├── src/
│   ├── kg/                     # 지식 그래프 모듈
│   │   ├── table_to_kg.py      # 표를 지식 그래프로 변환
│   │   └── kg_rag_system.py    # 지식 그래프 기반 RAG 시스템
│   ├── extractors/              # 표 추출 모듈
│   │   ├── hwp_extractor.py
│   │   ├── hwpx_extractor.py
│   │   └── pdf_extractor.py
│   ├── rag/                     # 기본 RAG 시스템
│   │   └── rag_system.py
│   └── utils/                   # 공통 유틸리티
│       ├── table_converter.py
│       └── config_loader.py
├── hwp5-table-extractor/        # HWP5 표 추출 도구
└── docs/
    ├── KG_RAG_SETUP.md          # 상세 설치 가이드
    └── QUICKSTART_KG_RAG.md     # 빠른 시작 가이드
```

## 사용 방법

### 프로그래밍 방식

```python
from src.kg.kg_rag_system import KnowledgeGraphRAGSystem
from src.utils import convert_hwp5_extractor_json_to_rag_format

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

## LLM 설정

### Ollama (로컬, 추천)

```bash
ollama pull llama3.2
```

### OpenAI

```yaml
api:
  provider: openai
  model: gpt-4o-mini
  api_key: your-openai-api-key
```

### Google Gemini

```yaml
api:
  provider: gemini
  model: gemini-2.0-flash-exp
  api_key: your-gemini-api-key
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

## 데이터 수집

### DART 공시 문서 다운로드

재무·공시 PDF 파일을 자동으로 다운로드:

```bash
export DART_API_KEY='your-api-key'
python scripts/download_dart_reports.py
```

자세한 내용은 [DART 다운로드 가이드](docs/DART_DOWNLOAD_GUIDE.md)를 참조하세요.

## 문서

- [상세 설치 가이드](docs/KG_RAG_SETUP.md)
- [빠른 시작 가이드](docs/QUICKSTART_KG_RAG.md)
- [DART 다운로드 가이드](docs/DART_DOWNLOAD_GUIDE.md)
- [PDF 표 추출 조사 보고서](docs/PDF_TABLE_EXTRACTION_RESEARCH.md)

## PDF 표 추출 실험

보고서에서 언급한 여러 PDF 파서를 비교 실험할 수 있습니다.

### 파서 비교 실험

```bash
# 모든 파서 비교 (5개 파일)
python3 scripts/compare_pdf_extractors.py --max-files 5

# 특정 파서만 비교
python3 scripts/compare_pdf_extractors.py --max-files 5 --methods pdfplumber camelot

# 단일 파일 테스트
python3 scripts/compare_pdf_extractors.py --single-file "data/dart_pdfs/your_file.pdf"
```

### 한국어 문서 문제점 분석

```bash
# 병합 셀, 다중 페이지, 한글 폰트 문제 분석
python3 scripts/analyze_korean_pdf_issues.py --input "data/dart_pdfs/your_file.pdf"
```

자세한 내용은 [PDF 비교 실험 가이드](scripts/README_PDF_COMPARISON.md)를 참조하세요.

## 라이선스

MIT License

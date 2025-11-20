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
│   └── utils/                 # 공통 유틸리티
│       ├── table_converter.py
│       └── config_loader.py
├── scripts/                     # 유틸리티 스크립트
│   ├── test_single_pdf_extraction.py  # 단일 PDF 다중 파서 테스트
│   ├── compare_pdf_extractors.py      # 파서 비교 실험
│   ├── analyze_korean_pdf_issues.py   # 한국어 PDF 문제 분석
│   ├── run_table_kg_experiments.py    # 표 유형별 KG 매핑 & RDF 내보내기
│   └── visualize_kg.py                # 단일 표 지식 그래프 시각화
├── analysis/                     # 실험 결과 분석 데이터
│   ├── parser_comparison.json   # 파서 비교 결과
│   └── kg_*.json                # 지식 그래프 통계
├── hwp5-table-extractor/        # HWP5 표 추출 도구
└── docs/
    ├── KG_RAG_SETUP.md          # 상세 설치 가이드
    ├── QUICKSTART_KG_RAG.md     # 빠른 시작 가이드
    └── DMS_KG_PDF_Report.md     # 다중 파서 KG 구축 보고서
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

## 최근 실험 결과

### 다중 파서 비교 및 지식 그래프 구축 실험

DMS_[기재정정]사업보고서(308페이지, 표 544개)를 대상으로 pdfplumber, PyMuPDF, Camelot 세 파서를 비교하고 지식 그래프로 변환한 실험 결과:

**주요 결과**:
- **pdfplumber/PyMuPDF**: 표 544개 전수 추출, None 비율 30% 이상 표 18개
- **Camelot**: 표 504개 추출, None 비율 30% 이상 표 0개 (병합 손실 최소화)
- **지식 그래프**: 세 파서 모두 4.5만~4.8만 노드 규모의 KG 생성 성공
- None 비율이 높은 표에서도 평균 228개 이상의 노드 보존 (구조 정보 유지)

자세한 내용은 [다중 파서 KG 구축 보고서](docs/DMS_KG_PDF_Report.md)를 참조하세요.

### 표 유형별 지식 그래프 매핑 실험 (hwp5-table-extractor 샘플)

`extracted_tables_hwp5_extractor_improved.json`에 포함된 45개 표를 8개 유형(단일 셀 안내, 2열 요약, 2열 목차, 3~4열 숫자 매트릭스, 다열 숫자/양식/텍스트 매트릭스, 기타)으로 자동 분류한 뒤 각 유형별 대표 표를 KG로 변환했습니다.

#### 기본 실험 (NetworkX + RDFLib)

```bash
# 유형별 샘플 수를 2개로 늘려서 실행
python scripts/run_table_kg_experiments.py \
  --input extracted_tables_hwp5_extractor_improved.json \
  --samples-per-type 2 \
  --output-dir outputs/kg_table_experiments
```

생성물:
- `outputs/kg_table_experiments/kg_table_experiments.json`: 유형 분포, 표별 메트릭, `Table -> Column -> Row -> Value` 체인 샘플 등 세부 리포트
- `outputs/kg_table_experiments/rdf_exports/*.ttl`: 각 표를 RDF(Turtle)로 직렬화한 파일. 외부 그래프 DB나 SPARQL 툴에 바로 적재 가능

#### 확장 실험 (최신 KG 라이브러리 통합)

최신 KG 구축 라이브러리들을 사용한 확장 실험:

```bash
# 유형별 3개 샘플로 확장 실험 실행
python scripts/run_advanced_kg_experiments.py \
  --input extracted_tables_hwp5_extractor_improved.json \
  --samples-per-type 3 \
  --output-dir outputs/kg_table_experiments_advanced
```

**사용된 라이브러리:**
- **NetworkX**: 그래프 구조 표현 및 분석
- **RDFLib**: RDF/Turtle 형식으로 직렬화
- **PyKEEN**: Knowledge Graph Embeddings (선택사항)
- **SPARQL**: 구조화된 쿼리 지원

**생성물:**
- `outputs/kg_table_experiments_advanced/advanced_kg_experiments.json`: 상세 매핑 분석 리포트
- `outputs/kg_table_experiments_advanced/rdf_exports/*.ttl`: RDF/Turtle 형식 KG
- `outputs/kg_table_experiments_advanced/pykeen_exports/*.json`: PyKEEN 형식 트리플 데이터

**표 데이터 → KG 매핑 분석 결과:**

각 표 유형별로 데이터가 KG의 어떤 부분으로 매핑되는지 상세 분석:

1. **구조 엔티티 (Structure Entities)**:
   - `Table`: 표 전체를 나타내는 루트 엔티티
   - `Row`: 각 행을 나타내는 엔티티 (행 인덱스 포함)
   - `Column`: 각 열을 나타내는 엔티티 (열 인덱스 및 헤더 정보 포함)
   - `Cell`: 개별 셀을 나타내는 엔티티 (행/열 위치 정보 포함)

2. **내용 엔티티 (Content Entities)**:
   - `Header`: 헤더 텍스트를 나타내는 엔티티 (계층 구조 지원, level 속성으로 깊이 표현)
   - `Value`: 의미 있는 값(숫자, 텍스트)을 나타내는 엔티티

3. **관계 (Relations)**:
   - `has_row`, `has_column`: 표 구조 관계
   - `has_header`: 헤더 관계 (계층 구조 지원)
   - `contains`: 포함 관계 (Table → Cell)
   - `has_value`: 값 관계 (Cell → Value)
   - `belongs_to`: 소속 관계 (Value → Header)

**유형별 매핑 특징:**

- **단일 셀 안내형 표**: 구조 노드(Table/Row/Column)만 존재, 길고 복잡한 문단도 Table/Row/Column 레벨에서 관리
- **다열 숫자 매트릭스**: 헤더 2단 계층, 값들을 모두 Value 노드로 승격해 수치 비교 질의 지원
- **양식 템플릿 표**: 병합 셀 구조를 Column/Row/Value 링크로 기록해 필드-값 대응 체인 명확히 추적
- **2열 목차형 표**: 120행 규모의 대용량 표도 효율적으로 변환 (263 노드, 331 엣지)
- **다열 텍스트 매트릭스**: 헤더 계층 깊이 3단까지 지원하여 복잡한 중첩 구조 처리

**실험 통계 (45개 표, 유형별 3개 샘플):**
- 총 19개 표 실험 완료
- 평균 노드 수: 80.2개
- 평균 엣지 수: 100.3개
- 헤더 계층 깊이: 0~3단
- RDF 내보내기 성공률: 100%

## 문서

- [상세 설치 가이드](docs/KG_RAG_SETUP.md)
- [빠른 시작 가이드](docs/QUICKSTART_KG_RAG.md)
- [DART 다운로드 가이드](docs/DART_DOWNLOAD_GUIDE.md)
- [PDF 표 추출 조사 보고서](docs/PDF_TABLE_EXTRACTION_RESEARCH.md)
- [다중 파서 KG 구축 보고서](docs/DMS_KG_PDF_Report.md) ⭐ 최신

## PDF 표 추출 실험

보고서에서 언급한 여러 PDF 파서를 비교 실험할 수 있습니다.

### 파서 비교 실험

```bash
# 모든 파서 비교 (5개 파일)
python3 scripts/compare_pdf_extractors.py --max-files 5

# 특정 파서만 비교
python3 scripts/compare_pdf_extractors.py --max-files 5 --methods pdfplumber camelot

# 단일 파일 테스트 (다중 파서 비교)
python3 scripts/test_single_pdf_extraction.py "data/dart_pdfs/your_file.pdf"
```

### 지식 그래프 변환

추출된 표 데이터를 지식 그래프로 변환:

```bash
# 표 데이터를 KG로 변환
python3 test_kg_conversion.py
```

### 한국어 문서 문제점 분석

```bash
# 병합 셀, 다중 페이지, 한글 폰트 문제 분석
python3 scripts/analyze_korean_pdf_issues.py --input "data/dart_pdfs/your_file.pdf"
```

자세한 내용은 [PDF 비교 실험 가이드](scripts/README_PDF_COMPARISON.md)를 참조하세요.

## 라이선스

MIT License

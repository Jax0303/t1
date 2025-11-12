# HWPX vs PDF Table Extraction RAG 실험

본 실험은 HWP/HWPX 문서에서 표를 추출하고 이를 기반으로 RAG 시스템을 구성하는 것이 핵심입니다. 구조적으로 명시적 표 정보를 갖는 HWPX의 장점을 활용하여 PDF보다 빠르고 정확한 표 추출이 가능하며, 이를 통해 RAG 성능을 향상시킬 수 있는지 검증합니다.

## 실험 목표

### 표 추출 성능
- **F1-score**: PDF 대비 15% 이상 향상
- **처리 속도**: PDF 대비 30% 이상 향상

### RAG 성능
- **EM Score**: PDF 기반 RAG 대비 10% 이상 향상
- **Hit@K**: Hit@1 ≥ 0.70, Hit@3 ≥ 0.85, Hit@5 ≥ 0.90

## 베이스라인 모델

### PDF 표 추출
- **pdfplumber**: Python 기반 PDF 파싱 라이브러리
- **camelot**: PDF 표 추출 전용 라이브러리 (lattice/stream 방법)
- **tabula**: Java 기반 PDF 표 추출 도구

### HWP 표 추출
- **hwp5-table-extractor**: HWP5 파일에서 표를 직접 추출하는 검증된 도구
  - OLE2 구조를 활용한 HWP5 파일 직접 파싱
  - 레코드 트리 구조 분석을 통한 정확한 표 추출
  - HWPX 변환 없이도 표 추출 가능 (45개 표 추출 성공 사례)
- **HWPX 변환 기반**: HWP → HWPX 변환 후 XML 파싱

## 평가 메트릭

### 표 추출 메트릭
- **F1-score**: 표 추출 정확도 (Precision과 Recall의 조화 평균)
- **Precision**: 추출된 표 중 정확한 표의 비율
- **Recall**: 실제 표 중 추출된 표의 비율
- **처리 속도**: 문서당 평균 추출 시간

### RAG 메트릭
- **EM Score (Exact Match)**: 정답과 완전히 일치하는 답변의 비율
- **Hit@K**: 상위 K개 검색 결과 중 관련 문서가 포함된 비율
- **BLEU Score**: 답변의 품질 평가 (n-gram 기반)

## 프로젝트 구조

```
t1/
├── config.yaml              # 실험 설정 파일
├── requirements.txt         # Python 의존성
├── run_experiment.py        # 실험 실행 스크립트
├── src/
│   ├── extractors/          # 표 추출 모듈
│   │   ├── hwpx_extractor.py
│   │   └── pdf_extractor.py
│   ├── rag/                 # RAG 시스템
│   │   └── rag_system.py
│   └── evaluation/          # 평가 메트릭
│       └── metrics.py
└── data/                    # 데이터셋 (사용자 제공 예정)
    ├── hwpx/               # HWPX 문서
    ├── pdf/                # PDF 문서
    ├── ground_truth/       # 정답 데이터
    └── questions.json      # 평가용 질문
```

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# NLTK 데이터 다운로드 (BLEU score 계산용)
python -c "import nltk; nltk.download('punkt')"

# hwp5-table-extractor 설정 (선택사항, HWP 직접 파싱용)
# 프로젝트에 포함된 hwp5-table-extractor 디렉토리를 사용하거나
# 직접 클론하여 사용할 수 있습니다:
# git clone https://github.com/hallazzang/hwp5-table-extractor.git
```

## 사용 방법

### 데이터셋 준비

프로젝트에는 두 가지 데이터셋이 사용됩니다:

1. **149.표 정보 질의 응답 데이터** (`data/raw/dataset1/`)
   - Training/Validation 원천데이터 및 라벨링데이터
   - JSON 형식의 질문-답변 쌍 및 표 라벨

2. **개정 표준취업규칙** (`data/raw/dataset2/`)
   - 같은 내용의 여러 형식 파일 (HWP, HWPX, PDF)
   - 형식별 추출 결과 비교용

### 실험 실행

#### 1. 다중 형식 비교 실험
같은 내용의 여러 형식 파일들을 파싱하여 결과 비교:
```bash
python run_multi_format_experiment.py
```

#### 2. 라벨링 데이터 비교 실험
JSON 라벨링 데이터와 추출 결과 비교:
```bash
python run_all_experiments.py
```

#### 3. 전체 실험 (권장)
모든 실험을 순차적으로 실행:
```bash
python run_all_experiments.py
```

#### 4. HWP 직접 파싱 vs HWPX 변환 비교 실험
HWP 파일을 직접 파싱하는 방법과 HWPX 변환 기반 방법을 비교:
```bash
python run_hwp_comparison.py
```

이 실험은 다음을 수행합니다:
- 동일한 HWP 파일에 대해 두 가지 파싱 방법 적용
- 각 방법으로 추출된 표를 기반으로 RAG 시스템 구축
- EM, F1, Hit@K 지표로 질의응답 성능 비교
- 성능 개선 효과 정량적 입증

#### 5. hwp5-table-extractor를 사용한 HWP 표 추출
검증된 hwp5-table-extractor 도구를 사용하여 HWP 파일에서 표 추출:
```bash
python test_hwp5_table_extractor_improved.py
```

**주요 성과**:
- HWP 파일에서 45개 표 성공적으로 추출 (HWPX의 43개보다 많음)
- HWPX 변환 없이도 직접 파싱 가능함을 입증
- 결과는 `extracted_tables_hwp5_extractor_improved.json`에 저장됨

## 데이터 형식

### 질의응답 데이터셋 형식
```json
{
  "Dataset": "...",
  "data": [
    {
      "doc_id": "21002084",
      "doc_title": "...",
      "paragraphs": [...],
      "tables": [...]
    }
  ]
}
```

### 라벨링 데이터 형식
표 정보는 다양한 형식으로 저장될 수 있습니다:
- `data`: 2D 배열 형식
- `rows`: 행 단위 데이터
- `cells`: 셀 단위 데이터 (row, col, value)

### 다중 형식 데이터셋
같은 내용의 파일들이 여러 형식으로 제공됩니다:
- `.hwp`: 한글 문서 (구버전, OLE2 기반)
  - **직접 파싱**: hwp5-table-extractor 사용 (권장)
  - **변환 후 파싱**: HWP → HWPX 변환 후 XML 파싱
- `.hwpx`: 한글 문서 (XML 기반, 최신 형식)
- `.pdf`: PDF 문서

## 결과

실험 결과는 다음 파일들에 저장됩니다:

### `results_multi_format.json`
- 형식별 표 추출 결과
- 파일 그룹별 비교
- 처리 시간 통계

### `results_label_comparison.json`
- 라벨링 데이터와의 비교 결과
- 형식별 F1-score, Precision, Recall
- 상세 비교 정보

### `results_hwp_comparison.json`
- HWP 직접 파싱 vs HWPX 변환 비교 결과
- 표 추출 개수 및 처리 시간 비교
- RAG 성능 지표 (EM, F1, Hit@K)

### 결과 항목
- 표 추출 성능 (F1-score, 처리 속도)
- 형식별 비교 통계
- 라벨링 데이터 매칭 결과
- HWP 파싱 방법별 성능 비교

## 주요 발견 사항

### HWP 직접 파싱 가능성 입증
초기 가설("HWP 직접 파싱은 실패하거나 제한적이다")과 달리, **hwp5-table-extractor**를 사용하여 HWP 파일에서 직접 표 추출이 가능함을 입증했습니다:

- ✅ **45개 표 추출 성공** (HWPX의 43개보다 많음)
- ✅ HWPX 변환 없이도 표 추출 가능
- ✅ 검증된 라이브러리를 활용한 안정적인 파싱

이를 통해 HWP 파일을 HWPX로 변환하는 단계 없이도 직접 파싱이 가능하다는 것을 확인했습니다.

## 기술 스택

- **표 추출**: 
  - **HWP**: 
    - hwp5-table-extractor (HWP5 직접 파싱)
    - pyhwp (HWP5 레코드 구조 파싱)
    - olefile (OLE2 구조 파싱)
  - **HWPX**: XML 파싱 (lxml, BeautifulSoup)
  - **PDF**: pdfplumber, camelot-py, tabula-py
  
- **RAG 시스템**:
  - LLM: Google Gemini 2.0 Flash Exp
  - 임베딩: Google text-embedding-004
  - 벡터 DB: ChromaDB
  - 프레임워크: LangChain

- **평가**:
  - scikit-learn (Precision, Recall, F1)
  - NLTK (BLEU Score)

## 참고 문헌

- TableRAG: SQL 기반 표 RAG 접근법
- HD-RAG: 계층적 행-열 수준 표 요약
- PDF 표 추출 연구들

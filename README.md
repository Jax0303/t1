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

PDF 표 추출을 위한 베이스라인:
- **pdfplumber**: Python 기반 PDF 파싱 라이브러리
- **camelot**: PDF 표 추출 전용 라이브러리 (lattice/stream 방법)
- **tabula**: Java 기반 PDF 표 추출 도구

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
- `.hwp`: 한글 문서 (구버전)
- `.hwpx`: 한글 문서 (XML 기반)
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

### 결과 항목
- 표 추출 성능 (F1-score, 처리 속도)
- 형식별 비교 통계
- 라벨링 데이터 매칭 결과

## 기술 스택

- **표 추출**: 
  - HWPX: XML 파싱 (lxml, BeautifulSoup)
  - PDF: pdfplumber, camelot-py, tabula-py
  
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

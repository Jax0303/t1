# 3축 PDF 추출기 빠른 시작 가이드

## 설치

비용 발생 없는 추출기들을 설치합니다:

```bash
# 기본 전통 베이스라인 (이미 설치되어 있을 수 있음)
pip install pypdf

# RAG/GenAI 지향 추출기
pip install docling-core
pip install "unstructured[pdf]"
pip install marker-pdf
```

또는 requirements.txt에서 설치:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 개별 추출기 테스트

```python
from src.extractors.pdf_extractor import PDFTableExtractor

# pypdf (전통 베이스라인)
extractor = PDFTableExtractor(method='pypdf')
tables = extractor.extract_tables('data/dart_pdfs/your_file.pdf')

# Docling (RAG/GenAI 지향)
extractor = PDFTableExtractor(method='docling')
tables = extractor.extract_tables('data/dart_pdfs/your_file.pdf')

# unstructured (RAG/GenAI 지향)
extractor = PDFTableExtractor(method='unstructured')
tables = extractor.extract_tables('data/dart_pdfs/your_file.pdf')

# Marker (RAG/GenAI 지향)
extractor = PDFTableExtractor(method='marker')
tables = extractor.extract_tables('data/dart_pdfs/your_file.pdf')
```

### 2. 비교 실험 실행

```bash
# 모든 추출기 비교 (기존 + 새로운)
python3 scripts/compare_pdf_extractors.py --max-files 5

# 특정 추출기만 비교
python3 scripts/compare_pdf_extractors.py --max-files 5 --methods pypdf docling unstructured marker

# 전통 베이스라인만 비교
python3 scripts/compare_pdf_extractors.py --max-files 5 --methods pdfplumber pymupdf pypdf

# RAG/GenAI 지향만 비교
python3 scripts/compare_pdf_extractors.py --max-files 5 --methods docling unstructured marker
```

## 추출기별 특징

### (A) 전통 베이스라인

| 추출기 | 특징 | 장점 | 단점 |
|--------|------|------|------|
| **pypdf** | 가장 단순한 텍스트 추출 | 설치 간단, 빠름 | 표 구조 추출 제한적 |
| **PyMuPDF** | 빠른 속도 | 속도 최고, 최신 버전에서 표 지원 | 표 추출 기능 제한적 |
| **pdfplumber** | 표/레이아웃 포함 | 높은 정확도, 표 구조 보존 | 상대적으로 느림 |

### (B) RAG/GenAI 지향

| 추출기 | 특징 | 장점 | 단점 |
|--------|------|------|------|
| **Docling** | IBM 오픈소스, 멀티 포맷 | 레이아웃 보존 우수 | API 문서 확인 필요 |
| **unstructured** | Element 기반 청킹 | RAG 최적화, 구조화된 출력 | GPU 선택적 필요 |
| **Marker** | 로컬 Markdown 변환 | LLM 활용, 높은 정확도 | GPU 권장, 모델 다운로드 |

## 주의사항

1. **Marker**: GPU가 있으면 자동으로 사용하지만, CPU만으로도 실행 가능 (느릴 수 있음)
2. **unstructured**: `strategy` 옵션 조정 가능 (`"fast"`, `"hi_res"`, `"ocr_only"`)
3. **Docling**: 실제 API 구조에 따라 코드 조정 필요할 수 있음
4. **pypdf**: 표 추출이 제한적이므로 텍스트 추출용으로만 권장

## 결과 확인

비교 실험 실행 후:
- `pdf_extractor_comparison.json`: 상세 결과 데이터
- `pdf_extractor_comparison.md`: 요약 리포트

## 문제 해결

### Docling 설치 오류
```bash
pip install docling-core --upgrade
```

### unstructured 설치 오류
```bash
pip install "unstructured[pdf]" --upgrade
# 또는
pip install unstructured[pdf] pillow
```

### Marker 설치 오류
```bash
pip install marker-pdf --upgrade
# GPU 없이 사용하려면 CPU 모드로 실행
```

## 다음 단계

1. 각 추출기로 테스트 파일 실행
2. 결과 비교 및 성능 분석
3. 필요시 코드 조정 (특히 Docling, Marker API)
4. VLM 추출기 추가 (Nougat, TATR) - GPU 필요



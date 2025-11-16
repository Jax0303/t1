# PDF 표 추출 도구 비교 실험 가이드

## 개요

이 스크립트는 보고서에서 언급한 여러 PDF 파서를 비교하여 성능을 평가합니다.

## 지원하는 파서

1. **pdfplumber** - 가장 높은 정확도 (보고서 기준)
2. **camelot** - Lattice/Stream 모드 지원
3. **tabula** - Java 기반, 안정적
4. **pymupdf** (Fitz) - 가장 빠른 속도

## 사용 방법

### 기본 사용 (모든 파서 비교)

```bash
python3 scripts/compare_pdf_extractors.py --max-files 5
```

### 특정 파서만 비교

```bash
python3 scripts/compare_pdf_extractors.py --max-files 5 --methods pdfplumber camelot
```

### 단일 파일 테스트

```bash
python3 scripts/compare_pdf_extractors.py --single-file "data/dart_pdfs/your_file.pdf"
```

### 옵션

- `--input`: 다운로드 목록 JSON 파일 (기본값: `data/dart_pdfs/download_list.json`)
- `--output`: 결과 JSON 파일 (기본값: `pdf_extractor_comparison.json`)
- `--methods`: 비교할 파서 리스트 (기본값: 모두)
- `--max-files`: 최대 처리 파일 수 (기본값: 5)
- `--single-file`: 단일 파일만 테스트

## 출력 결과

1. **JSON 파일** (`pdf_extractor_comparison.json`): 상세 결과 데이터
2. **마크다운 리포트** (`pdf_extractor_comparison.md`): 요약 리포트

## 결과 해석

### 성능 지표

- **성공 파일 수**: 표를 성공적으로 추출한 파일 수
- **총 추출 표**: 모든 파일에서 추출된 표의 총 개수
- **평균 표/파일**: 파일당 평균 표 개수
- **평균 시간/파일**: 파일당 평균 처리 시간

### 보고서 기준 예상 결과

- **pdfplumber**: 가장 높은 정확도, 중간 속도
- **camelot**: Lattice 모드에서 격자형 표에 우수, Stream 모드에서 공백 구분 표에 우수
- **tabula**: 안정적이지만 정확도는 pdfplumber보다 낮음
- **pymupdf**: 가장 빠른 속도, 표 추출 기능 제한적

## 문제 해결

### PyMuPDF 설치

```bash
pip install pymupdf
```

### Camelot 설치 (Ubuntu/Debian)

```bash
sudo apt-get install ghostscript python3-tk
pip install camelot-py[cv]
```

### Tabula 설치

Java가 필요합니다:

```bash
# Java 설치 확인
java -version

# Tabula 설치
pip install tabula-py
```

## 한국어 문서 특화 문제점

실험 결과에서 다음 문제점들을 확인할 수 있습니다:

1. **병합 셀**: 모든 파서에서 정확도 저하
2. **다중 페이지 표**: 표가 여러 페이지에 걸쳐 있을 때 추출 실패 가능
3. **한글 폰트**: 일부 파서에서 인식 문제
4. **스캔본 PDF**: OCR이 필요한 경우 추가 처리 필요

## 다음 단계

1. 실험 결과를 바탕으로 최적 파서 선택
2. 선택한 파서의 파라미터 튜닝
3. 하이브리드 접근 방식 고려 (여러 파서 조합)
4. 딥러닝 기반 표 검출 모델 고려 (Table Transformer 등)



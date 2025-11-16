# DART PDF 표 추출 가이드

## 개요

50개의 DART 공시 PDF 파일에서 표를 추출하고 지식 그래프로 변환하는 파이프라인입니다.

## 다음 단계

### 1. PDF 표 추출 테스트 (단일 파일)

```bash
# 가상환경 활성화 (필요시)
source venv/bin/activate

# 단일 파일 테스트
python3 -c "from src.extractors.pdf_extractor import PDFTableExtractor; extractor = PDFTableExtractor(); tables = extractor.extract_tables('data/dart_pdfs/3S_반기보고서 (2024.09)_20241114002413.pdf'); print(f'{len(tables)}개 표 추출됨')"
```

### 2. 일괄 표 추출 실행

```bash
# 모든 PDF 파일에서 표 추출
python3 scripts/batch_extract_pdf_tables.py

# 옵션:
# --input: 다운로드 목록 파일 경로 (기본값: data/dart_pdfs/download_list.json)
# --output: 출력 JSON 파일 경로 (기본값: extracted_tables_dart_pdfs.json)
# --method: 추출 방법 (pdfplumber, camelot, tabula) (기본값: pdfplumber)
# --max-files: 최대 처리 파일 수 (테스트용)

# 예시: 처음 5개 파일만 테스트
python3 scripts/batch_extract_pdf_tables.py --max-files 5

# 예시: camelot 방법 사용
python3 scripts/batch_extract_pdf_tables.py --method camelot
```

### 3. 추출 결과 확인

```bash
# 추출된 표 개수 확인
python3 -c "import json; data = json.load(open('extracted_tables_dart_pdfs.json')); print(f'총 {len(data)}개 표 추출됨')"

# 통계 확인
cat extracted_tables_dart_pdfs_stats.json
```

### 4. 지식 그래프 변환 및 RAG 파이프라인 실행

```bash
# 추출된 표를 지식 그래프로 변환하고 RAG 시스템 구축
python3 run_kg_rag_pipeline.py
```

**주의**: `run_kg_rag_pipeline.py`는 현재 `extracted_tables_hwp5_extractor_improved.json`을 사용합니다.
PDF 추출 결과를 사용하려면 스크립트를 수정하거나, 추출된 JSON 파일명을 변경해야 합니다.

## 파일 구조

```
data/dart_pdfs/
├── download_list.json          # 다운로드된 파일 목록
├── 3S_반기보고서 (2024.09)_*.pdf
├── 3S_분기보고서 (2024.12)_*.pdf
└── ... (50개 PDF 파일)

extracted_tables_dart_pdfs.json  # 추출된 표 데이터 (생성됨)
extracted_tables_dart_pdfs_stats.json  # 추출 통계 (생성됨)
```

## 추출 방법 비교

### pdfplumber (기본값)
- **장점**: 설치 간단, 대부분의 PDF에서 잘 동작
- **단점**: 복잡한 표 구조에서 정확도 낮을 수 있음

### camelot
- **장점**: 격자형 표에서 높은 정확도
- **단점**: 설치 복잡 (ghostscript 필요), 느림

### tabula
- **장점**: Java 기반, 안정적
- **단점**: Java 설치 필요, 일부 표에서 실패 가능

## 문제 해결

### pandas 모듈 없음 오류
```bash
pip install pandas
```

### pdfplumber 모듈 없음 오류
```bash
pip install pdfplumber
```

### camelot 사용 시
```bash
# Ubuntu/Debian
sudo apt-get install ghostscript python3-tk

# 또는 conda 사용
conda install -c conda-forge camelot-py
```

## 다음 작업

1. ✅ PDF 파일 다운로드 완료 (50개)
2. ✅ 일괄 표 추출 스크립트 작성 완료
3. ⏳ 표 추출 실행
4. ⏳ 지식 그래프 변환
5. ⏳ RAG 파이프라인 구축 및 테스트



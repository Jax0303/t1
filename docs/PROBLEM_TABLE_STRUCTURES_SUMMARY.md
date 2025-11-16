# 문제 표 구조 요약

## 핵심 발견 사항

### 1. 가장 흔한 문제: 헤더만 있는 표 (1행 표)

**발생 빈도**: 전체 빈 표의 대부분

**원인**:
- 현재 코드: `df = pd.DataFrame(table[1:], columns=table[0])`
- 1행만 있으면 `table[1:]`이 빈 리스트 → 빈 DataFrame 생성

**해결책**:
```python
# 1행만 있는 경우 처리
if len(table) == 1:
    df = pd.DataFrame([table[0]], columns=[f'col_{i}' for i in range(len(table[0]))])
```

### 2. 실제 데이터는 있지만 빈 표로 변환되는 경우

**발견**:
- 원본 PDF에는 표 데이터가 있음 (예: 11행 x 5열)
- 하지만 DataFrame 변환 과정에서 빈 표로 변환됨
- 예외 처리 부족으로 인한 문제

**해결책**:
- 예외 처리 강화
- 변환 실패 시 원본 데이터 보존
- 상세 로깅 추가

## 문제 표 유형별 발생 빈도

1. **헤더만 있는 표 (1행)**: 가장 많음 ⭐
2. **모든 셀이 None**: 중간
3. **모든 셀이 빈 문자열**: 적음
4. **단일 셀 표**: 적음

## 즉시 적용 가능한 해결책

### 코드 수정 (우선순위 높음)

```python
# src/extractors/pdf_extractor.py의 _extract_with_pdfplumber 수정

def _extract_with_pdfplumber(self, pdf_path: str, start_time: float) -> List[Dict]:
    tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            
            for idx, table in enumerate(page_tables):
                if table:
                    try:
                        # 개선된 변환 로직
                        if len(table) == 0:
                            continue
                        elif len(table) == 1:
                            # 1행만 있는 경우: 헤더를 데이터로 사용
                            df = pd.DataFrame([table[0]], columns=[f'col_{i}' for i in range(len(table[0]))])
                        else:
                            # 2행 이상: 첫 행을 헤더로 사용
                            headers = table[0]
                            # None 헤더 처리
                            if all(h is None or h == '' for h in headers):
                                headers = [f'col_{i}' for i in range(len(headers))]
                            df = pd.DataFrame(table[1:], columns=headers)
                        
                        # 빈 DataFrame 체크
                        if df.empty:
                            continue
                        
                        extraction_time = time.time() - start_time
                        tables.append({
                            'table_id': f"{Path(pdf_path).stem}_page{page_num}_table{idx}",
                            'dataframe': df,
                            'source_file': pdf_path,
                            'page_number': page_num + 1,
                            'extraction_method': 'pdfplumber',
                            'extraction_time': extraction_time
                        })
                    except Exception as e:
                        print(f"pdfplumber 표 처리 오류 (페이지 {page_num}, 표 {idx}): {e}")
    
    return tables
```

## 예상 효과

- **빈 표 감소**: 1행 표 문제 해결로 빈 표 대폭 감소 예상
- **데이터 보존**: 헤더만 있는 표도 데이터로 보존
- **안정성 향상**: 예외 처리 강화로 오류 감소


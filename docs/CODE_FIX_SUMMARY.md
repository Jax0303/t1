# 코드 수정 요약

## 수정 내용

### 1. 안전한 DataFrame 변환 함수 추가

`_safe_dataframe_conversion()` 메서드를 추가하여 다음 문제들을 해결:

1. **1행 표(헤더만) 처리**
   - 기존: `table[1:]`이 빈 리스트 → 빈 DataFrame 생성
   - 수정: 헤더를 데이터로 사용하여 단일 행 DataFrame 생성

2. **예외 처리 강화**
   - None 헤더 자동 생성
   - 빈 데이터 행 필터링
   - 변환 실패 시 None 반환

3. **파싱 툴 문제는 그대로 둠**
   - 모든 셀이 None인 경우: 제외 (파싱 툴 문제)
   - 모든 셀이 빈 문자열인 경우: 제외 (파싱 툴 문제)

### 2. 적용된 추출기

- ✅ **pdfplumber**: `_safe_dataframe_conversion()` 사용
- ✅ **pymupdf**: `_safe_dataframe_conversion()` 사용

### 3. 수정 전/후 비교

**동진쎄미켐_반기보고서**:
- 수정 전: 559개 표 중 45개 빈 표
- 수정 후: 558개 표 중 0개 빈 표
- **개선: 빈 표 45개 감소**

**1행 표 처리**:
- 수정 전: 빈 DataFrame으로 변환되어 손실
- 수정 후: 헤더를 데이터로 사용하여 보존 (136개)

## 코드 변경 사항

### 추가된 메서드

```python
def _safe_dataframe_conversion(self, table: List[List]) -> Optional[pd.DataFrame]:
    """
    안전한 DataFrame 변환
    1행 표(헤더만) 처리 및 예외 처리 강화
    """
    # 1행만 있는 경우: 헤더를 데이터로 사용
    if len(table) == 1:
        headers = table[0]
        if all(cell is None or cell == '' for cell in headers):
            return None  # 파싱 툴 문제는 제외
        df = pd.DataFrame([headers], columns=[f'col_{i}' for i in range(len(headers))])
        return df
    
    # 2행 이상: 첫 행을 헤더로 사용
    # ... (상세 로직)
```

### 수정된 메서드

- `_extract_with_pdfplumber()`: `_safe_dataframe_conversion()` 사용
- `_extract_with_pymupdf()`: `_safe_dataframe_conversion()` 사용

## 테스트 결과

### 동진쎄미켐_반기보고서
- ✅ 빈 표: 0개 (이전 45개)
- ✅ 1행 표: 136개 (헤더만 있는 표 보존)
- ✅ 정상 표: 422개

### 다른 파일들
- 휴온스글로벌: 빈 표 0개
- 파멥신: 빈 표 0개

## 효과

1. **빈 표 문제 해결**: 코드로 보완 가능한 빈 표 문제 완전 해결
2. **데이터 보존**: 헤더만 있는 표도 데이터로 보존
3. **안정성 향상**: 예외 처리 강화로 오류 감소

## 남은 문제 (파싱 툴 문제)

다음 문제들은 파싱 툴 자체의 한계이므로 그대로 둠:

1. 모든 셀이 None인 표 (텍스트 레이어 없음)
2. 이미지 기반 표 (OCR 필요)
3. 폰트 문제로 인한 텍스트 추출 실패

이러한 경우는 VLM/OCR 기반 추출기(Nougat, TATR)를 사용해야 함.


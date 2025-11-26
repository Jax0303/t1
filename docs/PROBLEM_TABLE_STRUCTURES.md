# 문제 표 구조 상세 분석

## 개요

PDF 추출 과정에서 빈 표(Empty Tables)가 발생하는 원인과 패턴을 분석한 문서입니다.

## 발견된 문제 유형

### 1. 모든 셀이 None인 표

**특징**:
- 표 구조는 감지되었지만 모든 셀 값이 `None`
- 텍스트 추출이 완전히 실패한 경우

**발생 원인**:
- PDF 내부의 텍스트 레이어가 없는 경우
- 이미지로만 구성된 표
- 폰트 임베딩 문제로 텍스트 추출 불가
- 복잡한 레이아웃으로 인한 파싱 실패

**예시**:
```
표 구조: 3행 x 4열
셀 값: [None, None, None, None]
       [None, None, None, None]
       [None, None, None, None]
```

### 2. 모든 셀이 빈 문자열인 표

**특징**:
- 표 구조는 있지만 모든 셀이 빈 문자열(`''`)
- 텍스트는 있지만 내용이 없는 경우

**발생 원인**:
- 공백만 있는 셀
- 숨겨진 텍스트
- 폰트 렌더링 문제

**예시**:
```
표 구조: 2행 x 3열
셀 값: ['', '', '']
       ['', '', '']
```

### 3. 단일 셀만 있는 표

**특징**:
- 1행 x 1열 표
- 실제 표가 아닌 다른 구조를 표로 잘못 인식

**발생 원인**:
- 표가 아닌 텍스트 블록을 표로 오인식
- 페이지 번호나 헤더/푸터를 표로 인식
- 단일 셀 구조

**예시**:
```
표 구조: 1행 x 1열
셀 값: ['페이지 번호']
```

### 4. 헤더만 있고 데이터 행이 없는 표 ⚠️ 가장 흔한 문제

**특징**:
- 헤더 행만 존재하고 데이터 행이 없음
- 표 구조는 있지만 내용이 비어있음
- **현재 코드 로직상 빈 DataFrame으로 변환됨**

**발생 원인**:
- 표 템플릿만 있고 데이터가 채워지지 않은 경우
- 페이지 나누기로 인한 표 분할
- 표의 일부만 추출된 경우
- **코드 로직**: `table[1:]`이 빈 리스트가 되어 빈 DataFrame 생성

**예시**:
```
원본 표 구조: 1행 x 5열
셀 값: ['항목1', '항목2', '항목3', '항목4', '항목5']
(데이터 행 없음)

현재 코드 동작:
  df = pd.DataFrame(table[1:], columns=table[0])
  → table[1:] = [] (빈 리스트)
  → 빈 DataFrame 생성
```

**실제 발생 사례**:
- 페이지 11: 표 #2 (1행 x 7열) - 헤더만 있음
- 페이지 15: 표 #2 (1행 x 4열) - 헤더만 있음  
- 페이지 18: 표 #3 (1행 x 3열) - 헤더만 있음

## 문제가 발생하는 파일/페이지

### 동진쎄미켐_반기보고서 (2025.06)
- **빈 표 45개** 발견
- 문제 페이지: 11, 15, 18, 21, 23, 29, 37, 53, 54, 57 등

### 휴온스글로벌_반기보고서 (2025.06)
- **빈 표 22개** 발견

### 파멥신_분기보고서 (2025.03)
- **빈 표 14개** 발견

## 실제 문제 원인 (코드 분석 결과)

### 핵심 문제: DataFrame 변환 로직

**현재 코드**:
```python
df = pd.DataFrame(table[1:], columns=table[0] if table else None)
```

**문제점**:
1. **1행만 있는 표**: `table[1:]`이 빈 리스트가 되어 빈 DataFrame 생성
2. **헤더만 있는 표**: 데이터 행이 없어서 빈 DataFrame 생성
3. **예외 처리 부족**: 변환 실패 시 빈 표로 저장됨

**실제 사례**:
- 원본 PDF에는 표 데이터가 있음 (11행 x 5열)
- 하지만 DataFrame 변환 시 빈 표로 변환됨
- 특히 1행만 있는 표(헤더만)가 주된 원인

## 기술적 원인 분석

### 1. PDF 구조 문제

**텍스트 레이어 없음**:
- PDF가 이미지로만 구성
- 스캔된 문서
- 텍스트 추출 불가

**복잡한 레이아웃**:
- 다중 컬럼 레이아웃
- 병합 셀
- 계층적 헤더 구조
- 표가 페이지 경계를 넘어감

### 2. 추출기 한계

**pdfplumber**:
- 표 경계 감지는 성공
- 셀 내용 추출 실패
- 복잡한 병합 셀 처리 한계

**pymupdf**:
- 동일한 문제 발생
- 표 감지 로직이 pdfplumber와 유사

### 3. 데이터 품질 문제

**인코딩 문제**:
- 한글 폰트 인코딩 이슈
- 특수 문자 처리 실패

**폰트 문제**:
- 임베딩되지 않은 폰트
- 커스텀 폰트 사용
- 폰트 매핑 실패

## 해결 방안

### 1. DataFrame 변환 로직 개선 ⭐ 우선순위 높음

```python
def safe_dataframe_conversion(table):
    """안전한 DataFrame 변환"""
    if not table or len(table) == 0:
        return None
    
    # 1행만 있는 경우: 헤더를 데이터로 사용
    if len(table) == 1:
        # 헤더만 있는 표는 단일 행 DataFrame으로 생성
        df = pd.DataFrame([table[0]], columns=[f'col_{i}' for i in range(len(table[0]))])
        return df
    
    # 2행 이상: 첫 행을 헤더로 사용
    headers = table[0]
    data_rows = table[1:]
    
    # 헤더가 모두 None이면 자동 생성
    if all(h is None or h == '' for h in headers):
        headers = [f'col_{i}' for i in range(len(headers))]
    
    try:
        df = pd.DataFrame(data_rows, columns=headers)
        # 빈 DataFrame 체크
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"DataFrame 변환 오류: {e}")
        return None
```

### 2. 빈 표 필터링

```python
def filter_empty_tables(tables):
    """빈 표 제거"""
    filtered = []
    for table in tables:
        df = table.get('dataframe')
        if df is not None and not df.empty:
            # 최소 크기 검증 (헤더만 있는 표도 유지하려면 조건 완화)
            if len(df) >= 1 and len(df.columns) >= 1:
                filtered.append(table)
    return filtered
```

### 2. 표 품질 검증

```python
def validate_table_quality(table):
    """표 품질 검증"""
    df = table.get('dataframe')
    if df is None or df.empty:
        return False
    
    # None 값 비율 확인
    none_ratio = df.isna().sum().sum() / (len(df) * len(df.columns))
    if none_ratio > 0.5:
        return False
    
    # 최소 데이터 행 확인
    data_rows = df.dropna(how='all').shape[0]
    if data_rows < 2:
        return False
    
    return True
```

### 3. 후처리 개선

- 빈 표 자동 제거 옵션
- 표 품질 점수 계산
- 문제 표 상세 로깅
- 원본 PDF 구조 분석

### 4. 대안 추출 방법

- OCR 기반 추출 (스캔 문서용)
- 이미지 기반 표 감지
- 레이아웃 분석 강화
- VLM 기반 추출 (Nougat, TATR)

## 권장 사항

1. **빈 표 자동 필터링**: 추출 후 빈 표 제거
2. **품질 검증**: 표 유효성 검사 후 저장
3. **문제 표 로깅**: 어떤 표가 문제인지 기록
4. **사용자 알림**: 빈 표가 많은 파일 경고

## 다음 단계

1. ✅ 문제 표 구조 분석 완료
2. ⏳ 빈 표 필터링 기능 구현
3. ⏳ 표 품질 검증 로직 추가
4. ⏳ 문제 표 상세 리포트 생성


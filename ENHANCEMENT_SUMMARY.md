# HWP/HWPX 파싱 개선 사항 요약

## 개선 완료 사항

### 1. HWPX 추출기 개선 ✅

**hp:tbl, hp:tr, hp:tc 네임스페이스 처리 강화**

- ElementTree를 사용한 네임스페이스 처리 추가
- `hp:tbl` → `hp:tr` → `hp:tc` → `hp:t` 구조 직접 파싱
- RAG 시스템에서 바로 SQL/데이터프레임으로 변환 가능한 구조

**주요 변경사항**:
- `_parse_hp_tbl_element()`: hp:tbl 요소를 직접 파싱
- `_extract_text_from_hp_tc()`: hp:tc 요소에서 텍스트 추출
- 네임스페이스 `http://www.hancom.co.kr/hwpml/2011/schema` 처리

### 2. HWP 파싱 전략 개선 ✅

**우선순위 기반 파싱 전략 구현**

1. **HWPX 변환 시도 (가장 안전)**: 가능한 경우 HWPX로 변환 후 처리
2. **olefile + zlib 활용**: 스토리지/스트림 읽기 및 레코드 구조 해석
3. **pyhwp 사용**: HWP5 형식 직접 파싱
4. **바이너리 직접 파싱**: 대체 방법

**주요 변경사항**:
- `_extract_with_olefile_and_zlib()`: olefile과 zlib를 활용한 스토리지/스트림 읽기
- `_parse_hwp_records()`: HWP 레코드 구조 해석
- `_extract_table_from_record()`: 레코드 데이터에서 표 추출

### 3. 레코드 구조 해석 구현 ✅

**HWP 레코드 기반 구조 파싱**

- 레코드 헤더 파싱 (태그, 크기)
- 스트림 데이터에서 표 정보 추출
- zlib 압축 해제 지원
- 텍스트 패턴 매칭을 통한 표 추출

## 기술적 세부사항

### HWPX 네임스페이스 처리

```python
# hp:tbl 태그 찾기
tbl_elements = root.findall('.//{http://www.hancom.co.kr/hwpml/2011/schema}tbl')

# hp:tr 태그 찾기
tr_elements = tbl_elem.findall('.//{http://www.hancom.co.kr/hwpml/2011/schema}tr')

# hp:tc 태그 찾기
tc_elements = tr_elem.findall('.//{http://www.hancom.co.kr/hwpml/2011/schema}tc')

# hp:t 태그에서 텍스트 추출
t_elements = tc_elem.findall('.//{http://www.hancom.co.kr/hwpml/2011/schema}t')
```

### HWP 레코드 구조 해석

```python
# 레코드 헤더 파싱 (4바이트 태그 + 4바이트 크기)
tag = struct.unpack('<I', data[pos:pos+4])[0]
size = struct.unpack('<I', data[pos+4:pos+8])[0]

# 레코드 데이터 추출
record_data = data[pos+8:pos+8+size]

# 표 데이터 추출 시도
table_data = self._extract_table_from_record(record_data, tag)
```

### olefile + zlib 활용

```python
# 스트림 읽기
stream_data = ole.openstream(stream_name).read()

# 방법 1: 레코드 구조 해석
extracted_tables = self._parse_hwp_records(stream_data, hwp_path, table_idx)

# 방법 2: zlib 압축 해제 시도
decompressed = zlib.decompress(stream_data)
extracted_tables = self._parse_hwp_records(decompressed, hwp_path, table_idx)
```

## 비교 실험 설계

동일한 문서에 대해:
1. **HWPX 변환 기반**: HWP → HWPX 변환 → 표 추출 → RAG
2. **HWP 직접 파싱**: HWP → 직접 파싱 → 표 추출 → RAG

**평가 지표**:
- EM Score: 정답 일치율
- F1 Score: 답변 품질
- Hit@K: 검색 정확도 (Hit@1, Hit@3, Hit@5)

## 사용 방법

### HWPX 추출 (개선됨)

```python
from src.extractors import HWPXTableExtractor

extractor = HWPXTableExtractor()
tables = extractor.extract_tables("document.hwpx")
# hp:tbl, hp:tr, hp:tc 태그를 직접 추출하여 DataFrame으로 변환
```

### HWP 직접 파싱 (개선됨)

```python
from src.extractors import HWPTableExtractor

extractor = HWPTableExtractor(use_direct_parsing=True)
tables = extractor.extract_tables("document.hwp", method='direct')
# 우선순위: HWPX 변환 → olefile+zlib → pyhwp → 바이너리 파싱
```

## 기대 효과

1. **파싱 속도 향상**: HWPX 변환 과정 제거로 속도 향상
2. **정확성 향상**: 레코드 구조 해석으로 더 정확한 표 추출
3. **호환성 향상**: 다양한 HWP 버전 지원 (HWP5, 구버전)
4. **RAG 최적화**: 구조화된 데이터로 RAG 성능 향상

## 참고

- HWP 파일 형식은 복잡하므로 모든 파일에서 완벽한 파싱이 보장되지 않을 수 있습니다.
- 실제 성능은 파일의 구조와 내용에 따라 달라질 수 있습니다.
- olefile과 zlib는 표준 라이브러리이지만, pyhwp는 선택적 설치가 필요합니다.


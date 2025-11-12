# HWP 파싱 개선안

## 현재 상황 분석

### 문제점
1. **레코드 구조 파싱 실패**: Section1 스트림(212KB)에서 레코드를 찾지 못함
2. **표 추출 실패**: 1개만 추출되었으나 데이터가 깨져있음
3. **압축 해제 미완료**: zlib 압축 여부 확인 필요

### 성공 사례
- **HWPX 파싱**: 43개 표 성공적으로 추출
- **파일 구조 인식**: OLE2 형식, Section 스트림 4개 확인

## 개선 방안

### 1. HWP 레코드 구조 정확한 파싱

#### 1.1 레코드 헤더 구조 개선
```python
# 현재 문제: 단순한 4바이트 태그 + 4바이트 크기 구조만 시도
# 개선: HWP 버전별 레코드 구조 지원

def _parse_hwp_records_v5(self, data: bytes):
    """
    HWP 5.0 레코드 구조:
    - 레코드 헤더: 4바이트 태그 + 4바이트 크기
    - 레코드 본문: 크기만큼의 데이터
    - 레코드 간격: 4바이트 정렬
    """
    pos = 0
    records = []
    
    while pos < len(data) - 8:
        # 레코드 헤더 읽기
        tag = struct.unpack('<I', data[pos:pos+4])[0]
        size = struct.unpack('<I', data[pos+4:pos+8])[0]
        
        # 유효성 검사 강화
        if not self._is_valid_record(tag, size, pos, len(data)):
            pos += 1
            continue
        
        # 레코드 데이터 추출
        record_data = data[pos+8:pos+8+size]
        records.append({
            'tag': tag,
            'size': size,
            'data': record_data,
            'offset': pos
        })
        
        # 다음 레코드로 이동 (4바이트 정렬)
        pos += 8 + size
        pos = (pos + 3) & ~3  # 4바이트 정렬
    
    return records

def _is_valid_record(self, tag: int, size: int, pos: int, total_size: int) -> bool:
    """레코드 유효성 검사"""
    # 1. 크기 범위 확인
    if size <= 0 or size > 1000000:  # 1MB 제한
        return False
    
    # 2. 범위 초과 확인
    if pos + 8 + size > total_size:
        return False
    
    # 3. 태그 패턴 확인 (HWP 5.0 표 관련 태그)
    # 표 관련 태그: 0x000B0000 ~ 0x000BFFFF (예상)
    # 실제 태그 값은 HWP 버전별로 다를 수 있음
    if tag & 0xFF000000 == 0x00000000:  # 상위 바이트가 0인 경우
        return True
    
    return False
```

#### 1.2 표 레코드 식별 개선
```python
# HWP 표 관련 태그 정의
TABLE_TAGS = {
    0x000B0001: 'TABLE_START',      # 표 시작
    0x000B0002: 'TABLE_ROW',        # 행
    0x000B0003: 'TABLE_CELL',       # 셀
    0x000B0004: 'TABLE_END',        # 표 끝
}

def _identify_table_records(self, records: List[Dict]) -> List[Dict]:
    """표 관련 레코드 식별"""
    table_records = []
    current_table = []
    
    for record in records:
        tag = record['tag']
        
        # 표 시작 태그 확인
        if tag in TABLE_TAGS or (tag & 0xFFFF0000) == 0x000B0000:
            if TABLE_TAGS.get(tag) == 'TABLE_START':
                if current_table:
                    table_records.append(current_table)
                current_table = [record]
            elif current_table:
                current_table.append(record)
                if TABLE_TAGS.get(tag) == 'TABLE_END':
                    table_records.append(current_table)
                    current_table = []
    
    if current_table:
        table_records.append(current_table)
    
    return table_records
```

### 2. 압축 해제 개선

#### 2.1 다중 압축 형식 지원
```python
def _decompress_stream(self, data: bytes) -> Optional[bytes]:
    """다양한 압축 형식 시도"""
    # 1. zlib 압축 시도
    try:
        return zlib.decompress(data)
    except:
        pass
    
    # 2. zlib 압축 (wbits=15) 시도
    try:
        return zlib.decompress(data, wbits=15)
    except:
        pass
    
    # 3. zlib 압축 (wbits=-15) 시도
    try:
        return zlib.decompress(data, wbits=-15)
    except:
        pass
    
    # 4. 압축되지 않은 경우 원본 반환
    return data
```

#### 2.2 스트림별 압축 여부 확인
```python
def _extract_with_olefile_improved(self, hwp_path: str) -> List[Dict]:
    """개선된 olefile 파싱"""
    import olefile
    
    tables = []
    ole = olefile.OleFileIO(hwp_path)
    
    # 모든 스트림 확인
    for stream_name in ole.listdir():
        stream_data = ole.openstream(stream_name).read()
        
        # 압축 해제 시도
        decompressed = self._decompress_stream(stream_data)
        
        # 레코드 파싱
        records = self._parse_hwp_records_v5(decompressed)
        
        # 표 레코드 식별
        table_records = self._identify_table_records(records)
        
        # 표 데이터 추출
        for table_record_group in table_records:
            table_data = self._extract_table_from_records(table_record_group)
            if table_data is not None:
                tables.append(table_data)
    
    ole.close()
    return tables
```

### 3. 표 데이터 추출 개선

#### 3.1 셀 데이터 파싱 개선
```python
def _extract_table_from_records(self, records: List[Dict]) -> Optional[pd.DataFrame]:
    """레코드 그룹에서 표 추출"""
    rows = []
    current_row = []
    
    for record in records:
        tag = record['tag']
        data = record['data']
        
        if tag == TABLE_TAGS.get('TABLE_ROW'):
            if current_row:
                rows.append(current_row)
            current_row = []
        elif tag == TABLE_TAGS.get('TABLE_CELL'):
            # 셀 데이터 추출
            cell_text = self._extract_cell_text(data)
            current_row.append(cell_text)
    
    if current_row:
        rows.append(current_row)
    
    if rows:
        # 행 길이 맞추기
        max_cols = max(len(row) for row in rows) if rows else 0
        for row in rows:
            while len(row) < max_cols:
                row.append("")
        
        return pd.DataFrame(rows)
    
    return None

def _extract_cell_text(self, cell_data: bytes) -> str:
    """셀 데이터에서 텍스트 추출"""
    # 방법 1: UTF-16 LE 인코딩 시도
    try:
        text = cell_data.decode('utf-16-le', errors='ignore')
        # 제어 문자 제거
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    except:
        pass
    
    # 방법 2: UTF-8 인코딩 시도
    try:
        text = cell_data.decode('utf-8', errors='ignore')
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    except:
        pass
    
    # 방법 3: CP949 인코딩 시도 (한글)
    try:
        text = cell_data.decode('cp949', errors='ignore')
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text.strip()
    except:
        pass
    
    return ""
```

### 4. pyhwp 라이브러리 활용

#### 4.1 pyhwp 설치 및 활용
```bash
pip install pyhwp
```

```python
def _extract_with_pyhwp_improved(self, hwp_path: str) -> List[Dict]:
    """pyhwp 라이브러리 활용"""
    try:
        from pyhwp import hwp5
        from pyhwp.hwp5 import plat
        
        # HWP 파일 열기
        hwp5file = hwp5.open(hwp_path)
        
        tables = []
        
        # BodyText 섹션 순회
        if hasattr(hwp5file, 'bodytext'):
            for section in hwp5file.bodytext.sections:
                # 표 찾기
                for record in section:
                    if record.tagname == 'HWPTAG_TABLE':
                        # 표 데이터 추출
                        table_data = self._parse_pyhwp_table(record)
                        if table_data is not None:
                            tables.append(table_data)
        
        return tables
    
    except ImportError:
        print("pyhwp 라이브러리가 설치되지 않았습니다.")
        return []
    except Exception as e:
        print(f"pyhwp 파싱 오류: {e}")
        return []
```

### 5. HWPX 변환 활용 (하이브리드 접근)

#### 5.1 HWP → HWPX 변환 후 파싱
```python
def _convert_hwp_to_hwpx(self, hwp_path: str) -> Optional[str]:
    """HWP를 HWPX로 변환"""
    try:
        # 한컴오피스 명령줄 도구 사용
        # 또는 pyhwp의 변환 기능 활용
        import subprocess
        
        hwpx_path = hwp_path.replace('.hwp', '_converted.hwpx')
        
        # 한컴오피스 변환 명령 (예시)
        # 실제 명령은 한컴오피스 버전에 따라 다를 수 있음
        result = subprocess.run(
            ['hwp5proc', 'xml', hwp_path, hwpx_path],
            capture_output=True,
            timeout=30
        )
        
        if result.returncode == 0 and os.path.exists(hwpx_path):
            return hwpx_path
        
    except Exception as e:
        print(f"HWPX 변환 실패: {e}")
    
    return None

def _extract_hybrid(self, hwp_path: str) -> List[Dict]:
    """하이브리드 접근: 직접 파싱 실패 시 HWPX 변환"""
    # 1. 직접 파싱 시도
    tables = self._extract_direct(hwp_path)
    
    if len(tables) > 0:
        return tables
    
    # 2. HWPX 변환 후 파싱
    hwpx_path = self._convert_hwp_to_hwpx(hwp_path)
    if hwpx_path:
        tables = self.hwpx_extractor.extract_tables(hwpx_path)
        # 임시 파일 삭제
        os.remove(hwpx_path)
        return tables
    
    return []
```

### 6. 디버깅 및 로깅 개선

#### 6.1 상세 로깅 추가
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _parse_hwp_records_v5(self, data: bytes):
    """로깅이 추가된 레코드 파싱"""
    logger.debug(f"레코드 파싱 시작: {len(data)} bytes")
    
    records = []
    pos = 0
    
    while pos < len(data) - 8:
        tag = struct.unpack('<I', data[pos:pos+4])[0]
        size = struct.unpack('<I', data[pos+4:pos+8])[0]
        
        logger.debug(f"pos={pos:08x}, tag=0x{tag:08x}, size={size}")
        
        if self._is_valid_record(tag, size, pos, len(data)):
            records.append({'tag': tag, 'size': size, 'offset': pos})
            pos += 8 + size
        else:
            pos += 1
    
    logger.info(f"총 {len(records)}개 레코드 발견")
    return records
```

## 구현 우선순위

### Phase 1: 즉시 구현 가능
1. ✅ 압축 해제 개선 (다중 형식 지원)
2. ✅ 레코드 유효성 검사 강화
3. ✅ 인코딩 다중 시도 (UTF-16, UTF-8, CP949)

### Phase 2: 중기 개선
1. ⚠️ pyhwp 라이브러리 통합
2. ⚠️ HWP 버전별 레코드 구조 지원
3. ⚠️ 표 태그 매핑 정확도 향상

### Phase 3: 장기 개선
1. 🔄 HWPX 변환 자동화
2. 🔄 한컴오피스 SDK 활용
3. 🔄 머신러닝 기반 표 구조 인식

## 예상 효과

- **표 추출 성공률**: 현재 0% → 목표 80%+
- **표 데이터 정확도**: 현재 0% → 목표 90%+
- **처리 속도**: 현재 0.02초 → 목표 1초 이내

## 참고 자료

- HWP 파일 포맷 스펙 (비공개, 한컴오피스)
- pyhwp 라이브러리: https://github.com/mete0r/pyhwp
- HWP-MCP 서버: https://www.mcp.pizza/mcp-server/gzMP/hwp-mcp


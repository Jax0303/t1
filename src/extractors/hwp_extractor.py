"""
HWP 파일에서 표를 추출하는 모듈
HWP는 바이너리 형식이므로 HWPX로 변환하거나 직접 파싱 필요

파싱 전략:
1. 가능한 경우 HWPX로 변환 후 처리 (가장 안전)
2. 변환이 불가능하면 olefile과 zlib를 이용해 스토리지/스트림 읽기 및 레코드 구조 해석
3. 텍스트 정규화 및 표 구조 보존 강화
"""
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import time
import re
import zlib
import struct

# HWPX 추출기 재사용 (베이스라인용)
from .hwpx_extractor import HWPXTableExtractor


class HWPTableExtractor:
    """HWP 문서에서 표를 추출하는 클래스
    
    두 가지 방법 제공:
    1. HWPX 변환 기반 (베이스라인)
    2. 직접 파싱 (새로운 방법)
    """
    
    def __init__(self, use_direct_parsing: bool = True):
        """
        Args:
            use_direct_parsing: True면 직접 파싱 사용, False면 HWPX 변환 사용
        """
        self.hwpx_extractor = HWPXTableExtractor()
        self.use_direct_parsing = use_direct_parsing
        self._check_dependencies()
    
    def _check_dependencies(self):
        """필요한 라이브러리 확인"""
        self.has_pyhwp = False
        self.has_olefile = False
        self.has_zlib = True  # zlib는 표준 라이브러리
        
        try:
            import pyhwp
            self.has_pyhwp = True
        except ImportError:
            pass
        
        try:
            import olefile
            self.has_olefile = True
        except ImportError:
            pass
    
    def extract_tables(self, hwp_path: str, method: str = 'auto') -> List[Dict]:
        """
        HWP 파일에서 표 추출
        
        Args:
            hwp_path: HWP 파일 경로
            method: 'direct' (직접 파싱), 'hwpx' (HWPX 변환), 'auto' (자동 선택)
            
        Returns:
            표 정보 리스트
        """
        start_time = time.time()
        tables = []
        
        try:
            if method == 'hwpx' or (method == 'auto' and not self.use_direct_parsing):
                # 방법 1: HWPX로 변환 후 추출 (베이스라인)
                tables = self._extract_via_hwpx(hwp_path)
                extraction_method = 'hwp_via_hwpx'
            else:
                # 방법 2: 직접 파싱 (새로운 방법)
                tables = self._extract_direct(hwp_path)
                extraction_method = 'hwp_direct_parsing'
            
            # 메타데이터 업데이트
            for table in tables:
                table['source_file'] = hwp_path
                table['extraction_method'] = extraction_method
                table['original_format'] = 'hwp'
        
        except Exception as e:
            print(f"HWP 추출 오류 ({hwp_path}): {e}")
            import traceback
            traceback.print_exc()
        
        extraction_time = time.time() - start_time
        
        # 각 표에 처리 시간 추가
        for table in tables:
            table['extraction_time'] = extraction_time / len(tables) if tables else extraction_time
        
        return tables
    
    def _extract_via_hwpx(self, hwp_path: str) -> List[Dict]:
        """HWPX 변환 기반 추출 (베이스라인)"""
        hwpx_path = self._convert_to_hwpx(hwp_path)
        
        if hwpx_path and os.path.exists(hwpx_path):
            tables = self.hwpx_extractor.extract_tables(hwpx_path)
            return tables
        
        return []
    
    def _extract_direct(self, hwp_path: str) -> List[Dict]:
        """HWP 파일 직접 파싱 (새로운 방법)
        
        우선순위:
        1. HWPX 변환 시도 (가장 안전)
        2. olefile + zlib를 이용한 스토리지/스트림 읽기 및 레코드 구조 해석
        3. pyhwp 사용 (HWP5 형식)
        4. 바이너리 직접 파싱
        """
        tables = []
        
        # 방법 0: 먼저 HWPX 변환 시도 (가장 안전)
        try:
            hwpx_tables = self._extract_via_hwpx(hwp_path)
            if hwpx_tables:
                print(f"  HWPX 변환 성공: {len(hwpx_tables)}개 표 추출")
                return hwpx_tables
        except Exception as e:
            print(f"  HWPX 변환 실패: {e}")
        
        # 방법 1: olefile + zlib를 이용한 스토리지/스트림 읽기 및 레코드 구조 해석
        if self.has_olefile:
            try:
                tables = self._extract_with_olefile_and_zlib(hwp_path)
                if tables:
                    print(f"  olefile+zlib 파싱 성공: {len(tables)}개 표 추출")
                    return tables
            except Exception as e:
                print(f"  olefile+zlib 파싱 실패: {e}")
        
        # 방법 2: pyhwp 사용 (HWP5 형식)
        if self.has_pyhwp:
            try:
                tables = self._extract_with_pyhwp(hwp_path)
                if tables:
                    print(f"  pyhwp 파싱 성공: {len(tables)}개 표 추출")
                    return tables
            except Exception as e:
                print(f"  pyhwp 파싱 실패: {e}")
        
        # 방법 3: 바이너리 직접 파싱 (간단한 버전)
        try:
            tables = self._extract_with_binary_parsing(hwp_path)
            if tables:
                print(f"  바이너리 파싱 성공: {len(tables)}개 표 추출")
                return tables
        except Exception as e:
            print(f"  바이너리 파싱 실패: {e}")
        
        if not tables:
            print(f"  경고: 모든 직접 파싱 방법 실패.")
        
        return tables
    
    def _extract_with_pyhwp(self, hwp_path: str) -> List[Dict]:
        """pyhwp를 사용한 표 추출"""
        try:
            from pyhwp import hwp5
            from pyhwp.hwp5 import plat
            
            # HWP 파일 열기
            hwp5file = hwp5.open(hwp_path)
            
            tables = []
            table_idx = 0
            
            # 본문 스트림 찾기
            bodytext = hwp5file.bodytext
            
            if bodytext is None:
                return []
            
            # 섹션별로 처리
            for section in bodytext.sections:
                # 표 찾기
                for record in section.records:
                    if record.tagname == 'Table':
                        table_data = self._parse_pyhwp_table(record)
                        if table_data is not None and not table_data.empty:
                            tables.append({
                                'table_id': f"{Path(hwp_path).stem}_pyhwp_{table_idx}",
                                'dataframe': table_data,
                                'source_file': hwp_path,
                                'extraction_method': 'hwp_direct_pyhwp'
                            })
                            table_idx += 1
            
            return tables
        
        except Exception as e:
            print(f"  pyhwp 추출 오류: {e}")
            return []
    
    def _parse_pyhwp_table(self, table_record) -> Optional[pd.DataFrame]:
        """pyhwp 테이블 레코드를 DataFrame으로 변환"""
        try:
            rows = []
            
            # 행 정보 추출
            if hasattr(table_record, 'rows'):
                for row in table_record.rows:
                    cells = []
                    if hasattr(row, 'cells'):
                        for cell in row.cells:
                            # 셀 내용 추출
                            cell_text = self._extract_cell_text(cell)
                            cells.append(cell_text)
                    
                    if cells:
                        rows.append(cells)
            
            if rows:
                # 행 길이 맞추기
                max_cols = max(len(row) for row in rows) if rows else 0
                for row in rows:
                    while len(row) < max_cols:
                        row.append("")
                
                return pd.DataFrame(rows)
        
        except Exception as e:
            print(f"  pyhwp 테이블 파싱 오류: {e}")
            return None
    
    def _extract_cell_text(self, cell) -> str:
        """셀에서 텍스트 추출"""
        texts = []
        
        try:
            if hasattr(cell, 'paragraphs'):
                for para in cell.paragraphs:
                    if hasattr(para, 'chars'):
                        for char in para.chars:
                            if hasattr(char, 'text'):
                                texts.append(char.text)
            
            # 정규화
            text = " ".join(texts)
            text = self._normalize_text(text)
            return text
        
        except Exception:
            return ""
    
    def _extract_with_olefile_and_zlib(self, hwp_path: str) -> List[Dict]:
        """olefile과 zlib를 이용한 스토리지/스트림 읽기 및 레코드 구조 해석"""
        try:
            import olefile
            
            if not olefile.isOleFile(hwp_path):
                return []
            
            ole = olefile.OleFileIO(hwp_path)
            tables = []
            table_idx = 0
            
            # 스트림 목록 확인
            stream_list = ole.listdir()
            
            # 본문 스트림 찾기
            # HWP5 형식: 'BodyText/Section0', 'BodyText/Section1' 등
            # 구버전: 'BodyText', 'Section0' 등
            body_streams = []
            for stream_name in stream_list:
                stream_str = '/'.join(stream_name) if isinstance(stream_name, tuple) else str(stream_name)
                if 'BodyText' in stream_str or 'Section' in stream_str:
                    body_streams.append(stream_name)
            
            for stream_name in body_streams:
                try:
                    stream_data = ole.openstream(stream_name).read()
                    
                    # 방법 1: 레코드 구조 해석
                    extracted_tables = self._parse_hwp_records(stream_data, hwp_path, table_idx)
                    if extracted_tables:
                        tables.extend(extracted_tables)
                        table_idx += len(extracted_tables)
                        continue
                    
                    # 방법 2: zlib 압축 해제 시도
                    try:
                        decompressed = zlib.decompress(stream_data)
                        extracted_tables = self._parse_hwp_records(decompressed, hwp_path, table_idx)
                        if extracted_tables:
                            tables.extend(extracted_tables)
                            table_idx += len(extracted_tables)
                            continue
                    except:
                        pass
                    
                    # 방법 3: 바이너리 스트림 파싱
                    extracted_tables = self._parse_binary_stream(stream_data, hwp_path, table_idx)
                    tables.extend(extracted_tables)
                    table_idx += len(extracted_tables)
                    
                except Exception as e:
                    print(f"    스트림 처리 오류 ({stream_name}): {e}")
                    continue
            
            ole.close()
            return tables
        
        except Exception as e:
            print(f"  olefile+zlib 추출 오류: {e}")
            return []
    
    def _parse_hwp_records(self, data: bytes, source_path: str, start_idx: int) -> List[Dict]:
        """HWP 레코드 구조 해석
        
        HWP 파일은 레코드 기반 구조로 되어 있음
        각 레코드는 헤더(태그, 크기 등)와 본문으로 구성
        """
        tables = []
        
        try:
            pos = 0
            record_count = 0
            max_records = 10000  # 무한 루프 방지
            
            while pos < len(data) - 4 and record_count < max_records:
                # 레코드 헤더 읽기 시도
                # HWP 레코드 구조는 버전에 따라 다를 수 있음
                try:
                    # 간단한 레코드 헤더 파싱 (4바이트 태그 + 4바이트 크기)
                    if pos + 8 > len(data):
                        break
                    
                    tag = struct.unpack('<I', data[pos:pos+4])[0]
                    size = struct.unpack('<I', data[pos+4:pos+8])[0]
                    
                    # 유효성 검사
                    if size > len(data) or size < 0:
                        pos += 1
                        continue
                    
                    # 표 관련 태그인지 확인 (일반적인 HWP 태그 값)
                    # 실제 태그 값은 HWP 버전에 따라 다를 수 있음
                    record_data = data[pos+8:pos+8+size] if pos+8+size <= len(data) else data[pos+8:]
                    
                    # 표 데이터 추출 시도
                    table_data = self._extract_table_from_record(record_data, tag)
                    if table_data is not None and not table_data.empty:
                        tables.append({
                            'table_id': f"{Path(source_path).stem}_record_{start_idx + len(tables)}",
                            'dataframe': table_data,
                            'source_file': source_path,
                            'extraction_method': 'hwp_record_parsing'
                        })
                    
                    pos += 8 + size
                    record_count += 1
                    
                except Exception:
                    # 레코드 파싱 실패 시 다음 위치로
                    pos += 1
            
            # 레코드 파싱으로 표를 찾지 못한 경우, 텍스트 패턴 매칭 시도
            if not tables:
                extracted_tables = self._parse_binary_stream(data, source_path, start_idx)
                tables.extend(extracted_tables)
        
        except Exception as e:
            print(f"  레코드 파싱 오류: {e}")
        
        return tables
    
    def _extract_table_from_record(self, record_data: bytes, tag: int) -> Optional[pd.DataFrame]:
        """레코드 데이터에서 표 추출 시도"""
        try:
            # 텍스트로 디코딩 시도
            text = None
            for encoding in ['utf-16-le', 'utf-8', 'cp949']:
                try:
                    text = record_data.decode(encoding, errors='ignore')
                    break
                except:
                    continue
            
            if text:
                # 표 형태의 데이터 찾기
                lines = text.split('\n')
                rows = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 탭이나 여러 공백으로 구분된 데이터
                    if '\t' in line or len(line.split()) > 2:
                        cells = re.split(r'\t+|\s{2,}', line)
                        if len(cells) >= 2:
                            rows.append([self._normalize_text(cell) for cell in cells])
                
                if rows:
                    max_cols = max(len(row) for row in rows) if rows else 0
                    for row in rows:
                        while len(row) < max_cols:
                            row.append("")
                    
                    return pd.DataFrame(rows)
        
        except Exception:
            pass
        
        return None
    
    def _extract_with_olefile(self, hwp_path: str) -> List[Dict]:
        """olefile을 사용한 표 추출 (레거시, _extract_with_olefile_and_zlib 사용 권장)"""
        return self._extract_with_olefile_and_zlib(hwp_path)
    
    def _extract_with_binary_parsing(self, hwp_path: str) -> List[Dict]:
        """바이너리 직접 파싱 (간단한 버전)"""
        try:
            with open(hwp_path, 'rb') as f:
                data = f.read()
            
            tables = []
            table_idx = 0
            
            # HWP 파일 시그니처 확인
            if not data.startswith(b'HWP Document File'):
                # HWP5 형식일 수 있음 (OLE2)
                return []
            
            # 간단한 패턴 매칭으로 표 찾기 시도
            # 실제로는 더 정교한 파싱이 필요하지만, 기본 구조만 추출
            extracted_tables = self._parse_binary_stream(data, hwp_path, table_idx)
            tables.extend(extracted_tables)
            
            return tables
        
        except Exception as e:
            print(f"  바이너리 파싱 오류: {e}")
            return []
    
    def _parse_binary_stream(self, stream_data: bytes, source_path: str, start_idx: int) -> List[Dict]:
        """바이너리 스트림에서 표 정보 파싱"""
        tables = []
        
        # 간단한 텍스트 추출 시도
        # 실제로는 HWP 바이너리 구조를 정확히 파싱해야 함
        try:
            # UTF-16 또는 UTF-8로 디코딩 시도
            text = None
            for encoding in ['utf-16-le', 'utf-8', 'cp949']:
                try:
                    text = stream_data.decode(encoding, errors='ignore')
                    break
                except:
                    continue
            
            if text:
                # 표 형태의 데이터 찾기 (탭이나 공백으로 구분된 행)
                lines = text.split('\n')
                current_table_rows = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 탭이나 여러 공백으로 구분된 데이터가 있으면 표로 간주
                    if '\t' in line or len(line.split()) > 2:
                        cells = re.split(r'\t+|\s{2,}', line)
                        if len(cells) >= 2:  # 최소 2개 열
                            current_table_rows.append(cells)
                        elif current_table_rows:
                            # 표 종료
                            table_data = self._rows_to_dataframe(current_table_rows)
                            if table_data is not None and not table_data.empty:
                                tables.append({
                                    'table_id': f"{Path(source_path).stem}_binary_{start_idx + len(tables)}",
                                    'dataframe': table_data,
                                    'source_file': source_path,
                                    'extraction_method': 'hwp_direct_binary'
                                })
                            current_table_rows = []
                
                # 마지막 표 처리
                if current_table_rows:
                    table_data = self._rows_to_dataframe(current_table_rows)
                    if table_data is not None and not table_data.empty:
                        tables.append({
                            'table_id': f"{Path(source_path).stem}_binary_{start_idx + len(tables)}",
                            'dataframe': table_data,
                            'source_file': source_path,
                            'extraction_method': 'hwp_direct_binary'
                        })
        
        except Exception as e:
            print(f"  스트림 파싱 오류: {e}")
        
        return tables
    
    def _rows_to_dataframe(self, rows: List[List[str]]) -> Optional[pd.DataFrame]:
        """행 리스트를 DataFrame으로 변환"""
        if not rows:
            return None
        
        try:
            # 행 길이 맞추기
            max_cols = max(len(row) for row in rows) if rows else 0
            normalized_rows = []
            
            for row in rows:
                normalized_row = [self._normalize_text(cell) for cell in row]
                while len(normalized_row) < max_cols:
                    normalized_row.append("")
                normalized_rows.append(normalized_row)
            
            df = pd.DataFrame(normalized_rows)
            
            # 첫 행을 헤더로 사용할지 결정 (모든 셀이 비어있지 않으면)
            if len(df) > 1:
                first_row_all_filled = all(str(cell).strip() for cell in df.iloc[0])
                if first_row_all_filled:
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)
            
            return df
        
        except Exception:
            return None
    
    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화 (RAG 최적화)"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 특수 문자 정리 (일부는 보존)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        # 줄바꿈 정리
        text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
        
        return text.strip()
    
    def _convert_to_hwpx(self, hwp_path: str) -> Optional[str]:
        """
        HWP를 HWPX로 변환 시도
        
        한글과컴퓨터의 변환 도구가 필요합니다.
        """
        hwp_file = Path(hwp_path)
        hwpx_path = hwp_file.with_suffix('.hwpx')
        
        # 이미 HWPX 파일이 있는지 확인
        if hwpx_path.exists():
            return str(hwpx_path)
        
        # 변환 시도 (한글과컴퓨터 변환 도구 필요)
        # 예: hwp5conv 또는 다른 변환 도구
        try:
            # subprocess로 변환 도구 실행 시도
            # 실제 환경에 맞게 수정 필요
            # result = subprocess.run(['hwp5conv', hwp_path, str(hwpx_path)], 
            #                        capture_output=True, timeout=30)
            # if result.returncode == 0:
            #     return str(hwpx_path)
            pass
        except Exception:
            pass
        
        return None
    
    def get_table_count(self, hwp_path: str) -> int:
        """HWP 파일의 표 개수 반환"""
        tables = self.extract_tables(hwp_path)
        return len(tables)


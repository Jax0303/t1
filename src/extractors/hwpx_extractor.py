"""
HWPX 문서에서 표를 추출하는 모듈
HWPX는 XML 기반 구조화된 형식이므로 표 정보가 명시적으로 인코딩되어 있음
"""
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from bs4 import BeautifulSoup
import time


class HWPXTableExtractor:
    """HWPX 문서에서 표를 추출하는 클래스
    
    HWPX는 XML 기반 구조로 hp:tbl, hp:tr, hp:tc 태그를 직접 추출 가능
    RAG 시스템에서 바로 SQL/데이터프레임으로 변환하기 쉬운 구조
    """
    
    def __init__(self):
        # HWPX 네임스페이스 정의
        self.namespace = {
            'hp': 'http://www.hancom.co.kr/hwpml/2011/schema',
            'hwp': 'http://www.hancom.co.kr/hwpml/2011/schema',  # 호환성
            'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0'
        }
    
    def extract_tables(self, hwpx_path: str) -> List[Dict]:
        """
        HWPX 파일에서 모든 표를 추출
        
        Args:
            hwpx_path: HWPX 파일 경로
            
        Returns:
            표 정보 리스트 (각 표는 DataFrame과 메타데이터 포함)
        """
        start_time = time.time()
        tables = []
        
        try:
            # HWPX는 ZIP 파일 형식
            with zipfile.ZipFile(hwpx_path, 'r') as zip_ref:
                # Contents 폴더에서 본문 내용 찾기
                file_list = zip_ref.namelist()
                
                # 본문 XML 파일 찾기
                content_files = [f for f in file_list if 'Contents' in f and f.endswith('.xml')]
                
                # Section 파일들에서 표 찾기 (우선 처리)
                section_files = sorted([f for f in file_list if 'section' in f.lower() and f.endswith('.xml')])
                for section_idx, section_file in enumerate(section_files):
                    section_xml = zip_ref.read(section_file)
                    section_tables = self._extract_from_section(section_xml, hwpx_path, section_idx)
                    tables.extend(section_tables)
                
                # Contents 파일에서도 표 찾기 (추가)
                for content_file in content_files:
                    # header.xml은 건너뛰기 (일반적으로 표가 없음)
                    if 'header' in content_file.lower():
                        continue
                    
                    content_xml = zip_ref.read(content_file)
                    
                    # 방법 1: ElementTree를 사용한 네임스페이스 처리 (hp:tbl, hp:tr, hp:tc)
                    try:
                        root = ET.fromstring(content_xml)
                        # hp:tbl 태그 찾기
                        tbl_elements = root.findall('.//{http://www.hancom.co.kr/hwpml/2011/schema}tbl')
                        if not tbl_elements:
                            # 네임스페이스 없이 찾기
                            tbl_elements = root.findall('.//tbl')
                        
                        for idx, tbl_elem in enumerate(tbl_elements):
                            table_data = self._parse_hp_tbl_element(tbl_elem)
                            if table_data is not None and not table_data.empty:
                                # 빈 표나 단일 셀 표는 제외 (실제 표가 아닐 수 있음)
                                if len(table_data) > 1 or (len(table_data) == 1 and len(table_data.columns) > 1):
                                    tables.append({
                                        'table_id': f"{Path(hwpx_path).stem}_content_{idx}",
                                        'dataframe': table_data,
                                        'source_file': hwpx_path,
                                        'extraction_method': 'hwpx_hp_tbl'
                                    })
                    except Exception as e:
                        pass
                    
                    # 방법 2: BeautifulSoup을 사용한 대체 방법
                    soup = BeautifulSoup(content_xml, 'xml')
                    
                    # hp:tbl 태그 찾기 (네임스페이스 포함)
                    tbl_elements = soup.find_all(['tbl', 'hp:tbl'])
                    
                    for idx, tbl_elem in enumerate(tbl_elements):
                        table_data = self._parse_table_element(tbl_elem)
                        if table_data is not None and not table_data.empty:
                            # 빈 표나 단일 셀 표는 제외
                            if len(table_data) > 1 or (len(table_data) == 1 and len(table_data.columns) > 1):
                                # 중복 체크
                                table_id = f"{Path(hwpx_path).stem}_content_bs_{idx}"
                                if not any(t['table_id'] == table_id for t in tables):
                                    tables.append({
                                        'table_id': table_id,
                                        'dataframe': table_data,
                                        'source_file': hwpx_path,
                                        'extraction_method': 'hwpx_structure'
                                    })
        
        except Exception as e:
            print(f"HWPX 추출 오류 ({hwpx_path}): {e}")
            return []
        
        extraction_time = time.time() - start_time
        
        # 각 표에 처리 시간 추가
        for table in tables:
            table['extraction_time'] = extraction_time / len(tables) if tables else extraction_time
        
        return tables
    
    def _parse_hp_tbl_element(self, tbl_elem) -> Optional[pd.DataFrame]:
        """hp:tbl 요소를 DataFrame으로 변환 (HWPX 네임스페이스 처리)"""
        try:
            rows = []
            # hp:tr 태그 찾기
            tr_elements = tbl_elem.findall('.//{http://www.hancom.co.kr/hwpml/2011/schema}tr')
            if not tr_elements:
                tr_elements = tbl_elem.findall('.//tr')
            
            for tr_elem in tr_elements:
                cells = []
                # hp:tc 태그 찾기
                tc_elements = tr_elem.findall('.//{http://www.hancom.co.kr/hwpml/2011/schema}tc')
                if not tc_elements:
                    tc_elements = tr_elem.findall('.//tc')
                
                for tc_elem in tc_elements:
                    # hp:t 태그에서 텍스트 추출
                    text_content = self._extract_text_from_hp_tc(tc_elem)
                    cells.append(text_content)
                
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
            print(f"hp:tbl 파싱 오류: {e}")
            return None
    
    def _extract_text_from_hp_tc(self, tc_elem) -> str:
        """hp:tc 요소에서 텍스트 추출"""
        texts = []
        
        # hp:t 태그 찾기
        t_elements = tc_elem.findall('.//{http://www.hancom.co.kr/hwpml/2011/schema}t')
        if not t_elements:
            t_elements = tc_elem.findall('.//t')
        
        for t_elem in t_elements:
            if t_elem.text:
                texts.append(t_elem.text.strip())
        
        # 직접 텍스트 추출
        if not texts and tc_elem.text:
            texts.append(tc_elem.text.strip())
        
        # tail 텍스트도 포함
        if tc_elem.tail:
            texts.append(tc_elem.tail.strip())
        
        return " ".join(texts) if texts else ""
    
    def _parse_table_element(self, table_elem) -> Optional[pd.DataFrame]:
        """표 XML 요소를 DataFrame으로 변환 (BeautifulSoup용)"""
        try:
            rows = []
            # hp:tr 또는 tr 태그 찾기
            row_elements = table_elem.find_all(['tr', 'hp:tr', 'Row'])
            
            for row_elem in row_elements:
                cells = []
                # hp:tc 또는 tc 태그 찾기
                cell_elements = row_elem.find_all(['tc', 'hp:tc', 'Cell'])
                
                for cell_elem in cell_elements:
                    # 셀 내용 추출
                    text_content = self._extract_text_from_tc(cell_elem)
                    cells.append(text_content)
                
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
            print(f"표 파싱 오류: {e}")
            return None
    
    def _extract_text_from_element(self, elem) -> str:
        """XML 요소에서 텍스트 추출"""
        texts = []
        for text_node in elem.find_all(['Text', 't']):
            if text_node.string:
                texts.append(text_node.string.strip())
        return " ".join(texts) if texts else ""
    
    def _extract_from_section(self, section_xml: bytes, source_path: str, section_idx: int = 0) -> List[Dict]:
        """Section XML에서 표 추출"""
        tables = []
        try:
            # 방법 1: ElementTree로 hp:tbl 태그 직접 찾기
            try:
                root = ET.fromstring(section_xml)
                namespace = 'http://www.hancom.co.kr/hwpml/2011/schema'
                
                # hp:tbl 태그 찾기
                tbl_elements = root.findall(f'.//{{{namespace}}}tbl')
                
                for idx, tbl_elem in enumerate(tbl_elements):
                    table_data = self._parse_hp_tbl_element(tbl_elem)
                    if table_data is not None and not table_data.empty:
                        # 빈 표나 단일 셀 표는 제외 (실제 표가 아닐 수 있음)
                        if len(table_data) > 1 or (len(table_data) == 1 and len(table_data.columns) > 1):
                            tables.append({
                                'table_id': f"{Path(source_path).stem}_section{section_idx}_tbl_{idx}",
                                'dataframe': table_data,
                                'source_file': source_path,
                                'extraction_method': 'hwpx_section_tbl'
                            })
            except Exception as e:
                pass
            
            # 방법 2: BeautifulSoup으로 tbl 태그 찾기
            soup = BeautifulSoup(section_xml, 'xml')
            
            # tbl 태그 찾기 (hp:tbl 또는 tbl)
            tbl_elements = soup.find_all('tbl')
            
            for idx, tbl_elem in enumerate(tbl_elements):
                # ElementTree로 이미 처리한 표는 건너뛰기
                if idx < len([t for t in tables if 'tbl' in t.get('table_id', '')]):
                    continue
                
                table_data = self._parse_table_element(tbl_elem)
                if table_data is not None and not table_data.empty:
                    # 빈 표나 단일 셀 표는 제외
                    if len(table_data) > 1 or (len(table_data) == 1 and len(table_data.columns) > 1):
                        table_id = f"{Path(source_path).stem}_section{section_idx}_bs_tbl_{idx}"
                        if not any(t['table_id'] == table_id for t in tables):
                            tables.append({
                                'table_id': table_id,
                                'dataframe': table_data,
                                'source_file': source_path,
                                'extraction_method': 'hwpx_section_bs'
                            })
        
        except Exception as e:
            print(f"Section 추출 오류: {e}")
            import traceback
            traceback.print_exc()
        
        return tables
    
    def _parse_tr_tc_structure(self, trs) -> Optional[pd.DataFrame]:
        """tr, tc 구조를 DataFrame으로 변환"""
        try:
            rows = []
            for tr in trs:
                cells = []
                tcs = tr.find_all('tc')
                
                for tc in tcs:
                    # tc 내부의 텍스트 추출
                    text_content = self._extract_text_from_tc(tc)
                    cells.append(text_content)
                
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
            print(f"tr/tc 파싱 오류: {e}")
            return None
    
    def _extract_text_from_tc(self, tc_elem) -> str:
        """tc 요소에서 텍스트 추출"""
        texts = []
        
        # t 태그 찾기
        for t_elem in tc_elem.find_all('t'):
            if t_elem.string:
                texts.append(t_elem.string.strip())
            elif t_elem.get_text():
                texts.append(t_elem.get_text().strip())
        
        # 직접 텍스트 추출
        if not texts:
            direct_text = tc_elem.get_text()
            if direct_text:
                texts.append(direct_text.strip())
        
        return " ".join(texts) if texts else ""
    
    def get_table_count(self, hwpx_path: str) -> int:
        """HWPX 파일의 표 개수 반환"""
        tables = self.extract_tables(hwpx_path)
        return len(tables)


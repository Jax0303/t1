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
    """HWPX 문서에서 표를 추출하는 클래스"""
    
    def __init__(self):
        self.namespace = {
            'hwp': 'http://www.hancom.co.kr/hwpml/2011/schema',
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
                
                for content_file in content_files:
                    content_xml = zip_ref.read(content_file)
                    soup = BeautifulSoup(content_xml, 'xml')
                    
                    # 표 요소 찾기 (HWPX의 표는 <Table> 태그로 표현)
                    table_elements = soup.find_all('Table')
                    
                    for idx, table_elem in enumerate(table_elements):
                        table_data = self._parse_table_element(table_elem)
                        if table_data is not None:
                            tables.append({
                                'table_id': f"{Path(hwpx_path).stem}_{idx}",
                                'dataframe': table_data,
                                'source_file': hwpx_path,
                                'extraction_method': 'hwpx_structure'
                            })
                
                # 대체 방법: Section 파일들에서 표 찾기
                if not tables:
                    section_files = sorted([f for f in file_list if 'section' in f.lower() and f.endswith('.xml')])
                    for section_idx, section_file in enumerate(section_files):
                        section_xml = zip_ref.read(section_file)
                        section_tables = self._extract_from_section(section_xml, hwpx_path)
                        # table_id에 section 번호 추가
                        for table in section_tables:
                            table['table_id'] = f"{Path(hwpx_path).stem}_section{section_idx}_{table['table_id'].split('_')[-1]}"
                        tables.extend(section_tables)
        
        except Exception as e:
            print(f"HWPX 추출 오류 ({hwpx_path}): {e}")
            return []
        
        extraction_time = time.time() - start_time
        
        # 각 표에 처리 시간 추가
        for table in tables:
            table['extraction_time'] = extraction_time / len(tables) if tables else extraction_time
        
        return tables
    
    def _parse_table_element(self, table_elem) -> Optional[pd.DataFrame]:
        """표 XML 요소를 DataFrame으로 변환"""
        try:
            rows = []
            row_elements = table_elem.find_all('Row')
            
            for row_elem in row_elements:
                cells = []
                cell_elements = row_elem.find_all('Cell')
                
                for cell_elem in cell_elements:
                    # 셀 내용 추출
                    text_elem = cell_elem.find('Text')
                    if text_elem:
                        # 텍스트 내용 추출
                        text_content = self._extract_text_from_element(text_elem)
                        cells.append(text_content)
                    else:
                        cells.append("")
                
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
    
    def _extract_from_section(self, section_xml: bytes, source_path: str) -> List[Dict]:
        """Section XML에서 표 추출"""
        tables = []
        try:
            soup = BeautifulSoup(section_xml, 'xml')
            
            # 방법 1: Table 태그 찾기
            table_elements = soup.find_all('Table')
            
            for idx, table_elem in enumerate(table_elements):
                table_data = self._parse_table_element(table_elem)
                if table_data is not None:
                    tables.append({
                        'table_id': f"{Path(source_path).stem}_section_{idx}",
                        'dataframe': table_data,
                        'source_file': source_path,
                        'extraction_method': 'hwpx_section'
                    })
            
            # 방법 2: tr, tc 태그로 표 찾기 (HWPX 실제 구조)
            if not tables:
                trs = soup.find_all('tr')
                if trs:
                    # tr들을 그룹화하여 표로 만들기
                    table_data = self._parse_tr_tc_structure(trs)
                    if table_data is not None and not table_data.empty:
                        tables.append({
                            'table_id': f"{Path(source_path).stem}_section_tr",
                            'dataframe': table_data,
                            'source_file': source_path,
                            'extraction_method': 'hwpx_tr_tc'
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


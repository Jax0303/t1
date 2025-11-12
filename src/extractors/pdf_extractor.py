"""
PDF 문서에서 표를 추출하는 베이스라인 모듈
여러 방법론을 비교: pdfplumber, camelot, tabula
"""
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import time

# 선택적 import
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False


class PDFTableExtractor:
    """PDF 문서에서 표를 추출하는 클래스 (베이스라인)"""
    
    def __init__(self, method: str = "pdfplumber"):
        """
        Args:
            method: 추출 방법 ('pdfplumber', 'camelot', 'tabula')
        """
        self.method = method
    
    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """
        PDF 파일에서 모든 표를 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            표 정보 리스트
        """
        start_time = time.time()
        tables = []
        
        try:
            if self.method == "pdfplumber":
                tables = self._extract_with_pdfplumber(pdf_path, start_time)
            elif self.method == "camelot":
                tables = self._extract_with_camelot(pdf_path, start_time)
            elif self.method == "tabula":
                tables = self._extract_with_tabula(pdf_path, start_time)
            else:
                raise ValueError(f"지원하지 않는 방법: {self.method}")
        
        except Exception as e:
            print(f"PDF 추출 오류 ({pdf_path}, {self.method}): {e}")
            return []
        
        return tables
    
    def _extract_with_pdfplumber(self, pdf_path: str, start_time: float) -> List[Dict]:
        """pdfplumber를 사용한 표 추출"""
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                
                for idx, table in enumerate(page_tables):
                    if table:
                        try:
                            # 첫 번째 행을 헤더로 사용
                            df = pd.DataFrame(table[1:], columns=table[0] if table else None)
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
                            print(f"pdfplumber 표 처리 오류: {e}")
        
        return tables
    
    def _extract_with_camelot(self, pdf_path: str, start_time: float) -> List[Dict]:
        """camelot을 사용한 표 추출"""
        if not CAMELOT_AVAILABLE:
            print(f"  경고: camelot이 설치되지 않았습니다. pdfplumber를 사용하세요.")
            return []
        
        tables = []
        
        try:
            # lattice 방법 시도
            camelot_tables = camelot.read_pdf(pdf_path, flavor='lattice', pages='all')
            
            for idx, table in enumerate(camelot_tables):
                try:
                    df = table.df
                    extraction_time = time.time() - start_time
                    
                    tables.append({
                        'table_id': f"{Path(pdf_path).stem}_camelot_table{idx}",
                        'dataframe': df,
                        'source_file': pdf_path,
                        'page_number': table.page,
                        'extraction_method': 'camelot_lattice',
                        'extraction_time': extraction_time,
                        'accuracy': table.accuracy
                    })
                except Exception as e:
                    print(f"camelot 표 처리 오류: {e}")
        
        except Exception as e:
            print(f"camelot 추출 실패, stream 방법 시도: {e}")
            try:
                camelot_tables = camelot.read_pdf(pdf_path, flavor='stream', pages='all')
                for idx, table in enumerate(camelot_tables):
                    df = table.df
                    extraction_time = time.time() - start_time
                    tables.append({
                        'table_id': f"{Path(pdf_path).stem}_camelot_stream_table{idx}",
                        'dataframe': df,
                        'source_file': pdf_path,
                        'page_number': table.page,
                        'extraction_method': 'camelot_stream',
                        'extraction_time': extraction_time,
                        'accuracy': table.accuracy
                    })
            except Exception as e2:
                print(f"camelot stream도 실패: {e2}")
        
        return tables
    
    def _extract_with_tabula(self, pdf_path: str, start_time: float) -> List[Dict]:
        """tabula를 사용한 표 추출"""
        if not TABULA_AVAILABLE:
            print(f"  경고: tabula가 설치되지 않았습니다. pdfplumber를 사용하세요.")
            return []
        
        tables = []
        
        try:
            dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            for idx, df in enumerate(dfs):
                if df is not None and not df.empty:
                    extraction_time = time.time() - start_time
                    tables.append({
                        'table_id': f"{Path(pdf_path).stem}_tabula_table{idx}",
                        'dataframe': df,
                        'source_file': pdf_path,
                        'extraction_method': 'tabula',
                        'extraction_time': extraction_time
                    })
        
        except Exception as e:
            print(f"tabula 추출 오류: {e}")
        
        return tables
    
    def get_table_count(self, pdf_path: str) -> int:
        """PDF 파일의 표 개수 반환"""
        tables = self.extract_tables(pdf_path)
        return len(tables)


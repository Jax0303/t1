"""
PDF 문서에서 표를 추출하는 베이스라인 모듈
여러 방법론을 비교: pdfplumber, camelot, tabula, PyMuPDF(Fitz)
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

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

try:
    from marker.convert import convert_single_pdf
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False

try:
    import warnings
    warnings.filterwarnings('ignore')
    from nougat.model import NougatModel, NougatConfig
    from nougat.utils.dataset import ImageDataset
    from nougat.utils.checkpoint import get_checkpoint
    import torch
    NOUGAT_AVAILABLE = True
except ImportError:
    NOUGAT_AVAILABLE = False
except Exception:
    NOUGAT_AVAILABLE = False


class PDFTableExtractor:
    """PDF 문서에서 표를 추출하는 클래스 (베이스라인)"""
    
    def __init__(self, method: str = "pdfplumber"):
        """
        Args:
            method: 추출 방법 ('pdfplumber', 'camelot', 'tabula', 'pymupdf', 
                    'pypdf', 'docling', 'unstructured', 'marker', 'nougat')
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
            elif self.method == "pymupdf":
                tables = self._extract_with_pymupdf(pdf_path, start_time)
            elif self.method == "pypdf":
                tables = self._extract_with_pypdf(pdf_path, start_time)
            elif self.method == "docling":
                tables = self._extract_with_docling(pdf_path, start_time)
            elif self.method == "unstructured":
                tables = self._extract_with_unstructured(pdf_path, start_time)
            elif self.method == "marker":
                tables = self._extract_with_marker(pdf_path, start_time)
            elif self.method == "nougat":
                tables = self._extract_with_nougat(pdf_path, start_time)
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
                    if table and len(table) > 0:
                        try:
                            df = self._safe_dataframe_conversion(table)
                            
                            # 빈 DataFrame은 제외 (파싱 툴 문제는 그대로 둠)
                            if df is None or df.empty:
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
                            print(f"pdfplumber 표 처리 오류 (페이지 {page_num+1}, 표 {idx}): {e}")
        
        return tables
    
    def _safe_dataframe_conversion(self, table: List[List]) -> Optional[pd.DataFrame]:
        """
        안전한 DataFrame 변환
        1행 표(헤더만) 처리 및 예외 처리 강화
        """
        if not table or len(table) == 0:
            return None
        
        try:
            # 1행만 있는 경우: 헤더를 데이터로 사용
            if len(table) == 1:
                headers = table[0]
                # 모든 셀이 None이면 제외 (파싱 툴 문제)
                if all(cell is None or cell == '' for cell in headers):
                    return None
                # 헤더를 데이터로 사용하여 단일 행 DataFrame 생성
                df = pd.DataFrame([headers], columns=[f'col_{i}' for i in range(len(headers))])
                return df
            
            # 2행 이상: 첫 행을 헤더로 사용
            headers = table[0]
            data_rows = table[1:]
            
            # 헤더가 모두 None이면 자동 생성
            if all(h is None or h == '' for h in headers):
                headers = [f'col_{i}' for i in range(len(headers))]
            
            # 데이터 행이 모두 None이면 제외 (파싱 툴 문제)
            if all(all(cell is None or cell == '' for cell in row) for row in data_rows):
                return None
            
            df = pd.DataFrame(data_rows, columns=headers)
            
            # 빈 DataFrame 체크
            if df.empty:
                return None
            
            return df
            
        except Exception as e:
            print(f"DataFrame 변환 오류: {e}")
            return None
    
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
    
    def _extract_with_pymupdf(self, pdf_path: str, start_time: float) -> List[Dict]:
        """PyMuPDF(Fitz)를 사용한 표 추출"""
        if not PYMUPDF_AVAILABLE:
            print(f"  경고: PyMuPDF가 설치되지 않았습니다. pip install pymupdf를 실행하세요.")
            return []
        
        tables = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # PyMuPDF는 표 추출을 직접 지원하지 않으므로
                # 텍스트 블록과 이미지를 기반으로 표를 찾음
                # 또는 find_tables() 메서드 사용 (최신 버전)
                try:
                    # PyMuPDF 1.23.0+ 버전에서는 find_tables() 지원
                    if hasattr(page, 'find_tables'):
                        page_tables = page.find_tables()
                        
                        for idx, table_info in enumerate(page_tables):
                            try:
                                # 표 데이터 추출
                                table_data = table_info.extract()
                                
                                if table_data and len(table_data) > 0:
                                    # 안전한 DataFrame 변환 사용
                                    df = self._safe_dataframe_conversion(table_data)
                                    
                                    if df is not None and not df.empty:
                                        extraction_time = time.time() - start_time
                                        
                                        tables.append({
                                            'table_id': f"{Path(pdf_path).stem}_pymupdf_page{page_num}_table{idx}",
                                            'dataframe': df,
                                            'source_file': pdf_path,
                                            'page_number': page_num + 1,
                                            'extraction_method': 'pymupdf',
                                            'extraction_time': extraction_time
                                        })
                            except Exception as e:
                                print(f"PyMuPDF 표 처리 오류 (페이지 {page_num}, 표 {idx}): {e}")
                    else:
                        # find_tables()가 없는 경우 텍스트 블록으로 시도
                        # 이 경우 표 구조 파악이 어려움
                        text_blocks = page.get_text("blocks")
                        # 간단한 텍스트 추출만 수행
                        extraction_time = time.time() - start_time
                        print(f"  경고: PyMuPDF find_tables() 미지원. 텍스트만 추출 가능.")
                        
                except Exception as e:
                    print(f"PyMuPDF 페이지 {page_num} 처리 오류: {e}")
            
            doc.close()
        
        except Exception as e:
            print(f"PyMuPDF 추출 오류: {e}")
        
        return tables
    
    def _extract_with_pypdf(self, pdf_path: str, start_time: float) -> List[Dict]:
        """pypdf를 사용한 텍스트 추출 (표 추출 제한적)"""
        if not PYPDF_AVAILABLE:
            print(f"  경고: pypdf가 설치되지 않았습니다. pip install pypdf를 실행하세요.")
            return []
        
        tables = []
        
        try:
            reader = PdfReader(pdf_path)
            
            for page_num, page in enumerate(reader.pages):
                # pypdf는 표 추출을 직접 지원하지 않으므로 텍스트만 추출
                text = page.extract_text()
                
                # 간단한 표 감지 시도 (탭이나 공백으로 구분된 텍스트)
                lines = text.split('\n')
                potential_table_rows = []
                
                for line in lines:
                    # 탭이나 여러 공백으로 구분된 줄을 표로 간주
                    if '\t' in line or len(line.split()) > 3:
                        cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                        if len(cells) > 1:
                            potential_table_rows.append(cells)
                
                # 연속된 표 행이 있으면 표로 간주
                if len(potential_table_rows) > 1:
                    try:
                        # 최대 열 수 찾기
                        max_cols = max(len(row) for row in potential_table_rows)
                        # 모든 행을 같은 열 수로 맞춤
                        normalized_rows = []
                        for row in potential_table_rows:
                            while len(row) < max_cols:
                                row.append('')
                            normalized_rows.append(row[:max_cols])
                        
                        if normalized_rows:
                            df = pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0] if normalized_rows else None)
                            extraction_time = time.time() - start_time
                            
                            tables.append({
                                'table_id': f"{Path(pdf_path).stem}_pypdf_page{page_num}_table0",
                                'dataframe': df,
                                'source_file': pdf_path,
                                'page_number': page_num + 1,
                                'extraction_method': 'pypdf',
                                'extraction_time': extraction_time,
                                'note': 'pypdf는 표 구조 추출이 제한적입니다'
                            })
                    except Exception as e:
                        print(f"pypdf 표 처리 오류 (페이지 {page_num}): {e}")
        
        except Exception as e:
            print(f"pypdf 추출 오류: {e}")
        
        return tables
    
    def _extract_with_docling(self, pdf_path: str, start_time: float) -> List[Dict]:
        """Docling을 사용한 문서 변환 및 표 추출"""
        if not DOCLING_AVAILABLE:
            print(f"  경고: docling이 설치되지 않았습니다. pip install docling-core를 실행하세요.")
            return []
        
        tables = []
        
        try:
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            
            # Docling 결과에서 표 추출
            # Docling은 구조화된 문서 객체를 반환
            doc = result.document
            
            # 표 찾기 (Docling의 구조에 따라 조정 필요)
            table_idx = 0
            for page_num, page in enumerate(doc.pages if hasattr(doc, 'pages') else [doc]):
                # 페이지의 표 요소 찾기
                if hasattr(page, 'tables'):
                    for table in page.tables:
                        try:
                            # 표를 DataFrame으로 변환
                            rows = []
                            if hasattr(table, 'rows'):
                                for row in table.rows:
                                    cells = []
                                    if hasattr(row, 'cells'):
                                        for cell in row.cells:
                                            cell_text = cell.text if hasattr(cell, 'text') else str(cell)
                                            cells.append(cell_text)
                                    rows.append(cells)
                            
                            if rows:
                                # 첫 번째 행을 헤더로 사용
                                df = pd.DataFrame(rows[1:], columns=rows[0] if rows else None)
                                extraction_time = time.time() - start_time
                                
                                tables.append({
                                    'table_id': f"{Path(pdf_path).stem}_docling_page{page_num}_table{table_idx}",
                                    'dataframe': df,
                                    'source_file': pdf_path,
                                    'page_number': page_num + 1,
                                    'extraction_method': 'docling',
                                    'extraction_time': extraction_time
                                })
                                table_idx += 1
                        except Exception as e:
                            print(f"Docling 표 처리 오류 (페이지 {page_num}): {e}")
                
                # 대안: 전체 문서에서 표 찾기
                if hasattr(doc, 'tables') and not tables:
                    for table in doc.tables:
                        try:
                            rows = []
                            if hasattr(table, 'rows'):
                                for row in table.rows:
                                    cells = []
                                    if hasattr(row, 'cells'):
                                        for cell in row.cells:
                                            cell_text = cell.text if hasattr(cell, 'text') else str(cell)
                                            cells.append(cell_text)
                                    rows.append(cells)
                            
                            if rows:
                                df = pd.DataFrame(rows[1:], columns=rows[0] if rows else None)
                                extraction_time = time.time() - start_time
                                
                                tables.append({
                                    'table_id': f"{Path(pdf_path).stem}_docling_table{table_idx}",
                                    'dataframe': df,
                                    'source_file': pdf_path,
                                    'extraction_method': 'docling',
                                    'extraction_time': extraction_time
                                })
                                table_idx += 1
                        except Exception as e:
                            print(f"Docling 표 처리 오류: {e}")
        
        except Exception as e:
            print(f"Docling 추출 오류: {e}")
        
        return tables
    
    def _extract_with_unstructured(self, pdf_path: str, start_time: float) -> List[Dict]:
        """unstructured를 사용한 문서 파싱 및 표 추출"""
        if not UNSTRUCTURED_AVAILABLE:
            print(f"  경고: unstructured가 설치되지 않았습니다. pip install unstructured[pdf]를 실행하세요.")
            return []
        
        tables = []
        
        try:
            # unstructured로 PDF 파싱
            # poppler가 없으면 "fast" 전략 사용
            try:
                elements = partition_pdf(
                    filename=pdf_path,
                    strategy="hi_res",  # 또는 "fast", "ocr_only"
                    infer_table_structure=True,
                    include_page_breaks=True
                )
            except Exception as e:
                if "poppler" in str(e).lower():
                    # poppler가 없으면 fast 전략 사용
                    elements = partition_pdf(
                        filename=pdf_path,
                        strategy="fast",
                        infer_table_structure=True,
                        include_page_breaks=True
                    )
                else:
                    raise
            
            # 표 요소 찾기
            table_idx = 0
            current_page = 1
            
            for element in elements:
                # 페이지 브레이크 확인
                if hasattr(element, 'metadata'):
                    md = element.metadata
                    if hasattr(md, 'page_number') and md.page_number:
                        current_page = md.page_number
                
                # 표 요소 확인
                if hasattr(element, 'category') and element.category == 'Table':
                    try:
                        # unstructured의 표는 HTML 또는 텍스트 형태로 제공될 수 있음
                        # metadata에서 HTML 확인
                        html_table = None
                        if hasattr(element, 'metadata'):
                            md = element.metadata
                            if hasattr(md, 'text_as_html') and md.text_as_html:
                                html_table = md.text_as_html
                        
                        if html_table:
                            # HTML 표를 DataFrame으로 변환
                            try:
                                dfs = pd.read_html(html_table)
                                if dfs:
                                    df = dfs[0]
                                    extraction_time = time.time() - start_time
                                    
                                    tables.append({
                                        'table_id': f"{Path(pdf_path).stem}_unstructured_page{current_page}_table{table_idx}",
                                        'dataframe': df,
                                        'source_file': pdf_path,
                                        'page_number': current_page,
                                        'extraction_method': 'unstructured',
                                        'extraction_time': extraction_time
                                    })
                                    table_idx += 1
                                    continue
                            except Exception as e:
                                print(f"unstructured HTML 표 변환 오류: {e}")
                        
                        # 텍스트 형태의 표를 파싱 시도
                        if hasattr(element, 'text') and element.text:
                            text = element.text
                            lines = text.split('\n')
                            rows = []
                            for line in lines:
                                if line.strip():
                                    # 탭이나 공백으로 구분
                                    cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                                    if not cells:
                                        cells = [cell.strip() for cell in line.split() if cell.strip()]
                                    if cells and len(cells) > 1:  # 최소 2개 열 필요
                                        rows.append(cells)
                            
                            if len(rows) > 1:
                                # 최대 열 수 맞추기
                                max_cols = max(len(row) for row in rows)
                                normalized_rows = []
                                for row in rows:
                                    while len(row) < max_cols:
                                        row.append('')
                                    normalized_rows.append(row[:max_cols])
                                
                                if normalized_rows:
                                    df = pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0] if normalized_rows else None)
                                    extraction_time = time.time() - start_time
                                    
                                    tables.append({
                                        'table_id': f"{Path(pdf_path).stem}_unstructured_page{current_page}_table{table_idx}",
                                        'dataframe': df,
                                        'source_file': pdf_path,
                                        'page_number': current_page,
                                        'extraction_method': 'unstructured',
                                        'extraction_time': extraction_time
                                    })
                                    table_idx += 1
                    except Exception as e:
                        print(f"unstructured 표 처리 오류 (페이지 {current_page}): {e}")
        
        except Exception as e:
            print(f"unstructured 추출 오류: {e}")
        
        return tables
    
    def _extract_with_marker(self, pdf_path: str, start_time: float) -> List[Dict]:
        """Marker를 사용한 PDF 변환 및 표 추출"""
        if not MARKER_AVAILABLE:
            print(f"  경고: marker가 설치되지 않았습니다. pip install marker-pdf를 실행하세요.")
            return []
        
        tables = []
        
        try:
            # Marker로 PDF를 Markdown으로 변환
            # Marker는 로컬에서 실행되며 GPU가 있으면 사용
            output_path = Path(pdf_path).with_suffix('.marker.md')
            
            # Marker 변환 실행
            result = convert_single_pdf(
                pdf_path,
                output_path,
                batch_multiplier=1,
                max_pages=None,
                langs=None  # 언어 자동 감지
            )
            
            # 변환된 Markdown에서 표 추출
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # Markdown 표 파싱 (|로 구분된 표)
                lines = markdown_content.split('\n')
                current_table = []
                table_idx = 0
                
                for line in lines:
                    line = line.strip()
                    # Markdown 표 형식 감지 (|로 시작하고 끝남)
                    if line.startswith('|') and line.endswith('|'):
                        # 헤더 구분선 건너뛰기
                        if '---' in line or '===' in line:
                            continue
                        # 표 행 파싱
                        cells = [cell.strip() for cell in line.split('|')[1:-1]]
                        if cells:
                            current_table.append(cells)
                    elif current_table:
                        # 표 끝 - DataFrame 생성
                        if len(current_table) > 1:
                            try:
                                # 첫 번째 행을 헤더로 사용
                                df = pd.DataFrame(current_table[1:], columns=current_table[0] if current_table else None)
                                extraction_time = time.time() - start_time
                                
                                tables.append({
                                    'table_id': f"{Path(pdf_path).stem}_marker_table{table_idx}",
                                    'dataframe': df,
                                    'source_file': pdf_path,
                                    'extraction_method': 'marker',
                                    'extraction_time': extraction_time
                                })
                                table_idx += 1
                            except Exception as e:
                                print(f"Marker 표 처리 오류: {e}")
                        current_table = []
                
                # 마지막 표 처리
                if current_table and len(current_table) > 1:
                    try:
                        df = pd.DataFrame(current_table[1:], columns=current_table[0] if current_table else None)
                        extraction_time = time.time() - start_time
                        
                        tables.append({
                            'table_id': f"{Path(pdf_path).stem}_marker_table{table_idx}",
                            'dataframe': df,
                            'source_file': pdf_path,
                            'extraction_method': 'marker',
                            'extraction_time': extraction_time
                        })
                    except Exception as e:
                        print(f"Marker 마지막 표 처리 오류: {e}")
                
                # 임시 파일 삭제 (선택적)
                # output_path.unlink()
        
        except Exception as e:
            print(f"Marker 추출 오류: {e}")
        
        return tables
    
    def _extract_with_nougat(self, pdf_path: str, start_time: float) -> List[Dict]:
        """Nougat를 사용한 PDF 변환 및 표 추출"""
        if not NOUGAT_AVAILABLE:
            print(f"  경고: nougat가 설치되지 않았습니다. pip install nougat-ocr를 실행하세요.")
            return []
        
        tables = []
        
        try:
            import warnings
            warnings.filterwarnings('ignore')
            from PIL import Image
            import fitz  # PyMuPDF for PDF to image conversion
            import io
            import re
            
            # GPU 사용 가능 여부 확인
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Nougat 모델 로드
            checkpoint = get_checkpoint("nougat", model_tag="0.1.0-small")
            model = NougatModel.from_pretrained(checkpoint)
            model.to(device)
            model.eval()
            
            # PDF를 이미지로 변환
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # 페이지를 이미지로 렌더링
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x 해상도
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Nougat로 텍스트 추출
                with torch.no_grad():
                    # 이미지를 텍스트로 변환
                    # Nougat는 Markdown 형식으로 출력
                    try:
                        output = model.inference(image=img, early_stopping=True)
                        # output은 토큰 ID일 수 있으므로 디코딩 필요
                        if hasattr(model, 'tokenizer'):
                            markdown_text = model.tokenizer.decode(output, skip_special_tokens=True)
                        else:
                            markdown_text = str(output)
                    except Exception as e:
                        print(f"Nougat inference 오류 (페이지 {page_num}): {e}")
                        continue
                    
                    # Markdown 표 파싱 (|로 구분된 표)
                    lines = markdown_text.split('\n')
                    current_table = []
                    table_idx = 0
                    
                    for line in lines:
                        line = line.strip()
                        # Markdown 표 형식 감지
                        if line.startswith('|') and line.endswith('|'):
                            # 헤더 구분선 건너뛰기
                            if '---' in line or '===' in line:
                                continue
                            # 표 행 파싱
                            cells = [cell.strip() for cell in line.split('|')[1:-1]]
                            if cells:
                                current_table.append(cells)
                        elif current_table:
                            # 표 끝 - DataFrame 생성
                            if len(current_table) > 1:
                                try:
                                    # 첫 번째 행을 헤더로 사용
                                    df = pd.DataFrame(current_table[1:], columns=current_table[0] if current_table else None)
                                    extraction_time = time.time() - start_time
                                    
                                    tables.append({
                                        'table_id': f"{Path(pdf_path).stem}_nougat_page{page_num}_table{table_idx}",
                                        'dataframe': df,
                                        'source_file': pdf_path,
                                        'page_number': page_num + 1,
                                        'extraction_method': 'nougat',
                                        'extraction_time': extraction_time
                                    })
                                    table_idx += 1
                                except Exception as e:
                                    print(f"Nougat 표 처리 오류 (페이지 {page_num}): {e}")
                            current_table = []
                    
                    # 마지막 표 처리
                    if current_table and len(current_table) > 1:
                        try:
                            df = pd.DataFrame(current_table[1:], columns=current_table[0] if current_table else None)
                            extraction_time = time.time() - start_time
                            
                            tables.append({
                                'table_id': f"{Path(pdf_path).stem}_nougat_page{page_num}_table{table_idx}",
                                'dataframe': df,
                                'source_file': pdf_path,
                                'page_number': page_num + 1,
                                'extraction_method': 'nougat',
                                'extraction_time': extraction_time
                            })
                        except Exception as e:
                            print(f"Nougat 마지막 표 처리 오류: {e}")
            
            doc.close()
        
        except Exception as e:
            print(f"Nougat 추출 오류: {e}")
            import traceback
            traceback.print_exc()
        
        return tables
    
    def get_table_count(self, pdf_path: str) -> int:
        """PDF 파일의 표 개수 반환"""
        tables = self.extract_tables(pdf_path)
        return len(tables)


"""
HWP 파일에서 표를 추출하는 모듈
HWP는 바이너리 형식이므로 HWPX로 변환하거나 pyhwp 같은 라이브러리 필요
"""
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import time

# HWPX 추출기 재사용
from .hwpx_extractor import HWPXTableExtractor


class HWPTableExtractor:
    """HWP 문서에서 표를 추출하는 클래스"""
    
    def __init__(self):
        self.hwpx_extractor = HWPXTableExtractor()
    
    def extract_tables(self, hwp_path: str) -> List[Dict]:
        """
        HWP 파일에서 표 추출
        
        현재는 HWPX로 변환 후 추출 시도
        또는 직접 파싱 (구현 필요)
        
        Args:
            hwp_path: HWP 파일 경로
            
        Returns:
            표 정보 리스트
        """
        start_time = time.time()
        tables = []
        
        try:
            # 방법 1: HWPX로 변환 시도 (한글과컴퓨터 변환 도구 필요)
            hwpx_path = self._convert_to_hwpx(hwp_path)
            
            if hwpx_path and os.path.exists(hwpx_path):
                # HWPX 추출기 사용
                tables = self.hwpx_extractor.extract_tables(hwpx_path)
                # 메타데이터 업데이트
                for table in tables:
                    table['source_file'] = hwp_path
                    table['extraction_method'] = 'hwp_via_hwpx'
                    table['original_format'] = 'hwp'
            
            # 방법 2: 직접 파싱 (pyhwp 등 사용 - 구현 필요)
            if not tables:
                print(f"  경고: HWP 파일 직접 파싱은 아직 구현되지 않았습니다.")
                print(f"  HWPX로 변환 후 처리하거나 pyhwp 라이브러리 설치가 필요합니다.")
        
        except Exception as e:
            print(f"HWP 추출 오류 ({hwp_path}): {e}")
            import traceback
            traceback.print_exc()
        
        extraction_time = time.time() - start_time
        
        # 각 표에 처리 시간 추가
        for table in tables:
            table['extraction_time'] = extraction_time / len(tables) if tables else extraction_time
        
        return tables
    
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


"""
DART(금융감독원 전자공시) 공시 문서 다운로더
OpenDartReader를 사용하여 재무·공시 PDF 파일을 다운로드
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
import requests

try:
    import OpenDartReader
except ImportError:
    print("OpenDartReader가 설치되지 않았습니다. 설치 중...")
    print("pip install OpenDartReader")
    sys.exit(1)


class DARTDownloader:
    """DART 공시 문서 다운로더"""
    
    def __init__(self, api_key: str, download_dir: str = "data/dart_pdfs"):
        """
        Args:
            api_key: DART API 키 (https://opendart.fss.or.kr 에서 발급)
            download_dir: 다운로드 디렉토리
        """
        self.dart = OpenDartReader(api_key)
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # 다운로드 통계
        self.stats = {
            'total_downloaded': 0,
            'total_failed': 0,
            'companies': {}
        }
    
    def get_company_list(self, sectors: Optional[List[str]] = None, limit: int = 20) -> List[Dict]:
        """
        회사 목록 조회
        
        Args:
            sectors: 업종 필터 (예: ['정보통신업', '제조업'])
            limit: 최대 회사 수
            
        Returns:
            회사 정보 리스트
        """
        print(f"\n회사 목록 조회 중...")
        
        # 상장사 목록 조회
        try:
            # 최근 3개월 기간으로 제한 (API 제약)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            
            # 공시 목록 조회 (corp_code 없이 조회하면 최근 공시만)
            recent_reports = self.dart.list(start=start_str, end=end_str, kind='A')
            
            if recent_reports is None or len(recent_reports) == 0:
                raise ValueError("공시 목록 조회 실패")
            
            # 회사 코드 추출 (중복 제거)
            unique_companies = {}
            for idx, row in recent_reports.iterrows():
                corp_code = row.get('corp_code', '')
                corp_name = row.get('corp_name', '')
                
                if corp_code and corp_code not in unique_companies:
                    unique_companies[corp_code] = {
                        'corp_code': corp_code,
                        'corp_name': corp_name,
                        'stock_code': row.get('stock_code', ''),
                        'sector': row.get('업종', '')
                    }
            
            company_list = list(unique_companies.values())[:limit]
            
            print(f"  {len(company_list)}개 회사 선택됨")
            return company_list
            
        except Exception as e:
            print(f"  오류: 회사 목록 조회 실패: {e}")
            # 샘플 회사 리스트 반환 (주요 IT/제조업체 - 실제 corp_code 사용)
            return [
                {'corp_code': '00126380', 'corp_name': '삼성전자', 'stock_code': '005930', 'sector': '반도체'},
                {'corp_code': '00164742', 'corp_name': 'SK하이닉스', 'stock_code': '000660', 'sector': '반도체'},
                {'corp_code': '00164725', 'corp_name': '네이버', 'stock_code': '035420', 'sector': '정보통신업'},
                {'corp_code': '00164730', 'corp_name': '카카오', 'stock_code': '035720', 'sector': '정보통신업'},
                {'corp_code': '00164742', 'corp_name': 'LG전자', 'stock_code': '066570', 'sector': '제조업'},
            ]
    
    def get_reports(self, corp_code: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, report_type: str = 'all') -> List[Dict]:
        """
        특정 회사의 공시 문서 목록 조회
        
        Args:
            corp_code: 회사 코드
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            report_type: 보고서 타입 ('all', '사업보고서', '반기보고서', '분기보고서')
            
        Returns:
            공시 문서 리스트
        """
        if not start_date:
            # 기본: 최근 3년
            end = datetime.now()
            start = end - timedelta(days=3*365)
            start_date = start.strftime('%Y%m%d')
            end_date = end.strftime('%Y%m%d')
        
        try:
            # 공시 목록 조회
            reports = self.dart.list(corp_code, start=start_date, end=end_date, kind='A')
            
            if reports is None or len(reports) == 0:
                return []
            
            # 보고서 타입 필터링
            if report_type != 'all':
                reports = reports[reports['report_nm'].str.contains(report_type, na=False)]
            
            # 사업보고서, 반기보고서, 분기보고서만 선택
            target_reports = ['사업보고서', '반기보고서', '분기보고서']
            filtered = reports[reports['report_nm'].str.contains('|'.join(target_reports), na=False)]
            
            report_list = []
            for idx, row in filtered.iterrows():
                report_list.append({
                    'rcept_no': row['rcept_no'],
                    'corp_name': row['corp_name'],
                    'report_nm': row['report_nm'],
                    'rcept_dt': row['rcept_dt'],
                    'corp_code': corp_code
                })
            
            return report_list
            
        except Exception as e:
            print(f"    경고: 공시 목록 조회 실패 ({corp_code}): {e}")
            return []
    
    def _is_valid_pdf(self, filepath: Path) -> bool:
        """
        파일이 유효한 PDF인지 확인
        
        Args:
            filepath: 파일 경로
            
        Returns:
            PDF 여부
        """
        if not filepath.exists() or filepath.stat().st_size == 0:
            return False
        
        try:
            # PDF 파일 시그니처 확인 (%PDF)
            with open(filepath, 'rb') as f:
                header = f.read(4)
                if header == b'%PDF':
                    return True
            
            # HTML 파일인지 확인
            with open(filepath, 'rb') as f:
                content = f.read(1024)
                if b'<!DOCTYPE' in content or b'<html' in content.lower():
                    return False
            
            return False
        except Exception:
            return False
    
    def download_report_pdf(self, rcept_no: str, corp_name: str, report_nm: str) -> Optional[Path]:
        """
        공시 문서의 PDF 파일 다운로드
        
        Args:
            rcept_no: 접수번호
            corp_name: 회사명
            report_nm: 보고서명
            
        Returns:
            다운로드된 파일 경로
        """
        try:
            # 파일명 정리
            safe_corp_name = corp_name.replace('/', '_').replace('\\', '_')
            safe_report_nm = report_nm.replace('/', '_').replace('\\', '_')
            filename = f"{safe_corp_name}_{safe_report_nm}_{rcept_no}.pdf"
            filepath = self.download_dir / filename
            
            # 이미 다운로드된 파일은 건너뛰기 (유효한 PDF인지 확인)
            if filepath.exists():
                if self._is_valid_pdf(filepath):
                    return filepath
                else:
                    # 유효하지 않은 파일 삭제
                    filepath.unlink()
            
            # 방법 1: DART PDF 직접 다운로드 URL 사용 (가장 확실한 방법)
            try:
                # DART PDF 다운로드 URL 패턴: https://dart.fss.or.kr/pdf/download/main.do?rcpNo=접수번호
                pdf_url = f"https://dart.fss.or.kr/pdf/download/main.do?rcpNo={rcept_no}"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(pdf_url, headers=headers, timeout=30, stream=True, allow_redirects=True)
                
                if response.status_code == 200:
                    # Content-Type 확인
                    content_type = response.headers.get('Content-Type', '').lower()
                    
                    # PDF인지 확인 (Content-Type 또는 파일 내용으로)
                    if 'pdf' in content_type or 'application/pdf' in content_type:
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # PDF 유효성 검사
                        if self._is_valid_pdf(filepath):
                            return filepath
                        else:
                            filepath.unlink()
                    else:
                        # Content-Type이 PDF가 아니어도 파일 내용 확인
                        # 첫 4바이트가 %PDF인지 확인
                        content_preview = response.content[:4]
                        if content_preview == b'%PDF':
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            
                            if self._is_valid_pdf(filepath):
                                return filepath
                            else:
                                filepath.unlink()
            except Exception as e1:
                pass
            
            # 방법 2: sub_docs에서 dcmNo 추출 후 PDF 다운로드
            try:
                attachments = self.dart.sub_docs(rcept_no)
                
                if attachments is not None and len(attachments) > 0:
                    if hasattr(attachments, 'columns'):
                        # 첫 번째 문서의 URL에서 dcmNo 추출
                        first_doc = attachments.iloc[0]
                        viewer_url = str(first_doc.get('url', ''))
                        
                        # URL에서 dcmNo 추출 (예: ...&dcmNo=10883250&...)
                        import re
                        dcm_no_match = re.search(r'dcmNo=(\d+)', viewer_url)
                        
                        if dcm_no_match:
                            dcm_no = dcm_no_match.group(1)
                            
                            # 여러 PDF 다운로드 URL 패턴 시도
                            pdf_urls = [
                                f"https://dart.fss.or.kr/pdf/download/pdf.do?rcp_no={rcept_no}&dcm_no={dcm_no}",
                                f"http://dart.fss.or.kr/pdf/download/pdf.do?rcp_no={rcept_no}&dcm_no={dcm_no}",
                                f"https://dart.fss.or.kr/pdf/download/excel.do?rcp_no={rcept_no}&dcm_no={dcm_no}",  # 일부는 excel.do
                            ]
                            
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                                'Referer': f'https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}'
                            }
                            
                            for pdf_url in pdf_urls:
                                try:
                                    response = requests.get(pdf_url, headers=headers, timeout=30, stream=True, allow_redirects=True)
                                    
                                    if response.status_code == 200:
                                        # PDF 시그니처 확인
                                        content_preview = response.content[:4] if len(response.content) >= 4 else b''
                                        
                                        if content_preview == b'%PDF':
                                            with open(filepath, 'wb') as f:
                                                for chunk in response.iter_content(chunk_size=8192):
                                                    if chunk:
                                                        f.write(chunk)
                                            
                                            if self._is_valid_pdf(filepath):
                                                return filepath
                                            else:
                                                filepath.unlink()
                                except Exception as e:
                                    continue
            except Exception as e2:
                pass
            
            # 방법 3: sub_docs에서 dcm_no를 찾아서 PDF 다운로드 URL 시도
            try:
                attachments = self.dart.sub_docs(rcept_no)
                
                if attachments is not None and len(attachments) > 0:
                    if hasattr(attachments, 'columns'):
                        # 첫 번째 문서의 dcm_no 찾기
                        first_doc = attachments.iloc[0]
                        dcm_no = first_doc.get('dcm_no', '')
                        
                        if dcm_no:
                            # dcm_no를 사용한 PDF 다운로드 URL
                            pdf_url = f"https://dart.fss.or.kr/pdf/download/pdf.do?rcp_no={rcept_no}&dcm_no={dcm_no}"
                            
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            }
                            
                            response = requests.get(pdf_url, headers=headers, timeout=30, stream=True, allow_redirects=True)
                            
                            if response.status_code == 200:
                                content_type = response.headers.get('Content-Type', '').lower()
                                
                                if 'pdf' in content_type or response.content[:4] == b'%PDF':
                                    with open(filepath, 'wb') as f:
                                        for chunk in response.iter_content(chunk_size=8192):
                                            if chunk:
                                                f.write(chunk)
                                    
                                    if self._is_valid_pdf(filepath):
                                        return filepath
                                    else:
                                        filepath.unlink()
            except Exception as e3:
                pass
            
            return None
                
        except Exception as e:
            print(f"      경고: PDF 다운로드 실패 ({rcept_no}): {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_batch(self, companies: List[Dict], max_files_per_company: int = 5, 
                      target_total: int = 50) -> Dict:
        """
        여러 회사의 공시 문서를 일괄 다운로드
        
        Args:
            companies: 회사 리스트
            max_files_per_company: 회사당 최대 다운로드 파일 수
            target_total: 목표 총 파일 수
            
        Returns:
            다운로드 통계
        """
        print(f"\n공시 문서 다운로드 시작")
        print(f"  목표 파일 수: {target_total}개")
        print(f"  회사당 최대: {max_files_per_company}개")
        print("=" * 80)
        
        downloaded_files = []
        
        for company in companies:
            if len(downloaded_files) >= target_total:
                break
            
            corp_code = company['corp_code']
            corp_name = company['corp_name']
            
            print(f"\n[{corp_name}] 처리 중...")
            
            # 공시 목록 조회
            reports = self.get_reports(corp_code)
            
            if not reports:
                print(f"  공시 문서 없음")
                continue
            
            # 최신 순으로 정렬
            reports = sorted(reports, key=lambda x: x['rcept_dt'], reverse=True)
            
            # 회사당 최대 개수만큼 다운로드
            company_downloaded = 0
            for report in reports[:max_files_per_company]:
                if len(downloaded_files) >= target_total:
                    break
                
                rcept_no = report['rcept_no']
                report_nm = report['report_nm']
                
                print(f"  다운로드 중: {report_nm} ({report['rcept_dt']})")
                
                filepath = self.download_report_pdf(rcept_no, corp_name, report_nm)
                
                if filepath:
                    downloaded_files.append({
                        'filepath': str(filepath),
                        'corp_name': corp_name,
                        'report_nm': report_nm,
                        'rcept_no': rcept_no,
                        'rcept_dt': report['rcept_dt']
                    })
                    company_downloaded += 1
                    self.stats['total_downloaded'] += 1
                    print(f"    ✓ 다운로드 완료: {filepath.name}")
                else:
                    self.stats['total_failed'] += 1
                    print(f"    ✗ 다운로드 실패")
                
                # API 호출 제한 고려 (초당 1회)
                time.sleep(1)
            
            self.stats['companies'][corp_name] = company_downloaded
            print(f"  완료: {company_downloaded}개 파일 다운로드")
        
        print("\n" + "=" * 80)
        print("다운로드 완료")
        print(f"  총 다운로드: {self.stats['total_downloaded']}개")
        print(f"  실패: {self.stats['total_failed']}개")
        print(f"  저장 위치: {self.download_dir}")
        print("=" * 80)
        
        # 다운로드 목록 저장
        import json
        list_file = self.download_dir / 'download_list.json'
        with open(list_file, 'w', encoding='utf-8') as f:
            json.dump(downloaded_files, f, ensure_ascii=False, indent=2)
        print(f"\n다운로드 목록 저장: {list_file}")
        
        return {
            'stats': self.stats,
            'files': downloaded_files
        }


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DART 공시 문서 다운로더')
    parser.add_argument('--api-key', type=str, required=True, help='DART API 키')
    parser.add_argument('--download-dir', type=str, default='data/dart_pdfs', help='다운로드 디렉토리')
    parser.add_argument('--target-total', type=int, default=50, help='목표 파일 수')
    parser.add_argument('--max-per-company', type=int, default=5, help='회사당 최대 파일 수')
    parser.add_argument('--sectors', type=str, nargs='+', help='업종 필터 (예: 정보통신업 제조업)')
    
    args = parser.parse_args()
    
    # 다운로더 초기화
    downloader = DARTDownloader(args.api_key, args.download_dir)
    
    # 회사 목록 조회
    companies = downloader.get_company_list(sectors=args.sectors, limit=20)
    
    # 일괄 다운로드
    result = downloader.download_batch(
        companies,
        max_files_per_company=args.max_per_company,
        target_total=args.target_total
    )
    
    print(f"\n총 {result['stats']['total_downloaded']}개 파일 다운로드 완료!")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
DART API 테스트 스크립트
OpenDartReader의 실제 동작 확인
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import OpenDartReader

API_KEY = '26f7944be4228c4ac49916286d61b51da1834cff'

dart = OpenDartReader(API_KEY)

# 최근 공시 목록 조회
print("최근 공시 목록 조회...")
reports = dart.list(start='20241101', end='20241130', kind='A')
print(f"조회된 공시 수: {len(reports) if reports is not None else 0}")

if reports is not None and len(reports) > 0:
    # 첫 번째 공시 확인
    first_report = reports.iloc[0]
    print(f"\n첫 번째 공시:")
    print(f"  회사명: {first_report.get('corp_name', 'N/A')}")
    print(f"  보고서명: {first_report.get('report_nm', 'N/A')}")
    print(f"  접수번호: {first_report.get('rcept_no', 'N/A')}")
    
    rcept_no = first_report.get('rcept_no', '')
    
    # 첨부파일 목록 조회
    print(f"\n첨부파일 목록 조회 (접수번호: {rcept_no})...")
    try:
        attachments = dart.sub_docs(rcept_no)
        print(f"첨부파일 수: {len(attachments) if attachments is not None else 0}")
        
        if attachments is not None and len(attachments) > 0:
            print("\n첨부파일 목록:")
            print(attachments.head())
            
            # PDF 파일 찾기
            if hasattr(attachments, 'columns') and 'file_nm' in attachments.columns:
                pdf_files = attachments[attachments['file_nm'].str.endswith('.pdf', na=False)]
                print(f"\nPDF 파일 수: {len(pdf_files)}")
                if len(pdf_files) > 0:
                    print("\n첫 번째 PDF 파일:")
                    print(pdf_files.iloc[0])
    except Exception as e:
        print(f"첨부파일 조회 오류: {e}")
        import traceback
        traceback.print_exc()




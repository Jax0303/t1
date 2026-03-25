#!/usr/bin/env python3
"""
다운로드한 DART PDF 파일을 보거나 원본 링크를 여는 스크립트
"""
import sys
import json
import webbrowser
from pathlib import Path
import subprocess
import os

def get_download_list():
    """다운로드 목록 로드"""
    list_file = Path("data/dart_pdfs/download_list.json")
    if list_file.exists():
        with open(list_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def view_pdf_locally(pdf_path):
    """로컬 PDF 파일 열기"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"파일을 찾을 수 없습니다: {pdf_path}")
        return False
    
    # OS별 PDF 뷰어 열기
    if sys.platform == 'darwin':  # macOS
        subprocess.run(['open', str(pdf_path)])
    elif sys.platform == 'win32':  # Windows
        os.startfile(str(pdf_path))
    else:  # Linux
        # xdg-open, evince, okular 등 시도
        for cmd in ['xdg-open', 'evince', 'okular', 'atril']:
            try:
                subprocess.run([cmd, str(pdf_path)], check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        print("PDF 뷰어를 찾을 수 없습니다. 수동으로 열어주세요.")
        return False
    
    return True

def open_dart_original(rcept_no):
    """DART 원본 링크 열기"""
    url = f"http://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"
    print(f"DART 원본 링크: {url}")
    webbrowser.open(url)

def list_pdfs():
    """다운로드된 PDF 목록 보기"""
    pdf_dir = Path("data/dart_pdfs")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    print("=" * 80)
    print("다운로드된 PDF 파일 목록")
    print("=" * 80)
    
    download_list = get_download_list()
    file_map = {Path(item['filepath']).name: item for item in download_list}
    
    for i, pdf_file in enumerate(pdf_files, 1):
        file_size = pdf_file.stat().st_size / 1024  # KB
        file_info = file_map.get(pdf_file.name, {})
        
        print(f"\n[{i}] {pdf_file.name}")
        print(f"    크기: {file_size:.1f} KB")
        if file_info:
            print(f"    회사: {file_info.get('corp_name', 'N/A')}")
            print(f"    보고서: {file_info.get('report_nm', 'N/A')}")
            print(f"    접수번호: {file_info.get('rcept_no', 'N/A')}")
    
    return pdf_files

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DART PDF 파일 보기')
    parser.add_argument('--list', action='store_true', help='PDF 목록 보기')
    parser.add_argument('--open', type=int, help='PDF 파일 번호로 열기 (--list로 번호 확인)')
    parser.add_argument('--file', type=str, help='PDF 파일 경로로 열기')
    parser.add_argument('--original', type=str, help='접수번호로 DART 원본 링크 열기')
    parser.add_argument('--all', action='store_true', help='모든 PDF 파일 열기')
    
    args = parser.parse_args()
    
    if args.list or (not args.open and not args.file and not args.original and not args.all):
        pdf_files = list_pdfs()
        
        print("\n" + "=" * 80)
        print("사용 방법:")
        print("  python scripts/view_dart_pdf.py --open 1  # 첫 번째 파일 열기")
        print("  python scripts/view_dart_pdf.py --file '파일명.pdf'  # 특정 파일 열기")
        print("  python scripts/view_dart_pdf.py --original 접수번호  # DART 원본 링크 열기")
        print("  python scripts/view_dart_pdf.py --all  # 모든 파일 열기")
    
    elif args.open:
        pdf_files = sorted(Path("data/dart_pdfs").glob("*.pdf"))
        if 1 <= args.open <= len(pdf_files):
            pdf_file = pdf_files[args.open - 1]
            print(f"\n파일 열기: {pdf_file.name}")
            view_pdf_locally(pdf_file)
            
            # 원본 링크도 표시
            download_list = get_download_list()
            file_map = {Path(item['filepath']).name: item for item in download_list}
            file_info = file_map.get(pdf_file.name, {})
            if file_info.get('rcept_no'):
                print(f"\nDART 원본 링크:")
                print(f"http://dart.fss.or.kr/dsaf001/main.do?rcpNo={file_info['rcept_no']}")
        else:
            print(f"잘못된 번호입니다. 1-{len(pdf_files)} 사이의 숫자를 입력하세요.")
    
    elif args.file:
        pdf_path = Path("data/dart_pdfs") / args.file
        if pdf_path.exists():
            print(f"\n파일 열기: {pdf_path.name}")
            view_pdf_locally(pdf_path)
        else:
            print(f"파일을 찾을 수 없습니다: {pdf_path}")
    
    elif args.original:
        open_dart_original(args.original)
    
    elif args.all:
        pdf_files = sorted(Path("data/dart_pdfs").glob("*.pdf"))
        print(f"\n{len(pdf_files)}개 파일 열기 중...")
        for pdf_file in pdf_files[:10]:  # 최대 10개만
            print(f"  열기: {pdf_file.name}")
            view_pdf_locally(pdf_file)
            input("다음 파일로 이동하려면 Enter를 누르세요...")

if __name__ == "__main__":
    main()




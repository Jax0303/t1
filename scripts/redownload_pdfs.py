#!/usr/bin/env python3
"""
기존 HTML 파일을 삭제하고 PDF로 다시 다운로드하는 스크립트
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dart import DARTDownloader


def is_valid_pdf(filepath: Path) -> bool:
    """파일이 유효한 PDF인지 확인"""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False
    
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header == b'%PDF':
                return True
            
            # HTML 파일인지 확인
            f.seek(0)
            content = f.read(1024)
            if b'<!DOCTYPE' in content or b'<html' in content.lower():
                return False
        
        return False
    except Exception:
        return False


def main():
    """메인 함수"""
    import os
    
    # API 키 확인
    api_key = os.getenv('DART_API_KEY')
    if not api_key:
        print("DART_API_KEY 환경 변수가 설정되지 않았습니다.")
        api_key = input("DART API 키를 입력하세요: ").strip()
        if not api_key:
            print("API 키가 없어 종료합니다.")
            return
    
    download_dir = Path("data/dart_pdfs")
    download_list_file = download_dir / "download_list.json"
    
    # 다운로드 목록 로드
    if not download_list_file.exists():
        print(f"다운로드 목록 파일을 찾을 수 없습니다: {download_list_file}")
        return
    
    with open(download_list_file, 'r', encoding='utf-8') as f:
        download_list = json.load(f)
    
    print("=" * 80)
    print("PDF 재다운로드 스크립트")
    print("=" * 80)
    print(f"\n다운로드 목록: {len(download_list)}개 파일")
    
    # 유효하지 않은 파일 확인
    invalid_files = []
    for file_info in download_list:
        filepath = Path(file_info.get('filepath', ''))
        if not filepath.is_absolute():
            filepath = download_dir / filepath.name
        
        if filepath.exists():
            if not is_valid_pdf(filepath):
                invalid_files.append((filepath, file_info))
                print(f"  ✗ 유효하지 않은 파일: {filepath.name}")
    
    if not invalid_files:
        print("\n모든 파일이 유효한 PDF입니다!")
        return
    
    print(f"\n총 {len(invalid_files)}개 파일을 재다운로드합니다.")
    confirm = input("계속하시겠습니까? (y/N): ").strip().lower()
    if confirm != 'y':
        print("취소되었습니다.")
        return
    
    # 다운로더 초기화
    downloader = DARTDownloader(api_key, str(download_dir))
    
    # 재다운로드
    success_count = 0
    fail_count = 0
    
    for filepath, file_info in invalid_files:
        # 기존 파일 삭제
        if filepath.exists():
            filepath.unlink()
            print(f"\n삭제: {filepath.name}")
        
        # 재다운로드
        rcept_no = file_info.get('rcept_no', '')
        corp_name = file_info.get('corp_name', '')
        report_nm = file_info.get('report_nm', '')
        
        print(f"다운로드 중: {corp_name} - {report_nm}")
        
        new_filepath = downloader.download_report_pdf(rcept_no, corp_name, report_nm)
        
        if new_filepath and is_valid_pdf(new_filepath):
            print(f"  ✓ 성공: {new_filepath.name}")
            success_count += 1
        else:
            print(f"  ✗ 실패")
            fail_count += 1
        
        # API 호출 제한 고려
        import time
        time.sleep(1)
    
    print("\n" + "=" * 80)
    print("재다운로드 완료")
    print(f"  성공: {success_count}개")
    print(f"  실패: {fail_count}개")
    print("=" * 80)


if __name__ == "__main__":
    main()



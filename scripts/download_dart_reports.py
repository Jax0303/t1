#!/usr/bin/env python3
"""
DART 공시 문서 다운로드 스크립트
사용 예시:
  python scripts/download_dart_reports.py --api-key YOUR_API_KEY
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dart import DARTDownloader


def main():
    """메인 함수"""
    import os
    
    # API 키 확인
    api_key = os.getenv('DART_API_KEY')
    if not api_key:
        print("=" * 80)
        print("DART API 키가 필요합니다.")
        print("=" * 80)
        print("\n1. https://opendart.fss.or.kr 에서 회원가입")
        print("2. API 키 발급")
        print("3. 환경 변수 설정:")
        print("   export DART_API_KEY='your-api-key'")
        print("\n또는 스크립트에 직접 입력:")
        api_key = input("\nDART API 키를 입력하세요: ").strip()
        
        if not api_key:
            print("API 키가 없어 종료합니다.")
            return
    
    print("=" * 80)
    print("DART 공시 문서 다운로더")
    print("=" * 80)
    print(f"\nAPI 키: {api_key[:10]}...")
    print("다운로드 디렉토리: data/dart_pdfs")
    print("목표 파일 수: 50개")
    print("업종: 정보통신업, 제조업")
    print("\n다운로드를 시작합니다...")
    
    # 다운로더 초기화
    downloader = DARTDownloader(api_key, download_dir="data/dart_pdfs")
    
    # 회사 목록 조회 (정보통신업, 제조업)
    companies = downloader.get_company_list(
        sectors=['정보통신업', '제조업', '반도체'],
        limit=15
    )
    
    print(f"\n선택된 회사 목록:")
    for i, company in enumerate(companies[:10], 1):
        print(f"  {i}. {company['corp_name']} ({company.get('sector', 'N/A')})")
    
    # 일괄 다운로드
    result = downloader.download_batch(
        companies,
        max_files_per_company=5,
        target_total=50
    )
    
    print(f"\n✓ 완료: {result['stats']['total_downloaded']}개 파일 다운로드")
    print(f"  저장 위치: data/dart_pdfs/")


if __name__ == "__main__":
    main()


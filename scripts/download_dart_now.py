#!/usr/bin/env python3
"""
DART 공시 문서 다운로드 스크립트 (즉시 실행 버전)
API 키가 하드코딩되어 있어 바로 실행 가능
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# API 키 설정
DART_API_KEY = '26f7944be4228c4ac49916286d61b51da1834cff'

from src.dart import DARTDownloader


def main():
    """메인 함수"""
    print("=" * 80)
    print("DART 공시 문서 다운로더")
    print("=" * 80)
    print(f"\nAPI 키: {DART_API_KEY[:10]}...")
    print("다운로드 디렉토리: data/dart_pdfs")
    print("목표 파일 수: 50개")
    print("업종: 정보통신업, 제조업, 반도체")
    print("\n다운로드를 시작합니다...\n")
    
    # 다운로더 초기화
    downloader = DARTDownloader(DART_API_KEY, download_dir="data/dart_pdfs")
    
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




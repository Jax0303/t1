#!/usr/bin/env python3
"""
빠른 테스트용 실험 스크립트
PDF 파일이 없어도 실험 스크립트가 제대로 작동하는지 확인
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("PDF 표 추출 실험 준비 상태 확인")
print("=" * 80)

# 1. 필요한 라이브러리 확인
print("\n[1] 라이브러리 확인")
libraries = {
    'pdfplumber': False,
    'camelot': False,
    'tabula': False,
    'pymupdf': False
}

try:
    import pdfplumber
    libraries['pdfplumber'] = True
    print("  ✓ pdfplumber 설치됨")
except ImportError:
    print("  ✗ pdfplumber 미설치")

try:
    import camelot
    libraries['camelot'] = True
    print("  ✓ camelot 설치됨")
except ImportError:
    print("  ✗ camelot 미설치")

try:
    import tabula
    libraries['tabula'] = True
    print("  ✓ tabula 설치됨")
except ImportError:
    print("  ✗ tabula 미설치")

try:
    import fitz
    libraries['pymupdf'] = True
    print("  ✓ pymupdf 설치됨")
except ImportError:
    print("  ✗ pymupdf 미설치")

# 2. PDF 파일 확인
print("\n[2] PDF 파일 확인")
pdf_dir = Path("data/dart_pdfs")
pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []

valid_pdfs = []
for pdf_file in pdf_files:
    if pdf_file.exists():
        try:
            with open(pdf_file, 'rb') as f:
                header = f.read(4)
                if header == b'%PDF':
                    valid_pdfs.append(pdf_file)
        except:
            pass

print(f"  총 PDF 파일: {len(pdf_files)}개")
print(f"  유효한 PDF: {len(valid_pdfs)}개")

if len(valid_pdfs) > 0:
    print("\n  유효한 PDF 파일 목록:")
    for pdf in valid_pdfs[:5]:
        print(f"    - {pdf.name}")
    if len(valid_pdfs) > 5:
        print(f"    ... 외 {len(valid_pdfs) - 5}개")

# 3. 실험 스크립트 확인
print("\n[3] 실험 스크립트 확인")
scripts = {
    'compare_pdf_extractors.py': Path("scripts/compare_pdf_extractors.py").exists(),
    'analyze_korean_pdf_issues.py': Path("scripts/analyze_korean_pdf_issues.py").exists(),
    'redownload_pdfs.py': Path("scripts/redownload_pdfs.py").exists()
}

for script, exists in scripts.items():
    if exists:
        print(f"  ✓ {script}")
    else:
        print(f"  ✗ {script} 없음")

# 4. 다음 단계 안내
print("\n" + "=" * 80)
print("다음 단계")
print("=" * 80)

if len(valid_pdfs) == 0:
    print("\n⚠️  PDF 파일이 없습니다. 다음 중 하나를 선택하세요:")
    print("\n1. 재다운로드 (DART API 키 필요):")
    print("   export DART_API_KEY='your-api-key'")
    print("   python3 scripts/redownload_pdfs.py")
    print("\n2. 새로 다운로드:")
    print("   export DART_API_KEY='your-api-key'")
    print("   python3 scripts/download_dart_reports.py")
else:
    print(f"\n✓ {len(valid_pdfs)}개 PDF 파일로 실험을 시작할 수 있습니다!")
    print("\n실험 실행:")
    print("   python3 scripts/compare_pdf_extractors.py --max-files 5")
    print("   python3 scripts/analyze_korean_pdf_issues.py --input 'data/dart_pdfs/your_file.pdf'")

available_methods = [k for k, v in libraries.items() if v]
if available_methods:
    print(f"\n사용 가능한 파서: {', '.join(available_methods)}")
else:
    print("\n⚠️  사용 가능한 파서가 없습니다. requirements.txt를 설치하세요:")
    print("   pip install -r requirements.txt")

print()




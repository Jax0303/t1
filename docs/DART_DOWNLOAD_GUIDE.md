# DART 공시 문서 다운로드 가이드

## 개요

DART(금융감독원 전자공시)에서 재무·공시 PDF 파일을 자동으로 다운로드하는 가이드입니다.

## DART API 키 발급

1. **회원가입**: https://opendart.fss.or.kr
2. **API 키 발급**: 마이페이지 > API 인증키 관리
3. **API 키 확인**: 발급받은 키를 복사해두세요

## 설치

```bash
pip install OpenDartReader
```

또는 requirements.txt에 포함되어 있으므로:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 방법 1: 스크립트 실행 (권장)

```bash
# 환경 변수로 API 키 설정
export DART_API_KEY='your-api-key-here'

# 스크립트 실행
python scripts/download_dart_reports.py
```

### 방법 2: 프로그래밍 방식

```python
from src.dart import DARTDownloader

# 다운로더 초기화
downloader = DARTDownloader(
    api_key='your-api-key',
    download_dir='data/dart_pdfs'
)

# 회사 목록 조회
companies = downloader.get_company_list(
    sectors=['정보통신업', '제조업'],
    limit=15
)

# 일괄 다운로드
result = downloader.download_batch(
    companies,
    max_files_per_company=5,
    target_total=50
)
```

## 다운로드 설정

### 기본 설정
- **목표 파일 수**: 50개
- **회사당 최대 파일**: 5개
- **기간**: 최근 3년
- **보고서 타입**: 사업보고서, 반기보고서, 분기보고서

### 커스터마이징

```python
# 특정 업종만 선택
companies = downloader.get_company_list(
    sectors=['정보통신업', '반도체'],
    limit=20
)

# 더 많은 파일 다운로드
result = downloader.download_batch(
    companies,
    max_files_per_company=10,  # 회사당 10개
    target_total=100  # 총 100개
)
```

## 다운로드된 파일

### 저장 위치
```
data/dart_pdfs/
├── 삼성전자_사업보고서_20231231.pdf
├── SK하이닉스_반기보고서_20230630.pdf
├── ...
└── download_list.json  # 다운로드 목록 메타데이터
```

### 파일명 형식
```
{회사명}_{보고서명}_{접수번호}.pdf
```

### download_list.json 구조
```json
[
  {
    "filepath": "data/dart_pdfs/삼성전자_사업보고서_20231231.pdf",
    "corp_name": "삼성전자",
    "report_nm": "사업보고서",
    "rcept_no": "20231231000001",
    "rcept_dt": "20231231"
  }
]
```

## 주의사항

1. **API 호출 제한**: DART API는 초당 1회 호출 제한이 있습니다. 스크립트에 자동으로 1초 대기 시간이 포함되어 있습니다.

2. **파일 크기**: PDF 파일은 수십 MB일 수 있으므로 충분한 저장 공간을 확보하세요.

3. **중복 다운로드 방지**: 이미 다운로드된 파일은 자동으로 건너뜁니다.

4. **네트워크**: 대량 다운로드 시 안정적인 네트워크 연결이 필요합니다.

## 문제 해결

### API 키 오류
```
오류: API 키가 유효하지 않습니다.
```
→ DART 포털에서 API 키를 확인하고 다시 시도하세요.

### 다운로드 실패
```
경고: PDF 다운로드 실패
```
→ 네트워크 연결을 확인하거나 나중에 다시 시도하세요. 일부 파일은 접근 제한이 있을 수 있습니다.

### 회사 목록 조회 실패
→ 인터넷 연결을 확인하거나 DART API 서버 상태를 확인하세요.

## 다음 단계

다운로드 완료 후:

1. **표 추출**: `test_hwp5_table_extractor_improved.py`를 PDF용으로 수정하여 표 추출
2. **지식 그래프 구축**: 추출된 표를 지식 그래프로 변환
3. **RAG 파이프라인**: `run_kg_rag_pipeline.py`로 QA 시스템 구축

## 참고 자료

- [DART 공시시스템](https://dart.fss.or.kr)
- [Open DART 포털](https://opendart.fss.or.kr)
- [OpenDartReader GitHub](https://github.com/FinanceData/OpenDartReader)



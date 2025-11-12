# RAG 파이프라인 테스트 가이드

## 개요
032.표 이미지-텍스트 쌍 데이터와 149.표 정보 질의 응답 데이터를 사용하여 Gemini 2.5 Pro로 RAG 시스템을 테스트합니다.

## 사전 준비

### 1. 패키지 설치
```bash
# 가상환경 생성 (권장)
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
pip install pandas beautifulsoup4 lxml html5lib pyyaml tqdm
```

또는 시스템 패키지로 설치:
```bash
pip install --break-system-packages -r requirements.txt
pip install --break-system-packages pandas beautifulsoup4 lxml html5lib pyyaml tqdm
```

### 2. API 키 설정

#### 방법 1: config.yaml 파일 생성
```bash
cp config.yaml.example config.yaml
# config.yaml 파일을 열어서 gemini.api_key에 API 키 입력
```

#### 방법 2: 환경변수 설정
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 3. 데이터 준비 확인
데이터는 이미 압축 해제되어 있어야 합니다:
- `/home/user/t1/test_data/032/` - HTML 테이블 파일들
- `/home/user/t1/test_data/149/` - JSON 질의응답 파일들

## 실행 방법

```bash
python3 test_rag_pipeline.py
```

## 스크립트 기능

1. **HTML 테이블 추출**: 032 데이터셋의 HTML 파일에서 표를 추출
2. **JSON 표 추출**: 149 데이터셋의 JSON context에서 HTML 테이블 추출
3. **지식 베이스 구축**: 추출된 표들을 ChromaDB 벡터 스토어에 임베딩
4. **질의응답 테스트**: 149 데이터셋의 질문으로 RAG 시스템 테스트
5. **정확도 평가**: 예상 답변과 생성 답변 비교

## 출력 예시

```
================================================================================
RAG 파이프라인 테스트 시작
================================================================================
모델: gemini-2.5-pro
HTML 디렉토리: /home/user/t1/test_data/032
QA JSON 파일: /home/user/t1/test_data/149/...
================================================================================

[HTML 테이블 추출] 총 11개 파일 처리 중...
HTML 파싱: 100%|████████████| 11/11 [00:00<00:00, ...]
  추출 완료: 11개 표

[JSON context에서 표 추출] ...
  추출 완료: X개 표

총 XX개 표로 지식 베이스 구축 중...
지식 베이스 구축 완료!

[질의응답 테스트] 10개 질문 테스트 중...
...
```

## 문제 해결

### 패키지 설치 오류
- 가상환경 사용 권장
- 또는 `--break-system-packages` 플래그 사용

### API 키 오류
- config.yaml 파일 확인
- 또는 GEMINI_API_KEY 환경변수 확인

### 데이터 경로 오류
- test_data 디렉토리 확인
- ZIP 파일 압축 해제 확인


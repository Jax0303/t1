# 실험 현황 정리

## 📊 표 추출 결과

### HWPX 직접 파싱
- **추출된 표**: 43개 ✅
- **표 크기 분포**:
  - 3행 x 2열: 14개
  - 6행 x 5열: 4개
  - 8행 x 4열: 4개
  - 17행 x 6열: 3개
  - **134행 x 24열: 2개** (큰 표)
  - **136행 x 24열: 2개** (큰 표)
- **처리 시간**: 평균 0.42초/파일

### HWP 직접 파싱
- **추출된 표**: 0개 ❌
- **문제**: 모든 파싱 방법 실패
  - pyhwp: 미설치 또는 실패
  - olefile+zlib: 레코드 구조 해석 실패
  - 바이너리 파싱: 패턴 매칭 실패

## 🔧 수정 완료 사항

### HWPX 추출기 개선
1. **Section 파일 우선 처리**: section1.xml에 37개 tbl 태그 발견
2. **hp:tbl 네임스페이스 처리**: ElementTree와 BeautifulSoup 모두 지원
3. **빈 표 필터링**: 단일 셀 표 제외 로직 추가

### 문제 원인
- 이전: Contents/header.xml에서 잘못된 표 1개만 추출
- 수정: Section 파일들을 우선 처리하여 실제 표 43개 추출

## 🚀 API 상태

### Gemini API
- **상태**: 한도 초과 (429 에러)
- **문제**: 무료 티어 한도 초과
  - `generativelanguage.googleapis.com/generate_content_free_tier_requests`: limit 0
  - `generativelanguage.googleapis.com/generate_content_free_tier_input_token_count`: limit 0

### Ollama
- **상태**: 설치 확인 중
- **설정**: config.yaml에 ollama 설정 추가됨
  - base_url: http://localhost:11434
  - model: llama3.2

## 📁 데이터 파일

### dataset2
- `개정 표준취업규칙(2025년, 배포).hwp`: HWP 파일
- `개정 표준취업규칙hwpx(2025년, 배포).hwpx`: HWPX 파일 (43개 표 추출 성공)
- `개정+표준취업규칙(2025년,+배포).hwp.pdf`: PDF 파일

## 🎯 다음 단계

1. **Ollama 설치 및 실행**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve
   ollama pull llama3.2
   ```

2. **RAG 평가 재실행**
   - Ollama로 전환하여 RAG 평가 수행
   - 표 추출 결과 기반 질의응답 테스트

3. **HWP 직접 파싱 개선** (선택)
   - pyhwp 라이브러리 설치 시도
   - 레코드 구조 해석 로직 개선

## 📈 현재 성과

✅ **표 추출**: HWPX에서 43개 표 성공적으로 추출
- 큰 표 (134행 x 24열) 포함
- 다양한 크기의 표 정확히 파싱

❌ **RAG 평가**: API 한도로 인해 미완료
- 표 추출은 완료되어 RAG 시스템 구축 준비 완료

## 🔍 발견 사항

1. **HWPX 파일 구조**:
   - Section 파일들에 실제 표 데이터 존재
   - Contents/header.xml에는 표가 없음

2. **표 구조**:
   - 대부분 2-6열 구조
   - 일부 큰 표는 24열 구조 (목차/인덱스 형식)

3. **파싱 방법**:
   - ElementTree 네임스페이스 처리: 정확함
   - BeautifulSoup: 보조 방법으로 유용


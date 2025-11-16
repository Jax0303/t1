# VLM 추출기 실험 현황

## GPU 환경 확인

✅ **GPU 사용 가능**
- GPU: NVIDIA GeForce RTX 3060 Ti
- CUDA 버전: 12.8
- PyTorch 버전: 2.9.1+cu128

## 구현 현황

### Nougat (학술/수식 도메인)

**상태**: ⚠️ 부분 구현 완료, 호환성 문제 발생

**설치 완료**:
- `nougat-ocr` 패키지 설치 완료
- 모델 다운로드 완료 (956MB, 0.1.0-small 버전)
- GPU에서 모델 로드 성공

**문제점**:
1. **Transformers 버전 호환성 문제**
   - `BARTDecoder.prepare_inputs_for_inference()`에서 `cache_position` 파라미터 오류
   - 최신 transformers 버전과 Nougat의 BARTDecoder가 호환되지 않음

2. **용도 불일치**
   - Nougat는 학술 논문의 수식과 텍스트 추출에 특화
   - DART 공시 문서(한국어, 표 중심)에는 적합하지 않을 수 있음

**해결 방안**:
- transformers 버전 다운그레이드 (복잡도 높음)
- Nougat 코드 패치 (시간 소요)
- 학술 논문 데이터셋으로 별도 테스트 권장

### Table Transformer (TATR)

**상태**: ❌ 설치 실패

**문제점**:
- `detectron2` 설치 실패
- GitHub에서 직접 빌드 시도했으나 컴파일 오류 발생
- CUDA/PyTorch 버전 호환성 문제

**대안**:
- 더 간단한 표 감지 모델 사용 고려
- 또는 detectron2 설치 가이드 따르기 (복잡)

## 현재까지 실험 결과

### 전통 베이스라인 (완료)

| 추출기 | 표 추출 | 속도 | 효율성 |
|--------|---------|------|--------|
| **pdfplumber** | 1,184개 | 7.49초/파일 | 52.7 표/초 ⭐ |
| **pymupdf** | 1,179개 | 15.11초/파일 | 26.0 표/초 |
| **pypdf** | 0개 | 2.28초/파일 | 표 추출 불가 |

### RAG/GenAI 지향 (부분 완료)

| 추출기 | 상태 | 비고 |
|--------|------|------|
| **unstructured** | ✅ 설치 완료 | 표 감지 로직 개선 필요 |
| **Docling** | ❌ 설치 실패 | API 확인 필요 |
| **Marker** | ❌ 설치 실패 | API 확인 필요 |

### VLM/OCR SOTA (진행 중)

| 추출기 | 상태 | 비고 |
|--------|------|------|
| **Nougat** | ⚠️ 호환성 문제 | transformers 버전 이슈 |
| **TATR** | ❌ 설치 실패 | detectron2 필요 |

## 권장 사항

### 즉시 사용 가능한 추출기

1. **pdfplumber** ⭐⭐⭐⭐⭐
   - 한국어 DART 공시 문서에 최적
   - 높은 정확도와 적절한 속도

2. **pymupdf** ⭐⭐⭐⭐
   - 정확도는 우수하나 속도가 느림

### 추가 작업 필요

1. **Nougat**
   - 학술 논문 데이터셋으로 별도 테스트
   - transformers 버전 조정 또는 패치

2. **Table Transformer**
   - detectron2 설치 가이드 따르기
   - 또는 더 간단한 대안 모델 사용

3. **RAG 추출기**
   - unstructured 표 감지 로직 개선
   - Docling/Marker API 확인 및 구현

## 다음 단계

1. ✅ 전통 베이스라인 비교 완료
2. ⏳ RAG 추출기 구현 및 테스트
3. ⏳ VLM 추출기 호환성 문제 해결
4. ⏳ 통합 비교 리포트 작성


# HWP 직접 파싱 vs HWPX 변환 비교 실험 구현 요약

## 구현 완료 사항

### 1. HWP 직접 파싱 추출기 구현 ✅

**파일**: `src/extractors/hwp_extractor.py`

**주요 기능**:
- **pyhwp 기반 파싱**: HWP5 파일의 구조화된 데이터 직접 파싱
- **olefile 기반 파싱**: OLE2 구조의 HWP 파일 스트림 파싱
- **바이너리 직접 파싱**: 텍스트 패턴 매칭을 통한 표 추출
- **정규화 강화**: RAG에 최적화된 텍스트 정규화 및 표 구조 보존

**핵심 개선사항**:
1. HWPX 변환 과정 제거로 처리 속도 향상
2. 변환 과정에서의 정보 손실 최소화
3. 텍스트 정규화 강화 (공백, 줄바꿈, 특수문자 처리)
4. 표 구조 보존 (헤더 인식, 셀 정렬)

### 2. 비교 실험 프레임워크 구현 ✅

**파일**: `src/experiment_hwp_comparison.py`

**주요 기능**:
- 동일한 HWP 파일에 대해 세 가지 방법 비교:
  1. HWP 직접 파싱 (새로운 방법)
  2. HWPX 직접 파싱 (베이스라인 1)
  3. HWP->HWPX 변환 후 파싱 (베이스라인 2)
- 표 추출 성능 비교 (개수, 시간, F1-score)
- RAG 시스템 구축 및 질의응답 평가
- EM, F1, Hit@K 지표 계산 및 비교

### 3. 평가 지표 구현 ✅

**표 추출 평가**:
- 표 개수 비교
- 추출 시간 비교
- 표 내용 유사도 (F1-score)

**RAG 성능 평가**:
- **EM Score**: 정답과 완전히 일치하는 답변의 비율
- **F1 Score**: 답변의 정밀도와 재현율의 조화 평균
- **Hit@K**: 상위 K개 검색 결과 중 관련 문서 포함 비율
  - Hit@1, Hit@3, Hit@5
- **BLEU Score**: 답변 품질 평가 (n-gram 기반)

### 4. 실행 스크립트 및 문서화 ✅

**파일들**:
- `run_hwp_comparison.py`: 실험 실행 스크립트
- `HWP_COMPARISON_README.md`: 상세 사용 가이드
- `README.md`: 메인 README 업데이트

## 사용 방법

### 기본 실행

```bash
python run_hwp_comparison.py
```

### 필요한 설정

1. **config.yaml**: API 키 설정 필요
   ```yaml
   api:
     openai:
       api_key: "your-api-key"
       model: "gpt-4o-mini"
     # 또는
     gemini:
       api_key: "your-api-key"
       model: "gemini-2.0-flash-exp"
   ```

2. **데이터 준비**: `data/raw/dataset2/`에 HWP/HWPX 파일 배치

3. **쿼리 준비**: `sample_queries.json` (선택적)

### 선택적 라이브러리 설치

```bash
# HWP5 파일 파싱용 (선택적)
pip install pyhwp

# OLE2 파일 파싱용 (선택적)
pip install olefile
```

**참고**: 이 라이브러리들이 없어도 바이너리 파싱 방법이 작동합니다.

## 결과 해석

실험 결과는 다음 형식으로 출력됩니다:

```
[성능 개선 분석]
베이스라인: HWPX 직접 파싱

[HWP 직접 파싱] vs [HWPX 직접 파싱]
  EM Score: 0.8500 (+0.1000, +13.3%)
  F1 Score: 0.8200 (+0.0800, +10.8%)
  Hit@1: 0.7500 (+0.0500, +7.1%)
  Hit@3: 0.9000 (+0.1000, +12.5%)
  Hit@5: 0.9500 (+0.0500, +5.6%)
```

이 결과는:
- **EM Score**: 정답 일치율이 13.3% 향상
- **F1 Score**: 답변 품질이 10.8% 향상
- **Hit@K**: 검색 정확도가 전반적으로 향상

## 기술적 특징

### HWP 직접 파싱의 장점

1. **처리 속도**: HWPX 변환 과정 제거로 속도 향상
2. **정보 보존**: 변환 과정에서의 정보 손실 최소화
3. **정규화**: RAG에 최적화된 텍스트 정규화
4. **구조 보존**: 표 구조 정보 보존 강화

### 정규화 방법

1. **텍스트 정규화**:
   - 연속된 공백 제거
   - 특수 문자 정리
   - 줄바꿈 정리

2. **표 구조 정규화**:
   - 행 길이 맞추기
   - 헤더 자동 인식
   - 셀 내용 정규화

3. **RAG 최적화**:
   - 임베딩에 적합한 형식으로 변환
   - 구조 정보 보존

## 파일 구조

```
t1/
├── src/
│   ├── extractors/
│   │   └── hwp_extractor.py          # HWP 직접 파싱 구현
│   ├── experiment_hwp_comparison.py  # 비교 실험 프레임워크
│   └── evaluation/
│       └── metrics.py                # 평가 지표 (기존)
├── run_hwp_comparison.py              # 실행 스크립트
├── HWP_COMPARISON_README.md           # 상세 가이드
└── IMPLEMENTATION_SUMMARY.md          # 이 문서
```

## 다음 단계

1. 실제 데이터로 실험 실행
2. 결과 분석 및 성능 개선 효과 확인
3. 필요시 파싱 방법 개선
4. 추가 평가 지표 고려

## 문제 해결

### pyhwp/olefile 설치 오류
→ 바이너리 파싱 방법이 자동으로 사용됩니다.

### HWPX 파일 없음
→ HWP 직접 파싱 방법만 실행됩니다.

### API 키 오류
→ `config.yaml`에 올바른 API 키 설정 확인

## 참고사항

- HWP 파일 형식이 복잡하므로 모든 파일에서 완벽한 파싱이 보장되지 않을 수 있습니다.
- 실제 성능은 파일의 구조와 내용에 따라 달라질 수 있습니다.
- pyhwp나 olefile이 설치되지 않은 경우, 바이너리 파싱 방법이 자동으로 사용됩니다.


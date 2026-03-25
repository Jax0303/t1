# 3축 PDF 추출기 비교 실험 설계

## 개요

"전통 파서 vs RAG 프레임워크 vs VLM 기반 SOTA" 3축 비교를 통한 PDF 추출 성능 평가 실험 설계 문서입니다.

## 실험 구조

### (A) 전통 베이스라인

| 추출기 | 상태 | 설치 명령 | 특징 |
|--------|------|-----------|------|
| **pypdf** | ✅ 구현 가능 | `pip install pypdf` | 가장 단순한 텍스트 추출, 레이아웃 보존 제한적 |
| **PyMuPDF** | ✅ 이미 구현됨 | `pip install pymupdf` | 빠른 속도, 최신 버전에서 표 추출 지원 |
| **pdfplumber** | ✅ 이미 구현됨 | `pip install pdfplumber` | 표/레이아웃 포함, 높은 정확도 |

### (B) RAG/GenAI 지향

| 추출기 | 상태 | 설치 명령 | 특징 | 제약사항 |
|--------|------|-----------|------|----------|
| **Docling** | ✅ 구현 가능 | `pip install docling-core` | IBM 오픈소스, 멀티 포맷 + 레이아웃 | 추가 의존성 필요 (transformers 등) |
| **unstructured** | ✅ 구현 가능 | `pip install unstructured[pdf]` | Element 기반 청킹, RAG 최적화 | GPU 선택적, 모델 다운로드 필요 |
| **Marker** | ✅ 구현 가능 | `pip install marker-pdf` | 로컬 Markdown/JSON 변환, LLM 활용 | GPU 권장, 모델 다운로드 필요 |
| **LlamaParse** | ⚠️ 제한적 | API 키 필요 | SaaS, LLM 기반 SOTA RAG 파서 | 비용 발생, API 호출 제한 |

### (C) VLM/OCR + 테이블 SOTA

| 추출기 | 상태 | 설치 명령 | 특징 | 제약사항 |
|--------|------|-----------|------|----------|
| **olmOCR/olmOCR2** | ⚠️ 복잡 | GitHub 설치 | 전체 문서 OCR + 구조 보존 | GPU 필수, 복잡한 설치 과정 |
| **Nougat** | ✅ 구현 가능 | `pip install nougat-ocr` | 학술/수식 도메인 특화 | GPU 필수, 대용량 모델 (1.4GB+) |
| **Table Transformer (TATR)** | ✅ 구현 가능 | `pip install detectron2` | 표 구조 SOTA, Microsoft | GPU 권장, detectron2 필요 |

## 평가 디멘션 (Metrics)

### 1. 텍스트 품질 (Text Quality)
- **정확도**: 추출된 텍스트의 정확도 (문자 오류율)
- **완전성**: 전체 텍스트 추출률
- **한글 처리**: 한글 폰트/인코딩 문제 해결 여부

### 2. 표 구조 보존 (Table Structure Preservation)
- **표 감지율**: 표를 정확히 감지하는 비율
- **셀 정확도**: 셀 경계 및 내용 정확도
- **병합 셀 처리**: 병합된 셀 인식 및 재현
- **계층 헤더**: 다중 행/열 헤더 처리

### 3. 수식 처리 (Formula Handling)
- **수식 인식**: LaTeX/MathML 변환 정확도
- **수식 위치**: 수식의 원본 위치 보존

### 4. 스캔 문서 처리 (Scanned Document OCR)
- **OCR 정확도**: 스캔된 이미지에서 텍스트 추출 정확도
- **레이아웃 복원**: 원본 레이아웃 재현도

### 5. 속도 (Speed)
- **처리 시간**: 파일당 평균 처리 시간
- **처리량**: 초당 처리 페이지 수
- **초기화 시간**: 모델 로딩/초기화 시간

### 6. 비용 (Cost)
- **라이선스**: 오픈소스 vs 유료
- **API 비용**: SaaS 서비스 사용 비용 (LlamaParse 등)
- **인프라 비용**: GPU/서버 운영 비용

### 7. GPU 요구사항 (GPU Requirements)
- **필수 여부**: GPU 필수 vs 선택적
- **메모리**: 필요한 GPU 메모리 (GB)
- **CPU 대체 가능성**: CPU만으로 실행 가능 여부

### 8. 추가 디멘션
- **설치 복잡도**: 설치 및 설정 난이도
- **문서화**: 문서 및 예제 품질
- **커뮤니티**: 활발한 커뮤니티 지원 여부

## 실험 설계

### Phase 1: 전통 베이스라인 비교 (A축)
**목표**: 기본 파서들의 성능 기준선 확립

**추출기**: pypdf, PyMuPDF, pdfplumber

**평가 포인트**:
- 텍스트 품질
- 표 구조 보존
- 속도
- 설치/사용 편의성

### Phase 2: RAG/GenAI 지향 비교 (B축)
**목표**: RAG 파이프라인에 최적화된 추출기 성능 평가

**추출기**: Docling, unstructured, Marker, (LlamaParse)

**평가 포인트**:
- 텍스트 품질 (RAG 입력으로 적합한지)
- 구조 보존 (청킹에 유리한 구조화)
- 속도 vs 정확도 트레이드오프
- GPU 요구사항

### Phase 3: VLM/OCR SOTA 비교 (C축)
**목표**: 최신 기술의 한계와 가능성 탐색

**추출기**: Nougat, Table Transformer (TATR), (olmOCR)

**평가 포인트**:
- 스캔 문서 처리
- 수식 처리 (Nougat)
- 표 구조 SOTA (TATR)
- GPU 요구사항 및 비용

### Phase 4: 통합 비교 분석
**목표**: 3축 통합 비교 및 사용 사례별 추천

**분석 항목**:
- 디멘션별 최고 성능 추출기
- 사용 사례별 추천 (텍스트 중심, 표 중심, 스캔 문서 등)
- 비용-효율성 분석

## 테스트 데이터셋

### 1. DART 공시 문서 (현재 사용 중)
- **특징**: 한국어, 복잡한 표 구조, 병합 셀 다수
- **파일 수**: 5-10개 샘플
- **용도**: 표 구조 보존, 한글 처리 평가

### 2. 학술 논문 (추가 필요)
- **특징**: 수식 포함, 복잡한 레이아웃
- **용도**: 수식 처리 평가 (Nougat 특화)

### 3. 스캔 문서 (추가 필요)
- **특징**: 이미지 기반 PDF
- **용도**: OCR 성능 평가

## 구현 계획

### Step 1: 전통 베이스라인 확장
- [x] pdfplumber (이미 구현됨)
- [x] PyMuPDF (이미 구현됨)
- [ ] pypdf 추가 구현

### Step 2: RAG/GenAI 추출기 구현
- [ ] Docling 통합
- [ ] unstructured 통합
- [ ] Marker 통합
- [ ] LlamaParse API 통합 (선택적)

### Step 3: VLM/OCR 추출기 구현
- [ ] Nougat 통합
- [ ] Table Transformer (TATR) 통합
- [ ] olmOCR 통합 (선택적, 복잡도 고려)

### Step 4: 평가 메트릭 구현
- [ ] 텍스트 품질 평가 (문자 오류율, 완전성)
- [ ] 표 구조 평가 (셀 정확도, 병합 셀)
- [ ] 속도 측정
- [ ] GPU 사용량 측정

### Step 5: 통합 비교 스크립트
- [ ] 모든 추출기를 통합한 비교 스크립트
- [ ] 디멘션별 성능 리포트 생성
- [ ] 시각화 (성능 차트, 비교 테이블)

## 예상 결과 및 인사이트

### 전통 베이스라인 (A축)
- **예상**: pdfplumber가 표 구조에서 우수, PyMuPDF가 속도에서 우수
- **인사이트**: 단순한 문서에는 충분하지만, 복잡한 구조에는 한계

### RAG/GenAI 지향 (B축)
- **예상**: Marker가 정확도에서 우수, unstructured가 구조화에서 우수
- **인사이트**: RAG 파이프라인에 최적화된 구조화된 출력 제공

### VLM/OCR SOTA (C축)
- **예상**: Nougat가 수식에서 SOTA, TATR이 표 구조에서 SOTA
- **인사이트**: GPU 비용 대비 성능 향상이 명확한지 평가 필요

## 제약사항 및 주의사항

### 구현 제약
1. **olmOCR**: 설치 복잡도가 높아 우선순위 낮음
2. **LlamaParse**: API 비용 발생, 선택적 구현
3. **GPU 모델**: GPU 없는 환경에서 평가 제한적

### 평가 제약
1. **Ground Truth**: 수동 라벨링 필요 (표 구조 정확도 평가)
2. **비용 측정**: 실제 API 호출 비용 측정 필요
3. **GPU 측정**: GPU 메모리/사용률 측정 도구 필요

## 다음 단계

1. **즉시 시작 가능**: 전통 베이스라인 확장 (pypdf 추가)
2. **단기 목표**: RAG/GenAI 추출기 1-2개 구현 (Docling, unstructured)
3. **중기 목표**: VLM 추출기 1개 구현 (Nougat 또는 TATR)
4. **장기 목표**: 전체 비교 실험 및 리포트 생성

## 참고 자료

- [pdfplumber 문서](https://github.com/jsvine/pdfplumber)
- [PyMuPDF 문서](https://pymupdf.readthedocs.io/)
- [Marker GitHub](https://github.com/VikParuchuri/marker)
- [unstructured 문서](https://unstructured.io/)
- [Nougat GitHub](https://github.com/facebookresearch/nougat)
- [Table Transformer 논문](https://arxiv.org/abs/2110.00061)




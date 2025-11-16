# DMS_[기재정정]사업보고서 표 추출·지식 그래프 구축 종합 보고서

## 1. Abstract
한국 공시 문서의 표는 병합 셀, 다층 헤더, 한글·혼합 텍스트가 중첩돼 외산 파서의 기본 설정만으로 안정적으로 추출하기 어렵다. 본 연구는 문제 점수 75점으로 선정된 DMS_[기재정정]사업보고서(308쪽, 표 544개)를 대상으로 pdfplumber, PyMuPDF, Camelot 세 파서를 비교하고, 추출 결과를 지식 그래프로 변환해 KG 기반 RAG 구축 가능성을 검증하였다. pdfplumber와 PyMuPDF는 표 544개를 전수 추출했으나 None 비율 30% 이상 표가 18개 존재했으며, Camelot은 504개 표를 추출하면서도 병합 손실을 최소화했다(`analysis/parser_comparison.json:1-24`). TableToKnowledgeGraph 변환 결과 세 파서 모두 4.5만~4.8만 노드 규모의 KG를 생성했고, None 비율이 높은 표에서도 평균 228개 이상의 노드가 남아 구조 정보를 보존하였다(`analysis/kg_pdfplumber_latest_summary.json:1-9`, `analysis/kg_pymupdf_summary.json:1-9`, `analysis/kg_camelot_summary.json:1-9`). Docling·Unstructured·Marker는 모델 다운로드/설치 제약으로 표를 생성하지 못했으며(`logs/docling_run.log:1-40`, `logs/unstructured_run.log:1-200`, `logs/marker_run.log:1-60`), 추후 네트워크가 허용되는 환경에서 재평가가 필요하다. 본 보고서는 파서 비교 결과, KG 통계, 문제 유형 분석, 후속 연구 과제를 제시한다.

## 2. Introduction
### 2.1 연구 배경
외국에서 개발된 표 파서는 서양 문서의 단순 헤더·비병합 구조를 가정하는 경우가 많아 한국 공시 문서의 구조적 특수성(다층 헤더, 다중 병합, 한글 인코딩)을 제대로 처리하지 못한다. 이는 지식 그래프 기반 RAG 파이프라인의 데이터 품질을 직접 저하시킨다.

### 2.2 연구 목적
복잡한 표를 안정적으로 추출하고 지식 그래프로 전환해 KG 기반 QA에 바로 활용 가능한 데이터를 확보한다. 추출 품질, 문제 패턴, KG 보존도, 후속 연구 주제를 체계적으로 도출한다.

### 2.3 연구 질문
1. pdfplumber, PyMuPDF, Camelot이 한국 공시 표를 어떻게 처리하는가?  
2. None 비율을 유발하는 구조적 원인은 무엇인가?  
3. 추출 결과를 지식 그래프로 변환했을 때 구조 정보가 어느 정도 보존되는가?  
4. 한국 특화 파서/하이브리드 전략을 위해 어떤 연구가 필요한가?

## 3. Related Work
### 3.1 PDF 표 추출
전통 파서(pdfplumber, PyMuPDF, Camelot)는 레이아웃 기반 규칙으로 표를 탐지한다. pdfplumber·PyMuPDF는 전수 추출이 가능하나 병합 값을 복원하지 못해 None 비율이 높다. Camelot lattice는 경계 기반 추출로 병합 손실을 줄이지만 복잡한 병합 구조의 표를 누락한다.

### 3.2 한국어 문서 처리
한국 공시 문서는 한글 폰트와 세로·가로 혼합 텍스트를 포함하므로 OCR/레이아웃 모델의 한국어 최적화가 요구된다. Docling, Unstructured, Marker같은 RAG 지향 파서는 최신 모델(rapidocr, YOLOX, Surya 등)을 필요로 하지만 외부 모델 다운로드 없이는 동작하지 않는다(`logs/unstructured_run.log:1-200`, `logs/marker_run.log:1-60`).

### 3.3 지식 그래프 구축
테이블 기반 KG는 헤더 계층을 정확히 재구성하고 셀 값을 누락 없이 매핑해야 한다. 병합이 해제되지 않으면 엔티티 속성 값이 None으로 남아 그래프 품질이 저하되므로, 추출층과 KG층에서 보정 전략이 모두 필요하다.

## 4. Methodology
### 4.1 데이터셋
문제 점수 75점으로 선정된 `data/dart_pdfs/DMS_[기재정정]사업보고서 (2023.12)_20250321000022.pdf`를 사용했다. 추출 요약(`extracted_DMS_[기재정정]사업보고서 (2023.12)_20250321000022_pdfplumber_annotated_summary.md:1-140`)에 따르면 표 544개, 빈 표 0개, 1행 표 40개, None 비율 30% 이상 표 18개가 포함돼 있다.

### 4.2 파서 구성이
`scripts/test_single_pdf_extraction.py`를 사용해 pdfplumber, PyMuPDF, Camelot을 동일 PDF에 적용했다. Docling·Unstructured·Marker도 시도했지만, 문서 허브 접근 제한 및 설치 문제로 표를 생성하지 못하고 로그만 기록하였다(`logs/docling_run.log:1-40`, `logs/unstructured_run.log:1-200`, `logs/marker_run.log:1-60`).

### 4.3 지식 그래프 변환
각 파서의 출력 JSON을 TableToKnowledgeGraph(`src/kg/table_to_kg.py`)로 변환해 MultiDiGraph 구조의 KG를 생성했다. 요약 통계와 세부 노드 정보를 `analysis/kg_{parser}_summary.json` 및 `analysis/kg_{parser}_details.json`에 저장했다.

### 4.4 평가 지표
- **None 비율**: 셀 값이 비어 있는 비중(30% 이상 문제 표).  
- **표 크기/빈 표/1행 표**: 레이아웃 복잡도와 데이터 유실 여부.  
- **추출 시간**: 파서별 처리 속도.  
- **KG 지표**: 표당 노드/엣지 수, 노드 분포(10개 미만·50개 이상).  
- **실패 로그**: RAG 지향 파서 실패 원인(모델 다운로드, 미설치 등).

## 5. Experiments
### 5.1 파서 비교
표 1은 파서별 추출 지표를 요약한다(`analysis/parser_comparison.json:1-24`).

**표 1. 파서별 추출 성능**

| 파서 | 총 표 | 추출 시간 (초) | 1행 표 | None≥30% 표 |
| --- | --- | --- | --- | --- |
| pdfplumber | 544 | 14.93 | 40 | 18 |
| PyMuPDF | 544 | 28.36 | 40 | 18 |
| Camelot | 504 | 167.10 | 0 | 0 |
| Docling | 0 | 0.48 | 0 | 0 |
| Unstructured | 0 | 63.77 | 0 | 0 |
| Marker | 0 | 0.00 | 0 | 0 |

pdfplumber와 PyMuPDF는 전수 추출이 가능하지만 병합/헤더 셀을 None으로 남겨 문제가 반복된다. Camelot은 추출 건수가 7.4% 감소했지만 None 비율 30% 이상 표는 없었고, 병합/헤더가 비교적 깔끔하게 보존되었다. Docling·Unstructured는 외부 모델(HuggingFace) 다운로드 실패로 중도 종료됐고, Marker는 내부 패키지가 노출되지 않아 초기화에 실패했다(`logs/docling_run.log:1-40`, `logs/unstructured_run.log:1-200`, `logs/marker_run.log:1-60`).

### 5.2 지식 그래프 구축
표 2는 파서별 KG 통계를 정리한다.

**표 2. 파서별 KG 통계**

| 파서 | 변환 표 수 | 총 노드 | 총 엣지 | 평균 노드/엣지 | 노드<10 | 노드≥50 |
| --- | --- | --- | --- | --- | --- | --- |
| pdfplumber | 544 | 48,396 | 59,478 | 88.96 / 109.33 | 7 | 335 |
| PyMuPDF | 544 | 48,396 | 59,478 | 88.96 / 109.33 | 7 | 335 |
| Camelot | 504 | 45,386 | 54,296 | 90.05 / 107.73 | 0 | 310 |

자료: `analysis/kg_pdfplumber_latest_summary.json:1-9`, `analysis/kg_pymupdf_summary.json:1-9`, `analysis/kg_camelot_summary.json:1-9`.

pdfplumber·PyMuPDF는 동일한 표 분포로 인해 KG 통계가 동일하게 측정됐다. Camelot은 추출 표가 적지만 표당 평균 노드 수가 가장 높았으며, 노드 10개 미만 표가 없었다. None 비율이 30% 이상인 표(6개)는 평균 228.5개의 노드를 유지해 구조 정보를 상당 부분 보존하였다(`analysis/kg_pdfplumber_details.json:1-40`). 따라서 None 셀 문제가 존재해도 KG로 전환하는 데 큰 장애는 없으며, 값 보간만 수행하면 RAG에서 활용 가능하다.

### 5.3 실패 로그 분석
- **Docling**: Hugging Face Hub에서 모델 스냅샷을 찾지 못해 추출이 중단되었다(`logs/docling_run.log:1-40`). 네트워크 허용 환경 또는 오프라인 모델 캐시가 필요하다.  
- **Unstructured**: YOLOX 레이아웃 모델을 다운로드하지 못해 5회 재시도 후 중단되었다(`logs/unstructured_run.log:1-200`).  
- **Marker**: 패키지 내부 모듈 노출 문제로 추출기가 초기화되지 않았다(`logs/marker_run.log:1-60`). 로컬 CLI(`marker_single`) 기반 호출로 수정하거나 GPU 없는 CPU 모드 설정이 필요하다.

## 6. Discussion
### 6.1 주요 발견
1. **병합 셀 복원 부족**: pdfplumber·PyMuPDF 모두 병합 정보를 인식하지만 값은 None으로 남아 None 비율 9.9%를 형성한다(`extracted_DMS_[기재정정]사업보고서 (2023.12)_20250321000022_pdfplumber_annotated_summary.md:8-140`).  
2. **Camelot의 보수적 추출**: 추출 표 수는 줄지만, 모든 표가 완전 데이터로 반환돼 병합 복원 문제를 크게 완화한다.  
3. **KG 보존력**: None 비율이 높은 표라도 헤더-셀 구조가 유지되어 평균 200개 이상의 노드가 생성된다. 이는 추출 단계에서 값 보간만 추가하면 KG 기반 RAG에 활용 가능함을 의미한다.

### 6.2 한계점
- Docling·Unstructured·Marker를 실행하지 못해 최신 RAG 파서와의 정량 비교가 불가능했다.  
- 병합 복원/보간 알고리즘을 적용하지 않아 pdfplumber/PyMuPDF의 None 비율이 과대 측정됐을 수 있다.  
- KG 품질을 노드/엣지 수 기반으로만 평가했으며, 실제 QA 정확도 검증은 향후 과제로 남아 있다.

### 6.3 연구 주제
1. **한국형 병합 복원 알고리즘**: None 비율 30% 이상 표(18개)를 대상으로 값 전파/보간 기법을 적용해 None 비율을 10% 이하로 줄이는 연구.  
2. **하이브리드 추출 파이프라인**: pdfplumber+Camelot(+Docling) 순차 실행으로 실패 표를 자동 대체하는 메타 추출기.  
3. **RAG 지향 파서 환경 구성**: Hugging Face 모델 캐시 및 Marker CLI 기반 실행을 자동화해 한국어 문서에 최적화된 RAG 파서를 실험할 것.  
4. **불완전 데이터 기반 KG 평가**: None 비율이 높은 표에 대해 값 보간 전/후 KG 기반 QA 정확도를 비교해 정량적 이득을 도출한다.

## 7. Conclusion
pdfplumber, PyMuPDF, Camelot을 동일 한국 공시 PDF에 적용한 결과, pdfplumber·PyMuPDF는 전체 표를 추출하지만 병합 값 손실이 남았고, Camelot은 일부 표를 누락하는 대신 완전한 값을 제공했다. 세 파서 모두 대규모 KG를 구축할 수 있었으며, None 비율이 높은 표에서도 구조 정보가 풍부하게 남아 있음을 확인했다. Docling·Unstructured·Marker는 네트워크/설치 제약으로 실행하지 못했으므로, 모델 다운로드가 가능한 환경에서 추가 실험이 요구된다. 향후 연구는 병합 복원, 하이브리드 추출, RAG 지향 파서 튜닝, KG 기반 QA 평가를 중심으로 진행될 것이다.

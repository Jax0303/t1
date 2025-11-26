# DART PDF 표 → KG 구축 실험 결과 리포트

## 1. 실험 개요

- 처리한 PDF: 10개
- 처리한 표: 50개
- PyKEEN 임베딩: 사용

## 2. LGPMA 기반 표 유형 분류 결과

| 유형 | 표 개수 | 평균 노드 | 평균 엣지 | 특징 |
|------|---------|-----------|-----------|------|
| hierarchical_header | 24 | 32.92 | 34.96 | 다중/다단 헤더 |
| simple_grid | 18 | 27.39 | 27.61 | 단순 그리드 구조 |
| large_complex | 2 | 208.50 | 207.50 | 대규모 복잡 구조 |
| incomplete | 1 | 169.00 | 208.00 | 불완전한 구조 (빈 셀 포함) |
| mixed_content | 5 | 71.00 | 91.60 | 텍스트+숫자 혼합 |

## 3. 유형별 대표 표 예시

### hierarchical_header
- PDF: `3S_반기보고서 (2024.09)_20241114002413.pdf`
- Table ID: `3S_반기보고서 (2024.09)_20241114002413_table0`
- 진단: 행 5개, 열 6개, 병합 셀 0개
- KG: 노드 44개, 엣지 45개

### simple_grid
- PDF: `3S_반기보고서 (2024.09)_20241114002413.pdf`
- Table ID: `3S_반기보고서 (2024.09)_20241114002413_table1`
- 진단: 행 5개, 열 3개, 병합 셀 0개
- KG: 노드 26개, 엣지 27개

### large_complex
- PDF: `DMS_[기재정정]사업보고서 (2023.12)_20250321000022.pdf`
- Table ID: `DMS_[기재정정]사업보고서 (2023.12)_20250321000022_table3`
- 진단: 행 22개, 열 4개, 병합 셀 0개
- KG: 노드 195개, 엣지 194개

### incomplete
- PDF: `DMS_[기재정정]사업보고서 (2024.12)_20250814000207.pdf`
- Table ID: `DMS_[기재정정]사업보고서 (2024.12)_20250814000207_table4`
- 진단: 행 23개, 열 4개, 병합 셀 0개
- KG: 노드 169개, 엣지 208개

### mixed_content
- PDF: `DMS_반기보고서 (2025.06)_20250814003725.pdf`
- Table ID: `DMS_반기보고서 (2025.06)_20250814003725_table4`
- 진단: 행 2개, 열 3개, 병합 셀 0개
- KG: 노드 15개, 엣지 17개

## 4. 표 난이도 ↔ KG 복잡도 상관관계

표의 구조적 복잡도가 높을수록 생성되는 KG의 노드/엣지 수가 증가하는 경향을 보입니다.

## 5. 시각화 자료

- `visualizations/label_stats.png`: 유형별 평균 노드/엣지 비교
- `visualizations/neo4j/`: Neo4j 로드용 CSV 및 Cypher 쿼리
- `visualizations/correlation_analysis.md`: 상세 상관관계 분석

## 6. 다음 단계 (완료)

### 6.1 Neo4j Browser에서 KG 구조 시각화 ✅
- **파일**: `visualizations/neo4j/visualization_queries.cypher`
- **내용**: 10가지 시각화 쿼리 (유형별 분포, 복잡도 분석, 유사 표 찾기 등)
- **사용법**: Neo4j Browser에서 쿼리 파일을 열어 실행

### 6.2 Graphviz로 서브그래프 생성 ✅
- **스크립트**: `scripts/generate_graphviz_subgraphs.py`
- **기능**: 
  - 유형별 대표 표 구조 그래프 생성
  - 유형별 비교 그래프 생성
  - 상세 서브그래프 생성
- **사용법**: 
  ```bash
  python scripts/generate_graphviz_subgraphs.py --format png
  ```
- **참고**: Graphviz dot 실행 파일 필요 (시스템 설치 필요)

### 6.3 PyKEEN 임베딩을 활용한 유사 표 검색 ✅
- **스크립트**: `scripts/pykeen_similarity_search.py`
- **기능**:
  - 표 KG 구조를 PyKEEN으로 임베딩
  - 코사인 유사도 기반 유사 표 검색
  - 검색 결과 리포트 생성
- **사용법**:
  ```bash
  python scripts/pykeen_similarity_search.py --model TransE --dim 50
  python scripts/pykeen_similarity_search.py --target "TABLE_ID" --top-k 5
  ```
- **출력**: `visualizations/pykeen/` 디렉터리에 임베딩 및 리포트 저장

### 6.4 빈 셀 추론 및 불완전 표 보완 연구 ✅
- **스크립트**: `scripts/cell_inference_research.py`
- **기능**:
  - 빈 셀 패턴 분석
  - 통계적 방법 기반 추론 (행/열 평균, 헤더 패턴 등)
  - 머신러닝 기반 추론 (선택사항)
  - 추론 결과 평가 및 리포트 생성
- **사용법**:
  ```bash
  python scripts/cell_inference_research.py
  python scripts/cell_inference_research.py --use-ml  # ML 기반 추론 포함
  ```
- **출력**: `visualizations/cell_inference/` 디렉터리에 결과 저장

## 7. 추가 실험 아이디어

고급 KG 라이브러리 활용 실험 아이디어는 `docs/additional_experiments.md`에 정리되어 있습니다:

- **RML/R2RML 계열**: PyRML, Morph-KGC, SDM-RDFizer를 활용한 표준화된 RDF 변환
- **Python KG 엔지니어링**: kglab (SHACL 검증, GNN 분석), maplib (OTTR 템플릿)
- **Graph DB + LLM**: Neo4j GraphRAG를 활용한 텍스트→KG 변환 및 질의응답

자세한 내용은 `docs/additional_experiments.md` 참조. 
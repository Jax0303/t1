# 추가 실험 아이디어: 고급 KG 라이브러리 활용

이 문서는 언급된 라이브러리들을 활용한 추가 실험 아이디어를 제시합니다.

## 1. Declarative KGC (RML/R2RML 계열)

### 1.1 PyRML – Python-native RML 엔진

**목적**: Python 환경에서 직접 RML 매핑 실행

**실험 아이디어**:
- 표 데이터를 RML 매핑으로 자동 변환
- JSON/CSV → RDF 변환 파이프라인 구축
- 대규모 표 데이터의 배치 처리

**구현 예시**:
```python
# visualizations/rml/table_mapping.ttl 파일 사용
# PyRML 설치: pip install pyrml

from pyrml import RMLMapper
mapper = RMLMapper()
rdf_output = mapper.map("visualizations/rml/table_mapping.ttl", 
                       "visualizations/rml/table_runs.json")
```

**기대 효과**:
- 표준화된 RDF 변환
- 다른 RML 엔진과의 호환성
- 재사용 가능한 매핑 파일

### 1.2 Morph-KGC – 대규모 RDF materialization

**목적**: 대규모 데이터셋의 효율적인 RDF 변환

**실험 아이디어**:
- 50개 이상의 표를 한 번에 RDF로 변환
- 병렬 처리 성능 비교
- 다양한 데이터 소스 통합 (CSV, JSON, SQL)

**구현 예시**:
```python
# Morph-KGC는 설정 파일 기반
# config.ini 생성 필요

# morph-kgc -c config.ini 실행
```

**기대 효과**:
- 대규모 데이터 처리
- 성능 최적화
- 프로덕션 환경 적용 가능

### 1.3 SDM-RDFizer – 고성능 RML 엔진

**목적**: KGCW 챌린지 수준의 고성능 변환

**실험 아이디어**:
- 성능 벤치마크 비교 (PyRML vs Morph-KGC vs SDM-RDFizer)
- 메모리 사용량 분석
- 변환 품질 평가

**기대 효과**:
- 최적의 엔진 선택
- 성능 최적화 방향 제시

## 2. Python KG 엔지니어링

### 2.1 kglab – RDFLib + SHACL + GNN까지 묶는 Graph Data Science 레이어

**목적**: 통합 KG 분석 및 검증 플랫폼

**실험 아이디어**:

#### 2.1.1 SHACL 검증
- 표 KG의 구조적 무결성 검증
- 커스텀 SHACL 셰이프 정의
- 검증 리포트 자동 생성

**구현 예시**:
```python
import kglab

kg = kglab.KnowledgeGraph()
kg.load_rdf("visualizations/rdf/table_kg.ttl")

# SHACL 검증
shacl = kglab.Subgraph()
validation_result = shacl.validate(kg)
```

#### 2.1.2 GNN 분석
- 그래프 신경망을 활용한 표 유형 분류
- 패턴 학습 및 예측
- 유사 표 자동 발견

**구현 예시**:
```python
from kglab import KnowledgeGraph
import torch_geometric

# GNN 모델 학습
model = kglab.GNNModel()
model.train(kg)
predictions = model.predict(new_tables)
```

**기대 효과**:
- 통합 분석 환경
- 자동화된 검증
- 머신러닝 통합

### 2.2 maplib – OTTR 템플릿 기반 고성능 KG 생성 + SHACL 검증

**목적**: 템플릿 기반 KG 생성 및 검증

**실험 아이디어**:
- 표 유형별 OTTR 템플릿 정의
- 템플릿 기반 자동 KG 생성
- SHACL 검증 통합

**구현 예시**:
```python
# maplib은 OTTR 템플릿 사용
# 템플릿 정의 후 자동 생성

from maplib import MapLib

maplib = MapLib()
kg = maplib.generate_from_template("table_template.ottr", table_data)
validation = maplib.validate_shacl(kg, "table_shapes.ttl")
```

**기대 효과**:
- 표준화된 KG 생성
- 재사용 가능한 템플릿
- 자동 검증

### 2.3 kgforge / Nexus Forge, ExeKGLib (옵션)

**목적**: 엔터프라이즈급 KG 관리

**실험 아이디어**:
- 분산 KG 저장소 구축
- 버전 관리 및 협업
- API 기반 접근

## 3. Graph DB + LLM 기반 파이프라인

### 3.1 Neo4j GraphRAG Python – SimpleKGPipeline

**목적**: LLM 기반 텍스트→KG 변환 및 RAG

**실험 아이디어**:

#### 3.1.1 텍스트 기반 표 설명 → KG 변환
- 표의 텍스트 설명을 LLM이 KG로 변환
- 기존 추출 방법과 비교
- 품질 평가

**구현 예시**:
```python
from neo4j_graphrag import SimpleKGPipeline

pipeline = SimpleKGPipeline(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# 텍스트에서 KG 생성
kg = pipeline.text_to_kg(table_description)
```

#### 3.1.2 GraphRAG 질의응답
- KG 기반 질의응답 시스템
- 표 데이터에 대한 자연어 질의
- 정확도 평가

**구현 예시**:
```python
# KG가 Neo4j에 로드된 후
answer = pipeline.query("2024년 매출이 가장 높은 회사는?")
```

**기대 효과**:
- 자연어 기반 KG 생성
- 지능형 질의응답
- LLM과 KG의 결합

## 4. 통합 실험 파이프라인 제안

### 4.1 전체 파이프라인 구조

```
표 추출 (PDF)
    ↓
KG 구축 (현재 시스템)
    ↓
RML 변환 (PyRML/Morph-KGC)
    ↓
RDF 저장 (RDFLib/kglab)
    ↓
SHACL 검증 (kglab/maplib)
    ↓
Neo4j 로드
    ↓
GraphRAG 질의응답
```

### 4.2 단계별 실험 계획

#### Phase 1: RML 변환 (1-2주)
- PyRML로 기본 변환 구현
- Morph-KGC로 성능 테스트
- 변환 품질 평가

#### Phase 2: SHACL 검증 (1주)
- kglab으로 SHACL 셰이프 정의
- 검증 자동화
- 리포트 생성

#### Phase 3: Neo4j GraphRAG (2주)
- Neo4j에 데이터 로드
- GraphRAG 파이프라인 구축
- 질의응답 테스트

#### Phase 4: 통합 및 최적화 (1주)
- 전체 파이프라인 통합
- 성능 최적화
- 문서화

## 5. 구체적인 실험 스크립트

### 5.1 RML 변환 실험
```bash
# scripts/experiment_rml_conversion.py
python scripts/experiment_rml_conversion.py \
    --engine pyrml \
    --input visualizations/rml/table_runs.json \
    --output outputs/rdf/pyrml_output.ttl
```

### 5.2 SHACL 검증 실험
```bash
# scripts/experiment_shacl_validation.py
python scripts/experiment_shacl_validation.py \
    --rdf visualizations/rdf/table_kg.ttl \
    --shapes schemas/table_shapes.ttl \
    --output reports/shacl_validation.json
```

### 5.3 GraphRAG 실험
```bash
# scripts/experiment_graphrag.py
python scripts/experiment_graphrag.py \
    --neo4j-uri bolt://localhost:7687 \
    --queries sample_queries.json \
    --output reports/graphrag_results.json
```

## 6. 평가 지표

### 6.1 변환 품질
- 트리플 수 비교
- 엔티티/관계 커버리지
- RDF 표준 준수도

### 6.2 성능
- 변환 시간
- 메모리 사용량
- 처리량 (표/초)

### 6.3 검증 품질
- SHACL 위반 수
- 검증 시간
- 오류 분류

### 6.4 질의응답 품질
- 정확도 (Accuracy)
- 응답 시간
- 사용자 만족도

## 7. 참고 자료

- **PyRML**: https://github.com/RMLio/pyrml
- **Morph-KGC**: https://github.com/oeg-upm/morph-kgc
- **SDM-RDFizer**: https://github.com/SDM-TIB/SDM-RDFizer
- **kglab**: https://github.com/DerwenAI/kglab
- **maplib**: https://github.com/INCATools/maplib
- **Neo4j GraphRAG**: https://github.com/neo4j-labs/graphrag

## 8. 다음 단계

1. **즉시 시작 가능한 실험**:
   - PyRML로 RML 변환 구현
   - kglab으로 SHACL 검증 구현
   - Neo4j GraphRAG 기본 설정

2. **중기 실험**:
   - Morph-KGC 성능 비교
   - GNN 기반 패턴 분석
   - 통합 파이프라인 구축

3. **장기 실험**:
   - 엔터프라이즈 배포
   - 대규모 데이터셋 테스트
   - 프로덕션 환경 최적화


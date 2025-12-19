# HierTable-RAG

**Hierarchical Table-based Retrieval Augmented Generation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HierTable-RAG는 **계층적 테이블 구조**를 인식하고 이를 활용하여 정확한 테이블 QA를 수행하는 연구 프로젝트입니다.

## 🎯 연구 목표

기존 Table RAG 시스템의 한계를 극복:

| 문제점 | 기존 방식 | HierTable-RAG |
|--------|----------|---------------|
| 계층 구조 손실 | 테이블을 평면 텍스트로 변환 | **계층적 헤더 트리 보존** |
| 병합 셀 무시 | rowspan/colspan 정보 손실 | **병합 셀 의미 해석** |
| 비효율적 검색 | 전체 테이블 검색 | **Multi-Granularity 검색** |
| 구조 무인식 | 단순 유사도 검색 | **Structure-Aware Retrieval** |

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      HierTable-RAG Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ Table Image │───▶│     GraphTSR     │───▶│ Hetero Graph  │  │
│  │   or HTML   │    │ (Structure Rec.) │    │(Tab-Cell-Text)│  │
│  └─────────────┘    └──────────────────┘    └───────┬───────┘  │
│                                                     │          │
│                                  ┌──────────────────┘          │
│                                  │                             │
│                                  ▼                             │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │ Adaptive Router │───▶│   GNN Indexer    │                   │
│  │ (Query Classif.)│    │   (GraphSAGE)    │                   │
│  └─────────────────┘    └────────┬─────────┘                   │
│                                  │                             │
│                                  ▼                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │  ContextBundle  │───▶│ StructuredPrompt │───▶│  Answer   │  │
│  │ (cells + paths) │    │    Builder       │    │ + Citations│  │
│  └─────────────────┘    └──────────────────┘    └───────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ 주요 기능

### 1. GraphTSR (Graph-based Table Structure Recognition)
- **Cell Detection**: Faster R-CNN을 이용한 셀 영역 탐지
- **Edge Classification**: 셀 간의 관계(Row/Col/Parent)를 GNN으로 분류
- **Hierarchy Building**: Parent-Child 관계를 기반으로 계층 트리 구축

### 2. GNN-based Structure-Aware Indexing
- **Heterogeneous Graph**: Table-Cell-Text 이종 그래프 구축
- **GraphSAGE**: 이웃 셀 정보를 집계하여 구조 인식 임베딩 생성
- **Hierarchy Bonus**: 검색 시 계층 구조(헤더-데이터)를 반영하여 정확도 향상

### 3. Adaptive Query Routing
- **Exact Value**: 특정 셀 값 조회 → Cell Level 검색
- **Header Match**: 스키마/헤더 질문 → Text/Triple Level 검색
- **Aggregation**: 집계/요약 질문 → Table Level 검색

### 4. End-to-End Hierarchical Table Understanding
- **Vision + Semantic 통합 모델**
- **Structural Attention Bias**: 행/열 관계 기반 학습 가능한 Attention 편향
- **Retrieval-Aware Parsing**: 검색 최적화 구조 학습 (Contrastive Loss)

### 5. Structure-Aware RAG Pipeline
- 계층 경로 포함 프롬프트 생성
- 셀 좌표 기반 인용 (Citation)
- Chain-of-Thought 추론 지원

## 📊 실험 결과

### Baseline 비교 (HiTab Dataset, 20 QA pairs)

| 시스템 | Exact Match | F1 Score | 개선율 |
|--------|-------------|----------|--------|
| **HierTable-RAG** | **85.0%** | **85.0%** | - |
| TableRAG | 70.0% | 70.0% | +21.4% |
| FlatRAG | 65.0% | 65.0% | +30.8% |
| DirectLLM | 60.0% | 60.0% | +41.7% |

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/Jax0303/t1.git
cd t1

# 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 기본 사용법

```python
from src.hiertable_rag.core.rag import HierTableRAG

# 시스템 초기화
rag = HierTableRAG()

# 1. 테이블 추출 및 인덱싱
tables = rag.extract_tables("report.pdf")
processed_tables = rag.process_tables(tables)
rag.build_index(processed_tables)

# 2. 질의 수행 (Adaptive Routing 자동 적용)
result = rag.query("2023년 Q1 매출은 얼마인가요?")

# 3. 응답 생성
response = rag.generate_response("2023년 Q1 매출은 얼마인가요?", result)
print(response)
```

### E2E 모델 사용 (Vision 기반)

```python
from src.models.integration import E2ETableRAGPipeline

# 파이프라인 초기화
pipeline = E2ETableRAGPipeline(
    llm_provider="openai",
    llm_model="gpt-4",
    device="cuda"  # or "cpu"
)

# 이미지에서 직접 QA
result = pipeline.process_and_answer(
    image_path="table.png",
    question="2023년 Q1 매출은 얼마인가요?"
)

print(result["answer"].answer_text)
```

## 📁 프로젝트 구조

```
HierTable-RAG/
├── src/
│   ├── hiertable_rag/     # Core RAG Logic
│   │   └── core/
│   │       ├── rag.py         # Main Pipeline
│   │       ├── indexer.py     # Multi-Granularity Indexer
│   │       ├── gnn_indexer.py # GNN-based Indexer
│   │       ├── router.py      # Adaptive Router
│   │       └── graph.py       # Knowledge Graph Builder
│   │
│   ├── models/            # Deep Learning Models
│   │   ├── vision_encoder.py  # Multi-res Vision Encoder
│   │   ├── graph_tsr.py       # Graph-based TSR
│   │   └── e2e_hierarchical.py # E2E Model
│   │
│   ├── parsing/           # Legacy Parsing Utils
│   ├── retrieval/         # Legacy Retrieval Utils
│   ├── generation/        # Prompting & Generation
│   └── evaluation/        # Metrics & Benchmarks
│
├── experiments/           # Experiment Results
├── scripts/               # Training & Running Scripts
├── data/                  # Datasets
└── outputs/               # Artifacts
```

## 🔬 핵심 알고리즘

### 1. Hierarchical Header Tree Extraction

```python
# colspan 기반 부모-자식 관계 결정
def _find_parent_node(self, node, parent_level_nodes):
    for parent in parent_level_nodes:
        # 부모의 열 범위가 자식을 포함하면 연결
        if parent.col_start <= node.col_start and \
           parent.col_end >= node.col_end:
            return parent
```

### 2. Contrastive Learning for Hierarchy

```python
# 같은 부모 아래 헤더들은 유사한 임베딩
def compute_contrastive_loss(self, embeddings, parent_labels):
    # InfoNCE Loss
    positive_mask = parent_labels.unsqueeze(0) == parent_labels.unsqueeze(1)
    loss = -log(exp(pos_sim) / exp(all_sim))
```

### 3. Adaptive Retrieval Strategy

```python
def _retrieve_adaptive(self, query, top_k):
    query_type = self.classify_query_type(query)
    
    if query_type == QueryType.LOOKUP:
        return self._retrieve_cell_direct(query, top_k)
    elif query_type == QueryType.AGGREGATE:
        return self._retrieve_at_granularity(query, "row", top_k)
```

## 📈 향후 계획

- [ ] 더 큰 벤치마크 데이터셋 평가 (WTQ, SQA)
- [ ] Vision 모델 사전 학습 가중치 공개
- [ ] 한국어 테이블 특화 모델
- [ ] 실시간 테이블 QA 데모

## 📚 참고 문헌

- HiTab: A Hierarchical Table Dataset (ACL 2022)
- TableRAG: Million-Token Table Understanding (NeurIPS 2024)
- DETR: End-to-End Object Detection with Transformers
- LayoutLMv3: Pre-training for Document AI

## 📄 라이선스

MIT License



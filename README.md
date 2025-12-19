# Agentic-TableRAG

**Self-Evolving Agentic Retrieval-Augmented Generation for Hierarchical Tables**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Agentic-TableRAG는 **지능형 플래너(Agentic Planner)**를 통해 테이블의 계층 구조를 동적으로 파악하고, 최적의 검색 입도(Granularity)를 결정하여 정확한 답변과 인용(Citation)을 제공하는 연구 프로젝트입니다.

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

# 1. Agentic RAG 시스템 초기화 (Planner 타입 설정 가능)
rag = HierTableRAG(planner_type="rule_based") # or "trainable"

# 2. 데이터 준비 (Hierarchical Table Data)
table_data = {
    "id": "t1",
    "headers": [...],
    "cells": [...]
}

# 3. 인덱싱 (Table, Schema, Cell level 동시 인덱싱)
rag.ingest_table(table_data)

# 4. 에이전틱 질의 수행
# Planner가 질문을 분석하여 필요한 검색 레벨(Table/Schema/Cell)을 선택합니다.
result = rag.query("What is the Q1 revenue for North America?")

# 5. 근거 기반 응답 확인
print(f"Answer: {result['response']['answer']}")
print(f"Cited Evidence: {result['response']['cited_cell_ids']}")
```

### 실험 실행

```bash
# HiTab 데이터셋 대상 실험 실행
python scripts/agentic_table_rag_experiment.py --dataset hitab --input data/sample_dataset.json
```

## 📁 프로젝트 구조

```
HierTable-RAG/
├── src/
│   ├── hiertable_rag/     # Core Agentic RAG Logic
│   │   ├── core/          # RAG Pipeline, Indexer, Planner, Evaluator
│   │   ├── extractors/    # PDF/Table Extractors
│   │   ├── generators/    # Citation-grounded Generators
│   │   ├── retrievers/    # Multi-Granularity Retrievers
│   │   └── processors/    # Table Structure Processors
│   │
│   ├── models/            # Deep Learning Models (Structure Recognition)
│   │   ├── vision_encoder.py
│   │   ├── graph_tsr.py
│   │   └── e2e_hierarchical.py
│   │
│   └── utils/             # Common Utilities
│
├── scripts/               # Experiment & Training Scripts
├── data/                  # Datasets (HiTab, etc.)
├── experiments/           # Experiment Results & Logs
├── outputs/               # Generated Artifacts
└── verify_agentic_rag.py  # Core Pipeline Verification Script
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



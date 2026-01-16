# Agentic-TableRAG

**Self-Evolving Agentic Retrieval-Augmented Generation for Hierarchical Tables**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Agentic-TableRAG는 **지능형 플래너(Agentic Planner)**와 **TTA 기반 적응적 OCR 전략**을 통해 테이블의 계층 구조를 보존하고, 최적의 검색 입도를 결정하여 정확한 답변을 제공하는 연구 프로젝트입니다.

## 🎯 연구 목표

기존 Table RAG 시스템의 한계를 극복:

| 문제점 | 기존 방식 | Agentic-TableRAG |
|--------|----------|---------------|
| 계층 구조 손실 | 테이블을 평면 텍스트로 변환 | **계층적 헤더 트리 보존** |
| OCR 노이즈 | 고정된 OCR 파싱 (14% 성능 저하) | **TTA 기반 적응적 파싱 전략** |
| 비효율적 검색 | 전체 테이블 검색 | **Multi-Granularity 검색** |
| 불확실성 무시 | 결과의 신뢰도 판단 불가 | **TTA 기반 불확실성 정량화** |

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      Agentic-TableRAG Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  Table Image   │───▶│  Adaptive Router │───▶│ Optimal Parser │  │
│      or PDF    │    │ (TTA Confidence) │    │ (OCR/TSR/VLM) │  │
│  └─────────────┘    └────────┬─────────┘    └───────┬───────┘  │
│                              │                      │          │
│                              ▼                      ▼          │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐│
│  │  Agentic Planner│◀──▶│  Graph Indexer   │◀──▶│ Vector DB   ││
│  │ (Query Routing) │    │  (Table Graphs)  │    │ (Retrieval) ││
│  └─────────────────┘    └────────┬─────────┘    └─────────────┘│
│                                  │                             │
│                                  ▼                             │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │  ContextBundle  │───▶│  Answer Gen.     │───▶│  Answer   │  │
│  │ (cells + paths) │    │  (w/ Citation)   │    │ + Citations│  │
│  └─────────────────┘    └──────────────────┘    └───────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ 주요 기능

### 1. TTA-based Adaptive OCR Parsing (NEW)
- **TTA Uncertainty Quantification**: Test-Time Augmentation의 일관성을 분석하여 파싱 신뢰도 측정.
- **3-Tier Adaptive Routing**:
    - **High Confidence**: 고성능 Basic OCR (효율성)
    - **Mid Confidence**: OCR + Structural TSR (구조 보존)
    - **Low Confidence**: Vision-LLM Fallback (정확도)

### 2. Graph-based Table Structure Recognition
- **Hierarchy Building**: Parent-Child 관계를 기반으로 계층 트리 구축 및 보존.
- **Structure-Aware Indexing**: 테이블의 물리적 위치뿐만 아니라 논리적 계층 구조를 인덱싱에 반영.

### 3. Agentic Query Planning
- 질문의 의도에 따라 검색 입도(Table/Schema/Cell)를 동적으로 선택.
- **Citation Retrieval**: 생성된 답변의 근거가 되는 셀 좌표를 정확히 인용.

## 📊 실험 결과

### 도메인별 적응적 전략 성능 개선 (TEDS Accuracy)

| 도메인 | 기본 OCR | 적응적 전략 (TTA) | 개선율 |
|--------|----------|-------------------|--------|
| **금융 (Financial)** | 0.6179 | **0.8368** | **+35.4%** |
| **학술 (Academic)** | 0.7096 | **0.8684** | **+22.4%** |
| **일반 (General)** | 0.8745 | **0.8745** | **0.0% (효율 보존)** |

> [!NOTE]
> 적응적 전략을 통해 복잡한 표(금융/학술)에서 발생하는 OCR 노이즈 문제를 획기적으로 해결하였으며, 단순한 표에서는 기존의 효율적인 OCR 방식을 유지하여 연산 비용을 최적화하였습니다.

## 🚀 빠른 시작

### 설치

```bash
git clone https://github.com/Jax0303/t1.git
cd t1
pip install -r requirements.txt
```

### 실행 및 검증

```bash
# TTA 기반 적응적 OCR 성능 평가
export PYTHONPATH=$PYTHONPATH:.
python scripts/eval_adaptive_ocr.py
```

## 📁 프로젝트 구조

```
Agentic-TableRAG/
├── src/
│   ├── hiertable_rag/     # 핵심 로직
│   │   ├── core/          # Uncertainty Estimator, Planner, RAG Pipeline
│   │   ├── extraction/    # Adaptive Extractor, TSR/VLM Baselines
│   │   ├── retrievers/    # Multi-Granularity Retrievers
│   │   └── generators/    # Citation-grounded Generators
│   └── models/            # 테이블 구조 인식 모델 (Swin, GraphTSR 등)
├── scripts/               # 평가 및 실험 스크립트
└── README.md
```

## 📄 라이선스

MIT License

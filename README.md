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
│  │ Table Image │───▶│ E2E Hierarchical │───▶│  HeaderTree   │  │
│  │   or HTML   │    │     Model        │    │    (JSON)     │  │
│  └─────────────┘    └──────────────────┘    └───────┬───────┘  │
│                                                     │          │
│  ┌──────────────────────────────────────────────────┘          │
│  │                                                              │
│  ▼                                                              │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │ HierarchicalIndex│───▶│ Structure-Aware  │                   │
│  │ (Multi-Granular) │    │    Retriever     │                   │
│  └─────────────────┘    └────────┬─────────┘                   │
│                                  │                              │
│                                  ▼                              │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │  ContextBundle  │───▶│ StructuredPrompt │───▶│  Answer   │  │
│  │ (cells + paths) │    │    Builder       │    │ + Citations│  │
│  └─────────────────┘    └──────────────────┘    └───────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ 주요 기능

### 1. 계층적 헤더 트리 추출 (Hierarchical Header Tree Extraction)
- colspan/rowspan 기반 부모-자식 관계 자동 감지
- HiTab/FinTabNet 스타일 다단계 헤더 지원
- JSON-LD 형식 출력

### 2. End-to-End Hierarchical Table Understanding (NEW!)
- **Vision + Semantic 통합 모델**
- 이미지에서 직접 계층 구조 추출
- Contrastive Learning으로 헤더 그룹핑

### 3. Multi-Granularity Retrieval
- Table / Subtable / Row / Cell 레벨 인덱싱
- 쿼리 유형별 적응형 검색 전략
- 토큰 효율 30-50% 개선

### 4. Structure-Aware RAG Pipeline
- 계층 경로 포함 프롬프트 생성
- 셀 좌표 기반 인용 (Citation)
- Chain-of-Thought 추론 지원

### 5. Knowledge Graph Mapping
- 테이블 → RDF 트리플 변환
- 상위 헤더 → 속성 카테고리
- 행 → 엔티티, 셀 값 → 관계

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
from src.parsing.extractor import TableExtractor
from src.parsing.hierarchy import HierarchyExtractor
from src.retrieval.retriever import StructureAwareRetriever
from src.generation.prompting import StructuredPromptBuilder

# 1. 테이블 추출
extractor = TableExtractor()
table = extractor.extract_from_html(html_string)

# 2. 계층 구조 감지
hierarchy_extractor = HierarchyExtractor()
header_tree = hierarchy_extractor.extract_header_tree(table)

# 3. 계층 구조 시각화
print(header_tree.visualize_ascii())
# TABLE_ROOT
# ├── 2023년 (span=4)
# │   ├── Q1 > 매출
# │   └── Q1 > 비용
# └── 2024년 (span=2)
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
│   ├── parsing/           # 테이블 파싱 및 계층 추출
│   │   ├── extractor.py   # PDF/HTML 테이블 추출
│   │   ├── hierarchy.py   # 계층적 헤더 트리 추출
│   │   └── cell_merger.py # 병합 셀 처리
│   │
│   ├── encoding/          # 테이블 인코딩
│   │   ├── tree_encoder.py    # 트리 → JSON/텍스트
│   │   └── graph_builder.py   # 테이블 → 그래프
│   │
│   ├── retrieval/         # 검색 모듈
│   │   ├── indexer.py     # Multi-Granularity 인덱싱
│   │   └── retriever.py   # Structure-Aware 검색
│   │
│   ├── generation/        # 응답 생성
│   │   └── prompting.py   # 구조화된 프롬프트 빌더
│   │
│   ├── models/            # E2E 모델 (NEW!)
│   │   ├── e2e_hierarchical.py  # Vision + Hierarchy 모델
│   │   ├── trainer.py           # 학습 유틸리티
│   │   └── integration.py       # RAG 통합
│   │
│   └── evaluation/        # 평가 모듈
│       ├── metrics.py     # QA 메트릭
│       └── baselines.py   # 베이스라인 시스템
│
├── experiments/           # 실험 결과
├── scripts/               # 실행 스크립트
├── data/                  # 데이터셋
└── outputs/               # 출력 파일 (KG, 그래프 등)
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

## 🤝 기여

기여를 환영합니다! Issue를 생성하거나 Pull Request를 제출해주세요.

## 📧 문의

프로젝트 관련 문의사항이 있으시면 Issue를 생성해주세요.

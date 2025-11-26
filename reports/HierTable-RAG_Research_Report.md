# HierTable-RAG: Hierarchy-Preserving Table Parsing and Retrieval Framework

## 복잡 구조 테이블을 위한 계층 보존형 Table-RAG 프레임워크 연구 보고서

---

## 목차

1. [연구 개요](#1-연구-개요)
2. [관련 연구](#2-관련-연구)
3. [제안 프레임워크: HierTable-RAG](#3-제안-프레임워크-hiertable-rag)
4. [실험 설계](#4-실험-설계)
5. [예상 결과 및 기여](#5-예상-결과-및-기여)
6. [타겟 학회/저널](#6-타겟-학회저널)
7. [연구 일정](#7-연구-일정-timeline)
8. [리스크 및 대응 방안](#8-리스크-및-대응-방안)

---

## 1. 연구 개요

### 1.1 연구 배경 및 필요성

최근 Retrieval-Augmented Generation(RAG)은 대규모 언어모델(LLM)의 지식 한계를 보완하기 위한 핵심 패러다임으로 자리 잡았다. 그러나 실무 문서(재무제표, 공시, 실험 결과 보고서 등)에는 **복잡한 구조의 표(table)**가 다수 포함되며, 이들 표는 다음과 같은 특성을 가진다.

- **다계층 헤더**(hierarchical/multi-level headers)
- **병합 셀**(rowspan/colspan)
- 다양한 레이아웃과 비정형 구조
- 수치·텍스트가 혼합된 셀 내용

기존 RAG 시스템은 이러한 표를 주로 **단순 텍스트로 평탄화(linearization)**하거나 Markdown/JSON 형태로 변환해 사용한다. 이 과정에서:

- 상·하위 헤더 간 **계층 관계**
- **병합 셀의 범위 정보**
- 셀이 속한 **구조적 위치**(어떤 헤더 조합 아래에 있는지)

가 소실되는 문제가 발생한다. 그 결과, 복잡 표에 대한 질의응답(Table QA)에서:

- ❌ 잘못된 열/행을 참조하거나
- ❌ 병합 셀을 오해하여 잘못된 값을 보고하고
- ❌ 구조적 질문("이 표는 어떤 섹션으로 나뉘어 있는가?" 등)에 취약한

한계가 두드러진다.

최근 RealHiTBench, HiTab 등에서는 계층적 헤더를 명시적으로 모델에 제공하면 성능이 향상된다는 결과를 보이고 있으나:

1. **헤더 트리를 자동 추출**하고,
2. 이를 **RAG 파이프라인(인덱싱–검색–프롬프팅)에 엔드투엔드로 통합**하는

종합적인 프레임워크는 아직 부족하다.

**본 연구는 이러한 문제를 해결하기 위해, 계층 구조를 보존하는 테이블 파싱 및 검색 프레임워크인 "HierTable-RAG"를 제안하고, 복잡 구조 표 QA에서의 효과를 정량적으로 검증하고자 한다.**

---

### 1.2 연구 제목

| 언어 | 제목 |
|------|------|
| **영문** | HierTable-RAG: Hierarchy-Preserving Table Parsing and Retrieval Framework for Complex Structured Tables |
| **국문** | HierTable-RAG: 복잡 구조 테이블을 위한 계층 보존형 Table-RAG 프레임워크 |

---

### 1.3 핵심 연구 질문 (Research Questions)

본 연구는 다음 네 가지 연구 질문을 다룬다.

| RQ | 질문 | 검증 방법 |
|----|------|----------|
| **RQ1** | 계층적 헤더 구조를 명시적으로 추출하고 RAG에 통합하면, 복잡한 표 QA 성능이 향상되는가? | HiTab/CHiTab 벤치마크 실험, 기존 Table-RAG와 비교 |
| **RQ2** | 어떤 구조 표현 방식(트리 JSON, 자연어, 그래프)이 LLM의 표 이해에 가장 효과적인가? | Ablation Study (표현 방식별 비교) |
| **RQ3** | Multi-granularity retrieval이 단일 granularity 대비 컨텍스트 효율성과 정확도를 동시에 개선하는가? | Granularity별 Retrieval 실험 |
| **RQ4** | 병합 셀 정보를 명시적으로 제공하면 셀 참조 오류(hallucination)가 감소하는가? | Cell Citation Accuracy, Hallucination Rate 분석 |

---

### 1.4 핵심 가설 (Hypotheses)

| 가설 | 내용 |
|------|------|
| **H1** | 계층적 헤더 트리를 LLM 프롬프트에 명시적으로 주입하면, 평탄화된(Flat) 표 대비 **Exact Match(EM) 점수가 약 10pp 내외 향상**된다. |
| **H2** | Multi-granularity retrieval(테이블 → 서브테이블 → 행/셀)은 단일 레벨 retrieval 대비 **컨텍스트 토큰 사용량을 30–50% 절감**하면서, EM/F1 기준 동등 이상의 정확도를 유지한다. |
| **H3** | 병합 셀 좌표 및 범위를 명시적으로 제공하면 **셀 참조 hallucination rate가 50% 이상 감소**한다. |

---

### 1.5 연구 요약 (한 문장)

> **연구 주제**: 계층적 헤더/병합 셀 구조를 보존해서 인덱싱·검색·프롬프팅까지 반영하는 Hierarchy-Preserving Table-RAG 프레임워크를 만들고, 복잡한 표 QA에서 flat-RAG 대비 성능·효율·구조 이해도를 동시에 끌어올리는지 평가하는 연구.

> **예상 결과**: 기존 flat-RAG 대비 EM/F1은 두 자릿수(pp) 수준으로 향상되고, multi-granularity retrieval로 컨텍스트 토큰은 30~50% 절감되며, 제안한 HPS·Cell Citation Accuracy 지표 기준으로도 구조 이해도가 유의미하게 높아진다.

> **기대 효과**: 복잡한 재무/공시/엔터프라이즈 표를 다루는 RAG 시스템에서, "표를 그냥 텍스트로 펴버리는" 방식의 한계를 넘어서 계층 구조를 엔드투엔드로 활용하는 레퍼런스 아키텍처 + 평가 지표를 제시한다.

---

## 2. 관련 연구

### 2.1 Table QA 및 Table-RAG

| 접근법 | 설명 | 한계 |
|--------|------|------|
| **Flat-RAG** | 표를 단순 텍스트/Markdown으로 직렬화하여 전체를 인덱싱 후 검색 | 헤더 계층과 병합 셀 정보 소실, 복잡 표에서 오해석/오참조 빈번 |
| **TableRAG 계열** | 스키마(열 이름) + 셀을 함께 인덱싱하여 "헤더 + 셀" 단위로 검색 | 헤더 간 계층 구조는 부분적으로만 활용, multi-level/multi-span header 표현력 제한 |
| **T-RAG / Hierarchical Memory** | 문서 간/섹션 간 계층 구조를 반영한 hierarchical index 사용 | 주로 문단·문서 수준에 초점, 테이블 내부의 계층적 헤더 구조까지 세밀하게 반영하는 사례 드묾 |

### 2.2 계층 헤더 및 복잡 테이블 인식

- **HiTab, CHiTab, RealHiTBench** 등은:
  - 계층적 헤더를 갖는 테이블
  - 다양한 복잡도 유형(merged cells, multi-header, multi-format)을 포함한 벤치마크 제공
  - 헤더 트리를 명시적으로 모델에 제공할 경우 성능 향상 보고

- **한계**:
  - 대부분 테이블 구조 인식/표 이해 모델 자체의 성능 평가에 초점
  - RAG 파이프라인(인덱싱–retrieval–프롬프팅)에 구조 정보를 어떻게 녹일 것인지에 대한 설계는 상대적으로 부족

### 2.3 본 연구의 차별점

| 차별점 | 설명 |
|--------|------|
| **계층 구조 end-to-end 통합** | 헤더 트리 추출 → 구조 인코딩 → 계층 인식 검색 → 구조 주입 프롬프팅까지 전체 파이프라인을 체계적으로 설계 |
| **Multi-granularity Table-RAG** | Table/Subtable/Row/Cell 수준의 다중 granular index 정의, 질문 유형에 따라 적응적으로 granularity를 선택하는 retrieval 알고리즘 제안 |
| **구조 보존 평가 지표 제안** | 기존 EM/F1만으로는 측정하기 어려운 계층 이해도(HPS), Cell Citation Accuracy 등 구조 보존 중심의 새로운 메트릭 제안 |

---

## 3. 제안 프레임워크: HierTable-RAG

### 3.1 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HierTable-RAG Framework                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Stage 1    │    │   Stage 2    │    │   Stage 3    │    │  Stage 4  │ │
│  │   Structure  │───▶│   Structure  │───▶│  Hierarchy   │───▶│ Structure │ │
│  │   Extraction │    │   Encoding   │    │   Aware      │    │  Injected │ │
│  │              │    │              │    │   Retrieval  │    │ Prompting │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                  │                   │                   │       │
│         ▼                  ▼                   ▼                   ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ • 헤더 트리   │    │ • Tree JSON  │    │ • Query 분류 │    │ • 구조    │ │
│  │ • 병합 셀    │    │ • 그래프 표현 │    │ • 적응형     │    │   프롬프트│ │
│  │ • 셀 좌표    │    │ • 임베딩     │    │   Granularity│    │ • LLM     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Stage | 이름 | 목표 |
|-------|------|------|
| **Stage 1** | Hierarchical Structure Extraction | 복잡한 표에서 계층적 헤더 트리 + 병합 셀 좌표 + 정규화된 셀 그리드를 손실 없이 추출 |
| **Stage 2** | Structure-Aware Encoding | 추출된 구조를 LLM 및 벡터 검색이 활용 가능한 표현으로 인코딩하고, multi-granularity 인덱스 생성 |
| **Stage 3** | Hierarchy-Aware Retrieval | 질의 유형에 따라 적절한 granularity를 선택하고, 계층 컨텍스트를 확장해 컨텍스트 번들 구성 |
| **Stage 4** | Structure-Injected Prompting | 헤더 계층, 병합 정보, 셀 좌표를 LLM 프롬프트에 명시적으로 주입해 구조 인식 추론 유도 |

---

### 3.2 Stage 1: Hierarchical Structure Extraction

#### 3.2.1 입력과 출력

| 구분 | 내용 |
|------|------|
| **입력** | HTML/PDF 테이블, DataFrame (예: LGPMA 기반 테이블 파싱 결과) |
| **출력** | `HeaderTree` (계층적 헤더 구조 JSON), `CellMap` (병합 셀 좌표 매핑), `NormalizedTable` (정규화된 셀 그리드) |

#### 3.2.2 헤더 트리 추출 알고리즘

```
Algorithm 1: ExtractHeaderTree(table)
─────────────────────────────────────

Input: table T with cells C[i,j]
Output: HeaderTree H

1. header_rows ← DetectHeaderRows(T)  // rowspan/colspan 패턴 및 스타일 기반 탐지
2. H.root ← CreateNode("ROOT")
3. for level = 0 to max_header_level:
4.     for each cell c in header_rows[level]:
5.         node ← CreateNode(c.text, c.colspan, c.rowspan)
6.         parent ← FindParentByOverlap(H, c.col_range, level-1)
7.         parent.children.append(node)
8.         node.col_range ← c.col_range
9. return H
```

**핵심 구현 포인트:**

- **헤더 행 탐지**: `<th>` 태그, rowspan/colspan, 굵기/배경색 등 스타일 정보를 이용해 헤더 행 구간 추출
- **부모–자식 관계 추론**: 상위 레벨 헤더의 col_range와 하위 셀의 col_range 겹침 여부를 통해 계층 구조 결정
- **병합 셀 좌표 시스템**: 각 셀을 `(row_start, row_end, col_start, col_end)`로 표현

> **구현 참고**: Stage 1은 parser-agnostic 설계지만, 구현은 **LGPMA**를 기반으로 한다. LGPMA output에서 `(row, col, rowspan, colspan)`를 가져와 Algorithm 1의 HeaderTree를 실제로 구성한다.

---

### 3.3 Stage 2: Structure-Aware Encoding

#### 3.3.1 헤더 트리 직렬화 방식

| 방식 | 예시 | 용도 |
|------|------|------|
| **Indented Text** | `Revenue\n - Q1\n - Q2` | 프롬프트에 직접 삽입 |
| **Tree JSON** | `{"@type": "Header", "label": "Revenue", "children": [...]}` | 구조 저장/전달, 그래프 변환 |
| **Natural Language** | `"Revenue 아래에 Q1, Q2가 있음"` | Few-shot 예시, 설명형 프롬프트 |

> 어느 방식이 LLM에게 가장 효과적인지는 **RQ2의 Ablation Study**를 통해 비교한다.

#### 3.3.2 Multi-Granularity 인덱싱

```
┌─────────────────────────────────────────────────────────┐
│                  Hierarchical Index                     │
├─────────────────────────────────────────────────────────┤
│  Level 0: Table-level                                   │
│    └── "Financial Report 2024, 4 sections, 120 cells"   │
│                                                         │
│  Level 1: SubTable-level (by top-level headers)         │
│    ├── "Revenue section: Q1-Q4, 30 cells"               │
│    └── "Expenses section: Q1-Q4, 30 cells"              │
│                                                         │
│  Level 2: Row-level                                     │
│    ├── "Row 3: Product A, [100, 120, 90, 110]"          │
│    └── "Row 4: Product B, [80, 95, 85, 100]"            │
│                                                         │
│  Level 3: Cell-level                                    │
│    └── "Cell(3,2): 120, Header: Revenue > Q2"           │
└─────────────────────────────────────────────────────────┘
```

**각 엔트리 메타데이터:**
```json
{
  "table_id": "...",
  "hierarchy_path": ["ROOT", "Revenue", "Q2"],
  "coordinates": [3, 2],
  "parent_ids": ["..."],
  "child_ids": ["..."]
}
```

---

### 3.4 Stage 3: Hierarchy-Aware Retrieval

#### 3.4.1 질의 유형 분류 (Query Type Classification)

| 유형 | 예시 질문 | 최적 Granularity |
|------|----------|------------------|
| **LOOKUP** | "2024년 Q2 매출은?" | Cell-level |
| **AGGREGATE** | "전체 연간 매출 합계는?" | Row/SubTable-level |
| **COMPARISON** | "Q1 vs Q2 매출 비교" | Row-level |
| **STRUCTURAL** | "이 표에 몇 개의 열이 있나?" | Table-level |
| **MULTI-HOP** | "매출 성장률이 가장 높은 분기의 비용은?" | Adaptive (복수 레벨 탐색) |

#### 3.4.2 계층 인식 Retrieval 알고리즘

```
Algorithm 2: HierarchyAwareRetrieve(query, index)
─────────────────────────────────────────────────

Input: query Q, HierarchicalIndex I
Output: ContextBundle with retrieved items + hierarchy

1.  query_type ← ClassifyQueryType(Q)
2.  
3.  if query_type == LOOKUP:
4.      candidates ← I.cell_index.search(Q, top_k=10)
5.  elif query_type == AGGREGATE:
6.      candidates ← I.subtable_index.search(Q, top_k=5)
7.  elif query_type == STRUCTURAL:
8.      candidates ← I.table_index.search(Q, top_k=3)
9.  else:  // MULTI-HOP or unknown
10.     candidates ← AdaptiveSearch(Q, I)  // 여러 레벨 탐색
11.
12. // 계층 컨텍스트 확장
13. for each candidate c in candidates:
14.     c.hierarchy_path ← GetAncestorHeaders(c, I)
15.     c.siblings ← GetSiblingCells(c, I, hops=1)
16.
17. return ContextBundle(candidates, hierarchy_info)
```

**계층 컨텍스트 확장:**
- **계층 경로(ancestor headers)**: 예: `Table > Revenue > Q2 > Online`
- **Sibling 컨텍스트**: 같은 행/열에 위치한 주변 셀들을 함께 제공

---

### 3.5 Stage 4: Structure-Injected Prompting

#### 3.5.1 기본 프롬프트 템플릿

```markdown
## 테이블 구조 정보

### 헤더 계층 (Header Hierarchy)
- Level 0: "Financial Summary 2024" (전체 테이블 제목)
  - Level 1: "Revenue" (열 0-3)
    - Level 2: "Q1" (열 0), "Q2" (열 1), "Q3" (열 2), "Q4" (열 3)
  - Level 1: "Expenses" (열 4-7)
    - Level 2: "Q1" (열 4), "Q2" (열 5), "Q3" (열 6), "Q4" (열 7)

### 병합 셀 정보
- (1, 0-3): "Revenue" - 열 0~3에 걸쳐 병합됨
- (1, 4-7): "Expenses" - 열 4~7에 걸쳐 병합됨

### 검색된 관련 데이터
| 행        | Revenue Q1 | Revenue Q2 | Revenue Q3 | Revenue Q4 |
|----------|------------|------------|------------|------------|
| Product A| 100        | 120        | 90         | 110        |
| Product B| 80         | 95         | 85         | 100        |

## 질문
Q2 매출이 가장 높은 제품은 무엇인가요?

## 지시사항
1. 위의 헤더 계층을 참고하여 "Revenue > Q2" 열을 찾으세요.
2. 해당 열의 값들을 비교하세요.
3. 답변 시 셀 좌표를 명시하세요 (예: [행 3, 열 2]).
```

#### 3.5.2 Ablation용 프롬프트 변형

| 변형 | 설명 | 목적 |
|------|------|------|
| **Full Structure** | 계층 + 병합 + 좌표 모두 포함 | Main 실험 조건 |
| **No Hierarchy** | 계층 정보 제외 | H1 (계층 효과) 검증 |
| **No Merge Info** | 병합 정보 제외 | H3 (병합 효과) 검증 |
| **Flat Only** | 단순 Markdown 테이블만 제공 | Baseline (Flat-RAG) |

---

## 4. 실험 설계

### 4.1 데이터셋

| 데이터셋 | 출처 | 크기 | 특징 | 용도 | 우선순위 |
|---------|------|------|------|------|---------|
| **HiTab** | ACL 2022 | 10,672 QA | 계층적 표, 약 98%가 계층 구조 보유 | Main 실험 (RQ1, RQ3) | 🔴 필수 |
| **K-EnterpriseTable** | 자체 구축 | 500+ QA | 한국 공시/재무 테이블 | 도메인 특화 일반화 평가 | 🔴 필수 |
| **CHiTab** | 2025 | 3,610 QA | 중국어, 복잡 계층 구조 | Cross-lingual 검증 | 🟡 선택 |
| **RealHiTBench** | 2025 | 3,752 QA | 다양한 복잡 유형, 다중 포맷 | Robustness 분석 | 🟡 선택 |

---

### 4.2 베이스라인 시스템

| 시스템 | 설명 | 구조 활용 여부 |
|--------|------|---------------|
| **Flat-RAG** | 표를 단순 Markdown/텍스트로 평탄화 후 인덱싱 | ❌ |
| **TableRAG** | 스키마(열 이름) + 셀 retrieval 기반 Table-RAG | 부분적 (헤더 계층X) |
| **T-RAG** | 문서/테이블 간 계층적 메모리 인덱스 사용 | 테이블 간 계층 (내부 헤더 계층X) |
| **E5 Framework** | Zero-shot 계층 분석, 표 구조 추론 | 부분적 |
| **LLM Direct** | 전체 테이블을 그대로 입력하여 Direct QA | ❌ (구조 명시적 활용X) |
| **HierTable-RAG (Ours)** | 계층 구조를 end-to-end로 보존하고 활용 | ✅ |

---

### 4.3 평가 지표

#### 4.3.1 QA 성능 지표

| 지표 | 수식 | 설명 |
|------|------|------|
| **Exact Match (EM)** | $EM = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[pred_i = gold_i]$ | 정확히 일치하는 답변 비율 |
| **F1 Score** | $F1 = \frac{2PR}{P+R}$ | 토큰 수준 Precision/Recall의 조화평균 |
| **Numerical Accuracy** | $NumAcc = \mathbb{1}[\|pred - gold\| < \epsilon]$ | 수치 답변의 근사 정확도 |

#### 4.3.2 구조 보존 지표 (Structure Preservation Metrics)

**1. Structure Recall**

정답을 위해 필요한 계층 헤더 집합 $H_{required}$와 검색 컨텍스트 안에 포함된 헤더 집합 $H_{retrieved}$에 대해:

$$StructureRecall = \frac{\|H_{retrieved} \cap H_{required}\|}{\|H_{required}\|}$$

> **예시**: 정답 셀이 `Revenue > Q2 > Online` 아래에 있을 때, $H_{required} = \{Revenue, Q2, Online\}$. 검색 컨텍스트 안에 이 중 몇 개가 포함되었는지 비율로 평가.

**2. Hierarchy Preservation Score (HPS)**

각 정답 셀에 대해 gold 경로와 pred 경로의 깊이 일치도를 측정:

$$HPS = \frac{1}{N}\sum_{i=1}^{N}depth\_match(h_i^{pred}, h_i^{gold})$$

> **예시**:
> - Gold path: `[ROOT, Revenue, Q2, Online]` (길이 4)
> - Pred path: `[ROOT, Revenue, Q2]` (길이 3)
> - depth_match = 3/4 = **0.75**

**3. Cell Citation Accuracy**

모델이 인용한 셀 좌표 집합 $C_{pred}$와 정답 셀 좌표 집합 $C_{gold}$에 대해:

$$CellCitationAccuracy = \frac{\|C_{pred} \cap C_{gold}\|}{\|C_{pred}\|}$$

> 프롬프트에서 "답변에 사용한 셀 좌표를 함께 제시하라"고 지시한 뒤, 모델이 정확한 셀을 참조했는지 평가.

#### 4.3.3 효율성 지표

| 지표 | 설명 |
|------|------|
| **Context Token Ratio** | 사용된 컨텍스트 토큰 수 / 전체 테이블 토큰 수 |
| **Retrieval Latency** | 검색에 소요된 시간(ms) |
| **Useful Cell Ratio** | 답변에 실제로 활용된 셀 / 검색된 셀 비율 |

---

### 4.4 실험 구성

#### 4.4.1 Core 실험 세트 (필수)

| 실험 | 대상 RQ | 설명 | 데이터 | 메트릭 |
|------|--------|------|--------|--------|
| **실험 1** | RQ1 | HiTab 전체 테스트셋 메인 비교 | HiTab | EM, F1, HPS, Structure Recall |
| **실험 3** | RQ3 | Granularity 전략 비교 (Cell/Row/SubTable/Adaptive) | HiTab | EM, Context Token Ratio, Useful Cell Ratio |

> 이 두 실험이 본 연구의 **코어 실험 세트**이다.

#### 4.4.2 Secondary 실험 세트 (선택/확장)

| 실험 | 대상 RQ | 설명 | 데이터 | 메트릭 |
|------|--------|------|--------|--------|
| **실험 2** | RQ2 | 구조 표현 방식 비교 (Indented/JSON/NL/Graph) | HiTab 서브셋 (~1,000 QA) | EM, F1, HPS |
| **실험 4** | RQ4 | 병합 셀 정보 효과 (With/Without Merge Info) | 병합 셀 포함 테이블 필터링 | Cell Citation Accuracy, Hallucination Rate |
| **실험 5** | - | 구조적 Perturbation 강건성 | HiTab + RealHiTBench | EM 변화율 Δ |

**Perturbation 유형 (실험 5):**
- 헤더 동의어 치환 (예: Revenue → Sales)
- 열 순서 셔플
- 일부 셀 마스킹

---

## 5. 예상 결과 및 기여

### 5.1 예상 실험 결과

| 시스템 | HiTab EM | HiTab F1 | HPS | Context Ratio |
|--------|----------|----------|-----|---------------|
| Flat-RAG | 55% | 62% | 0.30 | 100% |
| TableRAG | 62% | 68% | 0.50 | 45% |
| T-RAG | 64% | 70% | 0.60 | 40% |
| LLM Direct | 68% | 74% | 0.40 | 100% |
| **HierTable-RAG (Ours)** | **~70–73%** | **~76–79%** | **0.85** | **35%** |

**핵심 기대 효과:**
- Flat-RAG 대비 **EM 기준 약 10pp 내외 향상**
- Context Ratio는 약 **35% 수준**으로, multi-granularity retrieval이 토큰 효율성을 크게 개선

---

### 5.2 주요 기여(Contributions)

| # | 기여 | 상세 |
|---|------|------|
| 1 | **계층적 표 구조를 RAG에 완전 통합하는 최초의 종합 프레임워크** | 헤더 트리 추출 → 구조 인코딩 → 계층 인식 검색 → 구조 주입 프롬프팅까지 엔드투엔드 설계. 복잡 구조 테이블을 다루는 실제 RAG 시스템에 적용 가능한 레퍼런스 아키텍처 제시 |
| 2 | **Multi-Granularity Adaptive Retrieval** | Table/SubTable/Row/Cell 레벨의 계층적 인덱스 설계. Query Type에 따른 적응형 granularity 선택 알고리즘 제안. 정확도와 컨텍스트 효율성을 동시에 개선하는 Table-RAG 패턴 제시 |
| 3 | **Structure Preservation Metrics 제안** | Hierarchy Preservation Score(HPS), Structure Recall, Cell Citation Accuracy 등. 기존 QA 메트릭으로는 포착하기 어려웠던 "구조를 얼마나 제대로 이해하고 있는가"를 정량적으로 평가 |
| 4 | **한국어 도메인 확장 가능성 및 벤치마크 구축** | K-EnterpriseTable(한국 공시/재무 테이블 기반) 벤치마크 구축 및 평가. 다국어 및 도메인 특화 Table-RAG 연구의 기반 제공 |

---

## 6. 타겟 학회/저널

| 순위 | 학회/저널 | 마감(예상) | 적합성 |
|------|----------|-----------|--------|
| 1 | **ACL 2026** | 2026년 2월 | NLP + Table Understanding 메인 타겟 |
| 2 | **EMNLP 2026** | 2026년 6월 | RAG + Structured Data에 적합 |
| 3 | **NeurIPS 2026** | 2026년 5월 | Datasets & Benchmarks 트랙 가능 |
| 4 | **SIGIR 2026** | 2026년 1월 | 정보검색 관점의 Table-RAG로 제출 가능 |

---

## 7. 연구 일정 (Timeline)

```
2025년 11월 (현재)
├── Week 4: 문헌 조사 완료, 프레임워크 설계 확정 (본 문서)

2025년 12월
├── Week 1-2: Stage 1 (Structure Extraction) 구현
├── Week 3-4: Stage 2 (Structure Encoding, Hierarchical Index) 구현

2026년 1월
├── Week 1-2: Stage 3 (Hierarchy-Aware Retrieval) 구현
├── Week 3-4: Stage 4 (Structure-Injected Prompting) 구현 및 E2E 파이프라인 연결

2026년 2월
├── Week 1-2: HiTab Main Experiment + Granularity 실험 (실험 1, 3)
├── Week 3-4: Ablation Study (실험 2, 4) 및 논문 초고 작성

2026년 3월
├── Week 1-2: Robustness Analysis (실험 5) + 추가 실험
├── Week 3-4: 논문 수정 및 내부 리뷰

2026년 4월
└── ACL 2026 제출 (또는 EMNLP 2026 준비)
```

### Gantt Chart (간략)

| 단계 | 12월 | 1월 | 2월 | 3월 | 4월 |
|------|------|-----|-----|-----|-----|
| Stage 1-2 구현 | ████ | | | | |
| Stage 3-4 구현 | | ████ | | | |
| Core 실험 (1, 3) | | | ████ | | |
| Ablation (2, 4) | | | ██ | | |
| Robustness (5) | | | | ██ | |
| 논문 작성 | | | ██ | ████ | ██ |
| 제출 | | | | | ██ |

---

## 8. 리스크 및 대응 방안

| 리스크 | 가능성 | 영향도 | 대응 방안 |
|--------|--------|--------|----------|
| **헤더 트리 추출 정확도 저하** | 중 | 높음 | LGPMA 등 SOTA 파서 + 휴리스틱 + LLM 보정 하이브리드 방식 적용 |
| **LLM API 비용 과다** | 중 | 중간 | Llama 3 등 오픈소스 LLM 병행, 서브셋 실험 후 점진 확대 |
| **HiTab 외 데이터셋 일반화 어려움** | 중 | 중간 | RealHiTBench, CHiTab, K-EnterpriseTable을 활용해 다양성 확보 |
| **기존 베이스라인 재현 어려움** | 낮 | 중간 | 공개 코드 우선 활용, 필요 시 저자에게 재현 환경 문의 |
| **K-EnterpriseTable 구축 공수** | 중 | 중간 | 우선 소규모(수백 테이블)부터 시작 후 점진 확장, 자동화 스크립트 도입 |

---

## 부록

### A. RQ ↔ Stage ↔ 실험 매핑 요약

| RQ | 관련 Stage | 관련 실험 |
|----|-----------|----------|
| **RQ1**: 계층 헤더 + RAG → QA 성능 향상? | Stage 1~4 전체 | 실험 1 (Main) + Ablation (No Hierarchy, Flat Only) |
| **RQ2**: 표현 방식 비교 | Stage 2 (Encoding) | 실험 2 |
| **RQ3**: Multi-granularity vs 단일 granularity | Stage 2 (Index) + Stage 3 (Retrieval) | 실험 3 |
| **RQ4**: 병합 셀 정보 → hallucination 감소? | Stage 1 (merge 좌표) + Stage 4 (Prompting) | 실험 4 |

### B. 스코프 분리: Core vs Option

**1차 타겟 (Core):**
- 데이터셋: HiTab + K-EnterpriseTable
- RQ: RQ1 + RQ3 중심
- 실험: 실험 1, 실험 3

**2차 확장 (Option):**
- 시간/체력 되면 RQ2, RQ4, RealHiTBench, CHiTab 추가
- 실험: 실험 2, 4, 5

> **발표 시 표현**: "시간 제약상, 이번 학기에는 RQ1, RQ3를 중심으로 실험을 우선 수행하고, RQ2, RQ4 및 Robustness 분석은 2차 확장 과제로 계획하고 있습니다."

---

## 참고문헌 (추후 추가)

- [1] HiTab: A Hierarchical Table Dataset for Question Answering and Natural Language Generation (ACL 2022)
- [2] RealHiTBench: Real-World Hierarchical Table Benchmark (2025)
- [3] CHiTab: Chinese Hierarchical Table QA (2025)
- [4] TableRAG: Million-Token Table Understanding with Language Models (2024)
- [5] T-RAG: Lessons from the LLM Trenches (2024)
- [6] LGPMA: Complicated Table Structure Recognition (ICCV 2021)

---

**문서 버전**: v1.0  
**작성일**: 2025년 11월 26일  
**작성자**: [연구자명]


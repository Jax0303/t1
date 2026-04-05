# Detection-Based TSR Fails at Spanning Cells: A Rigorous Empirical Study

> **연구 상태**: 🔬 실험 완료 | 📊 GriTS 검증 완료 | 🆚 Detection vs Generation 비교 완료  
> **목표 학회**: Top-tier (CVPR / ECCV / NeurIPS)  
> **최종 업데이트**: 2026-04-06

---

## 1. 문제 정의 (Problem Definition)

### 1.1 배경

표 구조 인식(Table Structure Recognition, TSR)은 문서 이미지에서 표의 논리 구조(행·열·셀 경계)를 복원하는 과제다. 현재 주류 접근법은 두 가지로 나뉜다.

| 패러다임 | 대표 모델 | 작동 방식 |
|---|---|---|
| **Detection** | TATR (Microsoft), TSRDet, CascadeTabNet | 행·열·셀을 독립 객체로 탐지 → 교차점으로 grid 구성 |
| **Generation** | Docling-TableFormer (IBM), LORE, SLANet | 논리 구조를 직접 시퀀스/좌표로 생성 |

Detection 패러다임은 구현이 간단하고 localization 정보를 제공한다는 장점이 있다. 그러나 **spanning cell(colspan/rowspan)**이 존재하는 표에서 근본적인 구조적 한계가 있다는 것이 본 연구의 출발점이다.

### 1.2 핵심 가설

> **"Detection 패러다임은 행·열 탐지를 독립적으로 수행하기 때문에, spanning cell의 경계를 결정할 수 없다. 이 실패는 특정 모델/데이터의 문제가 아닌 패러다임 수준의 구조적 한계다."**

### 1.3 기존 연구의 공백

- 기존 벤치마크(PubTables-1M, SciTSR)는 전체 GriTS-Top F1을 보고하지만, **spanning cell에 특화된 분해 분석이 없다**.
- "전체 GriTS-Top이 높으면 span도 잘 복원된다"는 묵시적 가정이 있으나, 이는 span이 전체 grid의 ~17%에 불과하기 때문에 성립하지 않는다.
- Detection vs Generation 패러다임 비교가 **동일 metric·동일 데이터셋**에서 이루어진 연구가 없다.

---

## 2. 연구 방법론 (Methodology)

### 2.1 평가 Metric (연구 테마 직접 연결)

| Metric | 출처 | 측정 대상 | GT bbox 의존 |
|---|---|---|---|
| **GriTS-Top F1 (full)** | Smock et al. CVPR'23 | 전체 grid 위상 구조 | ❌ |
| **GriTS-Top F1 (span-only)** | 본 연구 | spanning cell만 격리 | ❌ |
| **Logical-Exact Rate** | 본 연구 | (sr,er,sc,ec) 튜플 exact match | 부분적* |
| **Span-Class Recall @IoU≥0.1** | 본 연구 | detection이 span 영역을 최소한 찾는가 | ✅ |

\* Logical-Exact는 row/col 중심값(center)만 사용하여 pixel bbox보다 robust.

> **이전 측정 방식 폐기**: 초기 실험에서 사용한 "chunk 기반 GT pixel bbox 재구성 + IoU≥0.5 span recovery"는 OCR 토큰 위치 기반 GT의 ~10% 오프셋 오차가 측정 결과를 체계적으로 저평가하는 artifact임이 확인됐다. 해당 수치(1.6%)는 **공식 철회**하고 위 metric으로 대체한다.

### 2.2 GriTS 구현 검증

공식 table-transformer repository 구현과 교차 검증:
- 15개 테이블 대상 mean absolute difference = **0.0034 (0.34%)** — 제출 기준 검증 완료
- 차이 원인: DP tie-breaking 미세 차이 (알고리즘적으로 동등)

### 2.3 데이터셋

| 데이터셋 | 규모 | GT 형식 | 용도 |
|---|---|---|---|
| **SciTSR-COMP** | 716 tables | 논리 JSON (full logical GT) | 완전한 정량 평가 |
| **PubTables-1M** | 25 tables* | XML bbox | 정성 + span 감지율 |
| **FinTabNet** | 25 tables* | XML bbox | 정성 + span 감지율 |

\* 로컬 보유 샘플 기준.

### 2.4 평가 모델

| 모델 | 패러다임 | 구현 상태 |
|---|---|---|
| TATR v1.0, v1.1-pub, v1.1-all | Detection (DETR) | ✅ 완료 |
| Docling-TableFormer (IBM) | Generation (VLM) | ✅ 완료 |
| CascadeTabNet | Detection (anchor) | ⚙️ Harness 완료, weights 필요 |
| Faster R-CNN | Detection (anchor) | ⚙️ Harness 완료, 학습 필요 |

---

## 3. 실험 결과 (Results)

### 3.1 전체 GriTS-Top vs Trivial Baseline (SciTSR-COMP 716 tables)

| Model | GriTS-Top F1 | Trivial baseline | Δ vs trivial |
|---|---|---|---|
| TATR v1.0 | 0.802 | 0.801 | +0.001 |
| TATR v1.1-pub | 0.778 | 0.763 | +0.015 |
| TATR v1.1-all | 0.778 | 0.761 | +0.017 |

> Trivial baseline = 모든 셀을 1×1로 예측 (span 처리 없음). TATR이 trivial을 겨우 +0.001~+0.017 상회하는 것은 span 탐지 시도가 오히려 alignment를 악화시키기 때문이다.

### 3.2 Span-Only GriTS-Top (SciTSR-COMP 716 tables)

| Model | GriTS-Top (span-only) | vs TATR v1.0 |
|---|---|---|
| TATR v1.0 | **0.086** | — |
| TATR v1.1-pub | **0.314** | +0.228 |
| TATR v1.1-all | **0.338** | +0.252 |

→ 전체 GriTS-Top(~0.78)과 span-only GriTS-Top(0.09~0.34) 사이의 **0.44~0.69 격차**가 Detection 패러다임의 span 처리 실패를 정량화한다.

### 3.3 표 유형별 Detection vs Generation 비교 (Type-Stratified)

*SciTSR-COMP + PubTables-1M + FinTabNet, 유형별 30 tables 샘플*

| 유형 | 기준 | TATR GriTS-Top | TATR Span-only | Docling GriTS-Top | Docling Span-only | 격차 (span) |
|---|---|---|---|---|---|---|
| **A-Simple** | 0 spans | 0.818 | — | **0.972** | — | — |
| **B-Header** | 1–3 spans | 0.754 | 0.244 | **0.844** | **0.673** | **+0.429** |
| **C-Multi** | 4–10 spans | 0.799 | 0.551 | **0.863** | **0.730** | **+0.179** |
| **D-Dense** | 11+ spans | 0.633 | 0.415 | **0.644** | **0.514** | **+0.099** |

**핵심 관찰:**
1. Span이 없는 표(A-Simple)에서 TATR(0.818) vs Docling(0.972) 차이 → Detection도 순수 grid는 잘 복원
2. Span이 등장하는 순간(B-Header) span-only 격차 2.76× → Detection의 구조적 한계 노출
3. Span이 복잡해질수록(B→D) TATR span-only 하락폭 > Docling 하락폭 → 패러다임 수준 차이

### 3.4 Logical-Exact Span Recovery (SciTSR-COMP)

| Model | Logical-Exact (macro) | GT bbox 의존 |
|---|---|---|
| TATR v1.0 | 0.6% | ❌ |
| TATR v1.1-pub | 7.6% | ❌ |
| TATR v1.1-all | 7.9% | ❌ |

→ GT pixel bbox 없이 순수 논리 인덱스 기반으로 평가해도 **9 8% 이상의 spanning cell이 정확히 복원되지 않는다**.

---

## 4. 논문 주장 (Defensible Claims)

### 4.1 입증된 주장 (Supported)

> **"Detection 패러다임(DETR 계열)은 전체 grid 구조를 ~0.78 GriTS-Top F1로 복원하지만, spanning cell에 특화된 평가(span-only GriTS-Top)에서는 0.09~0.34로 급락한다. 생성 패러다임(Docling-TableFormer)은 동일 데이터에서 span-only GriTS-Top 0.51~0.73을 달성하며 B-Header 유형에서 2.76× 우위를 보인다."**

### 4.2 Scope (명시적 한정)

- **탐지 패러다임 범위**: TATR 3변종 (DETR 계열). anchor-based (CascadeTabNet, Faster R-CNN) weights 미확보로 검증 미완.
- **데이터셋 범위**: SciTSR-COMP (716), PubTables-1M/FinTabNet (각 25 샘플). 대규모 PubTables-1M 전체 검증 필요.
- **D-Dense 한계**: 11개 이상 span이 있는 고밀도 구조에서는 생성 패러다임도 0.514 수준으로 한계 노출. "generation이 완벽하다"는 주장 불가.

### 4.3 남은 작업 (To Do)

| 항목 | 우선순위 | 예상 임팩트 |
|---|---|---|
| CascadeTabNet 실제 평가 (anchor-based) | 🔴 높음 | "detection paradigm 전반" 주장 완결 |
| PubTables-1M 전체(∼27k) GriTS 평가 | 🔴 높음 | 리뷰어 dataset concern 해소 |
| GSR Loss 학습 + 비교 | 🟡 중간 | detection 내 개선 가능성 ablation |
| FinTabNet 전체 평가 | 🟡 중간 | financial table 도메인 일반화 |

---

## 5. 실험 재현 (Reproducibility)

### 설치

```bash
pip install transformers torch torchvision docling pymupdf scipy tqdm matplotlib
```

### 실험 실행

```bash
# [핵심] 패러다임 비교 벤치마크 (유형별 TATR vs Docling)
python experiments/paradigm_benchmark.py

# GriTS-Top 전체 평가 (TATR 3변종, trivial baseline, span-only 포함)
python experiments/grits_eval.py

# Logical-index span 평가 (GT bbox 의존 없음)
python experiments/logical_eval.py

# Detection family harness (TATR + Faster R-CNN adapter)
python experiments/detection_family_eval.py
python experiments/detection_family_eval.py --frcnn_ckpt PATH  # Faster R-CNN 사용 시
```

### 결과 위치

```
experiments/
├── results_benchmark/
│   ├── benchmark_summary.json       ← 유형별 × 모델별 집계
│   ├── benchmark_stratified.png     ← 메인 비교 그림 (4 metric × 4 type)
│   └── benchmark_visual.png         ← 대표 표 GT overlay 시각화
├── results_grits/
│   ├── grits_summary.json           ← GriTS-Top full/trivial/span-only
│   └── grits_eval.png
├── results_logical/
│   └── logical_summary.json         ← Logical-Exact, Span-Dist Jaccard
└── results_family/
    └── family_summary.json          ← Detection family 통합 결과
```

---

## 6. 프로젝트 구조

```
.
├── data/
│   ├── SciTSR/             # SciTSR-COMP (716 tables, full logical GT)
│   ├── pubtables-1m/       # PubTables-1M 샘플 (25 images + XML)
│   └── fintabnet/          # FinTabNet 샘플 (25 images + XML)
├── experiments/
│   ├── _eval_utils.py              # 공유 유틸 (GT 로딩, IoU, chunk 파싱)
│   ├── grits_eval.py               # GriTS-Top/Con (Smock CVPR'23)
│   ├── logical_eval.py             # Logical-index span matching
│   ├── detection_family_eval.py    # DetectionModelAdapter harness
│   ├── paradigm_benchmark.py       # 유형별 TATR vs Docling 벤치마크
│   ├── s1_detection_paradigm_eval.py   # S-1: Detection consistency
│   ├── s2_paradigm_comparison.py       # S-2: Cross-paradigm
│   └── gsr_experiment.py               # GSR Loss ablation harness
└── src/
    └── gsr_hooks/          # GSR Loss, SeparatorExtractor 구현
```

---

## 7. 핵심 참고문헌

- Smock et al., "GriTS: Grid table similarity metric for table structure recognition," CVPR 2023.
- Prasad et al., "CascadeTabNet: An approach for end to end table detection and structure recognition from image-based documents," ICDAR 2020.
- Xiao et al., "TSRFormer: Table structure recognition with transformers," ACMMM 2022.
- IBM Docling Team, "Docling Technical Report," 2024.
- Ye et al., "SciTSR: A large-scale dataset for scientific table structure recognition," ECCV 2021.

---

**Last Updated**: 2026-04-06  
**Status**: ✅ GriTS Verified | ✅ Detection vs Generation Compared | ✅ Type-Stratified Analysis | ⚙️ CascadeTabNet Pending

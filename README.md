# SAGR: Span-Aware Grid Reconstruction for Robust TSR

> **연구 상태**: ✅ Phase 3 완료 — SAGR-Cov, Baseline 대비 3개 지표 모두 개선  
> **목표 학회**: Top-tier Computer Vision Conference (CVPR / ICCV)  
> **최종 업데이트**: 2026-04-13

---

## 1. 연구 배경 및 동기 (Motivation)

기존의 탐지 기반(Detection-based) 표 구조 인식(TSR) 모델들(TATR 등)은 행, 열, 그리고 스패닝 셀(spanning cell)을 독립적인 객체로 탐지한 뒤, 이를 결합하여 최종 격자 구조를 생성합니다. 하지만 이 **Stage 2: Grid Reconstruction** 과정에서 다음과 같은 치명적인 한계가 발견되었습니다.

1. **기하학적 민감도**: 스패닝 셀 박스가 1~2픽셀만 경계를 벗어나도 인접 행/열을 침범하거나 할당에 실패하는 'off-by-1' 오류 발생.
2. **구조적 정합성 결여**: 독립적인 IoU 임계값(0.3)을 기준으로 셀을 할당함에 따라, 논리적으로 연속되어야 할 스패닝 셀이 파편화되거나 정렬되지 않는 문제.

**본 연구는 이러한 Stage 2의 병목을 해결하기 위해 구간 커버리지(Interval Coverage) 기반의 새로운 재구성 알고리즘인 SAGR(Span-Aware Grid Reconstruction)을 제안합니다.**

---

## 2. 3단계 연구 파이프라인 (3-Phase Pipeline)

### Phase 1: Problem Proof (`phase1_problem_proof.py`)
- **가설**: "현재의 탐지 기반 모델은 스패닝 셀에서 구조적으로 실패하며, 사실상 아무런 처리를 하지 않은 Trivial Baseline과 성능 차이가 없다."
- **검증**: SciTSR-COMP 및 FinTabNet 데이터를 대상으로 `Trivial-Delta` 지표를 산출하여 모델의 실질적 기여도가 미미함을 증명.

### Phase 2: Oracle Decomposition (`phase2_oracle_decomposition.py`)
- **가설**: "실패의 원인은 탐지(Detection)가 아닌 재구성(Reconstruction) 단계에 있다."
- **검증**: 탐지 결과가 완벽하다고 가정(Oracle)했을 때의 복원 성능을 측정하여, Stage 2 알고리즘 개선의 잠재적 이득(Potential Gain)을 확인.

### Phase 3: SAGR Evaluation (`phase3_sagr_eval.py`)
- **가설**: "제안하는 SAGR 알고리즘이 기존의 IoU 기반 방식을 뛰어넘는 구조적 정합성을 제공한다."
- **검증**: `Baseline`, `SAGR-Cov`, `SAGR-Con`, `SAGR-Full` 변종들을 대조 평가하여 개선 효과 입증.

---

## 3. 제안 방법: SAGR (Span-Aware Grid Reconstruction)

`experiments/sagr.py`에 구현된 SAGR의 핵심 메커니즘은 다음과 같습니다.

- **Row/Column Coverage**: 전체 박스 IoU 대신, 행(Row)과 열(Col) 각각에 대한 1차원 겹침 비율(`overlap / max_dim`)을 계산하여 할당 여부를 결정합니다. 이는 박스의 절대적 크기 오차에 덜 민감합니다.
- **Contiguity Constraint**: 스패닝 셀이 차지하는 인덱스 사이에 공백이 생길 경우 이를 자동으로 연결하여 논리적 연속성을 보장합니다.

---

## 4. Phase 3 최종 결과 (SciTSR-COMP, 716 complex tables)

| 지표 | Baseline | SAGR-Con | SAGR-Cov | SAGR-Full |
|------|----------|----------|----------|-----------|
| GriTS-Top F1 | 0.7737 | 0.7605 | **0.7760** | 0.7734 |
| Span-Only F1 | 0.2977 | 0.2972 | **0.3369** | **0.3367** |
| Logical-Exact | 0.0727 | 0.0722 | **0.0809** | **0.0805** |

- **SAGR-Cov** (coverage, thresh=0.3): Baseline 대비 3개 지표 모두 개선
- **최적 threshold**: 0.3 (Sweep 결과)
- **핵심 버그 수정**: `_interval_coverage`의 `min()` → `max()` 분모 변경

---

## 5. 학습 인프라 (Training Infrastructure)

Phase 3 이후, Detection paradigm의 한계를 학습 수준에서 보완하기 위해 **TATR-Span** (DETRSpan + ordinal regression head)과 두 가지 baseline을 구현했습니다.

### 5.1 디렉토리 구조

```
tatr_span/
  models/
    detr_span.py          # DETRSpan: TATR + rowspan/colspan ordinal regression head
  datasets/
    pubtables1m.py        # PubTables-1M PASCAL VOC XML 로더
    fintabnet_c.py        # FinTabNet.c (HuggingFace bsmock/FinTabNet.c)
    scitsr.py             # SciTSR / SciTSR-COMP (cell bbox → row/col 역추론)
    transforms.py         # DETR 호환 augmentation (RandomResize, RandomSizeCrop 등)
  training/
    train.py              # DDP 학습 스크립트 (AdamW, backbone/head 분리 LR)
    config.yaml           # 학습 설정
  scripts/
    download_weights.py   # bsmock/tatr-pubtables1m-v1.0 가중치 다운로드
    download_data.py      # PubTables-1M / FinTabNet.c / SciTSR 데이터 다운로드

baselines/
  deformable_detr/
    setup.py              # fundamentalvision/Deformable-DETR 클론 + CUDA ops 빌드
    finetune.py           # PubTables-1M 파인튜닝 (DDP 지원)
    config.yaml           # R50, 300 queries, with_box_refine=True
  tsrdet/
    cascade_rcnn.py       # Cascade R-CNN (IoU 0.5/0.6/0.7) + Single-Label Regularization
    train.py              # SGD + MultiStepLR (epoch 12/16)
    config.yaml           # MDPI Electronics 2024 재현 설정

tatr_base/                # microsoft/table-transformer (공식 코드)
```

### 5.2 DETRSpan 아키텍처

```
TATR (ResNet-18 + Transformer)
  ├── class_embed   → [B, Q, 6]         TATR 6-class 분류
  ├── bbox_embed    → [B, Q, 4]         bounding box (cxcywh)
  └── span_embed    → [B, Q, 18]        ordinal regression (9 + 9 thresholds)
                                         P(rs>1)..P(rs>9), P(cs>1)..P(cs>9)
```

- **Ordinal decoding**: `span = 1 + sum(P > 0.5)` (max = 10)
- **Loss 적용**: Hungarian matching에서 `table spanning cell` (class=4)로 매칭된 쿼리에만 BCE loss 적용

### 5.3 학습 설정

| 항목 | 값 |
|------|-----|
| Base model | TATR ResNet-18 (`bsmock/tatr-pubtables1m-v1.0`) |
| Optimizer | AdamW |
| LR (transformer/heads) | 1e-4 |
| LR (backbone) | 1e-5 |
| Weight decay | 1e-4 |
| Batch size | 2/GPU × 4 GPU = 8 |
| Epochs | 20 (LR step-drop at epoch 15) |
| Image size | min=800, max=1333 |
| Span loss coef | 1.0 (rowspan + colspan BCE) |
| Spanning cell class weight | 2.0 (emphasized) |

### 5.4 빠른 시작

```bash
# 1. 가중치 다운로드
python tatr_span/scripts/download_weights.py

# 2. 데이터 다운로드 (PubTables-1M ~120GB, D드라이브 권장)
python tatr_span/scripts/download_data.py --dataset pubtables1m --out-dir /mnt/d/pubtables1m
python tatr_span/scripts/download_data.py --dataset fintabnet_c

# 3. TATR-Span 학습 (4 GPU)
torchrun --nproc_per_node=4 tatr_span/training/train.py \
    --config tatr_span/training/config.yaml

# 4. Deformable-DETR baseline 설정
python baselines/deformable_detr/setup.py
torchrun --nproc_per_node=4 baselines/deformable_detr/finetune.py \
    --config baselines/deformable_detr/config.yaml

# 5. TSRDet baseline 학습
torchrun --nproc_per_node=4 baselines/tsrdet/train.py \
    --config baselines/tsrdet/config.yaml
```

---

## 6. Phase 1-3 평가 실행

```bash
# Phase 1: 문제 정의 및 TATR 성능 한계 증명
python experiments/phase1_problem_proof.py

# Phase 2: 원인 분석 (Oracle 실험)
python experiments/phase2_oracle_decomposition.py

# Phase 3: SAGR 알고리즘 평가 및 Ablation Study
python experiments/phase3_sagr_eval.py
```

---

## 7. 향후 과제 (Next Steps)

- [ ] **PubTables-1M 학습 실행**: 데이터 다운로드 후 DETRSpan 파인튜닝
- [ ] **FinTabNet 평가**: FinTabNet.c로 cross-dataset 일반화 검증
- [ ] **Baseline 비교**: Deformable-DETR / TSRDet와 Span-F1 정량 비교
- [ ] **논문 작성**: Phase 1-3 + 학습 실험 통합, SAGR 알고리즘 정식 기술
- [ ] **결과 시각화 갱신**: `results_final/`의 논문용 그래프 최종 업데이트

---

**Last Updated**: 2026-04-13  
**Status**: ✅ Phase 3 완료 | 🔧 Training Infrastructure 구현 완료

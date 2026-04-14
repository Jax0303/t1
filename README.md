# SAGR: Span-Aware Grid Reconstruction for Robust TSR

> **연구 상태**: ✅ Phase 1-3 완료 | ✅ TATR-base Ablation 인프라 완료 (STEP 1-6, TASK 1-7)
> **목표 학회**: Top-tier Computer Vision Conference (CVPR / ICCV)
> **최종 업데이트**: 2026-04-14

---

## 1. 연구 배경 및 동기 (Motivation)

기존의 탐지 기반(Detection-based) 표 구조 인식(TSR) 모델들(TATR 등)은 행, 열, 스패닝 셀(spanning cell)을 독립 객체로 탐지 후 격자로 결합합니다. 하지만 **Stage 2: Grid Reconstruction** 단계에서 다음 문제가 발생합니다.

1. **기하학적 민감도**: 스패닝 셀 박스가 1~2픽셀만 벗어나도 할당 실패 ('off-by-1' 오류).
2. **구조적 정합성 결여**: 독립 IoU 임계값 기반 할당으로 인해 논리적 연속성 깨짐.

**본 연구는 SAGR(Span-Aware Grid Reconstruction)과 TATR-Span 학습 개선으로 이 두 병목을 동시에 해결합니다.**

---

## 2. 3단계 연구 파이프라인

### Phase 1: 문제 증명 (`experiments/phase1_problem_proof.py`)
- TATR이 spanning cell 처리에서 Trivial Baseline과 성능 차이 없음을 정량 증명

### Phase 2: 원인 분석 (`experiments/phase2_oracle_decomposition.py`)
- Oracle 탐지 가정 시 Stage 2(격자 복원)가 병목임을 확인

### Phase 3: SAGR 평가 (`experiments/phase3_sagr_eval.py`)
- `SAGR-Cov`가 Baseline 대비 3개 지표 모두 개선

---

## 3. Phase 3 최종 결과 (SciTSR-COMP, 716 complex tables)

| 지표 | Baseline | SAGR-Con | SAGR-Cov | SAGR-Full |
|------|----------|----------|----------|-----------|
| GriTS-Top F1 | 0.7737 | 0.7605 | **0.7760** | 0.7734 |
| Span-Only F1 | 0.2977 | 0.2972 | **0.3369** | **0.3367** |
| Logical-Exact | 0.0727 | 0.0722 | **0.0809** | **0.0805** |

- **최적 threshold**: 0.3
- SAGR-Cov: 구간 커버리지 기반, 박스 크기 오차에 강건

---

## 4. TATR-base Ablation Study (E0~E3)

Phase 3 이후, **학습 수준에서 spanning cell 인식을 개선**하는 ablation 실험을 설계·구현했습니다. Microsoft TATR 공식 코드(`tatr_base/`)를 직접 수정합니다.

### 4.1 실험 설계

| ID | 설명 | 추가 구성 |
|----|------|-----------|
| **E0** | Baseline TATR (원본 재현) | `--no_span_branch --no_grid_snapping` |
| **E1** | + Span Attribute Branch | ordinal regression head (K=8) |
| **E2** | + Span Loss + Hard Grid-Snapping | curriculum warmup 5 epochs |
| **E3** | + Span Loss + Soft Grid-Snapping | curriculum warmup 5 epochs |

3 seeds (42/43/44) × 4 실험 = **12회 학습**, 20 epochs each.

### 4.2 TATR-base 수정 내역 (STEP 1-6)

**STEP 1 — Span Attribute Branch** (`tatr_base/detr/models/detr.py`)
```
TATR Transformer decoder hidden state
  └── span_embed: Linear(256→128) → ReLU → Linear(128→16)
                  [B, Q, 16] = P(rowspan≥k) + P(colspan≥k), k=1..8
```
- `--no_span_branch`로 E0 완전 재현 (backward compat)

**STEP 2 — Hungarian Matching with C_span** (`tatr_base/detr/models/matcher.py`)
- `spanning cell` 매칭 쿼리에 span L1 cost 추가
- `--set_cost_span 0.0` → 기존 매칭과 동일

**STEP 3 — Ordinal Span Loss** (`tatr_base/detr/models/detr.py`)
- `loss_span`: GT label=5(spanning cell)에만 BCE 적용
- Ordinal encoding: span k → `[1,..,1,0,..,0]` (threshold=1..K)
- Batch에 spanning cell 없으면 zero tensor 반환 (NaN 방지)

**STEP 4 — Grid-Snapping Loss** (`tatr_base/detr/models/grid_snapping.py`)
- `compute_snapping_target()`: 예측 bbox를 GT row/col 구분선 격자에 snap
- `curriculum_warmup(epoch, n_warm=5)`: epoch≥5부터 snapping 활성화
- `loss_boxes_with_snapping()`: spanning cell에만 snapped GIoU target 적용
- `SetCriterion._build_separators()`: GT row(label=2)/col(label=1) bbox에서 구분선 추출
- `SetCriterion.set_epoch()`: 매 epoch 시작 시 `main.py`가 호출

**STEP 5 — Dataset: rowspan/colspan 파싱** (`tatr_base/src/table_datasets.py`)
- `read_pascal_voc()` → 4-tuple 반환: `bboxes, labels, rowspans, colspans`
- PASCAL VOC `<attributes>` 태그에서 rowspan/colspan 파싱
- target dict에 `rowspans`, `colspans` 텐서 추가

**STEP 6 — Eval by Complexity** (`tatr_base/src/eval_by_complexity.py`)
- 테스트 XML에서 `max(rowspan×colspan)`으로 복잡도 분류
  - `simple` (k=1), `complex` (k>1)
  - k-bin: `k=1`, `k=2`, `k=3~4`, `k≥5`
- 기존 `metrics.json` 위에서 동작 (grits.py 무수정)

### 4.3 실행 스크립트 (TASK 3-4)

```bash
# 전체 12회 ablation 학습 (순차 실행)
bash run_all.sh /mnt/d/pubtables1m/PubTables-1M-Structure outputs/ablation

# 결과 집계 (mean ± std, Markdown 테이블)
python aggregate_results.py \
    --results_dir outputs/ablation \
    --output_csv  outputs/ablation/summary_results.csv
```

### 4.4 Sanity Check (TASK 1)

```bash
python sanity_check.py
```

7가지 검증 (CPU only, 실제 데이터 불필요):

| # | 검증 항목 |
|---|----------|
| 1 | E1 forward: `pred_spans` shape [B, Q, 16] 확인 |
| 2 | E0 backward compat: `pred_spans` 없음 확인 |
| 3 | Loss 계산: NaN/폭발 없음 |
| 4 | Backward: `span_embed` gradient 정상 흐름 |
| 5 | XML 파싱: rowspan=2, colspan=3 정상 파싱 |
| 6 | Grid snapping: consecutive 제약 및 shape 확인 |
| 7 | E3 criterion 연동: warmup 전후 GIoU 손실 전환 확인 |

---

## 5. 학습 인프라 (tatr_span + baselines)

### 5.1 DETRSpan 아키텍처 (`tatr_span/`)
```
TATR (ResNet-18 + Transformer)
  ├── class_embed   → [B, Q, 6]     6-class 분류
  ├── bbox_embed    → [B, Q, 4]     bounding box (cxcywh)
  └── span_embed    → [B, Q, 18]    ordinal regression (K=9)
```

### 5.2 baselines/
- `deformable_detr/`: Deformable-DETR PubTables-1M finetune
- `tsrdet/`: Cascade R-CNN (IoU 0.5/0.6/0.7) + Single-Label Regularization

---

## 6. 데이터 (PubTables-1M-Structure)

| 분할 | XML 수 | 이미지 |
|------|--------|--------|
| train | 86,284 | `images/` (94,959 jpg, 추출 완료) |
| val | 11,707 | 추출 완료 |
| test | - | `test/` XML 추출 완료 |

- 경로: `/mnt/d/pubtables1m/PubTables-1M-Structure/`
- `run_all.sh` 기본 `DATA_ROOT` = 위 경로

---

## 7. 디렉토리 구조

```
t1/
├── tatr_base/               # microsoft/table-transformer (수정본, 자체 git)
│   ├── detr/models/
│   │   ├── detr.py          # Span head, grid-snapping loss 추가
│   │   ├── matcher.py       # C_span Hungarian cost 추가
│   │   └── grid_snapping.py # Grid-snapping 유틸리티
│   └── src/
│       ├── main.py          # CLI args + set_epoch() 호출
│       ├── table_datasets.py# rowspan/colspan 파싱
│       └── eval_by_complexity.py
├── tatr_span/               # DETRSpan 학습 패키지
├── baselines/               # Deformable-DETR, TSRDet
├── experiments/             # Phase 1-3 + 시각화
│   ├── sagr.py              # SAGR 알고리즘
│   ├── phase1~3_*.py        # 3단계 평가
│   ├── labmeeting_figs.py   # 랩미팅 시각화 (4 figures)
│   └── results_*/           # 결과 그래프
├── sanity_check.py          # 7-check CPU 검증 스크립트
├── run_all.sh               # 12회 ablation 실행기
└── aggregate_results.py     # 결과 집계 (Table A/B Markdown)
```

---

## 8. 향후 계획

- [ ] **Colab T4 GPU로 ablation 학습 실행** (E0~E3, seeds 42/43/44)
- [ ] **`training_log.json` 구현**: 매 epoch GriTS_Loc 기록 → best checkpoint 저장
- [ ] **FinTabNet cross-dataset 평가**: 학습된 E3 모델 → FinTabNet.c 테스트
- [ ] **논문 작성**: Phase 1-3 + Ablation 결과 통합, SAGR + Grid-Snapping 기술

---

**Last Updated**: 2026-04-14
**Status**: ✅ Phase 1-3 완료 | ✅ STEP 1-6 + TASK 1-7 구현 완료 | 🔲 GPU 학습 대기중

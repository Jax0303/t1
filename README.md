# SAGR: Span-Aware Grid Reconstruction for Robust TSR

> **연구 상태**: ✅ Phase 1-3 완료 | ✅ STEP 1-6 + TASK 1-7 구현 완료 | 🔲 Colab GPU 학습 진행 중
> **목표 학회**: Top-tier Computer Vision Conference (CVPR / ICCV)
> **최종 업데이트**: 2026-04-14

---

## 1. 연구 개요

기존 탐지 기반 TSR(Table Structure Recognition) 모델(TATR 등)의 두 가지 병목을 해결:

1. **Stage 2 병목**: 격자 재구성 단계에서 spanning cell bbox 1~2픽셀 오차로 할당 실패
2. **학습 수준 한계**: spanning cell 속성(rowspan/colspan)을 학습에 활용하지 않음

**해결책**: SAGR 알고리즘(Stage 2 개선) + TATR-Span 학습 개선(E0~E3 ablation)

---

## 2. Phase 1-3 결과 (SciTSR-COMP, 716 complex tables)

| 지표 | Baseline | SAGR-Con | SAGR-Cov | SAGR-Full |
|------|----------|----------|----------|-----------|
| GriTS-Top F1 | 0.7737 | 0.7605 | **0.7760** | 0.7734 |
| Span-Only F1 | 0.2977 | 0.2972 | **0.3369** | **0.3367** |
| Logical-Exact | 0.0727 | 0.0722 | **0.0809** | **0.0805** |

---

## 3. TATR-base Ablation Study (E0~E3)

### 3.1 실험 설계

| ID | 설명 | 핵심 옵션 |
|----|------|-----------|
| **E0** | Baseline TATR 원본 재현 | `--no_span_branch --no_grid_snapping` |
| **E1** | + Span Attribute Branch | ordinal regression head (K=8) |
| **E2** | + Span Loss + Hard Grid-Snapping | curriculum warmup 5 epochs |
| **E3** | + Span Loss + Soft Grid-Snapping | curriculum warmup 5 epochs |

3 seeds (42/43/44) × 4 실험 = **12회 학습**, 20 epochs each

### 3.2 수정된 파일 (STEP 1-6)

| 파일 | 수정 내용 |
|------|-----------|
| `tatr_base/detr/models/detr.py` | span_embed 헤드, loss_span, loss_boxes with snapping, set_epoch() |
| `tatr_base/detr/models/matcher.py` | C_span Hungarian cost |
| `tatr_base/detr/models/grid_snapping.py` | compute_snapping_target, curriculum_warmup, loss_boxes_with_snapping |
| `tatr_base/src/table_datasets.py` | rowspan/colspan XML 파싱, target dict 추가 |
| `tatr_base/src/main.py` | CLI args, criterion.set_epoch() 호출 |
| `tatr_base/src/eval_by_complexity.py` | simple/complex/k_bin 복잡도별 평가 |

### 3.3 Sanity Check (7/7 pass)

```bash
python sanity_check.py
```

| # | 검증 항목 |
|---|----------|
| 1 | E1 forward: pred_spans [B, Q, 16] |
| 2 | E0 backward compat: pred_spans 없음 |
| 3 | Loss NaN/폭발 없음 |
| 4 | span_embed gradient 정상 |
| 5 | XML rowspan/colspan 파싱 |
| 6 | Grid snapping consecutive 제약 |
| 7 | E3 criterion warmup 전후 전환 |

### 3.4 실행

```bash
# 전체 12회 ablation (로컬, GPU 필요)
bash run_all.sh /path/to/PubTables-1M-Structure outputs/ablation

# 결과 집계
python aggregate_results.py \
    --results_dir outputs/ablation \
    --output_csv  outputs/ablation/summary_results.csv
```

---

## 4. Colab T4 GPU 학습 (`colab_ablation.ipynb`)

로컬 GPU 없이 Google Colab T4로 실험 실행.

### 4.1 데이터 준비

D드라이브 → Drive 업로드 불필요. **Colab에서 HuggingFace 직접 다운로드.**

```python
# bsmock/pubtables-1m 에서 자동 다운로드 + 올바른 서브폴더로 압축 해제
# tar 내부가 flat이므로 각 tar를 지정 폴더에 추출:
#   Annotations_Train → DATA_ROOT/train/
#   Annotations_Val   → DATA_ROOT/val/
#   Annotations_Test  → DATA_ROOT/test/
#   Images_*          → DATA_ROOT/images/
```

### 4.2 노트북 실행 순서

| 섹션 | 내용 | 예상 시간 |
|------|------|----------|
| 0 | GPU / 디스크 확인 | 10초 |
| 1 | pycocotools 설치, git clone | 1~2분 |
| 2 | 경로 설정 + HF 데이터 다운로드 | ~수분 |
| 3 | sanity_check.py 7/7 확인 | 1~2분 |
| 4 | E0 → E1 → E2 → E3 학습 | 10분/실험 (subset) |
| 5 | eval_by_complexity | 수분 |
| 6 | aggregate_results | 수분 |
| 7 | 시각화 | 수분 |
| 8 | Drive 백업 | 수분 |

### 4.3 모드 설정

```python
SUBSET_MODE = True   # 500샘플/3epoch — smoke test (~10분/실험)
SUBSET_MODE = False  # 86K샘플/20epoch — 실제 학습 (~3~4시간/실험)
```

### 4.4 알려진 이슈 및 수정 이력

| 이슈 | 원인 | 수정 |
|------|------|------|
| `sys.path ../detr` 오류 | CWD가 repo root일 때 경로 불일치 | subprocess `cwd=SRC_DIR` 지정 |
| 데이터 MISSING 오류 | tar가 flat — 한 폴더에 다 풀림 | 각 tar를 지정 서브폴더에 추출 |

---

## 5. 디렉토리 구조

```
t1/
├── tatr_base/                   # microsoft/table-transformer (수정본, 자체 git)
│   ├── detr/models/
│   │   ├── detr.py              # Span head + grid-snapping loss
│   │   ├── matcher.py           # C_span Hungarian cost
│   │   └── grid_snapping.py     # snapping 유틸리티
│   └── src/
│       ├── main.py              # CLI + set_epoch()
│       ├── table_datasets.py    # rowspan/colspan 파싱
│       └── eval_by_complexity.py
├── tatr_span/                   # DETRSpan 학습 패키지
├── baselines/                   # Deformable-DETR, TSRDet
├── experiments/                 # Phase 1-3 스크립트 + 시각화
│   ├── sagr.py
│   ├── phase1~3_*.py
│   └── results_*/
├── sanity_check.py              # 7-check CPU 검증
├── run_all.sh                   # 12회 ablation 실행기
├── aggregate_results.py         # 결과 집계 (Table A/B)
└── colab_ablation.ipynb         # Colab T4 실행 노트북
```

---

## 6. 앞으로 할 일 (Todo)

### 즉시 (Colab 실험)
- [ ] `SUBSET_MODE=True` smoke test 완료 확인
- [ ] `SUBSET_MODE=False` 전체 학습 실행 (E0 → E1 → E2 → E3)
- [ ] 실험 결과 Drive 백업

### 학습 완료 후
- [ ] `training_log.json` 구현: 매 epoch val GriTS_Loc 기록 → best checkpoint 저장
- [ ] E0~E3 결과 Table A/B 정리 (simple/complex, k_bin 분해)
- [ ] E3 vs E0 delta 분석 (spanning cell 복잡도별 개선폭)

### 논문 작성
- [ ] Phase 1-3 + Ablation 결과 통합
- [ ] SAGR 알고리즘 + Grid-Snapping 정식 기술
- [ ] FinTabNet.c cross-dataset 일반화 검증

---

**Last Updated**: 2026-04-14
**Status**: ✅ Phase 1-3 완료 | ✅ STEP 1-6 구현 완료 | 🔲 Colab 학습 진행 중

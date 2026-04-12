# SAGR: Span-Aware Grid Reconstruction for Robust TSR

> **연구 상태**: 🔬 3단계 검증 파이프라인 구축 완료 | 🛠️ SAGR 알고리즘 구현 완료 | ⚠️ Phase 3 디버깅 중
> **목표 학회**: Top-tier Computer Vision Conference (CVPR / ICCV)
> **최종 업데이트**: 2026-04-13

---

## 1. 연구 배경 및 동기 (Motivation)

기존의 탐지 기반(Detection-based) 표 구조 인식(TSR) 모델들(TATR 등)은 행, 열, 그리고 스패닝 셀(spanning cell)을 독립적인 객체로 탐지한 뒤, 이를 결합하여 최종 격자 구조를 생성합니다. 하지만 이 **Stage 2: Grid Reconstruction** 과정에서 다음과 같은 치명적인 한계가 발견되었습니다.

1.  **기하학적 민감도**: 스패닝 셀 박스가 1~2픽셀만 경계를 벗어나도 인접 행/열을 침범하거나 할당에 실패하는 'off-by-1' 오류 발생.
2.  **구조적 정합성 결여**: 독립적인 IoU 임계값(0.3)을 기준으로 셀을 할당함에 따라, 논리적으로 연속되어야 할 스패닝 셀이 파편화되거나 정렬되지 않는 문제.

**본 연구는 이러한 Stage 2의 병목을 해결하기 위해 구간 커버리지(Interval Coverage) 기반의 새로운 재구성 알고리즘인 SAGR(Span-Aware Grid Reconstruction)을 제안합니다.**

---

## 2. 3단계 연구 파이프라인 (3-Phase Pipeline)

### Phase 1: Problem Proof (`phase1_problem_proof.py`)
*   **가설**: "현재의 탐지 기반 모델은 스패닝 셀에서 구조적으로 실패하며, 사실상 아무런 처리를 하지 않은 Trivial Baseline과 성능 차이가 없다."
*   **검증**: SciTSR-COMP 및 FinTabNet 데이터를 대상으로 `Trivial-Delta` 지표를 산출하여 모델의 실질적 기여도가 미미함을 증명.

### Phase 2: Oracle Decomposition (`phase2_oracle_decomposition.py`)
*   **가설**: "실패의 원인은 탐지(Detection)가 아닌 재구성(Reconstruction) 단계에 있다."
*   **검증**: 탐지 결과가 완벽하다고 가정(Oracle)했을 때의 복원 성능을 측정하여, Stage 2 알고리즘 개선의 잠재적 이득(Potential Gain)을 확인.

### Phase 3: SAGR Evaluation (`phase3_sagr_eval.py`)
*   **가설**: "제안하는 SAGR 알고리즘이 기존의 IoU 기반 방식을 뛰어넘는 구조적 정합성을 제공한다."
*   **검증**: `Baseline`, `SAGR-Cov`, `SAGR-Con`, `SAGR-Full` 변종들을 대조 평가하여 개선 효과 입증.

---

## 3. 제안 방법: SAGR (Span-Aware Grid Reconstruction)

`experiments/sagr.py`에 구현된 SAGR의 핵심 메커니즘은 다음과 같습니다.

*   **Row/Column Coverage**: 전체 박스 IoU 대신, 행(Row)과 열(Col) 각각에 대한 1차원 겹침 비율(overlap / min_dim)을 계산하여 할당 여부를 결정합니다. 이는 박스의 절대적 크기 오차에 덜 민감합니다.
*   **Contiguity Constraint**: 스패닝 셀이 차지하는 인덱스 사이에 공백이 생길 경우 이를 자동으로 연결하여 논리적 연속성을 보장합니다.

---

## 4. 실행 방법 (How to Run)

```bash
# Phase 1: 문제 정의 및 TATR 성능 한계 증명
python experiments/phase1_problem_proof.py

# Phase 2: 원인 분석 (Oracle 실험)
python experiments/phase2_oracle_decomposition.py

# Phase 3: SAGR 알고리즘 평가 및 Ablation Study
python experiments/phase3_sagr_eval.py
```

---

## 5. 향후 과제 (Next Steps)

현재 Phase 3 평가까지 완료되었으나, 코드 상의 논리 오류로 인해 SAGR의 최종 지표가 예상보다 낮게 산출되고 있습니다.

- [ ] **SAGR 디버깅 (Priority: High)**: `sagr.py`의 `coverage_thresh` 및 할당 로직을 검토하여 `Baseline` 이상의 성능(Logical-Exact > 0)을 확보할 것.
- [ ] **Threshold Sweep 분석**: Coverage 임계값 변화에 따른 성능 민감도를 분석하여 최적의 하이퍼파라미터 도출.
- [ ] **결과 시각화 업데이트**: 개선된 성능 수치를 바탕으로 `results_final/`의 논문용 그래프들을 갱신.

---

**Last Updated**: 2026-04-13  
**Status**: ⚠️ Phase 3 Evaluation Debugging in Progress

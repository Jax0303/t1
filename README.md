# Detection-based TSR의 행 검출 오류 분석 및 Row-NMS 후처리

> **연구 상태**: ✅ 문제 증명 완료 | ✅ Row-NMS 평가 완료
> **최종 업데이트**: 2026-04-16

---

## 1. 연구 개요

Detection-based TSR(Table Structure Recognition) 모델 **TATR**의 성능 저하 원인을 분석하고, 단순한 후처리로 개선 가능함을 증명합니다.

### 핵심 발견

**"Spanning cell 복원 실패의 근본 원인은 span detection이 아니라 row detection의 중복 검출이다."**

이를 3단계 논리 체인으로 증명:

1. **Oracle Decomposition** (FinTabNet n=68): GT row/col로 교체하면 F1 0.830→0.993 (+0.163), GT span으로 교체하면 0.831 (+0.001) → **병목은 row/col, span 아님**
2. **Count Diagnostic** (양 데이터셋): Col 정확도 85~99%, Row 정확도 0~10% → **병목은 row 단독**
3. **Error Mode** (SciTSR n=716): 76.5%가 정확히 +1 행 초과검출, 100%에서 y-overlap>0.5 → **중복 행 검출, 기하학적으로 탐지 가능**

### 제안: Row-NMS 후처리

y축 1D overlap > threshold τ인 행 쌍을 union merge하는 단순 후처리 블록.
TATR 아키텍처 변경 없이, row bbox 분기에서 그리드 재구성 직전에 삽입.

---

## 2. 주요 결과

### Row-NMS (τ=0.5, complex tables)

| 데이터셋 | Metric | Baseline | +Row-NMS | Δ |
|---------|--------|----------|----------|---|
| SciTSR | GriTS-Top F1 | 0.8247 | **0.8634** | **+3.9%p** |
| FinTabNet | GriTS-Top F1 | 0.9196 | **0.9463** | **+2.7%p** |

---

## 3. 실험 재현

### 사전 요건
- Python 3.10+, CUDA GPU (추론용)
- `pip install -r requirements.txt`
- SciTSR / FinTabNet 데이터셋 (`data/` 디렉토리)

### 실행 순서

```bash
# 1. Phase 1: Simple vs Complex gap 증명
python experiments/phase1_problem_proof.py

# 2. Phase 2: Oracle decomposition (FinTabNet)
python experiments/phase2_oracle_decomposition.py

# 3. Count diagnostic: Row/Col 개수 불일치 분석
python experiments/diag_rowcol_bottleneck.py

# 4. Row-NMS 평가 + figure 생성
python experiments/row_nms_eval.py

# 5. PPT용 figure 생성
python experiments/make_ppt_figures.py

# 6. Row 중복 검출 시각화
python experiments/visualize_row_duplicates.py
```

---

## 4. 디렉토리 구조

```
t1/
├── experiments/
│   ├── _eval_utils.py                 ← 데이터 로더, bbox 유틸
│   ├── phase1_problem_proof.py        ← Simple vs Complex gap 증명
│   ├── phase2_oracle_decomposition.py ← Oracle 분해 — row/col 병목 증명
│   ├── diag_rowcol_bottleneck.py      ← Row/Col count mismatch 진단
│   ├── row_nms_eval.py                ← Row-NMS 평가 + threshold sweep
│   ├── grid_reconstruct.py            ← Baseline 격자 재구성 모듈
│   ├── make_ppt_figures.py            ← PPT용 figure 생성
│   ├── visualize_row_duplicates.py    ← Row 중복 검출 정성 평가
│   ├── results_phase1/                ← Phase 1 결과
│   ├── results_phase2/                ← Oracle 결과 + oracle_summary.json
│   ├── results_diag/                  ← Count diagnostic 결과
│   ├── results_rownms/                ← Row-NMS 결과 + 모든 figure
│   └── results_phase3/
│       └── tatr_inference_cache.pkl   ← TATR 추론 캐시 (456MB, 재사용)
├── data/                              ← SciTSR, FinTabNet 데이터
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 5. 생성되는 Figure 목록

| Figure | 위치 | 설명 |
|--------|------|------|
| `fig_pipeline.png` | `results_rownms/` | 데이터 흐름도: Row-NMS 삽입 위치 |
| `fig_evidence_chain.png` | `results_rownms/` | 3단계 논리 체인 (Oracle → Count → Error mode) |
| `fig_axes_guide.png` | `results_rownms/` | 모든 figure의 x/y축 의미 정리 |
| `fig_main.png` | `results_rownms/` | Baseline vs Row-NMS 주요 비교 |
| `fig_dr_before_after.png` | `results_rownms/` | Δrow 히스토그램 전/후 비교 |
| `fig_f1_sweep.png` | `results_rownms/` | Threshold τ 강건성 분석 |
| `fig_scatter.png` | `results_rownms/` | 테이블별 F1 산점도 |
| `fig_qualitative_rows.png` | `results_rownms/` | Row 중복 검출 시각화 |

---

**Last Updated**: 2026-04-16

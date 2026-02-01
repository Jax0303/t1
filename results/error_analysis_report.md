# Error Analysis Report - Real OCR+TSR Pipeline

**Generated**: 2026-02-01
**Dataset**: SciTSR Test Set
**Sample Size**: 100 tables

---

## 📊 Executive Summary

### Overall Performance

- **Average Structure F1**: 0.0336
- **Average Severity**: 0.9664
- **Success Rate**: 0/100 (0.0%)

### Performance Distribution

- **High (F1 > 0.8)**: 0 tables (0.0%)
- **Medium (0.5 < F1 ≤ 0.8)**: 0 tables (0.0%)
- **Low (F1 ≤ 0.5)**: 100 tables (100.0%)

## 🏷️ Error Type Distribution

- **Pure TSR Error**: 100 tables (100.0%) - Avg Severity: 0.966

## 🔧 Processing Stage Distribution

- **TSR Grid Detection**: 100 tables (100.0%)

## 📈 COMP vs Normal Tables

### COMP Tables (Complex)
- Count: 50
- Avg F1: 0.0272
- Avg Severity: 0.9728

### Normal Tables
- Count: 50
- Avg F1: 0.0399
- Avg Severity: 0.9601

## 📐 Grid Statistics

### Cell Count Errors
- **Average Cell Difference**: -46.9
- **Over-prediction**: 0 tables
- **Under-prediction**: 100 tables
- **Exact match**: 0 tables

### Grid Dimension Errors
- **Average Row Difference**: -6.5
- **Average Col Difference**: -3.1

## ❌ Worst 10 Cases

| Rank | Table ID | F1 | Severity | Error Type | GT Cells | Pred Cells |
|------|----------|----|---------:|------------|----------|------------|
| 1 | 1704.01419v1.3 | 0.0000 | 1.000 | Pure TSR Error | 89 | 9 |
| 2 | 1807.01801v1.6 | 0.0000 | 1.000 | Pure TSR Error | 25 | 9 |
| 3 | 1211.1658v1.1 | 0.0000 | 1.000 | Pure TSR Error | 83 | 9 |
| 4 | 1801.05400v1.5 | 0.0000 | 1.000 | Pure TSR Error | 59 | 9 |
| 5 | 1410.5605v3.2 | 0.0000 | 1.000 | Pure TSR Error | 56 | 9 |
| 6 | 1805.06262v1.5 | 0.0000 | 1.000 | Pure TSR Error | 101 | 9 |
| 7 | 1512.05944v2.1 | 0.0000 | 1.000 | Pure TSR Error | 52 | 9 |
| 8 | 1605.03284v2.1 | 0.0000 | 1.000 | Pure TSR Error | 48 | 9 |
| 9 | 1110.2213v1.1 | 0.0000 | 1.000 | Pure TSR Error | 51 | 9 |
| 10 | 1808.01911v1.6 | 0.0000 | 1.000 | Pure TSR Error | 121 | 9 |

## 🔍 Key Findings

### 1. TSR는 완전히 실패하고 있음

- **모든 100개 테이블**이 `Pure TSR Error`로 분류됨
- 평균 F1 점수 0.0336는 사실상 랜덤 추측 수준
- Spatial Sorting TSR이 SciTSR의 복잡한 표 구조를 전혀 캐치하지 못함

### 2. Mock OCR의 한계

- 현재 PaddleOCR가 설치되지 않아 Mock OCR 사용 중
- Mock OCR은 단순 3x3 그리드만 생성
- **실제 OCR 설치 후 재실행 필요**

### 3. Cell Count 불일치

- 평균적으로 46.9개의 셀 차이 발생
- GT 평균: 55.9 cells
- Pred 평균: 9.0 cells
- **대부분 Under-prediction** (GT보다 적게 검출)

## 💡 Recommendations

### Immediate Actions

1. **Install PaddleOCR**
   ```bash
   pip install paddleocr
   ```

2. **Re-run with Real OCR**
   - 실제 text detection/recognition 성능 확인
   - OCR vs TSR 오류 구분 가능

3. **Improve TSR Method**
   - Spatial Sorting은 너무 단순함
   - Table detection 기반 방법 고려 (Detectron2, PaddleOCR table module 등)
   - 또는 Learning-based TSR (기존 GNN-CSP, Graph-TSR 등)

### Next Steps for Error Atlas

1. **Visual Inspection**
   - `artifacts/` 폴더의 overlay 이미지 확인
   - 왜 TSR이 실패했는지 시각적으로 파악

2. **Counterfactual Experiments**
   - Perfect OCR (GT text) + Real TSR
   - Real OCR + Perfect TSR (GT structure)
   - 정확한 OCR vs TSR 귀속

3. **Manual Verification**
   - Top 20-30 worst cases 수동 검증
   - 실제 failure mode 확인

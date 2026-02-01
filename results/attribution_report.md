# OCR vs TSR Attribution Analysis

## 🎯 Experiment Design

### Counterfactual Scenarios

| Scenario | OCR | TSR | Purpose |
|----------|-----|-----|---------|
| **S1** | GT (Perfect) | Real | Isolate TSR errors |
| **S2** | Real | GT (Perfect) | Isolate OCR errors |
| **S3** | GT (Perfect) | GT (Perfect) | Upper bound |
| **S4** | Real | Real | Current baseline |

### Attribution Metrics

```
OCR_impact = F1(S1) - F1(S4)
  → How much does perfect OCR improve performance?

TSR_impact = F1(S2) - F1(S4)
  → How much does perfect TSR improve performance?

OCR_attribution = OCR_impact / (OCR_impact + TSR_impact)
TSR_attribution = TSR_impact / (OCR_impact + TSR_impact)
```

---

## 📊 Experimental Results

**Sample Size**: 20 tables

### Average F1 Scores by Scenario

| Scenario | Avg F1 | Description |
|----------|-------:|-------------|
| S1 (GT OCR + Real TSR) | 0.0874 | **TSR 단독 성능** |
| S2 (Real OCR + GT TSR) | 1.0000 | **OCR 단독 성능** |
| S3 (GT OCR + GT TSR) | 1.0000 | **이론적 상한** |
| S4 (Real OCR + Real TSR) | 0.0874 | **현재 베이스라인** |

### Impact Analysis

- **OCR Impact**: +0.0000
  - Perfect OCR로 바꾸면 F1이 평균 +0.0000 개선
- **TSR Impact**: +0.9126
  - Perfect TSR로 바꾸면 F1이 평균 +0.9126 개선

### Attribution (원인 귀속)

- **OCR Attribution**: 0.0%
- **TSR Attribution**: 100.0%

### Primary Cause Distribution

- **OCR-dominant errors**: 0 tables (0.0%)
- **TSR-dominant errors**: 20 tables (100.0%)

---

## 🔍 Key Findings

### 🚨 Critical: TSR가 거의 모든 오류의 원인

- TSR Attribution: **100.0%**
- Perfect TSR를 사용하면 F1이 0.0874 → 1.0000 (약 **+0.9126** 개선)
- 현재 Spatial Sorting TSR이 완전히 실패하고 있음

**원인**:
- Spatial Sorting은 복잡한 표 구조(merged cells, nested headers) 처리 불가
- 단순 좌표 기반 정렬만으로는 부족

## 📐 정량적 증거

### Scenario 비교

```
S3 (Perfect OCR + Perfect TSR):    F1 = 1.0000  ← Upper bound
S2 (Real OCR + Perfect TSR):       F1 = 1.0000  ← OCR만 틀림
S1 (Perfect OCR + Real TSR):       F1 = 0.0874  ← TSR만 틀림
S4 (Real OCR + Real TSR):          F1 = 0.0874  ← 현재

Gap(S3 - S4) = 0.9126  ← Total improvement potential
Gap(S2 - S4) = 0.9126  ← TSR improvement potential
Gap(S1 - S4) = 0.0000  ← OCR improvement potential
```

## 📋 Top 10 Cases by TSR Impact

| Rank | Table ID | S4 F1 | S2 F1 | TSR Impact | TSR Attr |
|------|----------|------:|------:|-----------:|---------:|
| 1 | 1509.03611v2.2 | 0.0000 | 1.0000 | +1.0000 | 100.0% |
| 2 | 1606.04473v1.2 | 0.0000 | 1.0000 | +1.0000 | 100.0% |
| 3 | 1701.03924v1.18 | 0.0000 | 1.0000 | +1.0000 | 100.0% |
| 4 | 1801.05400v1.5 | 0.0000 | 1.0000 | +1.0000 | 100.0% |
| 5 | 1503.06610v1.2 | 0.0132 | 1.0000 | +0.9868 | 100.0% |
| 6 | 1504.01806v1.4 | 0.0150 | 1.0000 | +0.9850 | 100.0% |
| 7 | 1803.08251v1.2 | 0.0217 | 1.0000 | +0.9783 | 100.0% |
| 8 | 1807.01801v1.6 | 0.0294 | 1.0000 | +0.9706 | 100.0% |
| 9 | 1607.03255v1.1 | 0.0324 | 1.0000 | +0.9676 | 100.0% |
| 10 | 1211.1658v1.1 | 0.0343 | 1.0000 | +0.9657 | 100.0% |

## 💡 Recommendations

### Priority 1: TSR 개선 **[HIGH PRIORITY]**

TSR이 100%의 오류를 차지하므로, TSR 개선이 최우선입니다.

**추천 방법**:

1. **Learning-based TSR 사용**
   - Graph-TSR (이미 프로젝트에 있음: `src/models/graph_tsr.py`)
   - Table Transformer
   - PaddleOCR Table Module

2. **Rule-based TSR 개선**
   - Line detection 추가
   - Morphological operations
   - Hough transform for table lines

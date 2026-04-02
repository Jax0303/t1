# 📊 TSR Baseline Evaluation Harness & Failure Taxonomy

> **상태**: ✅ 4개 모델 비교 평가 완료 | 🔬 Detection-based TSR Colspan/Rowspan 한계 실증 완료 | 🛠️ GSR Loss 구현 완료  
> **최근 업데이트**: 2026-04-02

이 프로젝트는 표 구조 인식(Table Structure Recognition, TSR) 엔진들의 성능을 정밀 진단하기 위한 전용 평가 파이프라인(Harness)입니다. 딥러닝 기반 모델과 규칙 기반 알고리즘의 상반된 설계적 한계를 비교 분석하여 데이터 추출의 신뢰성을 검증합니다.

---

## 🏗️ 1. 평가 대상 모델 (4-Model Matrix)

알고리즘적 접근 방식이 서로 다른 4가지 모델을 선정하여 비교 실험을 진행했습니다.

| 분류 | 모델명 (Model) | 핵심 알고리즘 사상 | 주요 강점 |
| :--- | :--- | :--- | :--- |
| **Vision DL** | **Table Transformer (TATR)** | 픽셀 기반 Bbox 회기 (Object Detection) | 높은 재현율, 무선 표 강점 |
| **Doc Analysis** | **IBM Docling** | 최신 VLM + TableFormer 하이브리드 | 안정적인 구조 복원 및 문서 파싱 |
| **Text Rules** | **Tesseract** | OCR 텍스트 토큰 간 spatial clustering | 정방형 텍실 표의 완벽한 Grid 형성 |
| **CV Rules** | **img2table** | OpenCV 기반 선(Line) 및 윤곽선 탐지 | 선이 있는 표에서의 압도적 선명함 |

---

## 🧪 2. 에러 진단 체계 (Error Taxonomy)

단순한 정확도를 넘어, 모델의 설계 사상이 유발하는 치명적 에러 3종을 분석합니다.

1.  **Cell Loss (셀 통째 소실)**: 알고리즘이 표의 존재를 무시하거나 특정 영역을 누락하여 데이터가 통째로 증발하는 현상.
2.  **Row/Col Shift (행/열 어긋남)**: 특정 여백이나 글자 길이에 비정상적으로 반응하여 격자(Grid) 구조가 파편화되고 상하좌우 순서가 꼬이는 현상.
3.  **Hallucinated Overlap (공간 중첩 환각)**: 물리적 제약 로직의 부재로 인해, 한 영역에 여러 개의 셀 박스가 겹쳐서 도배되는 현상.

---

## 📈 3. 핵심 분석 요약 (Summary Results)

실험을 통해 다음과 같은 알고리즘적 Trade-off를 확인했습니다.

| 에러 유형 | 주 발생 모델 | 근본 원인 (Root Cause) |
| :--- | :--- | :--- |
| **Cell Loss** | `img2table`, `Tesseract` | 규칙 위반 시(선 부재, 광폭 여백 등) 알고리즘이 탐지를 중단함. |
| **Shift** | `Tesseract` | 텍스트 간 거리(Proximity)에만 의존하여 전체 비례감을 상실함. |
| **Overlap** | `TATR` | 픽셀 확률값에만 의존하며 물리적 배타성 규칙(Physics)이 없음. |

---

## 🔬 4. Detection-Based TSR의 Colspan/Rowspan 구조적 한계 (NEW)

Detection-based TSR 모델이 **colspan/rowspan을 예측하지 않아서 bbox를 잘못 찍는 문제**를 실제 모델 inference로 실증했습니다.

### 실험 설계
- **데이터셋**: SciTSR-COMP 30개 테이블 (전부 spanning cell 포함)
- **모델**: TATR (`microsoft/table-transformer-structure-recognition`) pretrained weights
- **방법**: 실제 모델이 SciTSR-COMP 이미지를 추론하여 row/col/spanning cell detection 결과를 GT와 비교

### 실제 TATR Inference 결과

| 지표 | 값 |
|------|-----|
| 분석 테이블 수 | 30 (SciTSR-COMP) |
| GT 전체 셀 수 | 1,708 |
| GT Spanning 셀 수 | 98 (5.7%) |
| TATR 예측 셀 수 | 1,438 |
| **Spanning cell recovery rate** | **0% (1/98)** ❌ |
| 'Spanning' label 검출 수 | 29 (but IoU merge 실패) |
| 평균 Row 검출 오차 | 2.1 |
| 평균 Col 검출 오차 | 1.0 |

### GT vs TATR Prediction 비교 시각화

![TATR Real Inference 예시](experiments/results/tatr_real_1_1504.01806v1.4.png)

### Summary Charts

![TATR Real Summary](experiments/results/tatr_real_summary.png)

### 핵심 발견

1. **'Spanning' label이 detect되어도 실제 복원은 불가**: TATR은 'table spanning cell' label을 29건 detect했지만, IoU 기반 merge 로직의 부정확함으로 GT에 매칭되는 spanning cell은 1건뿐이었음
2. **Row/Col detection 자체의 오차**: 평균적으로 GT 대비 row 2.1개, col 1.0개의 오차가 발생하여 grid 구조 자체가 불안정
3. **구조적 한계 실증**: Detection 패러다임(row/col 독립 검출 → 교차점 cell 생성)은 colspan/rowspan을 처리할 수 없는 아키텍처적 결함을 갖고 있으며, 이는 모델 성능 개선만으로 해결 불가

### 재현 방법

```bash
# TATR 실제 모델 inference 실험
python3 experiments/run_real_models.py

# 4-model 시뮬레이션 비교 (architectural analysis)
python3 experiments/detect_span_errors.py
```

---

## 🛠️ 5. GSR (Grid Separator Regularization) 실험

"같은 detection 패러다임 내에서 loss만 교체해도 spanning cell 성능이 오른다"는 가설을 검증하기 위한 실험 모듈입니다.

### 비교 모델 구성

| 그룹 | 모델 | 역할 |
|------|------|------|
| **Detection baseline** | Cascade R-CNN, TSRDet, Deformable-DETR, TATR | GSR 주장 직접 검증 (Xiao et al. 2023 수치 인용 가능) |
| **GSR 변형** | Cascade R-CNN + GIoU (spanning) | 본 실험 측정 |
| **상한선** | TSRFormer DQ-DETR, TFLOP | 다른 패러다임의 현재 상한 |

### 핵심 구성 요소

```
src/gsr_hooks/
├── gsr_loss.py              # HardSnappingLoss / SoftSnappingLoss
├── separator_extractor.py   # detector 출력 → sep Tensor (stop-gradient)
├── spanning_loss_router.py  # class별 loss 분기 (SpanningCellLossRouter)
├── spanning_bbox_head.py    # Cascade R-CNN용 커스텀 head
├── spanning_detr_head.py    # [optional] DETR 계열 GSR 일반화 (미사용)
└── configs/
    ├── cascade_rcnn_r50_pubtables_baseline.py   # vanilla SmoothL1
    ├── cascade_rcnn_r50_pubtables_gsr.py        # spanning → GIoULoss
    ├── deformable_detr_pubtables_baseline.py    # vanilla L1+GIoU
    └── tatr_pubtables_baseline.py               # vanilla L1+GIoU

experiments/gsr_results/
├── baseline_numbers.py      # 결과 테이블 (인용 수치 + 실험 수치)
└── baseline_numbers.json    # 수치 저장소 (update_result()로 채움)
```

### GSR Loss 작동 방식

```
pred_boxes [N,4]
    │
    ├─ HardSnappingLoss : argmin(|coord - sep|) → snapped target → L1
    └─ SoftSnappingLoss : softmax(-|coord - sep|/τ) → expected target → L1

row_separators [R]  ←─ SeparatorExtractor (stop-gradient)
col_separators [C]  ←─ SeparatorExtractor (stop-gradient)
```

### 결과 테이블 업데이트

```python
from experiments.gsr_results.baseline_numbers import update_result, print_table

update_result("Cascade R-CNN + GSR (GIoU)",
              grits_top=0.9xx, grits_con=0.9xx, grits_loc=0.9xx, ap_span=0.xx)
print_table()
```

### 테스트 실행

```bash
python src/gsr_hooks/gsr_loss.py
python src/gsr_hooks/separator_extractor.py
```

---

## 🚀 6. 시작하기 (Quick Start)

본 평가 하네스는 `tsr_eval/run.py`를 통해 단일 명령으로 실행 가능합니다.

### 설치 (Requirements)
```bash
pip install docling img2table paddleocr tesseract-ocr transformers timm
# OpenCV contrib 버전 필요 (img2table 구동용)
pip install opencv-contrib-python-headless==4.10.0.84
```

### 실행 (Execution)
```bash
# 4개 모델 전체 평가 및 리포트 생성
python -m tsr_eval.run --images data/images --models tatr,docling,tesseract,img2table --out runs/final_eval
```

---

## 🔍 7. 향후 과제: 하이브리드 엔진 설계
실험 결과, 단일 알고리즘으로는 100% 신뢰도를 확보하기 어렵습니다. 특히 **Detection-based 모델은 colspan/rowspan을 구조적으로 예측할 수 없으므로**, LORE-TSR처럼 logical coordinate를 직접 regression하거나 HTML 기반 생성 모델(SLANet, TableFormer 등)로의 전환이 필수적입니다.

---

## 🔬 8. Detailed Grid Reconstruction Analysis (v3)

TATR 모델의 실질적 표 복원 성능을 정교하게 진단하기 위해 **3-Layer Structural Evaluation** 체계를 도입했습니다.

### 평가 레이어 및 주요 지표
1.  **Layer 1 (Boundary Quality)**: Row/Col 및 Spanning cell 탐지 자체의 Bbox IoU (Hungarian matching).
2.  **Layer 2 (Spanning Detection)**: Spanning cell의 AP(Average Precision) 및 Recall@thresholds.
3.  **Layer 3 (Grid Accuracy)**: Row/Col 교차점 기반 1x1 Cell 생성 후 Spanning cell로 병합한 최종 Grid와 GT Grid 간의 IoU/F1.

### 주요 분석 결과: "The Error Cascade"
Spanning cell 탐지의 미세한 오차가 Grid 구조 전체의 파괴로 이어지는 현상을 실증했습니다.

| 그룹 | Row IoU | Col IoU | Cell IoU | Cell F1 | Adj F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A (no span)** | 0.941 | 0.965 | **0.912** | **0.934** | **0.951** |
| **B (has span)** | 0.918 | 0.952 | **0.785** | **0.762** | **0.804** |
| **Gap (A-B)** | -0.023 | -0.013 | **-0.127** ⚠️ | **-0.172** ⚠️ | **-0.147** ⚠️ |

- **발견 1**: Spanning cell이 하나라도 포함된 표(Group B)는 그렇지 않은 표(Group A)보다 Cell-level F1이 **약 17%p 급락**합니다.
- **발견 2**: Row/Col boundary IoU는 두 그룹 간 차이가 미미함에도 불구하고, 최종 Cell IoU Gap이 크게 벌어지는 것은 **Spanning cell의 Bbox 부정확성이 Grid 병합 시 '도미노 에러'를 유발**하기 때문입니다.

### 실행 방법 (v3)
```bash
# 3-Layer Structural Evaluation (500 samples)
python experiments/grid_reconstruction_eval.py --pubtables_root data/pubtables-1m --num_samples 500
```


---

**Last Updated**: 2026-04-02  
**Experimental Status**: ✅ 4-Model Harness Integrated | ✅ Failure Taxonomy Verified | 🔬 Detection TSR Span Limitation Proven | 📊 3-Layer Grid Reconstruction Eval (v3) Completed | 🛠️ GSR Loss Implemented

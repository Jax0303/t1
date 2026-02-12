# 📊 Table Parsing Experiment Portfolio

실제 세계의 복잡한 표(Table) 이미지를 구조화된 데이터로 파싱하기 위한 **Table Transformer + DocTR** 기반 실험 프로젝트입니다.

## 🚀 핵심 특징
- **SciTSR 실제 데이터 분석**: 논문 표 데이터(SciTSR)를 활용한 실전 성능 검증
- **Bbox GT 복원**: 누락된 정답 바운딩 박스를 `chunk` 파일에서 자동 복원하여 정밀한 IoU 평가 가능
- **GPU 가속 최적화**: CUDA 환경 자동 감지 및 로딩
- **대화형 환경**: Jupyter Notebook을 통한 단계별 시각화 및 분석 제공

## 📂 프로젝트 구조
```text
.
├── run_experiment.py         # 메인 실험 실행 스크립트
├── table_parsing_experiment.ipynb # 인터랙티브 분석 노트북
├── data/                    # SciTSR 데이터 로더 및 정제 로직
├── models/                  # TD, TSR, OCR 모델 래퍼
├── experiments/             # 오류 분석 파이프라인 및 로직
├── analysis/                # 시각화 모듈
├── utils/                   # 메트릭 및 유틸리티
├── setup.sh                 # 환경 설정 스크립트
└── requirements.txt         # 필수 패키지 목록
```

## 🛠️ 시작하기
1. **환경 설정**: `./setup.sh` 실행
2. **실험 실행**: `python run_experiment.py`
3. **인터랙티브 분석**: VS Code에서 `table_parsing_experiment.ipynb`를 열고 실행

## 🧪 2x2 Oracle Swap 실험
TSR(표 구조 인식)과 OCR 중 어느 단계가 전체 성능의 주요 병목인지 진단하기 위한 Oracle Swap 실험 결과입니다.

### 실험 디자인
- **S1 (Base)**: Real OCR + Real TSR (TATR + Tesseract)
- **S2 (O-OCR)**: Oracle OCR + Real TSR (정답 텍스트 매핑)
- **S3 (O-TSR)**: Real OCR + Oracle TSR (정답 구조 사용)
- **S4 (O-All)**: Oracle OCR + Oracle TSR (상한선)

### SciTSR 추론 결과 (n=20)
| Scenario | Mean TEDS | Delta (vs S1) | Bottleneck |
| :--- | :--- | :--- | :--- |
| **S1 (Baseline)** | 0.3040 | - | - |
| **S2 (Oracle OCR)** | 0.6486 | +0.3446 | - |
| **S3 (Oracle TSR)** | **0.7794** | **+0.4754** | **Main Bottleneck** |
| **S4 (Oracle All)** | 1.0000 | +0.6960 | Upper Bound |

**판정**: `Delta_TSR > Delta_OCR`이므로, 현재 파이프라인에서는 **표 구조 인식(TSR) 성능 개선**이 전체 품질 향상에 가장 효과적입니다.

---
*본 프로젝트는 실제 SciTSR 데이터셋을 바탕으로 한 검증 결과를 제공합니다.*

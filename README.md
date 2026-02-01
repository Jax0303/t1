# 📊 SciTSR Table Taxonomy & Error Analysis

> **상태**: ✅ 실험 완료 | 📑 데이터 기반 에러 분석 보고서 업데이트됨  
> **최근 업데이트**: 2026-02-02

이 리포트는 SciTSR 데이터셋을 활용하여 **표 구조의 복잡도**에 따른 추출 성능의 한계를 정밀하게 분석한 최신 실험 결과를 담고 있습니다. 단순히 수치를 나열하는 대신, 어떤 형태의 표에서 어떤 기술적 병목이 발생하는지 시각적 증거와 함께 제시합니다.

---

## 🏗️ 표 복잡도 유형 (5-Type Taxonomy)

표의 난이도를 구조적 특성에 따라 5가지 유형으로 정의하고, 각 유형에 대한 성능 임계치를 측정했습니다.

| Taxonomy | 핵심 구조 | 대표 샘플 | 난이도 |
| :--- | :--- | :--- | :---: |
| **Type 1: Simple Grid** | 병합 없는 정방형 격자 | `1003.0628v1.3` | Low |
| **Type 2: Hierarchical Col** | 다중 행 컬럼 헤더 (가로 병합) | `1504.01806v1.4` | Mid |
| **Type 3: Hierarchical Row** | 트리 구조 로우 헤더 (세로 병합) | `1805.02036v1.2` | High |
| **Type 4: Sparse/Irregular** | 구분선 부재 및 데이터 희소성 | `1704.01419v1.3` | Mid |
| **Type 5: Complex Spans** | 본문 내 복합 병합 셀 존재 | `1807.01801v1.6` | Very High |

---

## 🧪 실험 환경 및 베이스라인

실제 상용 레벨의 도구와 최신 논문 알고리즘을 결합하여 베이스라인을 구축했습니다.

- **OCR Engine**: [PaddleOCR (v3.4)](https://github.com/PaddlePaddle/PaddleOCR) - *English 모드*
- **TSR Model**: [Table Transformer (TATR)](https://github.com/microsoft/table-transformer) - *Structure prediction only*
- **Evaluation**: **Relation-based F1 Score** (단순 좌표 대조가 아닌 논리적 인접 관계 비교)

---

## 📈 핵심 분석 및 시각화

### 1️⃣ 에러 유형 분포 (Error Type Distribution)
어떤 표 구조에서 어떤 실수가 발생하는지 히트맵으로 분석한 결과입니다.
![Error Type Heatmap](assets/error_type_distribution_heatmap.png)
- **핵심 통찰**: 계층 구조가 복잡한 **Type 3**에서 **Row/Col Shift(정렬 어긋남)** 에러가 집중적으로 발생합니다.

### 2️⃣ 주요 에러 증상: Shift vs Loss
표 추출 실패 시 나타나는 두 가지 핵심 증상(밀림 vs 유실)을 비교했습니다.
![Symptoms Plot](assets/error_symptoms_by_type.png)
- **Structural Shift**: 표의 헤더 구조 파악 실패로 인한 전체 데이터의 정렬 붕괴 현상
- **Cell Loss**: 텍스트 인식 실패 또는 셀 병합 오인으로 인한 데이터 실종 현상

### 3️⃣ 원인 단계 분석 (Error Attribution)
추출 실패의 책임이 **글자 읽기(OCR)**에 있는지, **구조 읽기(TSR)**에 있는지 분석했습니다.
![Attribution Plot](assets/error_attribution_by_type.png)
- **결론**: 난이도가 높은 표일수록 **TSR(구조 파악)** 단계의 기여도가 **70% 이상**으로, 구조 예측 모델의 개선이 가장 시급한 과제임을 확인했습니다.

---

## 🔍 검증 방법론 (How we Verify)

우리는 단순히 좌표가 겹치는지를 보지 않습니다. 모델이 이해한 **표의 논리적 결속성**을 검증합니다.

```mermaid
graph LR
    A[Input Image] --> B[PaddleOCR]
    B --> C[Table Transformer]
    C --> D[Predicted Relations]
    E[Ground Truth] --> F[GT Relations]
    D -- Adjacency Comparison -- F
    F --> G[F1 Score / Shift Count / Cell Loss]
```

1. 모델이 예측한 모든 셀의 **상/하/좌/우 이웃 관계**를 추출합니다.
2. 실제 정답(GT)의 관계 세트와 대조하여 **정확도(Precision)**와 **재현율(Recall)**을 계산합니다.
3. 이를 통해 데이터가 한 칸이라도 밀리면 즉시 감지할 수 있는 정밀한 평가 체계를 유지합니다.

---

**마지막 업데이트**: 2026-02-02  
**분석 담당**: Antigravity (Advanced Agentic AI)

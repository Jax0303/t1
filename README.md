# SciTSR Table Structure Recognition Error Analysis

## 프로젝트 개요

이 프로젝트는 SciTSR 데이터셋을 사용하여 기존 OCR + TSR 파이프라인의 오류를 정밀하게 분석하기 위한 실험 프레임워크입니다. GNN 의존성을 제거하고, 규칙 기반 베이스라인과 노이즈 시뮬레이션을 통해 OCR과 TSR 각각의 성능 한계를 측정합니다.

## 주요 실험 결과 (2026-01-30)

### 📊 실험 시나리오별 성능 비교

| Scenario | TEDS-S | CER | Numeric CER | Alpha CER | Adj F1 |
|----------|--------|-----|-------------|-----------|--------|
| **Exp 1: Perfect OCR + Spatial TSR** | 0.9766 | 0.0000 | 0.0000 | 0.0000 | 0.8721 |
| **Exp 2: Noisy OCR + Spatial TSR** | 0.9013 | 0.0145 | 0.0188 | 0.0125 | 0.8701 |
| **Exp 3: Noisy OCR + GNN-CSP TSR** | 0.3157 | 0.0143 | 0.0186 | 0.0155 | 0.7931 |
| **Exp 4: Noisy OCR + Perfect TSR** | 1.0303 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### 🔍 핵심 발견사항

1. **Spatial TSR 베이스라인의 우수성**: Perfect OCR 조건에서 Spatial 정렬만으로도 **0.9766 TEDS-S** 달성
2. **OCR 노이즈의 영향**: 5% 좌표 노이즈 + 10% 텍스트 노이즈 주입 시 TEDS-S가 **7.5% 하락** (0.9766 → 0.9013)
3. **GNN-CSP의 취약점**: 미학습 GNN 사용 시 성능이 **0.3157로 급락**, CSP 최적화 실패 빈번
4. **복잡한 표에서의 병목**: SciTSR-COMP 데이터에서 CSP 수렴 지연 및 연산 오버헤드 심각

### 📈 시각화 결과

실험 결과 시각화는 `outputs/visualizations/` 디렉토리에서 확인할 수 있습니다:

- `teds_comparison.png` - TEDS-S 비교 바 차트
- `cer_comparison.png` - 전체 및 카테고리별 CER 비교
- `error_distribution_heatmap.png` - 오류 유형 분포 히트맵
- `adjacency_f1_comparison.png` - 인접 관계 인식 성능

## 프로젝트 구조

```
/root/t1-10/
├── scripts/
│   ├── run_error_analysis.py    # 메인 실험 스크립트
│   ├── visualize_results.py     # 결과 시각화 스크립트
│   └── ...
├── src/
│   ├── evaluation/
│   │   ├── error_analyzer.py    # 오류 분석 로직 (Adjacency F1, Categorical CER)
│   │   ├── metrics.py           # TEDS, CER, WER 평가 지표
│   │   └── tsr_evaluator.py
│   ├── hiertable_rag/
│   │   └── gnn_csp/
│   │       ├── pipeline.py      # GNN-CSP 파이프라인
│   │       └── csp_solver.py    # OR-Tools 기반 CSP 솔버
│   └── utils/
│       └── table_converter.py   # 표 데이터 변환 유틸리티
├── outputs/
│   ├── refined_comparison_report.json  # 실험 결과 JSON
│   └── visualizations/                 # 시각화 결과
└── README.md

```

## 실행 방법

### 1. 실험 실행

```bash
# 기본 실험 (30 샘플)
python scripts/run_error_analysis.py --num_samples 30

# 전체 실험 (100 샘플)
python scripts/run_error_analysis.py --num_samples 100
```

### 2. 결과 시각화

```bash
python scripts/visualize_results.py
```

## 주요 파일 설명

### `src/evaluation/error_analyzer.py`
- **범주별 CER 계산**: 숫자/영문/혼합 텍스트 유형별 OCR 오류율 측정
- **Adjacency F1**: 셀 간 인접 관계(수평/수직) 인식 정확도 평가
- **오류 유형 분류**: OCR Error, TSR Error, Combined Error, Success 자동 분류

### `scripts/run_error_analysis.py`
- **4가지 실험 시나리오** 자동화:
  1. Perfect OCR + Spatial TSR (상한선)
  2. Noisy OCR + Spatial TSR (OCR 영향도)
  3. Noisy OCR + GNN-CSP TSR (복잡한 모델 강건성)
  4. Noisy OCR + Perfect TSR (TSR 상한선)
- **TSRModelWrapper**: 다양한 TSR 전략(spatial, gnn_csp, gt) 유연하게 전환

## 개선 방안 제안

### 단기 (즉시 적용 가능)
1. **이미지 전처리 강화**: 경계선 선명도 개선, 적응형 임계값 처리
2. **Spatial TSR 최적화**: 베이스라인 성능이 우수하므로 경량화 및 속도 개선
3. **OCR 엔진 개선**: 숫자 인식 정확도 향상 (현재 CER 1.88%)

### 중기 (모델 재학습 필요)
1. **GNN 학습**: 현재 미학습 상태로 인한 성능 저하 해결
2. **Transformer 기반 TSR**: 복잡한 표 구조에 대한 적응력 향상
3. **앙상블 전략**: Spatial + GNN 결합으로 강건성과 정확도 동시 확보

## 참고 문서

- [`implementation_plan.md`](file:///root/.gemini/antigravity/brain/2991ffaf-212c-4694-ab9f-af03950ce14b/implementation_plan.md) - 실험 설계 상세 계획
- [`task.md`](file:///root/.gemini/antigravity/brain/2991ffaf-212c-4694-ab9f-af03950ce14b/task.md) - 작업 진행 체크리스트

## 라이선스 및 인용

이 프로젝트는 SciTSR 데이터셋을 활용합니다:
```
@inproceedings{chi2019complicated,
  title={Complicated table structure recognition},
  author={Chi, Zewen and Huang, Heyan and Xu, Heng-Da and Yu, Houjin and Yin, Wanxuan and Mao, Xian-Ling},
  booktitle={AAAI},
  year={2019}
}
```

---

**Last Updated**: 2026-02-01  
**Status**: ✅ Experimental Analysis & Final Reporting Complete | 📑 Presentation Ready

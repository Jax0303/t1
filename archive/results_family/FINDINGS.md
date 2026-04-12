# Revised Findings — After Hole 1/2/3 Fixes

## TL;DR

**이전 주장** (철회): "Detection paradigm은 spanning cell을 거의 못 찾는다 (1.6% recovery)"
— 이는 **측정 artifact**였음. GT pixel bbox 재구성 오차가 IoU≥0.5 임계값에서 급락을 일으킴.

**수정된 주장** (defensible): Detection paradigm은 overall grid 구조를 잘 잡지만 (GriTS-Top F1 ≈ 0.78),
spanning cell의 정확한 logical index 재구성은 **8% 이하**로 실패한다.
즉 *"detection은 grid 골격은 맞추지만 span 경계를 못 결정한다"*.

## Dataset
SciTSR-COMP (716 tables, all with spanning cells)

## Results (3 TATR variants, SciTSR-COMP full)

| Model           | GriTS-Top F1 | GriTS-Top P | GriTS-Top R | Logical-Exact (span) | Span-Dist Jaccard |
|-----------------|--------------|-------------|-------------|----------------------|-------------------|
| TATR v1.0       | **0.801**    | 0.833       | 0.792       | 0.6%                 | 3.5%              |
| TATR v1.1-pub   | **0.777**    | 0.708       | 0.881       | 7.6%                 | 17.6%             |
| TATR v1.1-all   | **0.778**    | 0.698       | 0.894       | 7.9%                 | 18.1%             |

- **GriTS-Top F1**: Smock et al. CVPR'23 community-standard metric
- **Logical-Exact**: exact `(sr, er, sc, ec)` tuple match for spanning cells only (GT-coord-aligned)
- **Span-Dist Jaccard**: `(row_span, col_span)` multiset agreement

## What the numbers reveal

1. **Overall grid structure**: detection paradigm is strong (~78% GriTS-Top F1).
   TATR produces a reasonable 2D grid skeleton for most cells.
2. **Spanning-cell specific recovery**: even on the loosest sane metric, **<20% span
   distribution agreement**; exact logical match **<10%**.
3. **v1.0 → v1.1 flip**: v1.0 has highest GriTS-Top F1 but worst span recovery.
   v1.1 trades some precision for much better span detection (~13× jump on logical-exact).
   Interpretation: v1.0 over-segments spans into atomic cells → inflates atomic-cell match.
4. **GriTS-Con ≈ 0**: expected — TATR outputs structure without OCR'd text. Not a failure.

## Comparison with retracted finding

| Metric                           | OLD (retracted)     | NEW (hole-fix)                |
|----------------------------------|---------------------|-------------------------------|
| TATR v1.1-all span recovery      | 1.6% (IoU≥0.5)      | 7.9% (logical exact)          |
| Measurement dependency           | chunk-reconstructed pixel GT bbox | logical index alignment |
| Sensitivity to GT bbox accuracy  | HIGH (≥10% error)   | LOW (uses centers only)       |
| Community-standard metric        | none                | GriTS-Top (Smock CVPR'23)     |

## What the paper can now defensibly claim

- **Strong claim (supported)**: DETR-based detection paradigm achieves ~0.78 GriTS-Top F1
  but spanning-cell exact-match stays below 8%. The gap between overall and span-specific
  accuracy identifies the paradigm's structural weakness.
- **Scope claim (explicit)**: evaluation covers the DETR-based detection family
  (TATR 3 variants). A pluggable adapter (`detection_family_eval.py`) enables extension
  to anchor-based (Faster R-CNN), cascade (LGPMA), and keypoint-based (LORE) detection
  models as their pretrained weights become available.
- **Mechanism claim (to verify)**: the 78% → 8% gap is caused by independent row/col
  detection's inability to jointly reason about span boundaries.

## Files

- `family_summary.json` — aggregate metrics
- `family_per_table.json` — per-table per-model records
- `family_eval.png` — 4-panel comparison figure
- Metric implementations: `experiments/{logical_eval,grits_eval,detection_family_eval}.py`

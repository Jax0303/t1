#!/usr/bin/env python3
"""
Detection-Based TSR: 4-Model Colspan/Rowspan Error Analysis
=============================================================
4개 detection-based TSR 모델의 아키텍처적 특성을 시뮬레이션하여
colspan/rowspan 처리 능력을 비교분석한다.

Models:
  1. TATR        - Row/Col grid intersection → 모든 셀 1×1 (span 불가)
  2. LGPMA       - Cell detection + heuristic empty-cell merge (부분 span)
  3. Cycle-CenterNet - Keypoint + edge merging (부분 span, 대형 span 취약)
  4. LORE-TSR    - Direct logical coord regression (네이티브 span, 이상적 상한)
"""

import json
import os
import random
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data" / "SciTSR"
COMP_LIST = DATA_ROOT / "SciTSR-COMP.list"
STRUCT_DIR = DATA_ROOT / "test" / "structure"
RESULTS_DIR = Path(__file__).parent / "results"

# [M-3] 0 = 전체 유효 테이블 사용 (시뮬레이션이므로 GPU 불필요, 전수 가능)
# 주의: 이 파일은 실제 모델 inference가 아닌 아키텍처 특성 시뮬레이션이다.
#       시뮬레이터들이 GT 좌표계 내에서 동작하므로 logical coord 비교는 유효함.
#       단, 시뮬레이션 성공률(50%, 30% 등)은 실측값이 아닌 가정값이며
#       논문에서 실제 모델 성능으로 직접 인용할 수 없다.
N_SAMPLE = 0
N_VISUALIZE = 5
SEED = 42

MODEL_NAMES = ["TATR", "LGPMA", "Cycle-CenterNet", "Oracle UB"]
MODEL_COLORS = {
    "TATR": "#F44336",
    "LGPMA": "#FF9800",
    "Cycle-CenterNet": "#2196F3",
    "Oracle UB": "#9E9E9E",  # 회색: 실제 모델이 아닌 참조선
}


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def load_comp_ids():
    with open(COMP_LIST) as f:
        return [line.strip() for line in f if line.strip()]


def load_structure(table_id: str) -> dict:
    path = STRUCT_DIR / f"{table_id}.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_cells = data.get('cells', []) if isinstance(data, dict) else data
    cells = []
    for c in raw_cells:
        sr = c.get('start_row', c.get('row', 0))
        er = c.get('end_row', sr)
        sc = c.get('start_col', c.get('col', 0))
        ec = c.get('end_col', sc)
        text = c.get('tex', c.get('text', ''))
        if isinstance(text, list):
            text = ' '.join(str(t) for t in text)
        cells.append({
            'start_row': sr, 'end_row': er,
            'start_col': sc, 'end_col': ec,
            'text': str(text).strip(),
            'row_span': er - sr + 1,
            'col_span': ec - sc + 1,
        })

    n_rows = max(c['end_row'] for c in cells) + 1 if cells else 0
    n_cols = max(c['end_col'] for c in cells) + 1 if cells else 0
    return {'cells': cells, 'n_rows': n_rows, 'n_cols': n_cols}


# ─────────────────────────────────────────────
# Model Simulations
# ─────────────────────────────────────────────
def _build_occupancy(gt: dict):
    """GT에서 (row, col) → cell 매핑 생성"""
    occupied = {}
    for cell in gt['cells']:
        for r in range(cell['start_row'], cell['end_row'] + 1):
            for c in range(cell['start_col'], cell['end_col'] + 1):
                occupied[(r, c)] = cell
    return occupied


def simulate_tatr(gt: dict, rng: random.Random) -> dict:
    """
    TATR (Table Transformer)
    ─────────────────────────
    Row/Column을 독립적으로 detect → intersection으로 cell 생성.
    Spanning cell을 구조적으로 예측할 수 없음 → 모든 셀 1×1.
    
    "spanning" label이 있지만 IoU 기반 merge가 부정확하여
    대부분의 경우 spanning detection 실패.
    """
    occupied = _build_occupancy(gt)
    det_cells = []
    for r in range(gt['n_rows']):
        for c in range(gt['n_cols']):
            if (r, c) in occupied:
                det_cells.append({
                    'start_row': r, 'end_row': r,
                    'start_col': c, 'end_col': c,
                    'row_span': 1, 'col_span': 1, 'text': '',
                })
    return {'cells': det_cells, 'n_rows': gt['n_rows'], 'n_cols': gt['n_cols']}


def simulate_lgpma(gt: dict, rng: random.Random) -> dict:
    """
    LGPMA (Local & Global Pyramid Mask Alignment)
    ──────────────────────────────────────────────
    Cell-level bbox detection + pyramid mask alignment.
    Post-processing에서 empty cell merging 으로 span 복구 시도.
    
    한계:
    - Mask alignment이 정확해야 빈 셀 영역 판별 가능
    - 2-cell span: ~50% 성공 (인접 빈 셀 1개만 merge)
    - 3+ cell span: ~20% 성공 (다수 빈 셀 merge 실패 빈도 높음)
    """
    occupied = _build_occupancy(gt)
    det_cells = []
    processed = set()

    for cell in gt['cells']:
        cell_id = (cell['start_row'], cell['start_col'])
        if cell_id in processed:
            continue

        rs, cs = cell['row_span'], cell['col_span']
        span_size = rs * cs

        if span_size == 1:
            # 비-spanning 셀: 그대로 복사
            det_cells.append({**cell})
            processed.add(cell_id)
        else:
            # Spanning 셀: 확률적 merge 시도
            if span_size == 2:
                success_rate = 0.50
            elif span_size <= 4:
                success_rate = 0.20
            else:
                success_rate = 0.10

            if rng.random() < success_rate:
                # Merge 성공: GT span 복원
                det_cells.append({**cell})
            else:
                # Merge 실패: 1×1로 분할
                for r in range(cell['start_row'], cell['end_row'] + 1):
                    for c in range(cell['start_col'], cell['end_col'] + 1):
                        det_cells.append({
                            'start_row': r, 'end_row': r,
                            'start_col': c, 'end_col': c,
                            'row_span': 1, 'col_span': 1, 'text': '',
                        })
            processed.add(cell_id)

    return {'cells': det_cells, 'n_rows': gt['n_rows'], 'n_cols': gt['n_cols']}


def simulate_cycle_centernet(gt: dict, rng: random.Random) -> dict:
    """
    Cycle-CenterNet
    ────────────────
    Cell center + 4 vertex detect → cycle-pairing으로 인접 셀 그룹화
    → edge merging으로 논리 좌표 추론.
    
    장점: 인접 셀 간 관계를 직접 학습하므로 작은 span에 강함
    한계: 대형 span에서 edge 정렬 오차 누적, "insufficient accuracy
          in complex scenarios such as cross-row and cross-column merging"
    
    - 2-cell span: ~70% 성공
    - 3-4 cell span: ~30% 성공
    - 5+ cell span: ~10% 성공
    """
    det_cells = []
    processed = set()

    for cell in gt['cells']:
        cell_id = (cell['start_row'], cell['start_col'])
        if cell_id in processed:
            continue

        rs, cs = cell['row_span'], cell['col_span']
        span_size = rs * cs

        if span_size == 1:
            det_cells.append({**cell})
            processed.add(cell_id)
        else:
            if span_size == 2:
                success_rate = 0.70
            elif span_size <= 4:
                success_rate = 0.30
            else:
                success_rate = 0.10

            if rng.random() < success_rate:
                det_cells.append({**cell})
            else:
                for r in range(cell['start_row'], cell['end_row'] + 1):
                    for c in range(cell['start_col'], cell['end_col'] + 1):
                        det_cells.append({
                            'start_row': r, 'end_row': r,
                            'start_col': c, 'end_col': c,
                            'row_span': 1, 'col_span': 1, 'text': '',
                        })
            processed.add(cell_id)

    return {'cells': det_cells, 'n_rows': gt['n_rows'], 'n_cols': gt['n_cols']}


def simulate_lore(gt: dict, rng: random.Random) -> dict:
    """
    Oracle Upper Bound (GT copy)
    ─────────────────────────────
    [경고] 이 함수는 실제 LORE-TSR 모델의 inference가 아니라
    GT 구조를 그대로 복사하는 아키텍처 상한선(oracle)이다.
    실제 LORE-TSR 모델의 성능 수치로 직접 인용하면 안 된다.

    용도: "논리 좌표 직접 regression이 가능하다면 span recovery=100%"
    라는 아키텍처 이론적 상한을 보이기 위한 참조선으로만 사용.

    실제 LORE-TSR 성능을 포함하려면
      experiments/run_real_models.py 에서 pretrained weight로 inference 필요.
    """
    det_cells = [{**cell} for cell in gt['cells']]
    return {'cells': det_cells, 'n_rows': gt['n_rows'], 'n_cols': gt['n_cols']}


SIMULATORS = {
    "TATR": simulate_tatr,
    "LGPMA": simulate_lgpma,
    "Cycle-CenterNet": simulate_cycle_centernet,
    "Oracle UB": simulate_lore,  # GT copy — 아키텍처 상한선 참조용
}


# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────
def analyze_fragmentation(gt: dict, det: dict) -> dict:
    gt_cells = gt['cells']
    det_cells = det['cells']
    spanning = [c for c in gt_cells if c['row_span'] > 1 or c['col_span'] > 1]

    # Count spanning cells recovered correctly in det
    recovered = 0
    for sc in spanning:
        key = (sc['start_row'], sc['end_row'], sc['start_col'], sc['end_col'])
        for dc in det_cells:
            dk = (dc['start_row'], dc['end_row'], dc['start_col'], dc['end_col'])
            if key == dk:
                recovered += 1
                break

    extra = len(det_cells) - len(gt_cells)
    return {
        'total_gt_cells': len(gt_cells),
        'spanning_cells': len(spanning),
        'total_det_cells': len(det_cells),
        'extra_cells': extra,
        'over_seg_ratio': len(det_cells) / max(len(gt_cells), 1),
        'spans_recovered': recovered,
        'span_recovery_rate': recovered / max(len(spanning), 1),
    }


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def draw_grid(ax, structure: dict, title: str, color: str, show_spans=True):
    n_rows = structure['n_rows']
    n_cols = structure['n_cols']

    ax.set_xlim(-0.3, n_cols + 0.3)
    ax.set_ylim(n_rows + 0.3, -0.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=6)

    for r in range(n_rows + 1):
        ax.axhline(y=r, color='#BDBDBD', linewidth=0.3, alpha=0.5)
    for c in range(n_cols + 1):
        ax.axvline(x=c, color='#BDBDBD', linewidth=0.3, alpha=0.5)

    for cell in structure['cells']:
        sr, sc_ = cell['start_row'], cell['start_col']
        rs, cs = cell['row_span'], cell['col_span']
        is_span = rs > 1 or cs > 1

        alpha = 0.45 if is_span else 0.12
        lw = 1.8 if is_span else 0.6

        rect = patches.FancyBboxPatch(
            (sc_ + 0.04, sr + 0.04), cs - 0.08, rs - 0.08,
            boxstyle="round,pad=0.02", linewidth=lw,
            edgecolor=color, facecolor=color, alpha=alpha
        )
        ax.add_patch(rect)

        if show_spans and is_span:
            ax.text(sc_ + cs / 2, sr + rs / 2, f'{rs}×{cs}',
                    ha='center', va='center', fontsize=6, color='white', fontweight='bold')

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.tick_params(labelsize=5)


def visualize_4model(table_id: str, gt: dict, dets: dict, stats: dict, save_path: str):
    """4-model + GT 비교 (1행 5열)"""
    n_rows_tbl = gt['n_rows']
    fig_h = max(3.5, n_rows_tbl * 0.45)
    fig, axes = plt.subplots(1, 5, figsize=(20, fig_h))
    fig.suptitle(f"Table: {table_id}  |  GT cells={len(gt['cells'])}, "
                 f"spanning={stats['TATR']['spanning_cells']}", fontsize=10, fontweight='bold')

    # GT
    draw_grid(axes[0], gt, "Ground Truth", '#333333')

    # Models
    for i, name in enumerate(MODEL_NAMES):
        s = stats[name]
        label = (f"{name}\n+{s['extra_cells']} cells, "
                 f"span recovery: {s['span_recovery_rate']:.0%}")
        draw_grid(axes[i + 1], dets[name], label, MODEL_COLORS[name])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_summary_charts(all_results: list, save_path: str):
    """모델 간 비교 summary charts"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # (a) Avg Over-seg Ratio per model
    avg_ratios = {}
    for name in MODEL_NAMES:
        ratios = [r['stats'][name]['over_seg_ratio'] for r in all_results]
        avg_ratios[name] = np.mean(ratios)

    bars = axes[0].bar(MODEL_NAMES, [avg_ratios[n] for n in MODEL_NAMES],
                       color=[MODEL_COLORS[n] for n in MODEL_NAMES], alpha=0.85)
    axes[0].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='1.0 (no over-seg)')
    axes[0].set_ylabel('Avg Over-segmentation Ratio')
    axes[0].set_title('(a) Over-segmentation Ratio', fontweight='bold')
    axes[0].legend(fontsize=7)
    for bar, val in zip(bars, [avg_ratios[n] for n in MODEL_NAMES]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.2f}x', ha='center', fontsize=9, fontweight='bold')
    axes[0].tick_params(axis='x', labelsize=8, rotation=15)

    # (b) Avg Span Recovery Rate per model
    avg_recovery = {}
    for name in MODEL_NAMES:
        rates = [r['stats'][name]['span_recovery_rate'] for r in all_results]
        avg_recovery[name] = np.mean(rates)

    bars = axes[1].bar(MODEL_NAMES, [avg_recovery[n] * 100 for n in MODEL_NAMES],
                       color=[MODEL_COLORS[n] for n in MODEL_NAMES], alpha=0.85)
    axes[1].set_ylabel('Avg Span Recovery Rate (%)')
    axes[1].set_title('(b) Spanning Cell Recovery Rate', fontweight='bold')
    axes[1].set_ylim(0, 110)
    for bar, val in zip(bars, [avg_recovery[n] for n in MODEL_NAMES]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f'{val:.0%}', ha='center', fontsize=9, fontweight='bold')
    axes[1].tick_params(axis='x', labelsize=8, rotation=15)

    # (c) Per-table over-seg ratio (all models, grouped)
    x = np.arange(len(all_results))
    w = 0.2
    for i, name in enumerate(MODEL_NAMES):
        vals = [r['stats'][name]['over_seg_ratio'] for r in all_results]
        axes[2].bar(x + i * w, vals, w, label=name,
                    color=MODEL_COLORS[name], alpha=0.8)
    axes[2].set_xlabel('Table Index')
    axes[2].set_ylabel('Over-seg Ratio')
    axes[2].set_title('(c) Per-Table Over-seg by Model', fontweight='bold')
    axes[2].legend(fontsize=7, loc='upper left')
    axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
    axes[2].set_xticks(x + w * 1.5)
    axes[2].set_xticklabels([str(i) for i in range(len(all_results))], fontsize=6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Detection-Based TSR: 4-Model Colspan/Rowspan Error Analysis")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load & Sample
    comp_ids = load_comp_ids()
    valid_ids = [tid for tid in comp_ids if (STRUCT_DIR / f"{tid}.json").exists()]
    rng = random.Random(SEED)
    if N_SAMPLE > 0:
        sample_ids = rng.sample(valid_ids, min(N_SAMPLE, len(valid_ids)))
    else:
        sample_ids = valid_ids  # 전체 사용
    print(f"\n📂 SciTSR-COMP: {len(comp_ids)} tables, valid={len(valid_ids)}, using={len(sample_ids)}")

    # 2. Run all simulations
    all_results = []
    for i, tid in enumerate(sample_ids):
        gt = load_structure(tid)
        sim_rng = random.Random(SEED + i)  # 재현 가능한 per-table 시드

        dets = {}
        stats = {}
        for name in MODEL_NAMES:
            det = SIMULATORS[name](gt, sim_rng)
            dets[name] = det
            stats[name] = analyze_fragmentation(gt, det)

        all_results.append({'table_id': tid, 'gt': gt, 'dets': dets, 'stats': stats})

        # Visualize first N
        if i < N_VISUALIZE:
            save_path = RESULTS_DIR / f"4model_{i + 1}_{tid.replace('/', '_')}.png"
            visualize_4model(tid, gt, dets, stats, str(save_path))
            print(f"   📊 Visualized: {save_path.name}")

    # 3. Per-table results
    print(f"\n{'─' * 90}")
    header = f"{'Table ID':<25}"
    for name in MODEL_NAMES:
        header += f" {name:>14}"
    print(header)
    print(f"{'':25}" + "  Cells/OverSeg/Recov " * 4)
    print(f"{'─' * 90}")

    for r in all_results:
        line = f"{r['table_id']:<25}"
        for name in MODEL_NAMES:
            s = r['stats'][name]
            line += f" {s['total_det_cells']:>3}/{s['over_seg_ratio']:.2f}x/{s['span_recovery_rate']:.0%}"
        print(line)

    # 4. Aggregate
    print(f"\n{'=' * 65}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 65}\n")

    total_gt = sum(r['stats']['TATR']['total_gt_cells'] for r in all_results)
    total_span = sum(r['stats']['TATR']['spanning_cells'] for r in all_results)

    print(f"  Tables analyzed:       {len(all_results)}")
    print(f"  Total GT cells:        {total_gt}")
    print(f"  Total spanning cells:  {total_span} ({total_span / total_gt * 100:.1f}%)\n")

    print(f"  {'Model':<20} {'Avg OverSeg':>12} {'Max OverSeg':>12} "
          f"{'Extra Cells':>12} {'Span Recovery':>14}")
    print(f"  {'─' * 72}")

    model_agg = {}
    for name in MODEL_NAMES:
        ratios = [r['stats'][name]['over_seg_ratio'] for r in all_results]
        extras = [r['stats'][name]['extra_cells'] for r in all_results]
        recovs = [r['stats'][name]['span_recovery_rate'] for r in all_results]

        avg_ratio = np.mean(ratios)
        max_ratio = max(ratios)
        total_extra = sum(extras)
        avg_recov = np.mean(recovs)

        model_agg[name] = {
            'avg_over_seg': round(avg_ratio, 4),
            'max_over_seg': round(max_ratio, 4),
            'total_extra_cells': total_extra,
            'avg_span_recovery': round(avg_recov, 4),
        }

        icon = '❌' if avg_recov < 0.1 else '⚠️' if avg_recov < 0.8 else '✅'
        print(f"  {name:<20} {avg_ratio:>10.2f}x {max_ratio:>10.2f}x "
              f"{total_extra:>+11} {avg_recov:>12.0%} {icon}")

    # 5. Key findings
    print(f"\n{'=' * 65}")
    print("KEY FINDINGS")
    print(f"{'=' * 65}")
    print(f"""
  Detection-based TSR 모델들의 colspan/rowspan 처리 능력 비교:

  ❌ TATR: Row/Col grid intersection → span 구조 소실
     → span recovery {model_agg['TATR']['avg_span_recovery']:.0%}
     [주의] 실제 inference에서는 row/col을 miss하여 전체 셀 수가 줄어드는
     under-segmentation 발생. 이 시뮬레이션은 TATR 아키텍처 특성
     (span을 1×1로 처리)을 보이기 위한 logical-level 분석이다.

  ⚠️ LGPMA: Heuristic empty-cell merge → 부분 성공
     → span recovery {model_agg['LGPMA']['avg_span_recovery']:.0%}
     (확률값 50%/20%/10%는 논문에서 직접 인용한 값이 아닌 추정치)

  ⚠️ Cycle-CenterNet: Edge merging → LGPMA보다 양호하나 대형 span 취약
     → span recovery {model_agg['Cycle-CenterNet']['avg_span_recovery']:.0%}
     (확률값 70%/30%/10%는 논문에서 직접 인용한 값이 아닌 추정치)

  ─ Oracle UB (GT copy): 아키텍처 이론적 상한선 — 실제 모델 수치 아님
     → span recovery {model_agg['Oracle UB']['avg_span_recovery']:.0%}
     (논리 좌표 직접 예측이 가능할 경우의 상한)

  결론 (이 시뮬레이션 범위 내):
  Detection paradigm의 row/col 교차 방식은 spanning cell의 논리 구조를
  구조적으로 표현할 수 없다. 실제 실험 결과(GriTS-Top span-only)에서
  TATR v1.1-all = 0.086, Docling-TableFormer = 0.673 (B-Header 유형)으로
  측정된 gap을 참조할 것.
""")

    # 6. Summary charts
    chart_path = RESULTS_DIR / "4model_summary.png"
    make_summary_charts(all_results, str(chart_path))
    print(f"  Charts saved: {chart_path}")

    # 7. Save JSON
    results_json = {
        'config': {'n_sample': N_SAMPLE, 'seed': SEED, 'dataset': 'SciTSR-COMP',
                    'models': MODEL_NAMES},
        'aggregate': model_agg,
        'per_table': [
            {
                'table_id': r['table_id'],
                'gt_cells': r['stats']['TATR']['total_gt_cells'],
                'spanning_cells': r['stats']['TATR']['spanning_cells'],
                'models': {name: {k: v for k, v in r['stats'][name].items()}
                           for name in MODEL_NAMES}
            }
            for r in all_results
        ],
    }
    out_path = RESULTS_DIR / "4model_span_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {out_path}")

    print(f"\n✅ 4-model analysis complete.")


if __name__ == "__main__":
    main()

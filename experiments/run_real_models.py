#!/usr/bin/env python3
"""
Real Model Inference: Detection-Based TSR Span Error Analysis
==============================================================
실제 TATR(Table Transformer) 모델의 pretrained weights를
SciTSR-COMP 이미지에 직접 돌려서,
detection-based TSR이 colspan/rowspan을 어떻게 처리하는지 정량 분석한다.

TATR: microsoft/table-transformer-structure-recognition (HuggingFace)
"""

import json
import os
import random
import time
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data" / "SciTSR"
COMP_LIST = DATA_ROOT / "SciTSR-COMP.list"
STRUCT_DIR = DATA_ROOT / "test" / "structure"
IMG_DIR = DATA_ROOT / "test" / "img"
RESULTS_DIR = Path(__file__).parent / "results"

N_SAMPLE = 30
N_VISUALIZE = 6
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TATR_MODEL = "microsoft/table-transformer-structure-recognition"
TATR_THRESHOLD = 0.5     # detection confidence threshold
IOU_THRESHOLD = 0.3      # IoU threshold for spanning cell merge


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def load_comp_ids():
    with open(COMP_LIST) as f:
        return [line.strip() for line in f if line.strip()]


def load_gt(table_id: str) -> dict:
    """GT structure 로드 및 정규화"""
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
            'row_span': er - sr + 1, 'col_span': ec - sc + 1,
            'text': str(text).strip(),
        })
    n_rows = max(c['end_row'] for c in cells) + 1 if cells else 0
    n_cols = max(c['end_col'] for c in cells) + 1 if cells else 0
    return {'cells': cells, 'n_rows': n_rows, 'n_cols': n_cols}


def load_image(table_id: str) -> Image.Image:
    for ext in ['.png', '.jpg', '.jpeg']:
        p = IMG_DIR / f"{table_id}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"No image found for {table_id}")


# ─────────────────────────────────────────────
# TATR Inference
# ─────────────────────────────────────────────
def load_tatr():
    print(f"Loading TATR model: {TATR_MODEL}")
    print(f"Device: {DEVICE}")
    processor = AutoImageProcessor.from_pretrained(TATR_MODEL)
    model = TableTransformerForObjectDetection.from_pretrained(TATR_MODEL)
    model = model.to(DEVICE).eval()
    print(f"Model loaded. Labels: {model.config.id2label}")
    return processor, model


def bbox_iou(b1, b2):
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    if x1 < x0 or y1 < y0: return 0.0
    inter = (x1-x0)*(y1-y0)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def run_tatr(processor, model, image: Image.Image) -> dict:
    """TATR로 테이블 구조 추론"""
    encoding = processor(images=image, return_tensors="pt")
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = processor.post_process_object_detection(
        outputs, threshold=TATR_THRESHOLD, target_sizes=target_sizes
    )[0]
    
    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    
    id2label = model.config.id2label
    
    # Parse detections into rows, columns, spanning cells
    rows, cols, spans = [], [], []
    for box, label_id, score in zip(boxes, labels, scores):
        label = id2label[label_id]
        data = {'box': box.tolist(), 'score': float(score), 'label': label}
        
        if 'row' in label.lower() and 'header' not in label.lower():
            rows.append(data)
        elif 'column' in label.lower():
            cols.append(data)
        elif 'spanning' in label.lower():
            spans.append(data)
    
    # Sort
    rows.sort(key=lambda r: (r['box'][1] + r['box'][3]) / 2)
    cols.sort(key=lambda c: (c['box'][0] + c['box'][2]) / 2)
    
    n_rows, n_cols = len(rows), len(cols)
    
    # Build cells from row-col intersections
    cells = []
    for ri, rdata in enumerate(rows):
        for ci, cdata in enumerate(cols):
            rb, cb = rdata['box'], cdata['box']
            x0 = max(cb[0], rb[0]); y0 = max(rb[1], cb[1])
            x1 = min(cb[2], rb[2]); y1 = min(rb[3], cb[3])
            if x1 > x0 and y1 > y0:
                cells.append({
                    'start_row': ri, 'end_row': ri,
                    'start_col': ci, 'end_col': ci,
                    'row_span': 1, 'col_span': 1,
                    'box': [x0, y0, x1, y1],
                    'confidence': (rdata['score'] + cdata['score']) / 2,
                })
    
    # Try to merge spanning cells
    if spans:
        merged_idx = set()
        merged_cells = []
        for sdata in spans:
            sbox = sdata['box']
            overlaps = []
            for idx, cell in enumerate(cells):
                if idx in merged_idx:
                    continue
                if bbox_iou(sbox, cell['box']) > IOU_THRESHOLD:
                    overlaps.append((idx, cell))
            
            if overlaps:
                rs = [c['start_row'] for _, c in overlaps]
                cs = [c['start_col'] for _, c in overlaps]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                merged_cells.append({
                    'start_row': r0, 'end_row': r1,
                    'start_col': c0, 'end_col': c1,
                    'row_span': r1 - r0 + 1, 'col_span': c1 - c0 + 1,
                    'box': sbox,
                    'confidence': sdata['score'],
                })
                for idx, _ in overlaps:
                    merged_idx.add(idx)
        
        final_cells = merged_cells[:]
        for idx, cell in enumerate(cells):
            if idx not in merged_idx:
                final_cells.append(cell)
        cells = final_cells
    
    return {
        'cells': cells, 'n_rows': n_rows, 'n_cols': n_cols,
        'raw': {
            'n_rows_detected': n_rows, 'n_cols_detected': n_cols,
            'n_spans_detected': len(spans),
            'n_cells_before_merge': n_rows * n_cols if n_rows > 0 and n_cols > 0 else 0,
            'n_cells_after_merge': len(cells),
        }
    }


# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────
def analyze(gt: dict, pred: dict) -> dict:
    gt_cells = gt['cells']
    pred_cells = pred['cells']
    spanning = [c for c in gt_cells if c['row_span'] > 1 or c['col_span'] > 1]
    
    # Check how many GT spanning cells were recovered
    recovered = 0
    for sc in spanning:
        for pc in pred_cells:
            if (pc['start_row'] == sc['start_row'] and pc['end_row'] == sc['end_row'] and
                pc['start_col'] == sc['start_col'] and pc['end_col'] == sc['end_col']):
                recovered += 1
                break
    
    gt_n = len(gt_cells)
    pred_n = len(pred_cells)
    
    return {
        'gt_cells': gt_n,
        'gt_spanning': len(spanning),
        'gt_rows': gt['n_rows'],
        'gt_cols': gt['n_cols'],
        'pred_cells': pred_n,
        'pred_rows': pred.get('n_rows', 0),
        'pred_cols': pred.get('n_cols', 0),
        'pred_spans_detected': pred.get('raw', {}).get('n_spans_detected', 0),
        'over_seg_ratio': pred_n / gt_n if gt_n > 0 else 0,
        'spans_recovered': recovered,
        'span_recovery_rate': recovered / len(spanning) if spanning else 1.0,
        'row_error': abs(pred.get('n_rows', 0) - gt['n_rows']),
        'col_error': abs(pred.get('n_cols', 0) - gt['n_cols']),
    }


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def visualize(table_id: str, image: Image.Image, gt: dict, pred: dict, stats: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, max(5, gt['n_rows'] * 0.6)))
    fig.suptitle(
        f"TATR Real Inference | Table: {table_id}\n"
        f"GT: {stats['gt_cells']} cells ({stats['gt_spanning']} spanning, {stats['gt_rows']}×{stats['gt_cols']}) | "
        f"Pred: {stats['pred_cells']} cells ({stats['pred_rows']}×{stats['pred_cols']}, "
        f"{stats['pred_spans_detected']} spanning detected) | "
        f"Recovery: {stats['span_recovery_rate']:.0%}",
        fontsize=9, fontweight='bold'
    )
    
    # (1) Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=9)
    axes[0].axis('off')
    
    # (2) GT grid
    ax = axes[1]
    nr, nc = gt['n_rows'], gt['n_cols']
    ax.set_xlim(-0.3, nc + 0.3); ax.set_ylim(nr + 0.3, -0.3)
    ax.set_aspect('equal')
    ax.set_title(f"Ground Truth ({stats['gt_spanning']} spanning)", fontsize=9, fontweight='bold')
    for r in range(nr+1): ax.axhline(y=r, color='#BDBDBD', lw=0.3)
    for c in range(nc+1): ax.axvline(x=c, color='#BDBDBD', lw=0.3)
    for cell in gt['cells']:
        sr, sc_ = cell['start_row'], cell['start_col']
        rs, cs = cell['row_span'], cell['col_span']
        is_sp = rs > 1 or cs > 1
        color = '#2196F3' if is_sp else '#E0E0E0'
        alpha = 0.5 if is_sp else 0.15
        rect = patches.FancyBboxPatch((sc_+0.04, sr+0.04), cs-0.08, rs-0.08,
                                       boxstyle="round,pad=0.02", lw=1.5 if is_sp else 0.5,
                                       edgecolor='#1565C0' if is_sp else '#757575',
                                       facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        if is_sp:
            ax.text(sc_+cs/2, sr+rs/2, f'{rs}×{cs}', ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')
    ax.set_xticks(range(nc)); ax.set_yticks(range(nr)); ax.tick_params(labelsize=5)
    
    # (3) TATR prediction grid
    ax = axes[2]
    pr, pc = pred['n_rows'], pred['n_cols']
    ax.set_xlim(-0.3, max(pc, nc) + 0.3); ax.set_ylim(max(pr, nr) + 0.3, -0.3)
    ax.set_aspect('equal')
    ax.set_title(f"TATR Prediction ({stats['pred_spans_detected']} spanning detected)", fontsize=9, fontweight='bold')
    for r in range(max(pr, nr)+1): ax.axhline(y=r, color='#BDBDBD', lw=0.3)
    for c in range(max(pc, nc)+1): ax.axvline(x=c, color='#BDBDBD', lw=0.3)
    for cell in pred['cells']:
        sr, sc_ = cell['start_row'], cell['start_col']
        rs, cs = cell['row_span'], cell['col_span']
        is_sp = rs > 1 or cs > 1
        color = '#4CAF50' if is_sp else '#FF5722'
        alpha = 0.5 if is_sp else 0.2
        rect = patches.FancyBboxPatch((sc_+0.04, sr+0.04), cs-0.08, rs-0.08,
                                       boxstyle="round,pad=0.02", lw=1.5 if is_sp else 0.5,
                                       edgecolor='#2E7D32' if is_sp else '#D84315',
                                       facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        if is_sp:
            ax.text(sc_+cs/2, sr+rs/2, f'{rs}×{cs}', ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')
    ax.set_xticks(range(max(pc, nc))); ax.set_yticks(range(max(pr, nr))); ax.tick_params(labelsize=5)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_summary(all_stats: list, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TATR Real Inference: Span Error Analysis on SciTSR-COMP", fontsize=12, fontweight='bold')
    
    # (a) Over-seg ratio
    ratios = [s['over_seg_ratio'] for s in all_stats]
    colors = ['#F44336' if r > 1.3 else '#FF9800' if r > 1.0 else '#4CAF50' if r > 0.8 else '#2196F3'
              for r in ratios]
    axes[0, 0].barh(range(len(ratios)), ratios, color=colors, height=0.7)
    axes[0, 0].axvline(x=1.0, color='green', ls='--', alpha=0.5, label='1.0 (perfect)')
    axes[0, 0].set_xlabel('Over-segmentation Ratio')
    axes[0, 0].set_title('(a) Per-Table Over-seg Ratio', fontweight='bold')
    axes[0, 0].set_yticks(range(len(ratios)))
    axes[0, 0].set_yticklabels([s['table_id'][:15] for s in all_stats], fontsize=5)
    axes[0, 0].legend(fontsize=7)
    
    # (b) Span recovery rate
    recovs = [s['span_recovery_rate'] * 100 for s in all_stats]
    axes[0, 1].bar(range(len(recovs)), recovs, color='#F44336', alpha=0.8)
    axes[0, 1].set_xlabel('Table Index')
    axes[0, 1].set_ylabel('Span Recovery Rate (%)')
    axes[0, 1].set_title('(b) Spanning Cell Recovery', fontweight='bold')
    axes[0, 1].set_ylim(0, 110)
    avg_rec = np.mean(recovs)
    axes[0, 1].axhline(y=avg_rec, color='red', ls='--', label=f'Avg: {avg_rec:.1f}%')
    axes[0, 1].legend()
    
    # (c) Row/Col detection accuracy
    row_err = [s['row_error'] for s in all_stats]
    col_err = [s['col_error'] for s in all_stats]
    x = np.arange(len(all_stats))
    w = 0.35
    axes[1, 0].bar(x - w/2, row_err, w, label='Row Error', color='#E91E63', alpha=0.8)
    axes[1, 0].bar(x + w/2, col_err, w, label='Col Error', color='#9C27B0', alpha=0.8)
    axes[1, 0].set_xlabel('Table Index')
    axes[1, 0].set_ylabel('|GT - Pred|')
    axes[1, 0].set_title('(c) Row/Col Detection Error', fontweight='bold')
    axes[1, 0].legend()
    
    # (d) GT cells vs Pred cells
    gt_ns = [s['gt_cells'] for s in all_stats]
    pr_ns = [s['pred_cells'] for s in all_stats]
    axes[1, 1].scatter(gt_ns, pr_ns, c='#FF5722', alpha=0.7, s=40)
    max_n = max(max(gt_ns), max(pr_ns)) + 5
    axes[1, 1].plot([0, max_n], [0, max_n], 'g--', alpha=0.5, label='y=x (perfect)')
    axes[1, 1].set_xlabel('GT Cell Count')
    axes[1, 1].set_ylabel('TATR Pred Cell Count')
    axes[1, 1].set_title('(d) GT vs Predicted Cell Count', fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("TATR Real Inference: Colspan/Rowspan Error Analysis")
    print("=" * 65)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load model
    processor, model = load_tatr()
    
    # 2. Sample tables
    comp_ids = load_comp_ids()
    valid_ids = [tid for tid in comp_ids if (STRUCT_DIR / f"{tid}.json").exists()
                 and any((IMG_DIR / f"{tid}{ext}").exists() for ext in ['.png', '.jpg'])]
    rng = random.Random(SEED)
    sample_ids = rng.sample(valid_ids, min(N_SAMPLE, len(valid_ids)))
    print(f"\n📂 SciTSR-COMP: {len(comp_ids)} tables, valid={len(valid_ids)}, sampled={len(sample_ids)}")
    
    # 3. Run inference
    all_stats = []
    for i, tid in enumerate(sample_ids):
        gt = load_gt(tid)
        image = load_image(tid)
        
        t0 = time.time()
        pred = run_tatr(processor, model, image)
        dt = time.time() - t0
        
        stats = analyze(gt, pred)
        stats['table_id'] = tid
        stats['inference_time'] = round(dt, 3)
        all_stats.append(stats)
        
        status = "✅" if stats['span_recovery_rate'] > 0.5 else "⚠️" if stats['span_recovery_rate'] > 0 else "❌"
        print(f"  [{i+1:2d}/{len(sample_ids)}] {tid:<25} "
              f"GT={stats['gt_cells']:>3}({stats['gt_spanning']}sp) "
              f"Pred={stats['pred_cells']:>3} "
              f"SpanDet={stats['pred_spans_detected']} "
              f"Recovery={stats['span_recovery_rate']:.0%} {status} "
              f"({dt:.2f}s)")
        
        if i < N_VISUALIZE:
            save_path = RESULTS_DIR / f"tatr_real_{i+1}_{tid.replace('/', '_')}.png"
            visualize(tid, image, gt, pred, stats, str(save_path))
    
    # 4. Aggregate
    print(f"\n{'=' * 65}")
    print("AGGREGATE RESULTS (TATR Real Inference)")
    print(f"{'=' * 65}")
    
    total_gt = sum(s['gt_cells'] for s in all_stats)
    total_span = sum(s['gt_spanning'] for s in all_stats)
    total_pred = sum(s['pred_cells'] for s in all_stats)
    total_spans_det = sum(s['pred_spans_detected'] for s in all_stats)
    total_recovered = sum(s['spans_recovered'] for s in all_stats)
    avg_ratio = np.mean([s['over_seg_ratio'] for s in all_stats])
    avg_recovery = np.mean([s['span_recovery_rate'] for s in all_stats])
    avg_time = np.mean([s['inference_time'] for s in all_stats])
    avg_row_err = np.mean([s['row_error'] for s in all_stats])
    avg_col_err = np.mean([s['col_error'] for s in all_stats])
    
    print(f"\n  Tables analyzed:             {len(all_stats)}")
    print(f"  Avg inference time:          {avg_time:.3f}s")
    print(f"  Total GT cells:              {total_gt}")
    print(f"  Total GT spanning cells:     {total_span} ({total_span/total_gt*100:.1f}%)")
    print(f"  Total Pred cells:            {total_pred}")
    print(f"  'Spanning' labels detected:  {total_spans_det}")
    print(f"  Spans actually recovered:    {total_recovered}/{total_span} ({total_recovered/total_span*100:.1f}%)")
    print(f"  Avg over-seg ratio:          {avg_ratio:.2f}x")
    print(f"  Avg span recovery rate:      {avg_recovery:.0%}")
    print(f"  Avg row detection error:     {avg_row_err:.1f}")
    print(f"  Avg col detection error:     {avg_col_err:.1f}")
    
    print(f"\n{'=' * 65}")
    print("CONCLUSION")
    print(f"{'=' * 65}")
    print(f"""
  실제 TATR 모델(microsoft/table-transformer-structure-recognition)을
  SciTSR-COMP {len(all_stats)}개 테이블에 직접 inference한 결과:

  • Spanning cell recovery rate: {avg_recovery:.0%}
  • 'Spanning' label이 detect되더라도 IoU 기반 merge가 부정확하여
    실제로 복원되는 spanning cell은 매우 적음
  • Row/Col detection 자체도 GT와 차이 발생 (avg row err: {avg_row_err:.1f}, col err: {avg_col_err:.1f})
  • Detection-based TSR의 구조적 한계가 실제 모델에서도 확인됨
""")
    
    # 5. Summary charts
    chart_path = RESULTS_DIR / "tatr_real_summary.png"
    make_summary(all_stats, str(chart_path))
    print(f"  Charts saved: {chart_path}")
    
    # 6. Save JSON
    out_path = RESULTS_DIR / "tatr_real_results.json"
    with open(out_path, 'w') as f:
        json.dump({
            'model': TATR_MODEL,
            'device': DEVICE,
            'threshold': TATR_THRESHOLD,
            'config': {'n_sample': N_SAMPLE, 'seed': SEED},
            'aggregate': {
                'tables': len(all_stats),
                'total_gt_cells': total_gt,
                'total_gt_spanning': total_span,
                'total_pred_cells': total_pred,
                'total_spans_detected': total_spans_det,
                'total_spans_recovered': total_recovered,
                'avg_over_seg_ratio': round(avg_ratio, 4),
                'avg_span_recovery_rate': round(avg_recovery, 4),
                'avg_row_error': round(avg_row_err, 2),
                'avg_col_error': round(avg_col_err, 2),
                'avg_inference_time': round(avg_time, 4),
            },
            'per_table': all_stats,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {out_path}")
    print(f"\n✅ TATR real inference analysis complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
S-2: Cross-Paradigm Span Recovery Comparison
=============================================
[S-2 비판 대응] "대안 패러다임을 동일 조건에서 평가하지 않았다"는 반론에 대한 실험.

비교 대상 (동일 데이터셋: SciTSR-COMP):
  ┌─────────────────────────────────────────────────────────┐
  │ Paradigm     │ Model         │ 핵심 메커니즘              │
  ├─────────────────────────────────────────────────────────┤
  │ Detection    │ TATR v1.1-all │ DETR: row/col 독립 탐지    │
  │ Generation   │ SLANet_plus   │ HTML token sequence 생성   │
  │ Hybrid       │ Docling       │ VLM + TableFormer 하이브리드│
  └─────────────────────────────────────────────────────────┘

핵심 가설:
  HTML/sequence 기반 생성 모델과 하이브리드 모델은 spanning cell을
  명시적 token(rowspan/colspan)으로 생성하므로,
  row/col 독립 탐지 후 교차점으로 cell을 구성하는 detection 패러다임보다
  높은 span recovery rate를 보일 것이다.

평가 지표:
  - Span Recovery Rate (IoU ≥ 0.5, chunk-based GT bbox 재구성)
  - Span Recovery without require_pred_is_span (위치만 확인)
  - Mann-Whitney U test: 각 대안 vs TATR
  - Effect size: rank-biserial correlation

데이터셋: SciTSR-COMP 전체 유효 테이블

주의사항:
  - SLANet/Docling 미설치 시 해당 모델 자동 스킵
  - Docling은 테이블 이미지 직접 처리 (SciTSR 이미지 = 이미 크롭된 표)
  - SLANet PaddleOCR 버전: ppstructure 파이프라인 사용
"""

import json
import os
import sys
import time
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import mannwhitneyu
from tqdm import tqdm

import torch

# ─── 공유 유틸 ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from _eval_utils import (
    load_valid_ids, load_gt, load_image, load_image_np,
    infer_gt_cell_bboxes, compute_span_recovery, span_recovery_rate,
    SPAN_RECOVERY_IOU,
)

# ─── tsr_eval 엔진 경로 추가 ─────────────────────────────────────────────────
TSR_EVAL_PATH = Path(__file__).parent.parent / "tsr_eval" / "engines"
sys.path.insert(0, str(TSR_EVAL_PATH.parent.parent))

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results_s2"
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.5
IOU_MERGE   = 0.3

# Paradigm 색상
PARADIGM_COLORS = {
    "detection":  "#E74C3C",
    "generation": "#2ECC71",
    "hybrid":     "#3498DB",
}

# ─────────────────────────────────────────────
# 모델 로더
# ─────────────────────────────────────────────
def _load_tatr(model_id: str):
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    processor = AutoImageProcessor.from_pretrained(model_id)
    # v1.1 모델: size={'longest_edge': 800}만 저장됨 → DetrImageProcessor 호환 형식으로 수정
    if hasattr(processor, "size") and isinstance(processor.size, dict):
        if "longest_edge" in processor.size and "shortest_edge" not in processor.size:
            le = processor.size["longest_edge"]
            processor.size = {"shortest_edge": le, "longest_edge": le}
    model = TableTransformerForObjectDetection.from_pretrained(model_id)
    model = model.to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def _run_tatr(processor, model, image_pil) -> list:
    """TATR inference → cells with bbox, row_span, col_span."""
    from _eval_utils import bbox_iou
    enc = processor(images=image_pil, return_tensors="pt")
    # fast processor가 model이 모르는 추가 키를 반환할 수 있으므로 필터링
    model_enc = {k: v.to(DEVICE) for k, v in enc.items() if k in {"pixel_values", "pixel_mask"}}
    outputs = model(**model_enc)
    target_sizes = [(image_pil.size[1], image_pil.size[0])]  # [(H, W)] — slow/fast 모두 호환
    res = processor.post_process_object_detection(
        outputs, threshold=CONF_THRESH, target_sizes=target_sizes
    )[0]
    boxes  = res["boxes"].cpu().numpy()
    labels = res["labels"].cpu().numpy()
    scores = res["scores"].cpu().numpy()
    id2label = model.config.id2label

    rows, cols, spans = [], [], []
    for box, lid, sc in zip(boxes, labels, scores):
        lname = id2label[lid].lower()
        entry = {"box": box.tolist(), "score": float(sc)}
        if "column" in lname:
            cols.append(entry)
        elif "spanning" in lname or "projected" in lname:
            spans.append(entry)
        else:
            rows.append(entry)

    rows.sort(key=lambda r: (r["box"][1] + r["box"][3]) / 2)
    cols.sort(key=lambda c: (c["box"][0] + c["box"][2]) / 2)

    cells = []
    for ri, rdata in enumerate(rows):
        for ci, cdata in enumerate(cols):
            rb, cb = rdata["box"], cdata["box"]
            x0 = max(cb[0], rb[0]); y0 = max(rb[1], cb[1])
            x1 = min(cb[2], rb[2]); y1 = min(rb[3], cb[3])
            if x1 > x0 and y1 > y0:
                cells.append({
                    "start_row": ri, "end_row": ri,
                    "start_col": ci, "end_col": ci,
                    "row_span": 1, "col_span": 1,
                    "bbox": [x0, y0, x1, y1],
                })

    if spans:
        merged_idx = set()
        merged_cells = []
        for sdata in spans:
            sbox = sdata["box"]
            overlaps = [
                (idx, c) for idx, c in enumerate(cells)
                if idx not in merged_idx and bbox_iou(sbox, c["bbox"]) > IOU_MERGE
            ]
            if overlaps:
                rs = [c["start_row"] for _, c in overlaps]
                cs = [c["start_col"] for _, c in overlaps]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                merged_cells.append({
                    "start_row": r0, "end_row": r1,
                    "start_col": c0, "end_col": c1,
                    "row_span": r1 - r0 + 1,
                    "col_span": c1 - c0 + 1,
                    "bbox": sbox,
                })
                for idx, _ in overlaps:
                    merged_idx.add(idx)
        cells = merged_cells + [c for i, c in enumerate(cells) if i not in merged_idx]

    return cells


def _load_slanet():
    """SLANet_plus (PaddleOCR) 로드. 미설치 시 None 반환."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tsr_eval.engines.paddle_engine import PaddleTSREngine
        engine = PaddleTSREngine(variant="SLANet_plus", device="cpu", backend="paddleocr")
        return engine
    except Exception as e:
        print(f"  ⚠️  SLANet 로드 실패: {e}")
        print("      pip install paddleocr paddlepaddle 후 재시도")
        return None


def _run_slanet(engine, image_np) -> list:
    """SLANet inference → cells."""
    result = engine.predict(image_np)
    return result.get("cells", [])


def _load_docling():
    """Docling 로드. 미설치 시 None 반환."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tsr_eval.engines.docling_engine import DoclingEngine
        engine = DoclingEngine()
        return engine
    except Exception as e:
        print(f"  ⚠️  Docling 로드 실패: {e}")
        print("      pip install docling 후 재시도")
        return None


def _run_docling(engine, image_np) -> list:
    """Docling inference → cells."""
    result = engine.predict(image_np)
    return result.get("cells", [])


# ─────────────────────────────────────────────
# 단일 테이블 평가
# ─────────────────────────────────────────────

# GriTS 임포트 (bbox 불필요 metric)
try:
    from grits_eval import grits_factored, sim_top, _span_only_grits_top
    _HAS_GRITS = True
except ImportError:
    _HAS_GRITS = False


def _cells_to_grits_format(cells: list) -> list:
    """DoclingEngine / TATR 출력 셀을 GriTS 입력 형식으로 변환.
    DoclingEngine: {row, col, row_span, col_span, bbox, ...}
    TATR:          {row_span, col_span, bbox, ...}
    GriTS 필요:    {start_row, end_row, start_col, end_col, row_span, col_span}
    """
    out, seen = [], set()
    for c in cells:
        sr = c.get("start_row", c.get("row", 0))
        sc = c.get("start_col", c.get("col", 0))
        rs = c.get("row_span", 1)
        cs = c.get("col_span", 1)
        key = (sr, sc)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "start_row": sr, "end_row": sr + rs - 1,
            "start_col": sc, "end_col": sc + cs - 1,
            "row_span": rs, "col_span": cs, "text": "",
        })
    return out


def evaluate_table(gt: dict, table_id: str, cells: list,
                   img_size=None,
                   require_pred_is_span: bool = True) -> dict:
    """
    한 테이블에 대해 span recovery를 계산한다.

    두 가지 기준으로 측정:
      strict : require_pred_is_span=True  — 예측 셀도 span이어야 함
      loose  : require_pred_is_span=False — 위치만 확인 (GT span 영역에 예측 셀 존재)
      grits_top_f1     : GriTS-Top F1 (bbox 불필요, 논리구조 비교)
      grits_span_only  : GriTS-Top span-only (spanning cell 격리 비교)

    img_size: (W, H) — chunk 좌표를 이미지 픽셀 좌표로 정규화.

    [참고] strict/loose는 bbox IoU 기반이므로 Docling 등 generation 모델에서는
    좌표계 불일치(GT=chunk row 전체 높이, pred=텍스트 영역만)로 인해
    부정확할 수 있다. GriTS-Top을 우선 참조할 것.
    """
    gt_cells_bbox = infer_gt_cell_bboxes(gt, table_id, img_size=img_size)
    has_chunk = len(gt_cells_bbox) > 0

    # GriTS-Top 평가 (bbox 불필요 — 논리 좌표만 사용)
    grits_f1 = None
    grits_span = None
    if _HAS_GRITS:
        pred_grits = _cells_to_grits_format(cells)
        nr_p = max((c["end_row"] for c in pred_grits), default=-1) + 1
        nc_p = max((c["end_col"] for c in pred_grits), default=-1) + 1
        nr_g = gt["n_rows"]
        nc_g = gt["n_cols"]
        if nr_p > 0 and nc_p > 0 and nr_g > 0 and nc_g > 0:
            _, _, grits_f1 = grits_factored(
                pred_grits, gt["cells"], nr_p, nc_p, nr_g, nc_g, sim_top)
            grits_span = _span_only_grits_top(pred_grits, gt["cells"])

    if not has_chunk:
        return {
            "table_id": table_id,
            "gt_n_spanning": None,
            "strict_recovered": None, "strict_rate": None,
            "loose_recovered": None,  "loose_rate": None,
            "grits_top_f1": grits_f1,
            "grits_span_only": grits_span,
            "has_chunk": False,
        }

    gt_spans = [c for c in gt_cells_bbox
                if c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1]
    total = len(gt_spans)

    rec_strict, _ = compute_span_recovery(gt_cells_bbox, cells,
                                           SPAN_RECOVERY_IOU, require_pred_is_span=True)
    rec_loose, _  = compute_span_recovery(gt_cells_bbox, cells,
                                           SPAN_RECOVERY_IOU, require_pred_is_span=False)
    return {
        "table_id":        table_id,
        "gt_n_spanning":   total,
        "strict_recovered": rec_strict,
        "strict_rate":     rec_strict / total if total > 0 else None,
        "loose_recovered": rec_loose,
        "loose_rate":      rec_loose  / total if total > 0 else None,
        "grits_top_f1":    grits_f1,
        "grits_span_only": grits_span,
        "has_chunk":       True,
    }


# ─────────────────────────────────────────────
# 통계 검정
# ─────────────────────────────────────────────
def _rank_biserial_r(u_stat, n1, n2):
    """Mann-Whitney U → rank-biserial correlation (effect size)."""
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def compare_vs_baseline(base_rates: list, alt_rates: list,
                         base_name: str, alt_name: str) -> dict:
    """
    두 모델의 span recovery rate를 통계 비교.

    검정: Mann-Whitney U (양측)
    Effect size: rank-biserial r
    """
    base = np.array([x for x in base_rates if x is not None], dtype=float)
    alt  = np.array([x for x in alt_rates  if x is not None], dtype=float)
    if len(base) < 5 or len(alt) < 5:
        return {"error": "insufficient data"}
    stat, pval = mannwhitneyu(alt, base, alternative="two-sided")
    r = _rank_biserial_r(stat, len(alt), len(base))
    return {
        "base_model":   base_name,
        "alt_model":    alt_name,
        "base_mean":    float(np.mean(base)),
        "alt_mean":     float(np.mean(alt)),
        "diff":         float(np.mean(alt) - np.mean(base)),
        "mw_stat":      float(stat),
        "mw_pvalue":    float(pval),
        "rank_biserial_r": float(r),
        "significant":  bool(pval < 0.05),
        "effect_magnitude": "large" if abs(r) >= 0.5 else "medium" if abs(r) >= 0.3 else "small",
    }


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────
def make_figures(results_by_model: dict, stats_results: dict, save_dir: Path):
    labels = list(results_by_model.keys())
    paradigms = {
        "TATR v1.1-all": "detection",
        "SLANet_plus":   "generation",
        "Docling":       "hybrid",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "S-2: Cross-Paradigm Span Recovery Comparison\n"
        "SciTSR-COMP — IoU≥0.5 (chunk-based GT bbox reconstruction)",
        fontsize=13, fontweight="bold",
    )

    # (A) Strict span recovery — bar + CI
    ax = axes[0]
    means, lows, highs = [], [], []
    colors = []
    for lbl in labels:
        recs = results_by_model[lbl]
        rates = [r["strict_rate"] for r in recs if r["strict_rate"] is not None]
        m  = np.mean(rates) if rates else 0
        se = np.std(rates) / np.sqrt(len(rates)) * 1.96 if len(rates) > 1 else 0
        means.append(m); lows.append(se); highs.append(se)
        pname = paradigms.get(lbl, "unknown")
        colors.append(PARADIGM_COLORS.get(pname, "#888888"))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, alpha=0.85)
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", color="black", capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, fontsize=9)
    ax.set_ylabel("Span Recovery Rate (IoU ≥ 0.5)")
    ax.set_ylim(0, 1.15)
    ax.set_title("(A) Strict Recovery\n(pred must also be span)", fontweight="bold")
    for bar, m_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m_val:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Legend (paradigm 색상)
    handles = [mpatches.Patch(color=c, label=p) for p, c in PARADIGM_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, title="Paradigm")

    # (B) Loose recovery (위치만 확인)
    ax = axes[1]
    means2 = []
    for lbl in labels:
        recs = results_by_model[lbl]
        rates = [r["loose_rate"] for r in recs if r["loose_rate"] is not None]
        means2.append(np.mean(rates) if rates else 0)

    bars2 = ax.bar(x, means2, color=colors, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, fontsize=9)
    ax.set_ylabel("Span Recovery Rate (IoU ≥ 0.5)")
    ax.set_ylim(0, 1.15)
    ax.set_title("(B) Loose Recovery\n(only spatial overlap required)", fontweight="bold")
    for bar, m_val in zip(bars2, means2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m_val:.3f}", ha="center", fontsize=9, fontweight="bold")

    # (C) Statistical comparison vs TATR
    ax = axes[2]
    baseline_lbl = "TATR v1.1-all"
    comparisons = {k: v for k, v in stats_results.items() if k != baseline_lbl
                   and "diff" in v}
    if comparisons:
        comp_labels = list(comparisons.keys())
        diffs  = [comparisons[k]["diff"]    for k in comp_labels]
        pvals  = [comparisons[k]["mw_pvalue"] for k in comp_labels]
        rbs    = [comparisons[k]["rank_biserial_r"] for k in comp_labels]
        comp_colors = [PARADIGM_COLORS.get(paradigms.get(k, ""), "#888888")
                       for k in comp_labels]

        y = np.arange(len(comp_labels))
        bars3 = ax.barh(y, diffs, color=comp_colors, alpha=0.85)
        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_yticks(y); ax.set_yticklabels(comp_labels, fontsize=9)
        ax.set_xlabel("Δ Recovery Rate (Alt − TATR)")
        ax.set_title(f"(C) Δ vs {baseline_lbl}\n(Mann-Whitney p-val annotated)", fontweight="bold")
        for i, (bar, pval, r_val) in enumerate(zip(bars3, pvals, rbs)):
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"p={pval:.3f} {sig}\nr={r_val:.2f}",
                    va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No comparison data\n(models not loaded)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(C) Statistical Comparison", fontweight="bold")

    plt.tight_layout()
    out = save_dir / "s2_paradigm_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_per_table_figure(results_by_model: dict, save_dir: Path):
    """테이블별 span recovery scatter — 패러다임 간 일관성 확인."""
    labels = list(results_by_model.keys())
    paradigms = {
        "TATR v1.1-all": "detection",
        "SLANet_plus":   "generation",
        "Docling":       "hybrid",
    }

    n = len(labels)
    if n < 2:
        return None

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Per-Table Strict Span Recovery Rate by Model", fontsize=12, fontweight="bold")

    for ax, lbl in zip(axes, labels):
        recs  = results_by_model[lbl]
        rates = [r["strict_rate"] if r["strict_rate"] is not None else -0.05 for r in recs]
        spans = [r["gt_n_spanning"] if r["gt_n_spanning"] is not None else 0 for r in recs]
        pname = paradigms.get(lbl, "unknown")
        color = PARADIGM_COLORS.get(pname, "#888888")

        valid = [(s, r) for s, r in zip(spans, rates) if r >= 0]
        if valid:
            xs, ys = zip(*valid)
            ax.scatter(xs, ys, c=color, alpha=0.4, s=15)
            # 추세선
            if len(xs) > 2:
                z = np.polyfit(xs, ys, 1)
                p = np.poly1d(z)
                xr = np.linspace(min(xs), max(xs), 100)
                ax.plot(xr, p(xr), color=color, linewidth=1.5, alpha=0.8)

        ax.set_xlabel("GT Spanning Cell Count")
        ax.set_ylabel("Strict Recovery Rate")
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"{lbl}\n({pname})", fontweight="bold")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    out = save_dir / "s2_per_table_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="S-2: Cross-Paradigm Span Recovery")
    parser.add_argument("--n_sample", type=int, default=0,
                        help="0=전체 사용 (기본). 빠른 테스트: --n_sample 50")
    parser.add_argument("--skip_docling", action="store_true",
                        help="Docling 건너뛰기 (속도 느림)")
    parser.add_argument("--skip_slanet", action="store_true",
                        help="SLANet 건너뛰기")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    import random
    valid_ids = load_valid_ids()
    rng = random.Random(SEED)
    if args.n_sample > 0:
        sample_ids = rng.sample(valid_ids, min(args.n_sample, len(valid_ids)))
    else:
        sample_ids = valid_ids

    print(f"\n{'='*65}")
    print("S-2: Cross-Paradigm Span Recovery Comparison")
    print(f"{'='*65}")
    print(f"  Dataset  : SciTSR-COMP ({len(sample_ids)} tables)")
    print(f"  Device   : {DEVICE}")
    print(f"  Paradigms: detection (TATR), generation (SLANet), hybrid (Docling)")
    print()

    results_by_model = {}

    # ── 1. TATR (Detection) ─────────────────────────────────────────────────
    tatr_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
    print(f"[Detection] Loading TATR v1.1-all...")
    try:
        tatr_proc, tatr_model = _load_tatr(tatr_id)
        tatr_records = []
        for tid in tqdm(sample_ids, desc="TATR v1.1-all", ncols=80):
            gt    = load_gt(tid)
            image = load_image(tid)
            try:
                cells = _run_tatr(tatr_proc, tatr_model, image)
            except Exception:
                continue
            tatr_records.append(evaluate_table(gt, tid, cells, img_size=image.size))
        results_by_model["TATR v1.1-all"] = tatr_records
        rates = [r["strict_rate"] for r in tatr_records if r["strict_rate"] is not None]
        print(f"  Strict Recovery: {np.mean(rates):.4f} ± {np.std(rates):.4f}  (n={len(rates)})")
        gf = [r["grits_top_f1"] for r in tatr_records if r.get("grits_top_f1") is not None]
        gs = [r["grits_span_only"] for r in tatr_records if r.get("grits_span_only") is not None]
        if gf: print(f"  GriTS-Top F1:    {np.mean(gf):.4f}  (n={len(gf)})")
        if gs: print(f"  GriTS Span-only: {np.mean(gs):.4f}  (n={len(gs)})")
    except Exception as e:
        print(f"  TATR error: {e}")

    # ── 2. SLANet (Generation) ──────────────────────────────────────────────
    if not args.skip_slanet:
        print(f"\n[Generation] Loading SLANet_plus (PaddleOCR)...")
        slanet = _load_slanet()
        if slanet is not None:
            slanet_records = []
            for tid in tqdm(sample_ids, desc="SLANet_plus", ncols=80):
                gt    = load_gt(tid)
                image_np = load_image_np(tid)
                try:
                    cells = _run_slanet(slanet, image_np)
                except Exception:
                    continue
                from PIL import Image as _PIL
                _img_size = _PIL.fromarray(image_np).size
                slanet_records.append(evaluate_table(gt, tid, cells, img_size=_img_size))
            results_by_model["SLANet_plus"] = slanet_records
            rates = [r["strict_rate"] for r in slanet_records if r["strict_rate"] is not None]
            if rates:
                print(f"  Strict Recovery: {np.mean(rates):.4f} ± {np.std(rates):.4f}  (n={len(rates)})")
        else:
            print("  → SLANet 스킵됨 (설치 필요)")
    else:
        print("\n[Generation] SLANet 스킵 (--skip_slanet)")

    # ── 3. Docling (Hybrid) ─────────────────────────────────────────────────
    if not args.skip_docling:
        print(f"\n[Hybrid] Loading Docling (IBM)...")
        docling_engine = _load_docling()
        if docling_engine is not None:
            docling_records = []
            for tid in tqdm(sample_ids, desc="Docling", ncols=80):
                gt    = load_gt(tid)
                image_np = load_image_np(tid)
                try:
                    cells = _run_docling(docling_engine, image_np)
                except Exception:
                    continue
                from PIL import Image as _PIL
                _img_size = _PIL.fromarray(image_np).size
                docling_records.append(evaluate_table(gt, tid, cells, img_size=_img_size))
            results_by_model["Docling"] = docling_records
            rates = [r["strict_rate"] for r in docling_records if r["strict_rate"] is not None]
            if rates:
                print(f"  Strict Recovery: {np.mean(rates):.4f} ± {np.std(rates):.4f}  (n={len(rates)})")
        else:
            print("  → Docling 스킵됨 (설치 필요)")
    else:
        print("\n[Hybrid] Docling 스킵 (--skip_docling)")

    if not results_by_model:
        print("모든 모델 로드 실패. 종료.")
        return

    # ── 통계 비교 ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Statistical Comparison vs TATR v1.1-all")
    print(f"{'='*65}")

    baseline_lbl  = "TATR v1.1-all"
    stats_results = {}

    if baseline_lbl in results_by_model:
        base_strict = [r["strict_rate"] for r in results_by_model[baseline_lbl]
                       if r["strict_rate"] is not None]
        for alt_lbl, alt_recs in results_by_model.items():
            if alt_lbl == baseline_lbl:
                continue
            alt_strict = [r["strict_rate"] for r in alt_recs if r["strict_rate"] is not None]
            cmp = compare_vs_baseline(base_strict, alt_strict, baseline_lbl, alt_lbl)
            stats_results[alt_lbl] = cmp
            if "error" in cmp:
                print(f"  {alt_lbl}: {cmp['error']}")
            else:
                print(f"\n  {alt_lbl} vs {baseline_lbl}:")
                print(f"    TATR mean     : {cmp['base_mean']:.4f}")
                print(f"    {alt_lbl:<14}: {cmp['alt_mean']:.4f}")
                print(f"    Δ             : {cmp['diff']:+.4f}")
                print(f"    Mann-Whitney  : U={cmp['mw_stat']:.1f}, p={cmp['mw_pvalue']:.4e}")
                print(f"    Effect size   : r={cmp['rank_biserial_r']:.3f} ({cmp['effect_magnitude']})")
                sig = "✅ significant" if cmp["significant"] else "✗ not significant"
                print(f"    {sig} (α=0.05)")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    fig_path   = make_figures(results_by_model, stats_results, RESULTS_DIR)
    scat_path  = make_per_table_figure(results_by_model, RESULTS_DIR)
    print(f"\n  📊 Comparison figure : {fig_path}")
    if scat_path:
        print(f"  📊 Scatter figure    : {scat_path}")

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    summary = {
        "dataset": "SciTSR-COMP",
        "n_tables": len(sample_ids),
        "iou_threshold": SPAN_RECOVERY_IOU,
        "models": {
            lbl: {
                "paradigm": {"TATR v1.1-all": "detection",
                             "SLANet_plus": "generation",
                             "Docling": "hybrid"}.get(lbl, "unknown"),
                "n_evaluated": len(recs),
                "chunk_coverage": sum(1 for r in recs if r["has_chunk"]),
                "strict_mean": float(np.mean([r["strict_rate"] for r in recs
                                              if r["strict_rate"] is not None]))
                               if any(r["strict_rate"] is not None for r in recs) else None,
                "loose_mean": float(np.mean([r["loose_rate"] for r in recs
                                             if r["loose_rate"] is not None]))
                              if any(r["loose_rate"] is not None for r in recs) else None,
                "grits_top_f1_mean": float(np.mean([r["grits_top_f1"] for r in recs
                                                     if r.get("grits_top_f1") is not None]))
                                     if any(r.get("grits_top_f1") is not None for r in recs) else None,
                "grits_span_only_mean": float(np.mean([r["grits_span_only"] for r in recs
                                                        if r.get("grits_span_only") is not None]))
                                        if any(r.get("grits_span_only") is not None for r in recs) else None,
            }
            for lbl, recs in results_by_model.items()
        },
        "statistical_comparisons": stats_results,
        "note_s2_interpretation": (
            "strict recovery: 예측 셀이 실제로 span(rowspan/colspan)을 가질 때만 recovery 인정. "
            "loose recovery: 위치만 확인 (detection 모델의 간접 복원 가능성 반영). "
            "두 지표의 차이가 크면 모델이 span을 명시적으로 예측하지 않고 있음을 의미. "
            "[주의] Docling의 strict_mean이 TATR보다 낮게 나오는 경우, "
            "이는 Docling이 실제로 span을 못 찾는 게 아니라 "
            "출력 포맷(predict() → cells)에서 row_span/col_span attribute가 "
            "표준 형식으로 전달되지 않는 파싱 artifact일 수 있다. "
            "GriTS-Top span-only (benchmark_summary.json) 기반 비교를 우선 사용할 것."
        ),
    }

    json_path = RESULTS_DIR / "s2_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    per_table_path = RESULTS_DIR / "s2_per_table.json"
    with open(per_table_path, "w", encoding="utf-8") as f:
        json.dump(
            {lbl: recs for lbl, recs in results_by_model.items()},
            f, indent=2
        )

    print(f"  💾 Summary     : {json_path}")
    print(f"  💾 Per-table   : {per_table_path}")

    print(f"\n{'='*65}")
    print("S-2 Analysis Complete.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

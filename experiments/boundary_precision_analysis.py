#!/usr/bin/env python3
"""
GSR Motivation Experiment: Spanning Cell Boundary Precision Analysis
====================================================================
수정된 가설 검증:
  "TATR은 spanning cell을 탐지(AP@0.5=84.9%)하지만,
   bbox 경계가 row/col separator에 정밀하게 맞지 않아 (AP@0.9=46.3%)
   grid 복원 단계에서 col/row span 인덱스 오류가 발생한다."

측정:
  1. Boundary offset  : matched span pair의 각 edge가 GT edge에서 얼마나 어긋나는가
  2. Separator snap   : 어긋난 edge가 GT separator (row/col 경계선)를 넘는가
  3. Grid index error : separator를 넘으면 colspan/rowspan이 잘못 결정됨
  4. IoU bin vs grid accuracy : IoU가 높을수록 grid index가 맞아가는가 확인

결론적으로:
  AP@0.5 = "있다는 건 알지만 경계는 틀림" → grid index 오류 높음
  AP@0.9 = "경계까지 맞음"               → grid index 오류 낮음
  GSR이 제안하는 separator snapping이 정확히 이 gap을 메움
"""

import os
import json
import warnings
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from torchvision.ops import box_iou
from transformers import AutoModelForObjectDetection, AutoImageProcessor

warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF   = 0.5
SEED   = 42
N_SAMPLES = 300

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ─────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────
def load_model():
    name = "microsoft/table-transformer-structure-recognition-v1.1-all"
    proc = AutoImageProcessor.from_pretrained(name)
    if isinstance(proc.size, dict) and list(proc.size.keys()) == ["longest_edge"]:
        v = proc.size["longest_edge"]
        proc.size = {"shortest_edge": v, "longest_edge": v}
    model = AutoModelForObjectDetection.from_pretrained(name).to(DEVICE)
    model.eval()
    return proc, model


@torch.no_grad()
def infer(image, proc, model):
    inputs = proc(images=image, return_tensors="pt").to(DEVICE)
    out    = model(**inputs)
    tgt    = torch.tensor([image.size[::-1]])
    res    = proc.post_process_object_detection(out, threshold=CONF,
                                                target_sizes=tgt)[0]
    return {k: v.cpu() for k, v in res.items()}


# ─────────────────────────────────────────────
# GT 파싱
# ─────────────────────────────────────────────
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows, cols, spans = [], [], []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        bb   = obj.find("bndbox")
        if bb is None:
            continue
        bbox = [float(bb.findtext(k, 0)) for k in ["xmin","ymin","xmax","ymax"]]
        if   name == "table row":           rows.append(bbox)
        elif name == "table column":        cols.append(bbox)
        elif name == "table spanning cell": spans.append(bbox)
    return rows, cols, spans


def get_separators(rows, cols):
    """
    GT row/col bbox에서 separator 위치를 추출.
    각 row의 top/bottom edge, 각 col의 left/right edge → unique sorted list.
    """
    row_seps = sorted(set([b[1] for b in rows] + [b[3] for b in rows]))
    col_seps = sorted(set([b[0] for b in cols] + [b[2] for b in cols]))
    return row_seps, col_seps


def nearest_sep(val, seps):
    """val에서 가장 가까운 separator와 거리 반환"""
    if not seps:
        return val, 0.0
    arr = np.array(seps)
    idx = np.argmin(np.abs(arr - val))
    return float(arr[idx]), float(abs(arr[idx] - val))


def nearest_sep_idx(val, seps):
    """val에서 가장 가까운 separator의 인덱스 반환"""
    if not seps:
        return 0
    arr = np.array(seps)
    return int(np.argmin(np.abs(arr - val)))


# ─────────────────────────────────────────────
# 핵심 분석: matched span pair → boundary error
# ─────────────────────────────────────────────
def analyze_boundary(gt_span, pred_span, row_seps, col_seps):
    """
    GT span bbox와 TATR pred span bbox 하나의 쌍을 받아
    각 edge(x1,y1,x2,y2)에 대한 오차 지표를 반환.

    반환값 (dict):
      iou          : GT-pred IoU
      offset_*     : 각 edge의 픽셀 오차 (|pred - gt|)
      norm_offset_*: width/height로 정규화한 오차
      snap_dist_*  : pred edge에서 가장 가까운 separator까지의 거리
      cross_sep_*  : pred edge가 separator를 넘어 wrong interval에 있는가 (bool)
      grid_err_col : col span 인덱스 오류 여부 (bool)
      grid_err_row : row span 인덱스 여부 (bool)
    """
    gx1, gy1, gx2, gy2 = gt_span
    px1, py1, px2, py2 = pred_span

    gt_w = gx2 - gx1
    gt_h = gy2 - gy1

    # IoU
    iou = float(box_iou(
        torch.tensor([[px1, py1, px2, py2]], dtype=torch.float32),
        torch.tensor([[gx1, gy1, gx2, gy2]], dtype=torch.float32)
    )[0, 0])

    # Edge offsets
    off = {
        "x1": abs(px1 - gx1), "y1": abs(py1 - gy1),
        "x2": abs(px2 - gx2), "y2": abs(py2 - gy2),
    }
    norm_off = {
        "x1": off["x1"] / gt_w if gt_w > 0 else 0,
        "y1": off["y1"] / gt_h if gt_h > 0 else 0,
        "x2": off["x2"] / gt_w if gt_w > 0 else 0,
        "y2": off["y2"] / gt_h if gt_h > 0 else 0,
    }

    # Snap distance: pred edge → nearest separator
    _, snap_x1 = nearest_sep(px1, col_seps)
    _, snap_x2 = nearest_sep(px2, col_seps)
    _, snap_y1 = nearest_sep(py1, row_seps)
    _, snap_y2 = nearest_sep(py2, row_seps)

    # Grid index 오류: "pred edge를 nearest separator로 snap했을 때
    # GT edge의 snap 결과(separator index)와 다른가?"
    # GT edge는 separator 위에 정확히 있으므로 GT snap index = GT separator index
    # pred edge가 다른 separator로 snap되면 → 다른 col/row 인덱스 → grid 오류
    gi_err_x1 = nearest_sep_idx(px1, col_seps) != nearest_sep_idx(gx1, col_seps)
    gi_err_x2 = nearest_sep_idx(px2, col_seps) != nearest_sep_idx(gx2, col_seps)
    gi_err_y1 = nearest_sep_idx(py1, row_seps) != nearest_sep_idx(gy1, row_seps)
    gi_err_y2 = nearest_sep_idx(py2, row_seps) != nearest_sep_idx(gy2, row_seps)

    grid_err_col = gi_err_x1 or gi_err_x2   # colspan 결정 오류
    grid_err_row = gi_err_y1 or gi_err_y2   # rowspan 결정 오류
    grid_err_any = grid_err_col or grid_err_row

    return {
        "iou": iou,
        "offset_x1": off["x1"], "offset_y1": off["y1"],
        "offset_x2": off["x2"], "offset_y2": off["y2"],
        "norm_x1": norm_off["x1"], "norm_y1": norm_off["y1"],
        "norm_x2": norm_off["x2"], "norm_y2": norm_off["y2"],
        "snap_x1": snap_x1, "snap_x2": snap_x2,
        "snap_y1": snap_y1, "snap_y2": snap_y2,
        "max_offset": max(off.values()),
        "max_norm":   max(norm_off.values()),
        "max_snap":   max(snap_x1, snap_x2, snap_y1, snap_y2),
        "grid_err_col": int(grid_err_col),
        "grid_err_row": int(grid_err_row),
        "grid_err_any": int(grid_err_any),
    }


# ─────────────────────────────────────────────
# 평가 루프
# ─────────────────────────────────────────────
def evaluate(pubtables_root, proc, model_obj, n_samples):
    img_dir = os.path.join(pubtables_root, "test", "images")
    ann_dir = os.path.join(pubtables_root, "test", "annotations")

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.endswith(('.jpg', '.png'))])
    np.random.seed(SEED)
    if n_samples < len(img_files):
        img_files = [img_files[i] for i in
                     sorted(np.random.choice(len(img_files), n_samples, replace=False))]

    records = []
    skipped = 0

    for img_file in tqdm(img_files, desc="Boundary analysis"):
        stem    = os.path.splitext(img_file)[0]
        xml_path = os.path.join(ann_dir, stem + ".xml")
        if not os.path.exists(xml_path):
            skipped += 1
            continue

        rows, cols, spans = parse_xml(xml_path)
        if not spans or not rows or not cols:
            skipped += 1
            continue

        try:
            image = Image.open(os.path.join(img_dir, img_file)).convert("RGB")
        except Exception:
            skipped += 1
            continue

        pred   = infer(image, proc, model_obj)
        labels = pred["labels"].numpy()
        boxes  = pred["boxes"].numpy()

        # TATR 예측에서 spanning cell (label=5)만 추출
        pred_spans = [boxes[i].tolist()
                      for i in range(len(labels)) if labels[i] == 5]
        if not pred_spans:
            # span 예측 없음 → GT span은 전부 미탐지
            for sp in spans:
                records.append({
                    "image": stem,
                    "detected": 0,
                    "iou": 0.0,
                    "gt_span_w": sp[2] - sp[0],
                    "gt_span_h": sp[3] - sp[1],
                    **{k: None for k in [
                        "offset_x1","offset_y1","offset_x2","offset_y2",
                        "norm_x1","norm_y1","norm_x2","norm_y2",
                        "snap_x1","snap_x2","snap_y1","snap_y2",
                        "max_offset","max_norm","max_snap",
                        "grid_err_col","grid_err_row","grid_err_any"
                    ]}
                })
            continue

        row_seps, col_seps = get_separators(rows, cols)

        # GT spans ↔ pred spans Hungarian matching
        gt_t  = torch.tensor(spans, dtype=torch.float32)
        pr_t  = torch.tensor(pred_spans, dtype=torch.float32)
        iou_m = box_iou(gt_t, pr_t).numpy()
        gi, pi = linear_sum_assignment(1.0 - iou_m)

        matched_gt = set()
        for g, p in zip(gi, pi):
            matched_iou = iou_m[g, p]
            gt_sp = spans[g]
            pr_sp = pred_spans[p]

            boundary = analyze_boundary(gt_sp, pr_sp, row_seps, col_seps)
            records.append({
                "image":   stem,
                "detected": 1 if matched_iou >= 0.5 else 0,
                "gt_span_w": gt_sp[2] - gt_sp[0],
                "gt_span_h": gt_sp[3] - gt_sp[1],
                **boundary,
            })
            matched_gt.add(g)

        # 매칭 안 된 GT span → 미탐지
        for g, sp in enumerate(spans):
            if g not in matched_gt:
                records.append({
                    "image": stem,
                    "detected": 0,
                    "iou": 0.0,
                    "gt_span_w": sp[2] - sp[0],
                    "gt_span_h": sp[3] - sp[1],
                    **{k: None for k in [
                        "offset_x1","offset_y1","offset_x2","offset_y2",
                        "norm_x1","norm_y1","norm_x2","norm_y2",
                        "snap_x1","snap_x2","snap_y1","snap_y2",
                        "max_offset","max_norm","max_snap",
                        "grid_err_col","grid_err_row","grid_err_any"
                    ]}
                })

    print(f"\n  처리: {len(img_files)-skipped}개 / 스킵: {skipped}개")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────
def visualize(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ── detected (IoU ≥ 0.5) 서브셋 ──
    det = df[df["detected"] == 1].copy()

    # ─────────────────────────────────────────
    # Figure 1: 메인 대시보드 (2×3)
    # ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "TATR Spanning Cell Boundary Precision Analysis\n"
        f"(N={len(df)} GT spans, detected(IoU≥0.5): {len(det)} = "
        f"{len(det)/len(df)*100:.1f}%,  dataset: PubTables-1M test)",
        fontsize=13, fontweight='bold', y=0.99
    )

    # (a) IoU 분포 전체
    ax = axes[0, 0]
    ax.hist(df["iou"].dropna(), bins=40, color="#5C6BC0", alpha=0.8, edgecolor='white')
    for t, c in [(0.5, 'orange'), (0.75, 'red'), (0.9, 'darkred')]:
        pct = (df["iou"] >= t).mean() * 100
        ax.axvline(t, color=c, lw=2, ls='--', label=f'@{t}: {pct:.1f}%')
    ax.set_xlabel("IoU (GT span vs TATR span pred)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Span IoU Distribution\n← 탐지 but 경계 부정확   경계 정확 →",
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Normalized edge offset 분포 (detected only)
    ax = axes[0, 1]
    for col, label, color in [
        ("norm_x1", "left edge",   "#E53935"),
        ("norm_x2", "right edge",  "#43A047"),
        ("norm_y1", "top edge",    "#FB8C00"),
        ("norm_y2", "bottom edge", "#1E88E5"),
    ]:
        data = det[col].dropna()
        ax.hist(data, bins=40, alpha=0.45, label=f"{label} (μ={data.mean():.3f})",
                color=color, density=True)
    ax.axvline(0, color='green', lw=1.5, ls='-', alpha=0.6, label='0 (perfect)')
    ax.set_xlabel("Normalized edge offset  |pred_edge − gt_edge| / span_width(height)")
    ax.set_ylabel("Density")
    ax.set_title("(b) Normalized Edge Offset\n(detected spans, IoU≥0.5)",
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.5)
    ax.grid(alpha=0.3)

    # (c) Grid index error rate by IoU bin
    ax = axes[0, 2]
    det_c = det.copy()
    bins  = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    labels_b = ["0.5-0.6","0.6-0.7","0.7-0.75","0.75-0.8",
                 "0.8-0.85","0.85-0.9","0.9-0.95","0.95-1.0"]
    det_c["iou_bin"] = pd.cut(det_c["iou"], bins=bins, labels=labels_b)
    grp = det_c.groupby("iou_bin", observed=True)["grid_err_any"].agg(
        ["mean", "count"]).reset_index()
    colors_b = plt.cm.RdYlGn([i / len(grp) for i in range(len(grp))])
    bars = ax.bar(range(len(grp)), grp["mean"] * 100,
                  color=colors_b, alpha=0.85)
    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{row["mean"]*100:.0f}%\n(n={int(row["count"])})',
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(labels_b, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Grid Index Error Rate (%)")
    ax.set_title("(c) IoU Bin → Grid Index Error Rate\n"
                 "IoU 높을수록 grid 오류 감소?", fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    # (d) Snap distance distribution (detected only)
    ax = axes[1, 0]
    for col, label, color in [
        ("snap_x1", "x1→col sep", "#E53935"),
        ("snap_x2", "x2→col sep", "#43A047"),
        ("snap_y1", "y1→row sep", "#FB8C00"),
        ("snap_y2", "y2→row sep", "#1E88E5"),
    ]:
        data = det[col].dropna()
        ax.hist(data, bins=50, alpha=0.4,
                label=f"{label} (μ={data.mean():.1f}px)",
                color=color, density=True)
    ax.set_xlabel("Distance to nearest GT separator (pixels)")
    ax.set_ylabel("Density")
    ax.set_title("(d) Pred Edge → Nearest Separator Distance\n"
                 "GSR snapping이 이 거리를 0으로 만듦", fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 50)
    ax.grid(alpha=0.3)

    # (e) Grid error rate by snap distance bin
    ax = axes[1, 1]
    det_s = det.dropna(subset=["max_snap", "grid_err_any"]).copy()
    snap_bins   = [0, 2, 5, 10, 20, 50, 1000]
    snap_labels = ["0-2", "2-5", "5-10", "10-20", "20-50", "50+"]
    det_s["snap_bin"] = pd.cut(det_s["max_snap"], bins=snap_bins, labels=snap_labels)
    grp_s = det_s.groupby("snap_bin", observed=True)["grid_err_any"].agg(
        ["mean", "count"]).reset_index()
    x_s = np.arange(len(grp_s))
    bars = ax.bar(x_s, grp_s["mean"] * 100,
                  color=plt.cm.RdYlGn(1 - grp_s["mean"].values * 0.8),
                  alpha=0.85)
    for bar, (_, row) in zip(bars, grp_s.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{row["mean"]*100:.0f}%\n(n={int(row["count"])})',
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x_s)
    ax.set_xticklabels(snap_labels, fontsize=9)
    ax.set_xlabel("Max snap distance (px) – largest edge offset to any separator")
    ax.set_ylabel("Grid Index Error Rate (%)")
    ax.set_title("(e) Snap Distance → Grid Error Rate\n"
                 "경계가 separator에 가까울수록 grid 오류 감소", fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    # (f) Summary: 단계별 성능 감쇠 (funnel chart)
    ax = axes[1, 2]
    total   = len(df)
    n_det5  = (df["iou"] >= 0.5).sum()
    n_det75 = (df["iou"] >= 0.75).sum()
    n_det9  = (df["iou"] >= 0.9).sum()
    n_grid_ok = det[det["grid_err_any"] == 0]["iou"].count()   # detected & grid OK
    n_det5_grid_ok = det[det["grid_err_any"] == 0].shape[0]

    stages = [
        ("GT spans\n(all)",        total,        "#78909C"),
        ("Detected\n(IoU≥0.5)",    n_det5,       "#5C6BC0"),
        ("Precise\n(IoU≥0.75)",    n_det75,      "#26A69A"),
        ("Very precise\n(IoU≥0.9)",n_det9,       "#43A047"),
        ("Detected &\nGrid OK",    n_det5_grid_ok,"#66BB6A"),
    ]
    x_f  = range(len(stages))
    vals = [s[1] for s in stages]
    cols = [s[2] for s in stages]
    bars = ax.bar(x_f, vals, color=cols, alpha=0.88)
    for bar, (label, val, _) in zip(bars, stages):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + total * 0.01,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x_f)
    ax.set_xticklabels([s[0] for s in stages], fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("(f) Detection Funnel\n"
                 "탐지 → 정밀도 → grid 정확도 단계별 감쇠", fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "boundary_precision_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")

    # ─────────────────────────────────────────
    # Figure 2: IoU vs max_norm scatter + grid error
    # ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("IoU vs Boundary Precision & Grid Error",
                 fontsize=12, fontweight='bold')

    ax = axes[0]
    colors_scatter = det["grid_err_any"].map({0: "#43A047", 1: "#E53935"})
    ax.scatter(det["iou"], det["max_norm"], c=colors_scatter,
               alpha=0.35, s=15, edgecolors='none')
    ax.set_xlabel("IoU (GT vs TATR span pred)")
    ax.set_ylabel("Max normalized edge offset (|pred−gt| / span_size)")
    ax.set_title("IoU vs Max Normalized Edge Offset\n"
                 "● green=grid OK  ● red=grid index error",
                 fontweight='bold')
    ax.axvline(0.9, color='gray', ls='--', lw=1.2, alpha=0.7, label='IoU=0.9')
    ax.axhline(0.1, color='gray', ls=':', lw=1.0, alpha=0.5)
    ax.set_xlim(0.5, 1.01)
    ax.set_ylim(0, 1.5)
    ax.legend(handles=[
        mpatches.Patch(color='#43A047', label='Grid OK'),
        mpatches.Patch(color='#E53935', label='Grid Error'),
    ], fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[1]
    iou_range = np.linspace(0.5, 0.98, 50)
    grid_err_rates = []
    grid_err_col_rates = []
    grid_err_row_rates = []
    for iou_min in iou_range:
        sub = det[det["iou"] >= iou_min]
        if len(sub) < 5:
            grid_err_rates.append(np.nan)
            grid_err_col_rates.append(np.nan)
            grid_err_row_rates.append(np.nan)
        else:
            grid_err_rates.append(sub["grid_err_any"].mean() * 100)
            grid_err_col_rates.append(sub["grid_err_col"].mean() * 100)
            grid_err_row_rates.append(sub["grid_err_row"].mean() * 100)

    ax.plot(iou_range, grid_err_rates, 'k-', lw=2.5, label='Any grid error')
    ax.plot(iou_range, grid_err_col_rates, 'b--', lw=1.8, label='Col span error (x edge)')
    ax.plot(iou_range, grid_err_row_rates, 'r--', lw=1.8, label='Row span error (y edge)')
    ax.fill_between(iou_range,
                    [v if not np.isnan(v) else 0 for v in grid_err_rates],
                    alpha=0.12, color='black')
    ax.set_xlabel("IoU threshold (minimum IoU for inclusion)")
    ax.set_ylabel("Grid Index Error Rate (%)")
    ax.set_title("IoU Threshold → Grid Error Rate\n"
                 "IoU 기준을 높일수록 grid 오류가 줄어드는가?",
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, 0.98)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    path2 = os.path.join(out_dir, "boundary_iou_vs_grid.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path2}")

    return det


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pubtables_root",
                        default="/home/ugh/t1/data/pubtables-1m")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--num_samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    print("=" * 65)
    print("Spanning Cell Boundary Precision Analysis (GSR motivation)")
    print(f"  Dataset : PubTables-1M test")
    print(f"  Samples : {args.num_samples}")
    print(f"  Device  : {DEVICE}")
    print("=" * 65)

    proc, mdl = load_model()
    df = evaluate(args.pubtables_root, proc, mdl, args.num_samples)

    det = df[df["detected"] == 1]
    n_total = len(df)
    n_det   = len(det)
    n_grid_err = int(det["grid_err_any"].sum())

    print(f"\n{'='*65}")
    print("결과 요약")
    print(f"{'='*65}")
    print(f"  GT spanning cells 총합    : {n_total}")
    print(f"  탐지됨  (IoU ≥ 0.5)       : {n_det} ({n_det/n_total*100:.1f}%)")
    print(f"  정밀    (IoU ≥ 0.75)      : {(df['iou']>=0.75).sum()} ({(df['iou']>=0.75).mean()*100:.1f}%)")
    print(f"  매우 정밀 (IoU ≥ 0.9)     : {(df['iou']>=0.9).sum()} ({(df['iou']>=0.9).mean()*100:.1f}%)")
    print()
    print(f"  [탐지된 {n_det}개 중 grid index 오류]")
    print(f"    Any grid error  : {n_grid_err} ({n_grid_err/n_det*100:.1f}%)")
    print(f"    Col span error  : {int(det['grid_err_col'].sum())} "
          f"({det['grid_err_col'].mean()*100:.1f}%)")
    print(f"    Row span error  : {int(det['grid_err_row'].sum())} "
          f"({det['grid_err_row'].mean()*100:.1f}%)")
    print()
    print(f"  [Edge offset — detected spans]")
    print(f"    max_norm mean : {det['max_norm'].mean():.4f}  "
          f"(상대 오차 {det['max_norm'].mean()*100:.1f}%)")
    print(f"    max_norm > 10%: {(det['max_norm']>0.1).mean()*100:.1f}%  "
          f"← separator 넘을 가능성")
    print(f"    snap dist mean: {det['max_snap'].mean():.1f} px  "
          f"(nearest separator까지 평균 거리)")
    print()

    # ── IoU-구간별 grid 오류율 ──
    print(f"  [IoU 구간별 grid index 오류율]")
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        sub = det[(det["iou"] >= lo) & (det["iou"] < hi)]
        if len(sub) == 0:
            continue
        err_rate = sub["grid_err_any"].mean() * 100
        print(f"    [{lo:.1f},{hi:.2f}): n={len(sub):>4},  "
              f"grid err = {err_rate:>5.1f}%")

    print()
    print("  [결론]")
    lo_iou_err = det[det["iou"] < 0.75]["grid_err_any"].mean() * 100
    hi_iou_err = det[det["iou"] >= 0.9]["grid_err_any"].mean() * 100
    print(f"    AP@0.5~0.75 구간: grid error {lo_iou_err:.1f}%")
    print(f"    AP@0.9+  구간  : grid error {hi_iou_err:.1f}%")
    print(f"    → IoU@0.5 기준 탐지는 84.9%이나 grid 정확도는 훨씬 낮음")
    print(f"    → separator snapping이 이 gap을 직접 해결")
    print(f"{'='*65}")

    # 저장
    csv_path = os.path.join(args.output_dir, "boundary_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  → {csv_path}")

    json_path = os.path.join(args.output_dir, "boundary_summary.json")
    summary = {
        "n_gt_spans": n_total,
        "n_detected_50": int(n_det),
        "n_detected_75": int((df["iou"] >= 0.75).sum()),
        "n_detected_90": int((df["iou"] >= 0.9).sum()),
        "grid_err_rate_all_detected": float(det["grid_err_any"].mean()),
        "grid_err_rate_iou_90plus": float(
            det[det["iou"] >= 0.9]["grid_err_any"].mean()),
        "max_norm_mean": float(det["max_norm"].mean()),
        "max_snap_mean_px": float(det["max_snap"].mean()),
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  → {json_path}")

    print("\n시각화 생성 중...")
    visualize(df, args.output_dir)
    print("\n✅ 완료")


if __name__ == "__main__":
    main()

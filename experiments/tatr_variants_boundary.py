#!/usr/bin/env python3
"""
TATR Variants Boundary Precision Comparison
============================================
같은 DETR 아키텍처, 다른 학습 데이터 3종에서 동일 패턴 확인.

모델 세 가지:
  v1.0 : microsoft/table-transformer-structure-recognition
         학습: PubTables-1M (원본 split)
  v1.1-pub: microsoft/table-transformer-structure-recognition-v1.1-pub
         학습: PubTables-1M (개선 annotation)
  v1.1-all: microsoft/table-transformer-structure-recognition-v1.1-all
         학습: PubTables-1M + FinTabNet + SciTSR (이종 도메인)

결론: 학습 데이터가 달라도 IoU@0.5 → IoU@0.9 구간에서
      동일한 grid index error monotonic decrease 패턴 → regression target 구조 문제

사용법:
  python tatr_variants_boundary.py \
      --pubtables_root /home/ugh/t1/data/pubtables-1m \
      --output_dir ./results/variants_comparison \
      --num_samples 300 \
      --confidence 0.5
"""

import os
import json
import warnings
import argparse
import xml.etree.ElementTree as ET

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
from transformers import AutoModelForObjectDetection, AutoImageProcessor

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

MODELS = [
    {
        "key":  "v1.0",
        "name": "microsoft/table-transformer-structure-recognition",
        "desc": "v1.0 (PubTables-1M original)",
        "color": "#4C72B0",
    },
    {
        "key":  "v1.1-pub",
        "name": "microsoft/table-transformer-structure-recognition-v1.1-pub",
        "desc": "v1.1-pub (PubTables-1M improved)",
        "color": "#DD8452",
    },
    {
        "key":  "v1.1-all",
        "name": "microsoft/table-transformer-structure-recognition-v1.1-all",
        "desc": "v1.1-all (PubTables-1M + FinTabNet + SciTSR)",
        "color": "#55A868",
    },
]

IOU_BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
RECALL_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


# ─────────────────────────────────────────────
# 모델 로딩 — label ID 자동 탐색
# ─────────────────────────────────────────────
def load_model(model_cfg):
    """
    모델 로드 + spanning cell label ID를 id2label에서 자동 탐색.
    id2label에 없으면 5번 (v1.1-all 기본) 사용.
    """
    name = model_cfg["name"]
    print(f"  Loading {model_cfg['key']}: {name}")
    proc = AutoImageProcessor.from_pretrained(name)

    # 일부 transformers 버전 processor.size 패치
    if isinstance(proc.size, dict) and list(proc.size.keys()) == ["longest_edge"]:
        v = proc.size["longest_edge"]
        proc.size = {"shortest_edge": v, "longest_edge": v}

    model = AutoModelForObjectDetection.from_pretrained(name).to(DEVICE)
    model.eval()

    # spanning cell label ID 탐색
    span_label_id = _find_span_label(model)
    row_label_id  = _find_row_label(model)
    col_label_id  = _find_col_label(model)
    print(f"    -> label ids: row={row_label_id}, col={col_label_id}, span={span_label_id}")

    return proc, model, span_label_id, row_label_id, col_label_id


def _find_label(model, keywords):
    """id2label에서 keywords 중 하나를 포함하는 label ID를 반환."""
    id2label = getattr(model.config, "id2label", {})
    for lid, lname in id2label.items():
        lname_lower = lname.lower()
        if any(kw in lname_lower for kw in keywords):
            return int(lid)
    return None


def _find_span_label(model):
    lid = _find_label(model, ["spanning", "span"])
    return lid if lid is not None else 5


def _find_row_label(model):
    lid = _find_label(model, ["table row"])
    if lid is None:
        lid = _find_label(model, ["row"])
    return lid if lid is not None else 2


def _find_col_label(model):
    lid = _find_label(model, ["table column"])
    if lid is None:
        lid = _find_label(model, ["column"])
    return lid if lid is not None else 1


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
        bbox = [float(bb.findtext(k, 0)) for k in ["xmin", "ymin", "xmax", "ymax"]]
        if   name == "table row":           rows.append(bbox)
        elif name == "table column":        cols.append(bbox)
        elif name == "table spanning cell": spans.append(bbox)
    return rows, cols, spans


# ─────────────────────────────────────────────
# Separator 추출 & 분석 유틸
# ─────────────────────────────────────────────
def get_separators(rows, cols):
    row_seps = sorted(set([b[1] for b in rows] + [b[3] for b in rows]))
    col_seps = sorted(set([b[0] for b in cols] + [b[2] for b in cols]))
    return row_seps, col_seps


def nearest_sep_idx(val, seps):
    if not seps:
        return 0
    arr = np.array(seps)
    return int(np.argmin(np.abs(arr - val)))


def nearest_snap_dist(val, seps):
    if not seps:
        return 0.0
    arr = np.array(seps)
    return float(np.min(np.abs(arr - val)))


def analyze_pair(gt_span, pred_span, row_seps, col_seps):
    gx1, gy1, gx2, gy2 = gt_span
    px1, py1, px2, py2 = pred_span
    gt_w = max(gx2 - gx1, 1e-6)
    gt_h = max(gy2 - gy1, 1e-6)

    iou = float(box_iou(
        torch.tensor([[px1, py1, px2, py2]], dtype=torch.float32),
        torch.tensor([[gx1, gy1, gx2, gy2]], dtype=torch.float32),
    )[0, 0])

    max_norm = max(
        abs(px1 - gx1) / gt_w, abs(px2 - gx2) / gt_w,
        abs(py1 - gy1) / gt_h, abs(py2 - gy2) / gt_h,
    )
    max_snap = max(
        nearest_snap_dist(px1, col_seps), nearest_snap_dist(px2, col_seps),
        nearest_snap_dist(py1, row_seps), nearest_snap_dist(py2, row_seps),
    )

    gi_err = (
        nearest_sep_idx(px1, col_seps) != nearest_sep_idx(gx1, col_seps) or
        nearest_sep_idx(px2, col_seps) != nearest_sep_idx(gx2, col_seps) or
        nearest_sep_idx(py1, row_seps) != nearest_sep_idx(gy1, row_seps) or
        nearest_sep_idx(py2, row_seps) != nearest_sep_idx(gy2, row_seps)
    )
    return {"iou": iou, "max_norm": max_norm, "max_snap": max_snap, "grid_err": int(gi_err)}


# ─────────────────────────────────────────────
# 추론
# ─────────────────────────────────────────────
@torch.no_grad()
def infer(image, proc, model, conf, span_lid, row_lid, col_lid):
    inputs = proc(images=image, return_tensors="pt").to(DEVICE)
    out    = model(**inputs)
    tgt    = torch.tensor([image.size[::-1]])
    res    = proc.post_process_object_detection(out, threshold=conf, target_sizes=tgt)[0]
    labels = res["labels"].cpu().numpy()
    boxes  = res["boxes"].cpu().numpy()
    return {
        "spans": [boxes[i].tolist() for i in range(len(labels)) if labels[i] == span_lid],
        "rows":  [boxes[i].tolist() for i in range(len(labels)) if labels[i] == row_lid],
        "cols":  [boxes[i].tolist() for i in range(len(labels)) if labels[i] == col_lid],
    }


# ─────────────────────────────────────────────
# 평가 루프 (단일 모델)
# ─────────────────────────────────────────────
def evaluate_model(img_files, img_dir, ann_dir, proc, model, conf,
                   span_lid, row_lid, col_lid, model_key):
    records = []

    for img_file in tqdm(img_files, desc=f"  [{model_key}]"):
        stem     = os.path.splitext(img_file)[0]
        xml_path = os.path.join(ann_dir, stem + ".xml")
        if not os.path.exists(xml_path):
            continue

        rows, cols, spans = parse_xml(xml_path)
        if not spans or not rows or not cols:
            continue

        try:
            image = Image.open(os.path.join(img_dir, img_file)).convert("RGB")
        except Exception:
            continue

        pred       = infer(image, proc, model, conf, span_lid, row_lid, col_lid)
        row_seps, col_seps = get_separators(rows, cols)

        if not pred["spans"]:
            for sp in spans:
                records.append({"model": model_key, "detected": 0, "iou": 0.0,
                                 "max_norm": None, "max_snap": None, "grid_err": None})
            continue

        gt_t  = torch.tensor(spans,        dtype=torch.float32)
        pr_t  = torch.tensor(pred["spans"], dtype=torch.float32)
        iou_m = box_iou(gt_t, pr_t).numpy()
        gi, pi = linear_sum_assignment(1.0 - iou_m)

        matched = set()
        for g, p in zip(gi, pi):
            miou = iou_m[g, p]
            rec  = analyze_pair(spans[g], pred["spans"][p], row_seps, col_seps)
            records.append({
                "model":    model_key,
                "detected": 1 if miou >= 0.5 else 0,
                **rec,
            })
            matched.add(g)

        for g in range(len(spans)):
            if g not in matched:
                records.append({"model": model_key, "detected": 0, "iou": 0.0,
                                 "max_norm": None, "max_snap": None, "grid_err": None})

    return records


# ─────────────────────────────────────────────
# 통계 요약
# ─────────────────────────────────────────────
def summarize(df):
    summary = {}
    for model_key, grp in df.groupby("model"):
        det = grp[grp["detected"] == 1]
        n_gt  = len(grp)
        n_det = len(det)

        bin_stats = []
        for lo, hi in IOU_BINS:
            band = det[(det["iou"] >= lo) & (det["iou"] < hi)]
            n    = len(band)
            err  = int(band["grid_err"].sum()) if n > 0 else 0
            bin_stats.append({
                "iou_bin": f"[{lo},{hi})",
                "n": n,
                "grid_err_n": err,
                "grid_err_rate": err / n if n > 0 else None,
            })

        recall_curve = {}
        if len(det) > 0:
            for t in RECALL_THRESHOLDS:
                recall_curve[t] = float((det["iou"] >= t).sum()) / n_gt
        else:
            recall_curve = {t: 0.0 for t in RECALL_THRESHOLDS}

        summary[model_key] = {
            "n_gt_spans":         n_gt,
            "n_detected_50":      n_det,
            "detection_rate_50":  n_det / n_gt if n_gt > 0 else 0.0,
            "n_detected_90":      int((det["iou"] >= 0.9).sum()),
            "detection_rate_90":  float((det["iou"] >= 0.9).sum()) / n_gt if n_gt > 0 else 0.0,
            "grid_err_rate_all":  float(det["grid_err"].mean()) if n_det > 0 else 0.0,
            "grid_err_rate_90plus": 0.0,  # filled below
            "max_norm_mean":      float(det["max_norm"].mean()) if n_det > 0 else 0.0,
            "max_snap_mean":      float(det["max_snap"].mean()) if n_det > 0 else 0.0,
            "iou_bin_stats":      bin_stats,
            "recall_curve":       recall_curve,
        }
        # grid_err_rate@0.9+
        band_90 = det[det["iou"] >= 0.9]
        if len(band_90) > 0:
            summary[model_key]["grid_err_rate_90plus"] = float(band_90["grid_err"].mean())

    return summary


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────
def plot_comparison(summary, out_dir):
    """4-panel comparative figure."""
    keys   = [m["key"]   for m in MODELS]
    descs  = [m["desc"]  for m in MODELS]
    colors = [m["color"] for m in MODELS]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "TATR Variant Comparison: Boundary Precision\n"
        "(Same architecture, different training data — pattern is consistent)",
        fontsize=13, fontweight="bold",
    )

    # ── Panel A: Recall Curve ────────────────────────────────────────────
    ax = axes[0, 0]
    for key, desc, color in zip(keys, descs, colors):
        if key not in summary:
            continue
        rc = summary[key]["recall_curve"]
        xs = RECALL_THRESHOLDS
        ys = [rc[t] for t in xs]
        ax.plot(xs, ys, marker="o", color=color, linewidth=2, markersize=7, label=desc)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color=color)
    ax.set_title("(A) Span Detection Recall vs IoU Threshold", fontweight="bold")
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("Recall")
    ax.set_xticks(RECALL_THRESHOLDS)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # ── Panel B: Grid Error Rate by IoU Bin ─────────────────────────────
    ax = axes[0, 1]
    bin_labels = [f"[{lo},{hi})" for lo, hi in IOU_BINS]
    x = np.arange(len(bin_labels))
    width = 0.25
    offsets = [-width, 0, width]
    for i, (key, desc, color) in enumerate(zip(keys, descs, colors)):
        if key not in summary:
            continue
        rates = []
        for bs in summary[key]["iou_bin_stats"]:
            r = bs["grid_err_rate"]
            rates.append(r if r is not None else 0.0)
        bars = ax.bar(x + offsets[i], rates, width, label=desc, color=color, alpha=0.85)
        for bar, rate in zip(bars, rates):
            if rate > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{rate:.2f}", ha="center", va="bottom", fontsize=7, color=color)
    ax.set_title("(B) Grid Index Error Rate by IoU Bin", fontweight="bold")
    ax.set_xlabel("IoU Bin")
    ax.set_ylabel("Grid Error Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # ── Panel C: Detection Rate @0.5 vs @0.9 ────────────────────────────
    ax = axes[1, 0]
    x = np.arange(len(keys))
    width = 0.35
    r50 = [summary[k]["detection_rate_50"] if k in summary else 0 for k in keys]
    r90 = [summary[k]["detection_rate_90"] if k in summary else 0 for k in keys]
    b1 = ax.bar(x - width / 2, r50, width, label="Recall@IoU0.5", color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + width / 2, r90, width, label="Recall@IoU0.9", color="#C44E52", alpha=0.85)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for k, x_pos, r5, r9 in zip(keys, x, r50, r90):
        gap = r5 - r9
        ax.annotate(f"Δ={gap:.3f}", (x_pos, max(r5, r9) + 0.05),
                    ha="center", fontsize=9, color="#8B0000", fontweight="bold")
    ax.set_title("(C) Span Detection: Recall@0.5 vs Recall@0.9\n"
                 "(Δ = boundary precision gap)", fontweight="bold")
    ax.set_ylabel("Recall")
    ax.set_xticks(x)
    ax.set_xticklabels([m["key"] for m in MODELS])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── Panel D: Grid Error @detected vs @IoU≥0.9 ───────────────────────
    ax = axes[1, 1]
    gerr_all = [summary[k]["grid_err_rate_all"]   if k in summary else 0 for k in keys]
    gerr_90  = [summary[k]["grid_err_rate_90plus"] if k in summary else 0 for k in keys]
    b1 = ax.bar(x - width / 2, gerr_all, width, label="Grid Err (IoU≥0.5)", color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + width / 2, gerr_90,  width, label="Grid Err (IoU≥0.9)", color="#55A868", alpha=0.85)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("(D) Grid Index Error Rate\n"
                 "(GSR target: push all predictions into IoU≥0.9 zone → 0% error)",
                 fontweight="bold")
    ax.set_ylabel("Grid Error Rate")
    ax.set_xticks(x)
    ax.set_xticklabels([m["key"] for m in MODELS])
    ax.set_ylim(0, 0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "tatr_variants_comparison.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


def plot_recall_drop_table(summary, out_dir):
    """보조: recall drop 정리 테이블 figure."""
    rows = []
    for m in MODELS:
        k = m["key"]
        if k not in summary:
            continue
        s = summary[k]
        rc = s["recall_curve"]
        rows.append([
            m["key"],
            f"{rc[0.5]:.3f}",
            f"{rc[0.7]:.3f}",
            f"{rc[0.9]:.3f}",
            f"{rc[0.5]-rc[0.9]:.3f}",
            f"{s['grid_err_rate_all']:.3f}",
            f"{s['grid_err_rate_90plus']:.3f}",
        ])

    col_labels = ["Model", "Recall\n@0.5", "Recall\n@0.7", "Recall\n@0.9",
                  "Δ(0.5→0.9)", "GridErr\n(IoU≥0.5)", "GridErr\n(IoU≥0.9)"]

    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # 헤더 색
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Δ 컬럼 강조
    for i in range(1, len(rows) + 1):
        tbl[i, 4].set_facecolor("#fff0f0")

    fig.suptitle("TATR Variants: Boundary Precision Gap Summary\n"
                 "Same arch, different training data → same pattern",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "tatr_variants_summary_table.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pubtables_root", default="/home/ugh/t1/data/pubtables-1m")
    p.add_argument("--output_dir",    default="./results/variants_comparison")
    p.add_argument("--num_samples",   type=int, default=300)
    p.add_argument("--confidence",    type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("TATR Variants Boundary Precision Comparison")
    print("=" * 60)
    print(f"  num_samples : {args.num_samples}")
    print(f"  confidence  : {args.confidence}")
    print(f"  device      : {DEVICE}")

    img_dir = os.path.join(args.pubtables_root, "test", "images")
    ann_dir = os.path.join(args.pubtables_root, "test", "annotations")

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith((".jpg", ".png"))])
    np.random.seed(SEED)
    if args.num_samples < len(img_files):
        img_files = [img_files[i] for i in
                     sorted(np.random.choice(len(img_files), args.num_samples, replace=False))]
    print(f"  Using {len(img_files)} images\n")

    all_records = []

    for m_cfg in MODELS:
        print(f"\n[{m_cfg['key']}] {m_cfg['desc']}")
        proc, model, span_lid, row_lid, col_lid = load_model(m_cfg)

        recs = evaluate_model(
            img_files, img_dir, ann_dir,
            proc, model, args.confidence,
            span_lid, row_lid, col_lid,
            m_cfg["key"],
        )
        all_records.extend(recs)

        # 메모리 해제
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(args.output_dir, "variants_records.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  -> {csv_path}")

    summary = summarize(df)
    json_path = os.path.join(args.output_dir, "variants_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {json_path}")

    print("\n[Visualizations]")
    plot_comparison(summary, args.output_dir)
    plot_recall_drop_table(summary, args.output_dir)

    # 콘솔 요약
    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    print(f"{'Model':<12} {'Recall@0.5':>10} {'Recall@0.9':>10} {'Δ':>8}  "
          f"{'GridErr@det':>12} {'GridErr@0.9+':>13}")
    print("-" * 70)
    for m in MODELS:
        k = m["key"]
        if k not in summary:
            continue
        s  = summary[k]
        rc = s["recall_curve"]
        print(f"{k:<12} {rc[0.5]:>10.3f} {rc[0.9]:>10.3f} "
              f"{rc[0.5]-rc[0.9]:>8.3f}  "
              f"{s['grid_err_rate_all']:>12.3f} {s['grid_err_rate_90plus']:>13.3f}")
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()

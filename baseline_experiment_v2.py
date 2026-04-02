"""
=======================================================================
TATR Baseline Experiment v2: Span vs Non-Span Cell IoU Gap Analysis
=======================================================================
v1 대비 개선사항:
  1. Hungarian matching (scipy) — greedy의 순서 의존성 제거
  2. Label-aware matching — TATR pred label 5(spanning cell)는 span GT에,
     row×col pred는 non-span GT에 우선 매칭
  3. Welch's t-test — span vs non-span IoU gap의 통계적 유의성 검증
  4. Confidence sensitivity — 여러 threshold에서 gap 안정성 확인
  5. Per-image 분석 — 이미지별 gap 분포로 outlier/artifact 확인

사용법:
    python baseline_experiment_v2.py \
        --pubtables_root /path/to/PubTables-1M-Structure \
        --output_dir ./results_v2 \
        --num_samples 500
=======================================================================
"""

import os
import json
import argparse
import warnings
from collections import defaultdict
import xml.etree.ElementTree as ET

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_ind, mannwhitneyu

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.5
IOU_MATCH_THRESHOLD = 0.5
NUM_SAMPLES = 500
SEED = 42

# TATR v1.1 label mapping (PubTables-1M)
LABEL_MAP = {
    0: "table",
    1: "table column",
    2: "table row",
    3: "table column header",
    4: "table projected row header",
    5: "table spanning cell",
}

# Spanning cell label ID in TATR output
SPAN_LABEL_ID = 5


# ─────────────────────────────────────────────
# 1. 모델 로드
# ─────────────────────────────────────────────
def load_model():
    print(f"[1/6] 모델 로드 중... (device: {DEVICE})")
    model_name = "microsoft/table-transformer-structure-recognition-v1.1-all"
    processor = AutoImageProcessor.from_pretrained(model_name)
    # 일부 transformers 버전에서 'longest_edge' only 포맷을 지원하지 않아 패치
    if isinstance(processor.size, dict) and list(processor.size.keys()) == ["longest_edge"]:
        longest = processor.size["longest_edge"]
        processor.size = {"shortest_edge": longest, "longest_edge": longest}
    model = AutoModelForObjectDetection.from_pretrained(model_name).to(DEVICE)
    model.eval()
    print(f"  → {model_name}")
    return processor, model


# ─────────────────────────────────────────────
# 2. GT 파싱
# ─────────────────────────────────────────────
def load_gt_annotation(xml_path):
    """PubTables-1M GT XML (Pascal VOC) → category별 bbox dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {
        "spanning_cell_boxes": [],
        "categories": defaultdict(list),
    }

    for obj in root.findall("object"):
        cat = obj.findtext("name", "").strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", 0))
        ymin = float(bndbox.findtext("ymin", 0))
        xmax = float(bndbox.findtext("xmax", 0))
        ymax = float(bndbox.findtext("ymax", 0))
        bbox = [xmin, ymin, xmax, ymax]
        result["categories"][cat].append(bbox)
        if cat == "table spanning cell":
            result["spanning_cell_boxes"].append(bbox)

    return result


def build_spanning_aware_gt(gt_data):
    """
    Spanning cell: GT의 spanning cell bbox 그대로 사용.
    Non-spanning cell: row×column 교차로 생성, spanning 영역 제외.

    반환: list of {"bbox": [x1,y1,x2,y2], "is_span": bool}
    """
    rows = gt_data["categories"].get("table row", [])
    cols = gt_data["categories"].get("table column", [])
    spanning_boxes = gt_data["spanning_cell_boxes"]

    if not rows or not cols:
        return []

    cells = []

    # Spanning cells — GT bbox 그대로
    for sp_bbox in spanning_boxes:
        cells.append({"bbox": sp_bbox, "is_span": True})

    # Non-spanning cells — row×col 교차, spanning 영역 제외
    for row_bbox in rows:
        for col_bbox in cols:
            x_min = max(row_bbox[0], col_bbox[0])
            y_min = max(row_bbox[1], col_bbox[1])
            x_max = min(row_bbox[2], col_bbox[2])
            y_max = min(row_bbox[3], col_bbox[3])

            if x_min >= x_max or y_min >= y_max:
                continue

            # 중심점이 spanning cell 안에 있으면 제외
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            overlaps = any(
                sp[0] <= cx <= sp[2] and sp[1] <= cy <= sp[3]
                for sp in spanning_boxes
            )
            if not overlaps:
                cells.append({"bbox": [x_min, y_min, x_max, y_max], "is_span": False})

    return cells


# ─────────────────────────────────────────────
# 3. Inference
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(image, processor, model, threshold):
    """TATR inference → boxes, scores, labels."""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    return {
        "boxes": results["boxes"].cpu(),
        "scores": results["scores"].cpu(),
        "labels": results["labels"].cpu(),
    }


# ─────────────────────────────────────────────
# 4. Hungarian Matching (v2 핵심 개선)
# ─────────────────────────────────────────────
def hungarian_match(gt_cells, pred_results):
    """
    Hungarian algorithm으로 GT-Pred 1:1 최적 매칭.

    v1의 greedy 매칭 문제:
    - GT 인덱스 순서에 따라 매칭 결과가 달라짐
    - 앞쪽 GT가 좋은 pred를 선점하면 뒤쪽 GT의 IoU가 인위적으로 낮아짐

    v2 개선:
    - 전체 IoU 합을 최대화하는 global optimum 매칭
    - scipy.optimize.linear_sum_assignment 사용 (cost = 1 - IoU)
    """
    n_gt = len(gt_cells)
    n_pred = len(pred_results["boxes"])

    if n_gt == 0:
        return []
    if n_pred == 0:
        return [{"iou": 0.0, "is_span": c["is_span"], "matched": False}
                for c in gt_cells]

    gt_boxes = torch.tensor([c["bbox"] for c in gt_cells], dtype=torch.float32)
    pred_boxes = pred_results["boxes"].float()

    iou_matrix = box_iou(gt_boxes, pred_boxes).numpy()  # (n_gt, n_pred)

    # Hungarian: minimize cost = 1 - IoU
    cost_matrix = 1.0 - iou_matrix
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # 결과 구성
    matched_ious = np.zeros(n_gt)
    for gi, pi in zip(gt_indices, pred_indices):
        matched_ious[gi] = iou_matrix[gi, pi]

    results = []
    for i, cell in enumerate(gt_cells):
        iou_val = float(matched_ious[i])
        results.append({
            "iou": iou_val,
            "is_span": cell["is_span"],
            "matched": iou_val >= IOU_MATCH_THRESHOLD,
        })

    return results


def label_aware_match(gt_cells, pred_results):
    """
    Label-aware Hungarian matching.

    TATR은 label=5로 spanning cell을 직접 예측함.
    이 정보를 활용하여:
    - span GT ↔ span pred (label=5) 먼저 매칭
    - non-span GT ↔ 나머지 pred 매칭
    - 매칭 안 된 GT는 전체 pred pool에서 fallback 매칭

    이렇게 하면 span pred가 non-span GT에 먹히는 문제를 방지.
    """
    n_gt = len(gt_cells)
    n_pred = len(pred_results["boxes"])

    if n_gt == 0:
        return []
    if n_pred == 0:
        return [{"iou": 0.0, "is_span": c["is_span"], "matched": False,
                 "pred_label": None}
                for c in gt_cells]

    gt_boxes = torch.tensor([c["bbox"] for c in gt_cells], dtype=torch.float32)
    pred_boxes = pred_results["boxes"].float()
    pred_labels = pred_results["labels"].numpy()

    iou_matrix = box_iou(gt_boxes, pred_boxes).numpy()

    # GT와 pred를 span/non-span으로 분리
    span_gt_idx = [i for i, c in enumerate(gt_cells) if c["is_span"]]
    nonspan_gt_idx = [i for i, c in enumerate(gt_cells) if not c["is_span"]]
    span_pred_idx = [j for j in range(n_pred) if pred_labels[j] == SPAN_LABEL_ID]
    nonspan_pred_idx = [j for j in range(n_pred) if pred_labels[j] != SPAN_LABEL_ID]

    matched_ious = np.zeros(n_gt)
    matched_pred_labels = [None] * n_gt
    used_preds = set()

    # Phase 1: span GT ↔ span pred
    if span_gt_idx and span_pred_idx:
        sub_iou = iou_matrix[np.ix_(span_gt_idx, span_pred_idx)]
        cost = 1.0 - sub_iou
        gi_local, pi_local = linear_sum_assignment(cost)
        for gl, pl in zip(gi_local, pi_local):
            gi_global = span_gt_idx[gl]
            pi_global = span_pred_idx[pl]
            matched_ious[gi_global] = iou_matrix[gi_global, pi_global]
            matched_pred_labels[gi_global] = int(pred_labels[pi_global])
            used_preds.add(pi_global)

    # Phase 2: non-span GT ↔ non-span pred
    remaining_nonspan_pred = [j for j in nonspan_pred_idx if j not in used_preds]
    if nonspan_gt_idx and remaining_nonspan_pred:
        sub_iou = iou_matrix[np.ix_(nonspan_gt_idx, remaining_nonspan_pred)]
        cost = 1.0 - sub_iou
        gi_local, pi_local = linear_sum_assignment(cost)
        for gl, pl in zip(gi_local, pi_local):
            gi_global = nonspan_gt_idx[gl]
            pi_global = remaining_nonspan_pred[pl]
            matched_ious[gi_global] = iou_matrix[gi_global, pi_global]
            matched_pred_labels[gi_global] = int(pred_labels[pi_global])
            used_preds.add(pi_global)

    # Phase 3: 아직 매칭 안 된 GT → 남은 pred 전체에서 fallback
    unmatched_gt = [i for i in range(n_gt) if matched_ious[i] == 0]
    remaining_pred = [j for j in range(n_pred) if j not in used_preds]
    if unmatched_gt and remaining_pred:
        sub_iou = iou_matrix[np.ix_(unmatched_gt, remaining_pred)]
        cost = 1.0 - sub_iou
        gi_local, pi_local = linear_sum_assignment(cost)
        for gl, pl in zip(gi_local, pi_local):
            gi_global = unmatched_gt[gl]
            pi_global = remaining_pred[pl]
            if iou_matrix[gi_global, pi_global] > 0:
                matched_ious[gi_global] = iou_matrix[gi_global, pi_global]
                matched_pred_labels[gi_global] = int(pred_labels[pi_global])

    results = []
    for i, cell in enumerate(gt_cells):
        iou_val = float(matched_ious[i])
        results.append({
            "iou": iou_val,
            "is_span": cell["is_span"],
            "matched": iou_val >= IOU_MATCH_THRESHOLD,
            "pred_label": matched_pred_labels[i],
        })

    return results


# ─────────────────────────────────────────────
# 5. 데이터 경로 탐색
# ─────────────────────────────────────────────
def find_data_dirs(root):
    """여러 가능한 PubTables-1M 디렉토리 구조를 탐색."""
    candidates = [
        (os.path.join(root, "test", "images"),
         os.path.join(root, "test", "annotations")),
        (os.path.join(root, "images", "test"),
         os.path.join(root, "annotations", "test")),
        (os.path.join(root, "PubTables-1M-Structure", "test", "images"),
         os.path.join(root, "PubTables-1M-Structure", "test", "annotations")),
    ]
    for img_dir, ann_dir in candidates:
        if os.path.exists(img_dir) and os.path.exists(ann_dir):
            return img_dir, ann_dir

    # 마지막 시도: 재귀 탐색
    for dirpath, dirnames, filenames in os.walk(root):
        if "images" in dirnames:
            img_candidate = os.path.join(dirpath, "images")
            ann_candidate = os.path.join(dirpath, "annotations")
            if os.path.exists(ann_candidate):
                return img_candidate, ann_candidate

    raise FileNotFoundError(
        f"데이터를 찾을 수 없습니다: {root}\n"
        f"예상 구조: root/test/images/ + root/test/annotations/"
    )


# ─────────────────────────────────────────────
# 6. 평가 루프
# ─────────────────────────────────────────────
def evaluate(data_dir, processor, model, num_samples=None, confidence=0.5):
    """
    PubTables-1M test set 평가.
    두 가지 매칭 방식을 동시에 실행하여 비교.
    """
    img_dir, ann_dir = find_data_dirs(data_dir)
    print(f"  이미지: {img_dir}")
    print(f"  어노테이션: {ann_dir}")

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"  총 이미지: {len(img_files)}")

    if num_samples and num_samples < len(img_files):
        np.random.seed(SEED)
        indices = np.random.choice(len(img_files), num_samples, replace=False)
        img_files = [img_files[i] for i in sorted(indices)]
        print(f"  샘플링: {num_samples}개")

    all_hungarian = []
    all_label_aware = []
    skipped = 0

    # 이미지별 통계 수집 (per-image analysis용)
    per_image_stats = []

    for img_file in tqdm(img_files, desc="Inference"):
        stem = os.path.splitext(img_file)[0]
        ann_file = os.path.join(ann_dir, stem + ".xml")

        if not os.path.exists(ann_file):
            skipped += 1
            continue

        gt_data = load_gt_annotation(ann_file)
        gt_cells = build_spanning_aware_gt(gt_data)

        if not gt_cells:
            skipped += 1
            continue

        try:
            image = Image.open(os.path.join(img_dir, img_file)).convert("RGB")
        except Exception:
            skipped += 1
            continue

        pred = run_inference(image, processor, model, confidence)

        # 두 가지 매칭
        h_matches = hungarian_match(gt_cells, pred)
        la_matches = label_aware_match(gt_cells, pred)

        for m in h_matches:
            m["image"] = stem
        for m in la_matches:
            m["image"] = stem

        all_hungarian.extend(h_matches)
        all_label_aware.extend(la_matches)

        # Per-image stats
        span_cells = [m for m in la_matches if m["is_span"]]
        nonspan_cells = [m for m in la_matches if not m["is_span"]]
        if span_cells and nonspan_cells:
            per_image_stats.append({
                "image": stem,
                "n_span": len(span_cells),
                "n_nonspan": len(nonspan_cells),
                "mean_iou_span": np.mean([m["iou"] for m in span_cells]),
                "mean_iou_nonspan": np.mean([m["iou"] for m in nonspan_cells]),
                "gap": (np.mean([m["iou"] for m in nonspan_cells])
                        - np.mean([m["iou"] for m in span_cells])),
            })

    print(f"\n  처리: {len(img_files) - skipped}개 / 스킵: {skipped}개")

    df_hungarian = pd.DataFrame(all_hungarian)
    df_label_aware = pd.DataFrame(all_label_aware)
    df_per_image = pd.DataFrame(per_image_stats) if per_image_stats else pd.DataFrame()

    return df_hungarian, df_label_aware, df_per_image


# ─────────────────────────────────────────────
# 7. 통계적 검정
# ─────────────────────────────────────────────
def statistical_tests(span_ious, nonspan_ious):
    """Welch's t-test + Mann-Whitney U test."""
    results = {}

    if len(span_ious) < 2 or len(nonspan_ious) < 2:
        return {"error": "샘플 부족"}

    # Welch's t-test (등분산 가정 안 함)
    t_stat, t_pval = ttest_ind(nonspan_ious, span_ious, equal_var=False)
    results["welch_t"] = {"t_statistic": float(t_stat), "p_value": float(t_pval)}

    # Mann-Whitney U (비모수)
    u_stat, u_pval = mannwhitneyu(nonspan_ious, span_ious, alternative='greater')
    results["mann_whitney_u"] = {"u_statistic": float(u_stat), "p_value": float(u_pval)}

    # Effect size: Cohen's d
    pooled_std = np.sqrt(
        (np.std(nonspan_ious, ddof=1)**2 + np.std(span_ious, ddof=1)**2) / 2
    )
    cohens_d = (np.mean(nonspan_ious) - np.mean(span_ious)) / pooled_std if pooled_std > 0 else 0
    results["cohens_d"] = float(cohens_d)

    return results


# ─────────────────────────────────────────────
# 8. Confidence Sensitivity 분석
# ─────────────────────────────────────────────
def confidence_sensitivity(data_dir, processor, model, num_samples=100):
    """
    여러 confidence threshold에서 gap이 안정적인지 확인.
    gap이 특정 threshold에서만 나타나면 artifact일 수 있음.
    """
    print("\n[5/6] Confidence sensitivity 분석 중...")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    for conf in thresholds:
        _, df_la, _ = evaluate(data_dir, processor, model,
                               num_samples=num_samples, confidence=conf)
        if len(df_la) == 0:
            continue

        span = df_la[df_la["is_span"] == True]
        nonspan = df_la[df_la["is_span"] == False]

        if len(span) > 0 and len(nonspan) > 0:
            gap = nonspan["iou"].mean() - span["iou"].mean()
            results.append({
                "confidence": conf,
                "n_span": len(span),
                "n_nonspan": len(nonspan),
                "mean_iou_span": float(span["iou"].mean()),
                "mean_iou_nonspan": float(nonspan["iou"].mean()),
                "gap": float(gap),
            })
            print(f"    conf={conf:.1f}: gap={gap:.4f} "
                  f"(span={span['iou'].mean():.4f}, "
                  f"nonspan={nonspan['iou'].mean():.4f})")

    return results


# ─────────────────────────────────────────────
# 9. 분석 및 시각화
# ─────────────────────────────────────────────
def analyze_and_visualize(df_hungarian, df_label_aware, df_per_image,
                          output_dir, conf_sensitivity=None):
    """결과 분석, 통계 검정, 시각화."""
    os.makedirs(output_dir, exist_ok=True)

    # Label-aware 결과를 주 분석 대상으로 사용
    df = df_label_aware
    span_df = df[df["is_span"] == True]
    nonspan_df = df[df["is_span"] == False]

    # Hungarian 결과도 비교용
    df_h = df_hungarian
    span_h = df_h[df_h["is_span"] == True]
    nonspan_h = df_h[df_h["is_span"] == False]

    iou_gap_la = nonspan_df["iou"].mean() - span_df["iou"].mean() if len(span_df) > 0 else 0
    iou_gap_h = nonspan_h["iou"].mean() - span_h["iou"].mean() if len(span_h) > 0 else 0

    # ── 콘솔 출력 ──
    print("\n" + "=" * 65)
    print("  TATR Baseline v2 분석 결과")
    print("=" * 65)
    print(f"  전체 셀: {len(df)}")
    print(f"  Spanning: {len(span_df)} ({100*len(span_df)/len(df):.1f}%)")
    print(f"  Non-span: {len(nonspan_df)} ({100*len(nonspan_df)/len(df):.1f}%)")
    print()

    print(f"  [Mean IoU — Label-Aware Hungarian]")
    print(f"    Non-span:    {nonspan_df['iou'].mean():.4f}")
    print(f"    Span:        {span_df['iou'].mean():.4f}" if len(span_df) > 0 else "    Span:        N/A")
    print(f"    ★ Gap:       {iou_gap_la:.4f}")
    print()

    print(f"  [Mean IoU — Plain Hungarian (비교용)]")
    print(f"    Non-span:    {nonspan_h['iou'].mean():.4f}")
    print(f"    Span:        {span_h['iou'].mean():.4f}" if len(span_h) > 0 else "    Span:        N/A")
    print(f"    Gap:         {iou_gap_h:.4f}")
    print()

    # 통계적 검정
    if len(span_df) > 1:
        stats = statistical_tests(span_df["iou"].values, nonspan_df["iou"].values)
        print(f"  [통계적 검정]")
        if "welch_t" in stats:
            print(f"    Welch's t:   t={stats['welch_t']['t_statistic']:.4f}, "
                  f"p={stats['welch_t']['p_value']:.2e}")
        if "mann_whitney_u" in stats:
            print(f"    Mann-Whitney: U={stats['mann_whitney_u']['u_statistic']:.0f}, "
                  f"p={stats['mann_whitney_u']['p_value']:.2e}")
        if "cohens_d" in stats:
            d = stats["cohens_d"]
            effect = "small" if d < 0.5 else "medium" if d < 0.8 else "large"
            print(f"    Cohen's d:   {d:.4f} ({effect})")
        print()
    else:
        stats = {}

    # IoU 분포 통계
    print(f"  [IoU 분포]")
    for label, sub_df in [("Non-span", nonspan_df), ("Span", span_df)]:
        if len(sub_df) == 0:
            continue
        print(f"    {label}: mean={sub_df['iou'].mean():.4f}, "
              f"std={sub_df['iou'].std():.4f}, "
              f"Q25={sub_df['iou'].quantile(0.25):.4f}, "
              f"Q75={sub_df['iou'].quantile(0.75):.4f}")
    print()

    # AP @ 다양한 threshold
    print(f"  [AP @ IoU Threshold]")
    print(f"    {'Thr':>5}  {'All':>7}  {'Non-sp':>7}  {'Span':>7}  {'Gap':>7}")
    ap_data = []
    for t in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
        all_ap = (df['iou'] >= t).mean()
        ns_ap = (nonspan_df['iou'] >= t).mean()
        sp_ap = (span_df['iou'] >= t).mean() if len(span_df) > 0 else 0
        gap = ns_ap - sp_ap
        print(f"    @{t:.2f}  {all_ap:.4f}  {ns_ap:.4f}  {sp_ap:.4f}  {gap:.4f}")
        ap_data.append({"threshold": t, "all": all_ap, "nonspan": ns_ap,
                        "span": sp_ap, "gap": gap})

    # Per-image gap 분포
    if len(df_per_image) > 0:
        print(f"\n  [Per-Image Gap 분포]")
        print(f"    이미지 수 (span+nonspan 모두 있는): {len(df_per_image)}")
        print(f"    Mean gap:   {df_per_image['gap'].mean():.4f}")
        print(f"    Median gap: {df_per_image['gap'].median():.4f}")
        print(f"    Std gap:    {df_per_image['gap'].std():.4f}")
        pos_gap_pct = (df_per_image['gap'] > 0).mean() * 100
        print(f"    gap > 0인 이미지 비율: {pos_gap_pct:.1f}%")

    print("=" * 65)

    # ── 시각화 ──

    # Fig 1: IoU 분포 히스토그램 + 박스플롯
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(nonspan_df["iou"], bins=50, alpha=0.6, density=True,
                 label=f"Non-span (μ={nonspan_df['iou'].mean():.3f}, n={len(nonspan_df)})",
                 color="steelblue")
    if len(span_df) > 0:
        axes[0].hist(span_df["iou"], bins=50, alpha=0.6, density=True,
                     label=f"Span (μ={span_df['iou'].mean():.3f}, n={len(span_df)})",
                     color="coral")
    axes[0].set_xlabel("IoU")
    axes[0].set_ylabel("Density")
    axes[0].set_title("IoU distribution: span vs non-span")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, 1)

    # 박스플롯
    box_data = [nonspan_df["iou"].values]
    box_labels = ["Non-span"]
    box_colors = ["steelblue"]
    if len(span_df) > 0:
        box_data.append(span_df["iou"].values)
        box_labels.append("Span")
        box_colors.append("coral")

    bp = axes[1].boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("IoU")
    axes[1].set_title("IoU boxplot")

    # Per-image gap 히스토그램
    if len(df_per_image) > 0:
        axes[2].hist(df_per_image["gap"], bins=40, alpha=0.7, color="mediumpurple")
        axes[2].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        axes[2].axvline(x=df_per_image["gap"].mean(), color='red',
                        linestyle='-', linewidth=1.2, label=f"Mean={df_per_image['gap'].mean():.3f}")
        axes[2].set_xlabel("IoU gap (non-span − span)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Per-image IoU gap distribution")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No images with\nboth span & non-span",
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("Per-image gap (N/A)")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "iou_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  → {fig_path}")

    # Fig 2: AP curve
    fig, ax = plt.subplots(figsize=(8, 5))
    thresholds = np.arange(0.5, 0.96, 0.05)
    ns_aps = [(nonspan_df['iou'] >= t).mean() for t in thresholds]
    sp_aps = [(span_df['iou'] >= t).mean() if len(span_df) > 0 else 0 for t in thresholds]

    ax.plot(thresholds, ns_aps, 'o-', color="steelblue", label="Non-span", linewidth=2)
    ax.plot(thresholds, sp_aps, 's-', color="coral", label="Span", linewidth=2)
    ax.fill_between(thresholds, sp_aps, ns_aps, alpha=0.15, color="red", label="Gap")
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Match rate")
    ax.set_title("Match rate at various IoU thresholds")
    ax.legend()
    ax.set_xlim(0.5, 0.95)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, "ap_curve.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {fig_path}")

    # Fig 3: Confidence sensitivity (있으면)
    if conf_sensitivity and len(conf_sensitivity) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        confs = [r["confidence"] for r in conf_sensitivity]
        gaps = [r["gap"] for r in conf_sensitivity]
        span_ious = [r["mean_iou_span"] for r in conf_sensitivity]
        ns_ious = [r["mean_iou_nonspan"] for r in conf_sensitivity]

        ax.plot(confs, ns_ious, 'o-', color="steelblue", label="Non-span mean IoU", linewidth=2)
        ax.plot(confs, span_ious, 's-', color="coral", label="Span mean IoU", linewidth=2)
        ax.fill_between(confs, span_ious, ns_ious, alpha=0.15, color="red", label="Gap")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("Mean IoU")
        ax.set_title("Gap stability across confidence thresholds")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig_path = os.path.join(output_dir, "confidence_sensitivity.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → {fig_path}")

    # CSV + JSON 저장
    csv_path = os.path.join(output_dir, "cell_iou_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    if len(df_per_image) > 0:
        per_img_path = os.path.join(output_dir, "per_image_stats.csv")
        df_per_image.to_csv(per_img_path, index=False)
        print(f"  → {per_img_path}")

    summary = {
        "total_cells": len(df),
        "span_cells": len(span_df),
        "nonspan_cells": len(nonspan_df),
        "mean_iou_all": float(df['iou'].mean()),
        "mean_iou_span": float(span_df['iou'].mean()) if len(span_df) > 0 else None,
        "mean_iou_nonspan": float(nonspan_df['iou'].mean()),
        "iou_gap_label_aware": float(iou_gap_la),
        "iou_gap_hungarian": float(iou_gap_h),
        "match_rate_all": float(df['matched'].mean()),
        "match_rate_span": float(span_df['matched'].mean()) if len(span_df) > 0 else None,
        "match_rate_nonspan": float(nonspan_df['matched'].mean()),
        "statistical_tests": stats,
        "ap_by_threshold": ap_data,
        "confidence_sensitivity": conf_sensitivity,
        "per_image_gap_mean": float(df_per_image['gap'].mean()) if len(df_per_image) > 0 else None,
        "per_image_gap_positive_pct": float((df_per_image['gap'] > 0).mean() * 100) if len(df_per_image) > 0 else None,
        "model": "microsoft/table-transformer-structure-recognition-v1.1-all",
        "dataset": "PubTables-1M (test)",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → {summary_path}")

    return summary


# ─────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────
def main():
    global NUM_SAMPLES, CONFIDENCE_THRESHOLD

    parser = argparse.ArgumentParser(
        description="TATR Baseline v2: Span vs Non-span IoU Gap Analysis"
    )
    parser.add_argument("--pubtables_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results_v2")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES,
                        help="0=전체")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--skip_sensitivity", action="store_true",
                        help="Confidence sensitivity 분석 스킵")
    args = parser.parse_args()

    NUM_SAMPLES = args.num_samples if args.num_samples > 0 else None
    CONFIDENCE_THRESHOLD = args.confidence

    print("=" * 65)
    print("  TATR Baseline v2")
    print("  Span vs Non-Span Cell IoU Gap Analysis")
    print("=" * 65)
    print(f"  데이터:  {args.pubtables_root}")
    print(f"  출력:    {args.output_dir}")
    print(f"  샘플:    {NUM_SAMPLES or '전체'}")
    print(f"  Device:  {DEVICE}")
    print(f"  개선사항: Hungarian matching, label-aware, 통계 검정")
    print()

    # Step 1: 모델 로드
    processor, model = load_model()

    # Step 2: 데이터 확인
    print(f"\n[2/6] 데이터 확인 중...")

    # Step 3: 메인 평가
    print(f"\n[3/6] Inference + 매칭...")
    df_h, df_la, df_per = evaluate(
        args.pubtables_root, processor, model, NUM_SAMPLES, CONFIDENCE_THRESHOLD
    )

    if len(df_la) == 0:
        print("ERROR: 결과 없음. 데이터 경로 확인.")
        return

    # Step 4: Confidence sensitivity (선택)
    conf_sens = None
    if not args.skip_sensitivity:
        conf_sens = confidence_sensitivity(
            args.pubtables_root, processor, model,
            num_samples=min(100, NUM_SAMPLES or 100)
        )

    # Step 5: 분석 + 시각화
    print(f"\n[6/6] 분석 및 시각화...")
    summary = analyze_and_visualize(df_h, df_la, df_per, args.output_dir, conf_sens)

    # 결론
    print(f"\n{'='*65}")
    gap = summary.get("iou_gap_label_aware", 0)
    stats = summary.get("statistical_tests", {})
    p_val = stats.get("welch_t", {}).get("p_value", 1.0)

    if gap > 0.1 and p_val < 0.001:
        print(f"  ★ Gap = {gap:.4f}, p < 0.001")
        print(f"  → 문제정의 근거 충분. 통계적으로 유의미한 격차.")
    elif gap > 0.05:
        print(f"  ★ Gap = {gap:.4f}, p = {p_val:.2e}")
        print(f"  → 격차 존재. AP@0.75+ 에서 더 클 가능성 있음.")
    else:
        print(f"  ★ Gap = {gap:.4f}")
        print(f"  → Gap이 작음. 문제정의 재검토 필요.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

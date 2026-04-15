#!/usr/bin/env python3
"""
Row/Col Detection Bottleneck Diagnostic (v2 — fixed label routing)
===================================================================
Phase 2 Oracle (FinTabNet n=68) 결과를 SciTSR n=716으로 외삽 검증.

핵심 질문:
  TATR이 예측하는 row/col 개수가 GT와 얼마나 일치하는가?

수정 사항 (v1 버그):
  phase3_sagr_eval.run_tatr_raw의 버그:
    if "column" in lname: cols.append(b)
  → "table column header" (실제로는 header ROW) 도 cols로 잘못 들어감.

  v2:  "spanning" / "projected" → spans
       "table column header"    → rows   (header는 가로 span, row로 취급)
       "table column"           → cols
       "table row"              → rows
       "table"                  → 무시
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

sys.path.insert(0, str(Path(__file__).parent))
from _eval_utils import (
    load_comp_ids, load_gt, load_image,
    load_fintabnet_gt, load_fintabnet_image, FINTAB_ANN,
)
from phase1_problem_proof import grits_top
from grid_reconstruct import make_baseline

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID  = "microsoft/table-transformer-structure-recognition-v1.1-all"
CONF      = 0.5

RESULTS_DIR = Path(__file__).parent / "results_diag"
RESULTS_DIR.mkdir(exist_ok=True)


@torch.no_grad()
def run_tatr_raw_fixed(proc, mdl, image: Image.Image) -> dict:
    """v2 라벨 분기 — column header를 rows로."""
    enc = proc(images=image, return_tensors="pt")
    out = mdl(**{k: v.to(DEVICE) for k, v in enc.items()
                 if k in {"pixel_values", "pixel_mask"}})
    res = proc.post_process_object_detection(
        out, threshold=CONF,
        target_sizes=[(image.size[1], image.size[0])])[0]
    boxes  = res["boxes"].cpu().numpy()
    labels = res["labels"].cpu().numpy()
    id2l   = mdl.config.id2label

    rows, cols, spans = [], [], []
    for box, lid in zip(boxes, labels):
        lname = id2l[lid].strip().lower()
        b = box.tolist()
        if "spanning" in lname or "projected" in lname:
            spans.append(b)
        elif "column header" in lname:       # header ROW
            rows.append(b)
        elif lname == "table column":
            cols.append(b)
        elif lname == "table row":
            rows.append(b)
        # "table" → ignore
    rows.sort(key=lambda b: (b[1] + b[3]) / 2)
    cols.sort(key=lambda b: (b[0] + b[2]) / 2)
    return {"rows_bbox": rows, "cols_bbox": cols, "spans_bbox": spans}


# ─────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────
def load_scitsr_complex():
    items = []
    for tid in load_comp_ids():
        try:
            items.append({"gt": load_gt(tid), "image": load_image(tid)})
        except Exception:
            continue
    return items


def load_fintabnet_complex():
    items = []
    for xml_f in sorted(FINTAB_ANN.glob("*.xml")):
        img = load_fintabnet_image(xml_f)
        if img is None: continue
        gt = load_fintabnet_gt(xml_f)
        if gt is None or not gt["is_complex"]: continue
        items.append({"gt": gt, "image": img})
    return items


# ─────────────────────────────────────────────
# Analyze
# ─────────────────────────────────────────────
def analyse(proc, mdl, items: list, ds_name: str) -> dict:
    recon = make_baseline()
    records = []
    for item in tqdm(items, desc=ds_name):
        try:
            raw = run_tatr_raw_fixed(proc, mdl, item["image"])
        except Exception as e:
            print(f"[err {ds_name}] {e}")
            continue

        gt = item["gt"]
        n_gt_r, n_gt_c = gt["n_rows"], gt["n_cols"]
        n_pr_r, n_pr_c = len(raw["rows_bbox"]), len(raw["cols_bbox"])
        if n_pr_r == 0 or n_pr_c == 0 or n_gt_r == 0 or n_gt_c == 0:
            continue

        pred = recon.reconstruct(raw["rows_bbox"], raw["cols_bbox"], raw["spans_bbox"])
        try:
            _, _, f1 = grits_top(pred["cells"], gt["cells"],
                                 pred["n_rows"], pred["n_cols"], n_gt_r, n_gt_c)
        except Exception:
            continue
        records.append({
            "n_gt_r": n_gt_r, "n_gt_c": n_gt_c,
            "n_pr_r": n_pr_r, "n_pr_c": n_pr_c,
            "dr": n_pr_r - n_gt_r,
            "dc": n_pr_c - n_gt_c,
            "f1": float(f1),
        })

    drs = [r["dr"] for r in records]
    dcs = [r["dc"] for r in records]

    rows_exact = sum(1 for d in drs if d == 0)
    cols_exact = sum(1 for d in dcs if d == 0)
    both_exact = sum(1 for r in records if r["dr"] == 0 and r["dc"] == 0)

    f1_exact = [r["f1"] for r in records if r["dr"] == 0 and r["dc"] == 0]
    f1_wrong = [r["f1"] for r in records if r["dr"] != 0 or r["dc"] != 0]

    return {
        "dataset":       ds_name,
        "n":             len(records),
        "rows_exact":    rows_exact,
        "cols_exact":    cols_exact,
        "both_exact":    both_exact,
        "rows_exact_pct": rows_exact / len(records) * 100 if records else 0,
        "cols_exact_pct": cols_exact / len(records) * 100 if records else 0,
        "both_exact_pct": both_exact / len(records) * 100 if records else 0,
        "dr_hist":       dict(Counter(drs)),
        "dc_hist":       dict(Counter(dcs)),
        "dr_mean":       float(np.mean(drs)) if drs else None,
        "dc_mean":       float(np.mean(dcs)) if dcs else None,
        "f1_mean_when_exact": float(np.mean(f1_exact)) if f1_exact else None,
        "f1_mean_when_wrong": float(np.mean(f1_wrong)) if f1_wrong else None,
        "records":       records,
    }


def plot_hist(stats_list):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, st in enumerate(stats_list):
        drs = [r["dr"] for r in st["records"]]
        dcs = [r["dc"] for r in st["records"]]
        clip = 10
        drs_c = [max(-clip, min(clip, d)) for d in drs]
        dcs_c = [max(-clip, min(clip, d)) for d in dcs]
        bins = np.arange(-clip-0.5, clip+1.5)

        ax = axes[i, 0]
        ax.hist(drs_c, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="red", lw=1.2, linestyle="--")
        ax.set_title(f"{st['dataset']} — Δrows  "
                     f"(exact={st['rows_exact_pct']:.1f}%, mean={st['dr_mean']:+.2f})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("n_pred - n_gt (rows, clipped ±10)")
        ax.set_ylabel("# tables")
        ax.grid(axis="y", alpha=0.3)

        ax = axes[i, 1]
        ax.hist(dcs_c, bins=bins, color="#55A868", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="red", lw=1.2, linestyle="--")
        ax.set_title(f"{st['dataset']} — Δcols  "
                     f"(exact={st['cols_exact_pct']:.1f}%, mean={st['dc_mean']:+.2f})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("n_pred - n_gt (cols, clipped ±10)")
        ax.set_ylabel("# tables")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Row/Col Count Mismatch — TATR prediction vs GT  (fixed labeling)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = RESULTS_DIR / "rowcol_mismatch_hist.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


def plot_scatter(stats_list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, st in zip(axes, stats_list):
        recs = st["records"]
        err  = np.array([abs(r["dr"]) + abs(r["dc"]) for r in recs])
        f1s  = np.array([r["f1"] for r in recs])
        ax.scatter(err, f1s, s=14, alpha=0.5, c="#4C72B0")
        for e in sorted(set(err)):
            mask = err == e
            ax.plot(e, f1s[mask].mean(), "rD", markersize=7)
        corr = float(np.corrcoef(err, f1s)[0, 1]) if len(err) > 1 and err.std() > 0 else float("nan")
        ax.set_xlabel("|Δrows| + |Δcols|")
        ax.set_ylabel("GriTS-Top F1")
        ax.set_title(f"{st['dataset']}  —  corr={corr:+.3f}",
                     fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    plt.suptitle("Count Mismatch vs GriTS-Top F1 (baseline reconstruct)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = RESULTS_DIR / "rowcol_vs_f1.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")


def main():
    print("=" * 60)
    print("Row/Col bottleneck diagnostic (v2 fixed labeling)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("\n[Data] SciTSR complex ...")
    sc_items = load_scitsr_complex()
    print(f"  loaded: {len(sc_items)}")

    print("[Data] FinTabNet complex ...")
    ft_items = load_fintabnet_complex()
    print(f"  loaded: {len(ft_items)}")

    print(f"\n[Model] Loading {MODEL_ID}")
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    if hasattr(proc, "size"):
        if "longest_edge" in proc.size and ("shortest_edge" not in proc.size
                                            or proc.size["shortest_edge"] is None):
            le = proc.size["longest_edge"]
            proc.size["shortest_edge"], proc.size["longest_edge"] = le, le
    try:
        mdl = TableTransformerForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
    except Exception:
        import os, tempfile
        from huggingface_hub import hf_hub_download
        cf = hf_hub_download(repo_id=MODEL_ID, filename="config.json")
        cd = json.load(open(cf))
        if cd.get("backbone_config") and cd["backbone_config"].get("dilation") is None:
            cd["backbone_config"]["dilation"] = False
        if cd.get("dilation") is None:
            cd["dilation"] = False
        td = tempfile.mkdtemp()
        with open(os.path.join(td, "config.json"), "w") as f:
            json.dump(cd, f)
        mdl = TableTransformerForObjectDetection.from_pretrained(MODEL_ID, config=td).to(DEVICE).eval()

    sc = analyse(proc, mdl, sc_items, "SciTSR")
    ft = analyse(proc, mdl, ft_items, "FinTabNet")

    def dump(st):
        print(f"\n  [{st['dataset']}] n={st['n']}")
        print(f"    rows exact : {st['rows_exact']:4d}/{st['n']}  ({st['rows_exact_pct']:.1f}%)   meanΔ={st['dr_mean']:+.2f}")
        print(f"    cols exact : {st['cols_exact']:4d}/{st['n']}  ({st['cols_exact_pct']:.1f}%)   meanΔ={st['dc_mean']:+.2f}")
        print(f"    BOTH exact : {st['both_exact']:4d}/{st['n']}  ({st['both_exact_pct']:.1f}%)")
        print(f"    F1 exact   : {st['f1_mean_when_exact']}")
        print(f"    F1 wrong   : {st['f1_mean_when_wrong']}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    dump(sc); dump(ft)

    out = RESULTS_DIR / "rowcol_mismatch.json"
    with open(out, "w") as f:
        json.dump({"SciTSR":    {k: v for k, v in sc.items() if k != "records"},
                   "FinTabNet": {k: v for k, v in ft.items() if k != "records"}},
                  f, indent=2)
    print(f"\n[saved] {out}")

    plot_hist([sc, ft])
    plot_scatter([sc, ft])


if __name__ == "__main__":
    main()

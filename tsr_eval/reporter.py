"""
Report Generator
================
Generates `runs/<run_id>/report.md` summarising:
- Per-model summary metrics table
- Failure taxonomy with worst-case image links
- Failure type taxonomy (col drift, row drift, merged-cell split, OCR outlier, crop mismatch)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Taxonomy ──────────────────────────────────────────────────────────────────

FAILURE_TAXONOMY = {
    "col_drift": "Column boundaries shift significantly across rows (high col_x_align std).",
    "row_drift": "Row boundaries shift significantly across columns (high row_y_align std).",
    "merged_cell_split": "Merged/spanning cells incorrectly split into multiple cells (row_span=1 everywhere despite visual spans).",
    "ocr_outlier": "OCR bounding box far outside the visible table area (bbox x/y > image dimensions).",
    "crop_mismatch": "Model output bbox coordinates do not align with image content — usually a transform bug (scale/pad mismatch).",
    "monotonicity_violation": "Column or row bbox ordering is non-monotonic (column N bbox starts to the left of column N-1).",
    "col_overlap": "Adjacent column bounding boxes overlap — indicates duplicate or erroneous column detection.",
    "col_gap": "Large gap between adjacent columns — missing column or bbox not covering whitespace.",
}


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(
    run_dir: Path,
    all_metrics: List[Dict[str, Any]],
    worst_cases: List[Dict[str, Any]],
    models_used: List[str],
    images_dir: Path,
    gt_used: bool,
) -> Path:
    """
    Generate report.md in run_dir.

    Parameters
    ----------
    run_dir       : Output run directory.
    all_metrics   : Flat list of all per-(image,model) metric dicts.
    worst_cases   : Ranked worst-case records from visualizer.select_worst_cases().
    models_used   : List of model names that ran.
    images_dir    : Path to input images for context.
    gt_used       : Whether GT metrics were available.

    Returns
    -------
    Path to report.md.
    """
    run_dir = Path(run_dir)
    report_path = run_dir / "report.md"

    lines: List[str] = []

    # Header
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines += [
        f"# TSR Evaluation Report",
        f"",
        f"**Run directory:** `{run_dir}`  ",
        f"**Generated:** {ts}  ",
        f"**Models:** {', '.join(models_used)}  ",
        f"**GT metrics:** {'Yes' if gt_used else 'No — self-consistency only'}",
        f"",
    ]

    # Per-model aggregated summary table
    lines += _model_summary_table(all_metrics, models_used, gt_used)
    lines += [""]

    # Per-model worst-case breakdown
    lines += _worst_case_section(worst_cases, run_dir)
    lines += [""]

    # Failure taxonomy
    lines += _taxonomy_section(worst_cases)
    lines += [""]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written to {report_path}")
    return report_path


# ── Sections ──────────────────────────────────────────────────────────────────

def _model_summary_table(
    all_metrics: List[Dict[str, Any]],
    models: List[str],
    gt_used: bool,
) -> List[str]:
    """Build markdown summary table grouped by model."""
    lines = ["## Per-Model Summary", ""]

    gt_cols = "| TEDS | Cell-F1 |" if gt_used else ""
    lines.append(
        f"| Model | N images | Col-Align↑ | Row-Align↑ | Col-Mono-Viol↓ | Row-Mono-Viol↓ "
        f"| Col-OvlpRatio↓ | Summary Score↑ {gt_cols}|"
    )
    sep = f"|-------|----------|-----------|-----------|----------------|----------------"
    sep += f"|----------------|---------------|"
    if gt_used:
        sep += "------|---------|"
    lines.append(sep)

    for model in models:
        recs = [m for m in all_metrics if m.get("model") == model]
        if not recs:
            continue
        n = len(recs)
        avg = lambda key: np.mean([r.get(key, 0.0) for r in recs])
        col_a = f"{avg('col_x_align_score'):.3f}"
        row_a = f"{avg('row_y_align_score'):.3f}"
        col_mv = f"{avg('col_monotonicity_violations'):.1f}"
        row_mv = f"{avg('row_monotonicity_violations'):.1f}"
        col_ov = f"{avg('col_overlap_ratio'):.3f}"
        summ = f"{avg('summary_score'):.3f}"

        row = f"| **{model}** | {n} | {col_a} | {row_a} | {col_mv} | {row_mv} | {col_ov} | {summ} |"
        if gt_used:
            teds = [r.get("teds") for r in recs if r.get("teds") is not None]
            f1s = [r.get("cell_f1") for r in recs if r.get("cell_f1") is not None]
            t_str = f"{np.mean(teds):.3f}" if teds else "N/A"
            f1_str = f"{np.mean(f1s):.3f}" if f1s else "N/A"
            row = f"| **{model}** | {n} | {col_a} | {row_a} | {col_mv} | {row_mv} | {col_ov} | {summ} | {t_str} | {f1_str} |"
        lines.append(row)

    return lines


def _worst_case_section(
    worst_cases: List[Dict[str, Any]],
    run_dir: Path,
) -> List[str]:
    """Build worst-case image gallery section."""
    lines = ["## Worst-Case Images", ""]
    if not worst_cases:
        lines.append("_No worst cases identified (possibly too few images)._")
        return lines

    lines.append(
        "Images ranked by lowest summary score + highest monotonicity violations.\n"
    )
    lines.append("| # | Model | Image | Score | Col-Viol | Row-Viol | Col-OvlpR | Overlay |")
    lines.append("|---|-------|-------|-------|----------|----------|-----------|---------|")

    for rank, rec in enumerate(worst_cases, 1):
        model = rec.get("model", "?")
        image_id = rec.get("image_id", "?")
        score = f"{rec.get('summary_score', 0):.3f}"
        cv = rec.get("col_monotonicity_violations", 0)
        rv = rec.get("row_monotonicity_violations", 0)
        ov = f"{rec.get('col_overlap_ratio', 0):.3f}"
        overlay = rec.get("worst_case_overlay_path", rec.get("overlay_path", ""))
        if overlay:
            # Make relative to run_dir
            try:
                rel = Path(overlay).relative_to(run_dir)
                overlay_link = f"[overlay]({rel})"
            except ValueError:
                overlay_link = f"[overlay]({overlay})"
        else:
            overlay_link = "N/A"
        lines.append(f"| {rank} | {model} | `{image_id}` | {score} | {cv} | {rv} | {ov} | {overlay_link} |")

    return lines


def _taxonomy_section(worst_cases: List[Dict[str, Any]]) -> List[str]:
    """Failure type taxonomy with observed instances."""
    lines = ["## Failure Type Taxonomy", ""]

    # Detect failure types from worst cases
    observed: Dict[str, List[str]] = {k: [] for k in FAILURE_TAXONOMY}

    for rec in worst_cases:
        image_id = rec.get("image_id", "?")
        model = rec.get("model", "?")
        tag = f"`{image_id}` ({model})"

        if rec.get("col_monotonicity_violations", 0) > 0:
            observed["monotonicity_violation"].append(tag)
        if rec.get("col_x_align_score", 1.0) < 0.5:
            observed["col_drift"].append(tag)
        if rec.get("row_y_align_score", 1.0) < 0.5:
            observed["row_drift"].append(tag)
        if rec.get("col_overlap_ratio", 0.0) > 0.3:
            observed["col_overlap"].append(tag)
        if rec.get("col_gap_ratio", 0.0) > 0.3:
            observed["col_gap"].append(tag)

    for ftype, desc in FAILURE_TAXONOMY.items():
        instances = observed.get(ftype, [])
        obs_str = ", ".join(instances[:5]) if instances else "_none observed_"
        lines += [
            f"### `{ftype}`",
            f"{desc}",
            f"",
            f"**Observed in:** {obs_str}",
            f"",
        ]

    return lines


# ── CSV metrics export ─────────────────────────────────────────────────────────

def save_metrics_csv(
    all_metrics: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Export all metrics as a CSV file."""
    import csv

    if not all_metrics:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "image_id", "model",
        "col_x_align_score", "row_y_align_score",
        "col_monotonicity_violations", "row_monotonicity_violations",
        "col_overlap_ratio", "col_gap_ratio",
        "row_overlap_ratio", "row_gap_ratio",
        "summary_score",
        "teds", "cell_f1", "cell_precision", "cell_recall", "em",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)

    logger.info(f"Metrics CSV saved: {out_path}")

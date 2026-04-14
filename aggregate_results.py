"""
TASK 4 — Results Aggregation Script
=====================================
Reads all results_by_complexity.csv from E0~E3 and computes
mean ± std across 3 seeds. Prints two summary tables in Markdown.

Usage:
    python aggregate_results.py --results_dir outputs/ablation
    python aggregate_results.py --results_dir outputs/ablation \
        --output_csv outputs/ablation/summary_results.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_csv(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def collect_runs(results_dir: Path) -> List[dict]:
    """Collect all per-seed results_by_complexity.csv rows."""
    rows = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            csv_path = seed_dir / "results_by_complexity.csv"
            if not csv_path.exists():
                continue
            for row in load_csv(csv_path):
                row["experiment_id"] = exp_dir.name
                row["seed"]          = seed_dir.name
                rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────

def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n    = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var  = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def _fmt(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "—"
    return f"{mean:.4f} ±{std:.4f}"


def aggregate(rows: List[dict]) -> Dict[Tuple[str, str, str], List[float]]:
    """
    Returns: {(exp_id, split, metric): [values across seeds]}
    """
    buckets: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        exp  = row["experiment_id"]
        split = row.get("split", row.get("k_bin", "all"))
        for metric in ["GriTS_Top", "GriTS_Loc", "GriTS_Con"]:
            try:
                val = float(row[metric])
            except (KeyError, ValueError):
                continue
            buckets[(exp, split, metric)].append(val)
    return buckets


# ─────────────────────────────────────────────────────────────
# Table A — Main results (simple vs complex)
# ─────────────────────────────────────────────────────────────

def table_a(buckets: Dict, experiments: List[str]) -> List[dict]:
    """Build Table A rows: per-experiment simple/complex GriTS_Top + GriTS_Loc."""
    rows = []
    for exp in experiments:
        row = {"Experiment": exp}
        for split in ["simple", "complex", "all"]:
            for metric in ["GriTS_Top", "GriTS_Loc"]:
                vals = buckets.get((exp, split, metric), [])
                m, s = _mean_std(vals)
                col = f"{split.capitalize()} {metric}"
                row[col] = _fmt(m, s)
        rows.append(row)
    return rows


def print_table_a(rows: List[dict]):
    cols = list(rows[0].keys()) if rows else []
    widths = {c: max(len(c), max((len(r.get(c, "—")) for r in rows), default=4))
              for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep    = "-|-".join("-" * widths[c] for c in cols)
    print("\n**Table A — Main Results (mean ± std, 3 seeds)**\n")
    print(f"| {header} |")
    print(f"| {sep} |")
    for row in rows:
        line = " | ".join(row.get(c, "—").ljust(widths[c]) for c in cols)
        print(f"| {line} |")


# ─────────────────────────────────────────────────────────────
# Table B — Span complexity breakdown (E0 vs E3)
# ─────────────────────────────────────────────────────────────

K_BINS = ["k=1", "k=2", "k=3~4", "k>=5", "all"]


def table_b(buckets: Dict, e0: str, e3: str) -> List[dict]:
    rows = []
    for kbin in K_BINS:
        e0_vals = buckets.get((e0, kbin, "GriTS_Loc"), [])
        e3_vals = buckets.get((e3, kbin, "GriTS_Loc"), [])
        e0_m, e0_s = _mean_std(e0_vals)
        e3_m, e3_s = _mean_std(e3_vals)
        if math.isnan(e0_m) and math.isnan(e3_m):
            continue
        delta = (e3_m - e0_m) if not (math.isnan(e3_m) or math.isnan(e0_m)) else float("nan")
        rows.append({
            "k_bin":       kbin,
            f"{e0} GriTS_Loc": _fmt(e0_m, e0_s),
            f"{e3} GriTS_Loc": _fmt(e3_m, e3_s),
            "Delta":       f"{delta:+.4f}" if not math.isnan(delta) else "—",
        })
    return rows


def print_table_b(rows: List[dict], e0: str, e3: str):
    if not rows:
        print("\n**Table B — empty (no k_bin data found)**\n")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max((len(r.get(c, "—")) for r in rows), default=4))
              for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep    = "-|-".join("-" * widths[c] for c in cols)
    print(f"\n**Table B — Span Complexity Breakdown ({e0} vs {e3}, GriTS_Loc)**\n")
    print(f"| {header} |")
    print(f"| {sep} |")
    for row in rows:
        line = " | ".join(row.get(c, "—").ljust(widths[c]) for c in cols)
        print(f"| {line} |")


# ─────────────────────────────────────────────────────────────
# Save summary CSV
# ─────────────────────────────────────────────────────────────

def save_summary_csv(rows_a: List[dict], rows_b: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Combine into one file with a section marker
    all_rows = [{"_table": "A", **r} for r in rows_a] + \
               [{"_table": "B", **r} for r in rows_b]
    if not all_rows:
        return
    fields = sorted({k for r in all_rows for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"\n[saved] summary CSV → {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Aggregate ablation results")
    parser.add_argument("--results_dir", required=True,
                        help="Root ablation output dir (E0_baseline/, E1_span/, ...)")
    parser.add_argument("--output_csv", default="summary_results.csv")
    parser.add_argument("--e0", default="E0_baseline", help="E0 experiment dir name")
    parser.add_argument("--e3", default="E3_snap_soft", help="E3 experiment dir name")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Collect all rows from per-seed CSVs
    rows = collect_runs(results_dir)
    if not rows:
        # Fallback: look for a single results_by_complexity.csv in results_dir root
        root_csv = results_dir / "results_by_complexity.csv"
        if root_csv.exists():
            rows = load_csv(root_csv)
            print(f"[info] Using root CSV: {root_csv} ({len(rows)} rows)")
        else:
            print(f"[warn] No results found in {results_dir}")
            print("       Run eval_by_complexity.py first, or check --results_dir.")
            # Still proceed to print empty tables so script doesn't crash
    else:
        print(f"[info] Loaded {len(rows)} rows from {results_dir}")

    buckets      = aggregate(rows)
    experiments  = sorted({r["experiment_id"] for r in rows}) if rows else \
                   ["E0_baseline", "E1_span", "E2_snap_hard", "E3_snap_soft"]

    rows_a = table_a(buckets, experiments)
    rows_b = table_b(buckets, args.e0, args.e3)

    print_table_a(rows_a)
    print_table_b(rows_b, args.e0, args.e3)

    save_summary_csv(rows_a, rows_b, Path(args.output_csv))


if __name__ == "__main__":
    main()

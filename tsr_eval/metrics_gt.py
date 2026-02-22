"""
GT-Based Metrics
================
Computes TEDS, Cell-F1, and Exact Match (EM) when GT files are present.

GT file format (auto-detected)
--------------------------------
*.html  → HTML table <table>..</table>
*.json  → {"cells": [{"row":int,"col":int,"row_span":int,"col_span":int,"text":str},...]}
*.csv   → CSV where row/col position = position in file

A GT file matches an image if its stem == image_id.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_gt_metrics(
    normalized: Dict[str, Any],
    gt_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Compute GT-based metrics if a matching GT file exists.

    Parameters
    ----------
    normalized : Output of normalizer.normalize()
    gt_dir     : Directory containing GT files.

    Returns
    -------
    dict with TEDS, cell_f1, em — or None if no GT found.
    """
    image_id = normalized.get("image_id", "")
    gt_cells, gt_html = _load_gt(gt_dir, image_id)
    if gt_cells is None:
        return None

    tables = normalized.get("tables", [])
    if not tables:
        pred_cells = []
    else:
        pred_cells = tables[0].get("cells", [])

    teds = _teds(normalized, gt_html, gt_dir, image_id)
    f1, precision, recall = _cell_f1(pred_cells, gt_cells)
    em = _exact_match(pred_cells, gt_cells)

    return {
        "teds": teds,
        "cell_f1": round(f1, 4),
        "cell_precision": round(precision, 4),
        "cell_recall": round(recall, 4),
        "em": int(em),
        "gt_source": str(image_id),
    }


# ── GT loading ────────────────────────────────────────────────────────────────

def _load_gt(gt_dir: Path, image_id: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """Return (cell_list, html_str) or (None, None) if no GT found."""
    gt_dir = Path(gt_dir)
    for ext in (".html", ".json", ".csv"):
        candidate = gt_dir / f"{image_id}{ext}"
        if candidate.exists():
            return _parse_gt_file(candidate)
    logger.debug(f"No GT file found for {image_id} in {gt_dir}")
    return None, None


def _parse_gt_file(path: Path) -> Tuple[List[Dict], str]:
    """Parse GT file and return (cells, html_str)."""
    ext = path.suffix.lower()
    if ext == ".html":
        return _parse_html_gt(path.read_text(encoding="utf-8"))
    elif ext == ".json":
        return _parse_json_gt(json.loads(path.read_text(encoding="utf-8")))
    elif ext == ".csv":
        return _parse_csv_gt(path.read_text(encoding="utf-8"))
    return [], ""


def _parse_html_gt(html_str: str) -> Tuple[List[Dict], str]:
    """Parse HTML table into cell list."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_str, "html.parser")
        cells = []
        for ridx, tr in enumerate(soup.find_all("tr")):
            for cidx, td in enumerate(tr.find_all(["td", "th"])):
                cells.append({
                    "row_idx": ridx,
                    "col_idx": cidx,
                    "row_span": int(td.get("rowspan", 1)),
                    "col_span": int(td.get("colspan", 1)),
                    "text": _norm_text(td.get_text()),
                    "bbox": [0, 0, 0, 0],
                })
        return cells, html_str
    except Exception as exc:
        logger.warning(f"HTML GT parse error: {exc}")
        return [], html_str


def _parse_json_gt(data: Dict) -> Tuple[List[Dict], str]:
    """Parse JSON GT into cell list."""
    cells_raw = data.get("cells", [])
    cells = []
    for c in cells_raw:
        cells.append({
            "row_idx": int(c.get("row", c.get("row_idx", 0))),
            "col_idx": int(c.get("col", c.get("col_idx", 0))),
            "row_span": int(c.get("row_span", c.get("rowspan", 1))),
            "col_span": int(c.get("col_span", c.get("colspan", 1))),
            "text": _norm_text(str(c.get("text", ""))),
            "bbox": c.get("bbox", c.get("box", [0, 0, 0, 0])),
        })
    return cells, _cells_to_html(cells)


def _parse_csv_gt(csv_str: str) -> Tuple[List[Dict], str]:
    """Parse CSV GT (grid layout) into cell list."""
    reader = csv.reader(io.StringIO(csv_str))
    rows = list(reader)
    cells = []
    for ridx, row in enumerate(rows):
        for cidx, val in enumerate(row):
            cells.append({
                "row_idx": ridx,
                "col_idx": cidx,
                "row_span": 1,
                "col_span": 1,
                "text": _norm_text(val),
                "bbox": [0, 0, 0, 0],
            })
    return cells, _cells_to_html(cells)


# ── TEDS ─────────────────────────────────────────────────────────────────────

def _teds(
    normalized: Dict[str, Any],
    gt_html: Optional[str],
    gt_dir: Path,
    image_id: str,
) -> Optional[float]:
    """Compute Tree-Edit Distance Similarity (TEDS) on HTML.
    
    Falls back to structure-only TEDS if no text in pred.
    Returns None if lxml or zss not available.
    """
    if not gt_html:
        return None
    try:
        try:
            from teds import TEDS as TEDSCalc
            teds_calc = TEDSCalc(structure_only=False)
            pred_html = _normalized_to_html(normalized)
            return round(teds_calc.evaluate(pred_html, gt_html), 4)
        except ImportError:
            pass

        # Fallback: simplified structural TEDS via tag-sequence edit distance
        pred_html = _normalized_to_html(normalized)
        score = _simple_teds(pred_html, gt_html)
        return round(score, 4)
    except Exception as exc:
        logger.warning(f"TEDS computation failed: {exc}")
        return None


def _normalized_to_html(normalized: Dict[str, Any]) -> str:
    """Convert normalized JSON to HTML string."""
    tables = normalized.get("tables", [])
    if not tables:
        return "<table></table>"
    return _cells_to_html(tables[0].get("cells", []))


def _cells_to_html(cells: List[Dict]) -> str:
    """Build minimal HTML from cells."""
    if not cells:
        return "<table></table>"
    n_rows = max(c.get("row_idx", 0) + c.get("row_span", 1) for c in cells)
    # Build row-indexed bins
    slot: Dict[Tuple[int, int], Dict] = {}
    for c in cells:
        slot[(c.get("row_idx", 0), c.get("col_idx", 0))] = c

    lines = ["<table>"]
    for r in range(n_rows):
        row_cells = [(c, cidx) for (ridx, cidx), c in slot.items() if ridx == r]
        row_cells.sort(key=lambda x: x[1])
        if not row_cells:
            continue
        lines.append("<tr>")
        for c, _ in row_cells:
            rs = c.get("row_span", 1)
            cs = c.get("col_span", 1)
            text = c.get("text", "")
            attrs = ""
            if rs > 1: attrs += f' rowspan="{rs}"'
            if cs > 1: attrs += f' colspan="{cs}"'
            lines.append(f"<td{attrs}>{text}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _simple_teds(pred_html: str, gt_html: str) -> float:
    """Simple TEDS approximation by comparing tag sequences."""
    def tag_seq(html: str) -> List[str]:
        return re.findall(r"<[^>]+>", html)

    pred_tags = tag_seq(pred_html)
    gt_tags = tag_seq(gt_html)

    # Levenshtein distance on tag sequences
    m, n = len(pred_tags), len(gt_tags)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i]
        for j in range(1, n + 1):
            cost = 0 if pred_tags[i - 1] == gt_tags[j - 1] else 1
            new_dp.append(min(new_dp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost))
        dp = new_dp

    edit_dist = dp[n]
    similarity = 1.0 - edit_dist / max(m, n)
    return max(0.0, similarity)


# ── Cell F1 ───────────────────────────────────────────────────────────────────

def _cell_f1(
    pred_cells: List[Dict],
    gt_cells: List[Dict],
) -> Tuple[float, float, float]:
    """
    Cell-level F1: match cells by (row_idx, col_idx, row_span, col_span).
    Returns (f1, precision, recall).
    """
    def cell_key(c):
        return (
            c.get("row_idx", 0), c.get("col_idx", 0),
            c.get("row_span", 1), c.get("col_span", 1),
        )

    pred_keys = [cell_key(c) for c in pred_cells]
    gt_keys = [cell_key(c) for c in gt_cells]
    pred_set = set(pred_keys)
    gt_set = set(gt_keys)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


# ── Exact Match ───────────────────────────────────────────────────────────────

def _exact_match(pred_cells: List[Dict], gt_cells: List[Dict]) -> bool:
    """EM: True iff every (row_idx, col_idx, row_span, col_span, text) matches."""
    def cell_full_key(c):
        return (
            c.get("row_idx", 0), c.get("col_idx", 0),
            c.get("row_span", 1), c.get("col_span", 1),
            _norm_text(str(c.get("text", ""))),
        )

    pred_set = set(cell_full_key(c) for c in pred_cells)
    gt_set = set(cell_full_key(c) for c in gt_cells)
    return pred_set == gt_set


# ── Utilities ─────────────────────────────────────────────────────────────────

def _norm_text(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(text.split()).lower()

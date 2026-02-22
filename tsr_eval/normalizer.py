"""
Normalizer
==========
Converts raw engine output (cells/rows/cols dicts) into the standard
normalized JSON format, and saves it to `normalized/<image_id>_<model>.json`.

Standard format (per image per model)
--------------------------------------
{
  "image_id":    str,
  "image_path":  str,
  "image_size":  {"W": int, "H": int},
  "model":       str,
  "transform_log": { ... },
  "tables": [
    {
      "table_bbox": [x1, y1, x2, y2],
      "rows": [{"idx": int, "bbox": [...]}],
      "cols": [{"idx": int, "bbox": [...]}],
      "cells": [
        {
          "row_idx": int, "col_idx": int,
          "row_span": int, "col_span": int,
          "bbox": [x1, y1, x2, y2],
          "text": str,
          "ocr_tokens": [{"text": str, "bbox": [...], "conf": float}],
          "source_model": str
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def normalize(
    image_id: str,
    image_path: str,
    image_size: Dict[str, int],         # {"W": int, "H": int}
    model_name: str,
    engine_output: Dict[str, Any],     # raw engine output dict
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convert raw engine output to standard normalized JSON.

    Parameters
    ----------
    image_id    : Unique image identifier (stem of filename).
    image_path  : Absolute or relative path to the image file.
    image_size  : Original image dimensions.
    model_name  : Engine name string.
    engine_output : Raw dict from engine.predict().
    out_dir     : If given, writes normalized JSON here.

    Returns
    -------
    Normalized dict.
    """
    cells = engine_output.get("cells", [])
    rows = engine_output.get("rows", [])
    cols = engine_output.get("cols", [])
    transform_log = engine_output.get("transform_log", {})

    # Standardize cells
    norm_cells = []
    for c in cells:
        bbox = _to_xyxy(c.get("bbox", c.get("box", [0, 0, 0, 0])))
        norm_cells.append({
            "row_idx": int(c.get("row_idx", c.get("row", 0))),
            "col_idx": int(c.get("col_idx", c.get("col", 0))),
            "row_span": int(c.get("row_span", 1)),
            "col_span": int(c.get("col_span", 1)),
            "bbox": bbox,
            "text": str(c.get("text", "")),
            "ocr_tokens": _normalize_tokens(c.get("ocr_tokens", [])),
            "source_model": model_name,
        })

    # Standardize rows
    norm_rows = []
    for r in rows:
        norm_rows.append({
            "idx": int(r.get("idx", 0)),
            "bbox": _to_xyxy(r.get("bbox", [0, 0, 0, 0])),
        })

    # Standardize cols
    norm_cols = []
    for col in cols:
        norm_cols.append({
            "idx": int(col.get("idx", 0)),
            "bbox": _to_xyxy(col.get("bbox", [0, 0, 0, 0])),
        })

    # Compute table bbox (union of all cell bboxes)
    if norm_cells:
        all_bboxes = [c["bbox"] for c in norm_cells]
        table_bbox = [
            min(b[0] for b in all_bboxes),
            min(b[1] for b in all_bboxes),
            max(b[2] for b in all_bboxes),
            max(b[3] for b in all_bboxes),
        ]
    else:
        table_bbox = [0, 0, image_size["W"], image_size["H"]]

    table_entry = {
        "table_bbox": table_bbox,
        "rows": norm_rows,
        "cols": norm_cols,
        "cells": norm_cells,
    }

    doc = {
        "image_id": image_id,
        "image_path": str(image_path),
        "image_size": image_size,
        "model": model_name,
        "transform_log": transform_log,
        "tables": [table_entry],
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{image_id}_{model_name}.json"
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        logger.debug(f"Saved normalized JSON: {out_path}")

    return doc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_xyxy(bbox) -> List[float]:
    """Ensure bbox is [x0, y0, x1, y1] floats."""
    if not bbox or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def _normalize_tokens(tokens: List[Any]) -> List[Dict]:
    """Normalize OCR token list."""
    out = []
    for t in tokens:
        if isinstance(t, dict):
            out.append({
                "text": str(t.get("text", "")),
                "bbox": _to_xyxy(t.get("bbox", [0, 0, 0, 0])),
                "conf": float(t.get("conf", t.get("confidence", 1.0))),
            })
    return out

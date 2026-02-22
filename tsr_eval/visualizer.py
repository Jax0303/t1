"""
Visualizer
==========
Draws rows (blue bands), cols (green bands), and cells (red outlines) on images.
Also renders (row,col) index labels in each cell.

Outputs one PNG per (image, model) pair to `overlays/<model>/<image_id>.png`.
Worst cases ranked by summary_score are copied to `overlays/worst_cases/`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Colours (BGR for OpenCV → RGB for PIL)
_ROW_FILL = (173, 216, 230, 60)    # light-blue fill  (RGBA)
_ROW_EDGE = (0, 0, 200, 200)       # blue edge
_COL_FILL = (144, 238, 144, 60)    # light-green fill
_COL_EDGE = (0, 180, 0, 200)       # green edge
_CELL_EDGE = (200, 0, 0, 220)      # red edge
_TOKEN_EDGE = (255, 165, 0, 200)   # orange edge for OCR tokens
_TEXT_COLOR = (30, 30, 30)         # dark text for labels


def draw_overlay(
    image: np.ndarray,
    normalized: Dict[str, Any],
    out_path: Optional[Path] = None,
) -> "PIL.Image.Image":                 # type: ignore
    """
    Draw rows, cols, and cells on the image.

    Parameters
    ----------
    image      : H×W×3 uint8 RGB array.
    normalized : Output of normalizer.normalize().
    out_path   : If given, saves PNG here.

    Returns
    -------
    PIL.Image (RGB) with overlay.
    """
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load a default truetype font for labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    tables = normalized.get("tables", [])
    if not tables:
        result = Image.alpha_composite(pil_img, overlay).convert("RGB")
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            result.save(str(out_path))
        return result

    table = tables[0]

    # --- Rows (horizontal blue bands) ---
    for rdata in table.get("rows", []):
        b = rdata.get("bbox", [])
        if len(b) >= 4:
            draw.rectangle([b[0], b[1], b[2], b[3]], fill=_ROW_FILL, outline=_ROW_EDGE, width=1)

    # --- Cols (vertical green bands) ---
    for cdata in table.get("cols", []):
        b = cdata.get("bbox", [])
        if len(b) >= 4:
            draw.rectangle([b[0], b[1], b[2], b[3]], fill=_COL_FILL, outline=_COL_EDGE, width=1)

    # --- Cells and OCR Tokens ---
    for cell in table.get("cells", []):
        # Draw Tokens (orange outlines)
        for token in cell.get("ocr_tokens", []):
            tb = token.get("bbox", [])
            if len(tb) >= 4:
                draw.rectangle([tb[0], tb[1], tb[2], tb[3]], outline=_TOKEN_EDGE, width=1)

        b = cell.get("bbox", [])
        if len(b) < 4:
            continue
        
        # Draw Cell (red outline)
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=_CELL_EDGE, width=2)
        
        # Cell Label
        label = f"r{cell.get('row_idx',0)}c{cell.get('col_idx',0)}"
        tx, ty = b[0] + 2, b[1] + 2
        try:
            bbox_txt = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle(bbox_txt, fill=(255, 255, 255, 180))
        except Exception:
            pass
        draw.text((tx, ty), label, fill=_TEXT_COLOR, font=font)

    result = Image.alpha_composite(pil_img, overlay).convert("RGB")
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path))
        logger.debug(f"Saved overlay: {out_path}")

    return result


def select_worst_cases(
    metrics_records: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Select top-N worst-case images from aggregated metrics.

    Ranking: ascending summary_score, descending monotonicity violations.

    Parameters
    ----------
    metrics_records : List of dicts, each containing:
        image_id, model, summary_score, col_monotonicity_violations,
        row_monotonicity_violations, overlay_path
    top_n : Number of worst cases to return.

    Returns
    -------
    List of the worst records, sorted worst-first.
    """
    def rank_key(r):
        score = r.get("summary_score", 1.0)
        mono = r.get("col_monotonicity_violations", 0) + r.get("row_monotonicity_violations", 0)
        return (score, -mono)  # ascending score = worse first

    ranked = sorted(metrics_records, key=rank_key)
    return ranked[:top_n]


def copy_worst_cases(
    worst_records: List[Dict[str, Any]],
    overlays_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Copy overlay images for worst cases to `overlays/worst_cases/`.

    Returns the same records with `worst_case_overlay_path` added.
    """
    wc_dir = Path(overlays_dir) / "worst_cases"
    wc_dir.mkdir(parents=True, exist_ok=True)

    updated = []
    for rec in worst_records:
        src = rec.get("overlay_path")
        if src and Path(src).exists():
            dst = wc_dir / f"{rec.get('model','unk')}_{Path(src).name}"
            shutil.copy2(src, dst)
            rec = dict(rec, worst_case_overlay_path=str(dst))
        updated.append(rec)
    return updated

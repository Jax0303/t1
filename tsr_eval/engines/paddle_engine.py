"""
PaddleOCR / PaddleX TSR Engine Wrapper
========================================
Supports: SLANet, SLANet_plus, SLANet_zh  (via paddleocr ppstructure)
          SLANeXt_wired, SLANeXt_wireless  (via paddlex >= 3.0)

Outputs
-------
raw_outputs/<image_id>_<variant>_raw.html    — raw HTML token stream
raw_outputs/<image_id>_<variant>_raw.json    — bboxes + tokens (pre-parse)
raw_outputs/<image_id>_<variant>_post.json   — post-parse cells

Transform Logging
-----------------
PaddleOCR ppstructure auto-resizes images.  The engine records:
  input_size: (W, H) original
  (no explicit resize metadata exposed by paddleocr public API, so
   we note "resize details are internal to paddleocr and not exposed")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _poly_to_xyxy(polygon) -> List[float]:
    """Convert polygon [[x,y],...] or flat [x,y,x,y,...] to [x0,y0,x1,y1]."""
    if not polygon:
        return [0, 0, 0, 0]
    if isinstance(polygon[0], (list, tuple)):
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
    else:
        xs = [polygon[i] for i in range(0, len(polygon), 2)]
        ys = [polygon[i] for i in range(1, len(polygon), 2)]
    return [min(xs), min(ys), max(xs), max(ys)]


class PaddleTSREngine:
    """
    PaddleOCR / PaddleX table structure recognition engine.

    Parameters
    ----------
    variant : str
        One of: SLANet, SLANet_plus, SLANet_zh,
                SLANeXt_wired, SLANeXt_wireless
    device : str
        'cpu' or 'gpu:0'
    backend : str
        'paddleocr' or 'paddlex'
    """

    def __init__(
        self,
        variant: str = "SLANet_plus",
        device: str = "cpu",
        backend: str = "paddleocr",
    ) -> None:
        self.variant = variant
        self.device = device
        self.backend = backend
        self.name = f"paddle_{variant.lower()}"
        self._load_model()

    def _load_model(self) -> None:
        if self.backend == "paddlex":
            self._load_paddlex()
        else:
            self._load_paddleocr()

    def _load_paddleocr(self) -> None:
        try:
            from paddleocr import PPStructure
            use_gpu = self.device.startswith("gpu")

            # Map variant → table_model_dir hint (PPStructure selects internally)
            algo_map = {
                "SLANet": "SLANet",
                "SLANet_plus": "SLANet_plus",
                "SLANet_zh": "SLANet_zh",
            }
            algo = algo_map.get(self.variant, "SLANet_plus")

            logger.info(f"Loading PPStructure ({algo}) — use_gpu={use_gpu}")
            self._engine = PPStructure(
                show_log=False,
                use_gpu=use_gpu,
                table=True,
                ocr=False,          # We run OCR separately
                layout=False,
                table_model_dir=None,
            )
            self._mode = "ppstructure"
            logger.info(f"PPStructure ({algo}) loaded.")
        except Exception as exc:
            logger.error(f"PaddleOCR load failed: {exc}")
            raise

    def _load_paddlex(self) -> None:
        try:
            import paddlex  # type: ignore

            logger.info(f"Loading PaddleX {self.variant} on {self.device}")
            self._engine = paddlex.create_model(
                model_name=self.variant, device=self.device
            )
            self._mode = "paddlex"
            logger.info(f"PaddleX {self.variant} loaded.")
        except Exception as exc:
            logger.error(f"PaddleX load failed: {exc}")
            raise

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image: np.ndarray,
        raw_out_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run PaddleOCR/X TSR on a single image.

        Returns
        -------
        dict with keys: cells, rows, cols, num_rows, num_cols, transform_log
        """
        orig_h, orig_w = image.shape[:2]
        transform_log = {
            "input_size": {"W": orig_w, "H": orig_h},
            "note": "PaddleOCR internal resize details are not publicly exposed.",
            "inverse_formula": "coords are already in original image pixel space",
        }

        if self._mode == "ppstructure":
            result = self._predict_ppstructure(image, orig_w, orig_h)
        else:
            result = self._predict_paddlex(image, orig_w, orig_h)

        result["transform_log"] = transform_log

        if raw_out_dir is not None:
            raw_out_dir = Path(raw_out_dir)
            raw_out_dir.mkdir(parents=True, exist_ok=True)
            rp = result.get("_raw", {})
            (raw_out_dir / "raw.html").write_text(rp.get("html", ""), encoding="utf-8")
            (raw_out_dir / "raw.json").write_text(
                json.dumps({"html_tokens": rp.get("tokens", []),
                            "bboxes": rp.get("bboxes", [])}, indent=2)
            )
            post_cells = [
                {k: v for k, v in c.items() if k != "ocr_tokens"} for c in result["cells"]
            ]
            (raw_out_dir / "post.json").write_text(
                json.dumps({"cells": post_cells}, indent=2)
            )

        result.pop("_raw", None)
        return result

    # ── PPStructure backend ───────────────────────────────────────────────────

    def _predict_ppstructure(self, image: np.ndarray, orig_w: int, orig_h: int) -> Dict[str, Any]:
        """Run PPStructure table pipeline."""
        try:
            outputs = self._engine(image)
            table_region = None
            for region in outputs:
                if region.get("type", "").lower() == "table":
                    table_region = region
                    break
            if table_region is None and outputs:
                table_region = outputs[0]

            if table_region is None:
                logger.warning("PPStructure: no table region found")
                return self._empty()

            res = table_region.get("res", {})
            html_str = res.get("html", "")
            bboxes = res.get("bbox", [])      # list of polygon lists
            tokens = res.get("structure", res.get("structure_str", []))

            raw = {"html": html_str, "tokens": tokens, "bboxes": bboxes}
            return self._parse_html_and_bboxes(html_str, tokens, bboxes, raw)

        except Exception as exc:
            logger.error(f"PPStructure predict failed: {exc}")
            return self._empty()

    # ── PaddleX backend ───────────────────────────────────────────────────────

    def _predict_paddlex(self, image: np.ndarray, orig_w: int, orig_h: int) -> Dict[str, Any]:
        """Run PaddleX model pipeline."""
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image.astype(np.uint8))

            outputs = list(self._engine.predict(input=pil_img, batch_size=1))
            if not outputs:
                return self._empty()

            res = outputs[0].json.get("res", {}) if hasattr(outputs[0], "json") else {}
            bboxes = res.get("bbox", [])
            tokens = res.get("structure", [])
            html_str = "".join(tokens)
            if html_str and not html_str.startswith("<table"):
                html_str = f"<table>{html_str}</table>"

            raw = {"html": html_str, "tokens": tokens, "bboxes": bboxes}
            return self._parse_html_and_bboxes(html_str, tokens, bboxes, raw)

        except Exception as exc:
            logger.error(f"PaddleX predict failed: {exc}")
            return self._empty()

    # ── HTML + bbox parsing ───────────────────────────────────────────────────

    def _parse_html_and_bboxes(
        self,
        html_str: str,
        tokens: List[str],
        bboxes: List,
        raw: Dict,
    ) -> Dict[str, Any]:
        """Parse HTML structure tokens and match with bboxes."""
        from bs4 import BeautifulSoup

        if not html_str:
            logger.warning("Paddle TSR: empty HTML output")
            return dict(self._empty(), _raw=raw)

        try:
            soup = BeautifulSoup(html_str, "html.parser")
            # Track actual (row, col) with span accounting
            html_cells: List[Dict] = []
            col_cursor: Dict[int, int] = {}  # row → next free col

            for ridx, row in enumerate(soup.find_all("tr")):
                col_cursor.setdefault(ridx, 0)
                for td in row.find_all(["td", "th"]):
                    rs = int(td.get("rowspan", 1))
                    cs = int(td.get("colspan", 1))
                    # skip occupied cells
                    while col_cursor.get(ridx, 0) in [
                        occupied
                        for r2, occupied in col_cursor.items()
                        if r2 == ridx
                    ]:
                        break
                    cidx = col_cursor.get(ridx, 0)
                    html_cells.append({
                        "row": ridx,
                        "col": cidx,
                        "row_span": rs,
                        "col_span": cs,
                    })
                    # advance cursor
                    for dr in range(rs):
                        r2 = ridx + dr
                        col_cursor[r2] = col_cursor.get(r2, 0) + cs

        except Exception as exc:
            logger.warning(f"HTML parse failed: {exc}")
            html_cells = []

        # Convert bboxes
        xyxy_bboxes = [_poly_to_xyxy(b) for b in bboxes]

        n_html = len(html_cells)
        n_bbox = len(xyxy_bboxes)

        if n_html != n_bbox:
            logger.warning(
                f"Paddle TSR ({self.variant}): cell-bbox mismatch — "
                f"{n_html} HTML cells vs {n_bbox} bboxes"
            )

        cells = []
        for i, cell_info in enumerate(html_cells):
            bbox = xyxy_bboxes[i] if i < n_bbox else [0, 0, 0, 0]
            cells.append({
                "row_idx": cell_info["row"],
                "col_idx": cell_info["col"],
                "row_span": cell_info["row_span"],
                "col_span": cell_info["col_span"],
                "bbox": bbox,
                "text": "",
                "ocr_tokens": [],
                "confidence": 0.9,
            })

        # Derive rows / cols from cells
        rows_out, cols_out = self._derive_rows_cols(cells)

        return {
            "_raw": raw,
            "cells": cells,
            "rows": rows_out,
            "cols": cols_out,
            "num_rows": max((c["row_idx"] + c["row_span"] for c in cells), default=0),
            "num_cols": max((c["col_idx"] + c["col_span"] for c in cells), default=0),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _derive_rows_cols(cells: List[Dict]) -> Tuple[List, List]:
        """Derive row/col summary from cells for self-consistency metrics."""
        if not cells:
            return [], []

        row_map: Dict[int, List] = {}
        col_map: Dict[int, List] = {}
        for c in cells:
            row_map.setdefault(c["row_idx"], []).append(c["bbox"])
            col_map.setdefault(c["col_idx"], []).append(c["bbox"])

        def merge_bboxes(bboxes):
            xs0 = [b[0] for b in bboxes]
            ys0 = [b[1] for b in bboxes]
            xs1 = [b[2] for b in bboxes]
            ys1 = [b[3] for b in bboxes]
            return [min(xs0), min(ys0), max(xs1), max(ys1)]

        rows = [{"idx": k, "bbox": merge_bboxes(v)} for k, v in sorted(row_map.items())]
        cols = [{"idx": k, "bbox": merge_bboxes(v)} for k, v in sorted(col_map.items())]
        return rows, cols

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {"cells": [], "rows": [], "cols": [], "num_rows": 0, "num_cols": 0}

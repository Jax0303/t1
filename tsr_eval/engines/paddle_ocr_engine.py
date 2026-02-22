"""
PaddleOCR OCR Engine Wrapper
==============================
Runs PaddleOCR in pure OCR mode (detection + recognition) and clusters
the detected text boxes into a table grid — analogous to the Tesseract engine.

Outputs
-------
raw_outputs/<image_id>_paddle_ocr.json   — raw detection+recognition output
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _poly_to_xyxy(polygon) -> List[float]:
    """Convert polygon [[x,y],...] to [x0,y0,x1,y1]."""
    if not polygon:
        return [0, 0, 0, 0]
    if isinstance(polygon[0], (list, tuple)):
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
    else:
        xs = [polygon[i] for i in range(0, len(polygon), 2)]
        ys = [polygon[i] for i in range(1, len(polygon), 2)]
    return [min(xs), min(ys), max(xs), max(ys)]


class PaddleOCREngine:
    """
    PaddleOCR pure OCR engine (det + rec).

    Parameters
    ----------
    lang : str
        Language code (e.g. 'en', 'ch').
    use_angle_cls : bool
        Enable angle classification.
    show_log : bool
        Show PaddleOCR logging.
    device : str
        'cpu' or 'gpu:0'.
    cluster_row_tol : int
        Y-center tolerance for row grouping.
    cluster_col_tol : int
        X-center tolerance for col grouping.
    """

    name: str = "paddle_ocr"

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        show_log: bool = False,
        device: str = "cpu",
        cluster_row_tol: int = 10,
        cluster_col_tol: int = 15,
    ) -> None:
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self.device = device
        self.cluster_row_tol = cluster_row_tol
        self.cluster_col_tol = cluster_col_tol
        self._load_model()

    def _load_model(self) -> None:
        try:
            from paddleocr import PaddleOCR
            use_gpu = self.device.startswith("gpu")
            logger.info(f"Loading PaddleOCR (lang={self.lang}, use_gpu={use_gpu})")
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=self.show_log,
                use_gpu=use_gpu,
            )
            logger.info("PaddleOCR loaded.")
        except Exception as exc:
            logger.error(f"PaddleOCR load failed: {exc}")
            raise

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image: np.ndarray,
        raw_out_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run PaddleOCR and cluster into grid.

        Returns
        -------
        dict: cells, rows, cols, num_rows, num_cols, transform_log, ocr_tokens
        """
        orig_h, orig_w = image.shape[:2]
        transform_log = {
            "input_size": {"W": orig_w, "H": orig_h},
            "note": "PaddleOCR internally resizes; coordinates returned in original space.",
            "inverse_formula": "coords are already in original image pixel space",
        }

        words = self._run_ocr(image)

        if raw_out_dir is not None:
            raw_out_dir = Path(raw_out_dir)
            raw_out_dir.mkdir(parents=True, exist_ok=True)
            (raw_out_dir / "raw.json").write_text(
                json.dumps({"words": words}, indent=2), encoding="utf-8"
            )

        result = self._cluster_words(words, orig_w, orig_h)
        result["transform_log"] = transform_log
        result["ocr_tokens"] = words
        return result

    # ── OCR ───────────────────────────────────────────────────────────────────

    def _run_ocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run PaddleOCR and parse results into word dicts."""
        try:
            raw = self._ocr.ocr(image, cls=self.use_angle_cls)
        except Exception as exc:
            logger.error(f"PaddleOCR.ocr() failed: {exc}")
            return []

        words = []
        if not raw:
            return words

        page = raw[0] if isinstance(raw[0], list) else raw
        if page is None:
            return words

        for item in page:
            if not item:
                continue
            polygon, (text, conf) = item
            bbox = _poly_to_xyxy(polygon)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            words.append({
                "text": text,
                "bbox": bbox,
                "conf": float(conf),
                "cx": cx,
                "cy": cy,
            })

        return words

    # ── Clustering (identical strategy as Tesseract engine) ───────────────────

    def _cluster_words(
        self,
        words: List[Dict[str, Any]],
        orig_w: int,
        orig_h: int,
    ) -> Dict[str, Any]:
        if not words:
            return {"cells": [], "rows": [], "cols": [], "num_rows": 0, "num_cols": 0}

        sorted_words = sorted(words, key=lambda w: (w["cy"], w["cx"]))

        row_groups: List[List[Dict]] = []
        current_row: List[Dict] = []
        prev_cy: Optional[float] = None

        for wd in sorted_words:
            if prev_cy is None or abs(wd["cy"] - prev_cy) > self.cluster_row_tol:
                if current_row:
                    row_groups.append(current_row)
                current_row = [wd]
                prev_cy = wd["cy"]
            else:
                current_row.append(wd)
                prev_cy = (prev_cy + wd["cy"]) / 2

        if current_row:
            row_groups.append(current_row)

        all_x_centers = sorted({wd["cx"] for wd in words})
        col_boundaries: List[float] = []
        for cx in all_x_centers:
            if not col_boundaries or abs(cx - col_boundaries[-1]) > self.cluster_col_tol:
                col_boundaries.append(cx)

        n_cols = len(col_boundaries)

        def find_col(cx: float) -> int:
            best, best_d = 0, abs(cx - col_boundaries[0])
            for i, bc in enumerate(col_boundaries):
                d = abs(cx - bc)
                if d < best_d:
                    best_d = d; best = i
            return best

        cell_map: Dict[Tuple[int, int], List[Dict]] = {}
        for ridx, row in enumerate(row_groups):
            for wd in sorted(row, key=lambda w: w["cx"]):
                cidx = find_col(wd["cx"])
                cell_map.setdefault((ridx, cidx), []).append(wd)

        cells = []
        for (ridx, cidx), slot_words in sorted(cell_map.items()):
            merged_text = " ".join(w["text"] for w in slot_words)
            avg_conf = float(np.mean([w["conf"] for w in slot_words]))
            cells.append({
                "row_idx": ridx,
                "col_idx": cidx,
                "row_span": 1,
                "col_span": 1,
                "bbox": [
                    min(w["bbox"][0] for w in slot_words),
                    min(w["bbox"][1] for w in slot_words),
                    max(w["bbox"][2] for w in slot_words),
                    max(w["bbox"][3] for w in slot_words),
                ],
                "text": merged_text,
                "ocr_tokens": [
                    {"text": w["text"], "bbox": w["bbox"], "conf": w["conf"]}
                    for w in slot_words
                ],
                "confidence": avg_conf,
            })

        rows_out = []
        for ridx, row in enumerate(row_groups):
            rows_out.append({
                "idx": ridx,
                "bbox": [
                    min(w["bbox"][0] for w in row),
                    min(w["bbox"][1] for w in row),
                    max(w["bbox"][2] for w in row),
                    max(w["bbox"][3] for w in row),
                ],
            })

        cols_out = []
        for cidx in range(n_cols):
            col_words = [w for w in words if find_col(w["cx"]) == cidx]
            if col_words:
                cols_out.append({
                    "idx": cidx,
                    "bbox": [
                        min(w["bbox"][0] for w in col_words),
                        min(w["bbox"][1] for w in col_words),
                        max(w["bbox"][2] for w in col_words),
                        max(w["bbox"][3] for w in col_words),
                    ],
                })

        return {
            "cells": cells,
            "rows": rows_out,
            "cols": cols_out,
            "num_rows": len(row_groups),
            "num_cols": n_cols,
        }

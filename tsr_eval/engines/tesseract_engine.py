"""
Tesseract OCR Engine Wrapper
==============================
Runs pytesseract on table images and clusters words into rows/columns.

Outputs
-------
raw_outputs/<image_id>_tesseract.tsv     — raw TSV from tesseract image_to_data()
raw_outputs/<image_id>_tesseract.hocr    — hOCR output

Transform Logging
-----------------
Tesseract operates on the input image directly (no internal resize).
transform_log records only the original image dimensions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TesseractEngine:
    """
    Tesseract OCR engine that clusters words into a table grid.

    Parameters
    ----------
    lang : str
        Tesseract language pack (e.g. 'eng').
    psm : int
        Page segmentation mode: 6 = uniform block of text.
    oem : int
        OCR engine mode: 3 = default (LSTM + legacy).
    cluster_row_tol : int
        Y-center tolerance (pixels) for row grouping.
    cluster_col_tol : int
        X-center tolerance (pixels) for column grouping.
    """

    name: str = "tesseract"

    def __init__(
        self,
        lang: str = "eng",
        psm: int = 6,
        oem: int = 3,
        cluster_row_tol: int = 10,
        cluster_col_tol: int = 15,
    ) -> None:
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.cluster_row_tol = cluster_row_tol
        self.cluster_col_tol = cluster_col_tol
        self._check_tesseract()

    def _check_tesseract(self) -> None:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tess = pytesseract
            logger.info("Tesseract available.")
        except Exception as exc:
            logger.error(f"Tesseract not available: {exc}")
            raise

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image: np.ndarray,
        raw_out_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run Tesseract OCR and cluster words into rows/columns.

        Returns
        -------
        dict with keys: cells, rows, cols, num_rows, num_cols, transform_log,
                        ocr_tokens (all raw tokens with bboxes and confidence)
        """
        from PIL import Image as PILImage
        import pandas as pd

        orig_h, orig_w = image.shape[:2]
        transform_log = {
            "input_size": {"W": orig_w, "H": orig_h},
            "note": "Tesseract operates directly on the input image; no internal resize.",
            "inverse_formula": "coords are already in original image pixel space",
        }

        pil_img = PILImage.fromarray(image.astype(np.uint8))
        cfg = f"--psm {self.psm} --oem {self.oem}"

        # --- Raw TSV output ---
        tsv_str = self._tess.image_to_data(pil_img, lang=self.lang, config=cfg)

        # --- hOCR output ---
        hocr_str = ""
        try:
            hocr_str = self._tess.image_to_pdf_or_hocr(pil_img, lang=self.lang, config=cfg, extension="hocr").decode("utf-8")
        except Exception:
            pass

        # Save raw
        if raw_out_dir is not None:
            raw_out_dir = Path(raw_out_dir)
            raw_out_dir.mkdir(parents=True, exist_ok=True)
            (raw_out_dir / "raw.tsv").write_text(tsv_str, encoding="utf-8")
            if hocr_str:
                (raw_out_dir / "raw.hocr").write_text(hocr_str, encoding="utf-8")

        # Parse TSV into word records
        words = self._parse_tsv(tsv_str)

        # Cluster into rows → cols
        result = self._cluster_words(words, orig_w, orig_h)
        result["transform_log"] = transform_log
        result["ocr_tokens"] = words  # keep all raw tokens

        return result

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_tsv(tsv_str: str) -> List[Dict[str, Any]]:
        """Parse Tesseract TSV output into word dicts."""
        words = []
        lines = tsv_str.strip().splitlines()
        if len(lines) < 2:
            return words

        header = lines[0].split("\t")
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            text = row.get("text", "").strip()
            conf_str = row.get("conf", "-1")
            try:
                conf = float(conf_str)
            except ValueError:
                conf = -1.0

            if not text or conf < 0:
                continue

            try:
                x = int(row["left"])
                y = int(row["top"])
                w = int(row["width"])
                h = int(row["height"])
            except (ValueError, KeyError):
                continue

            words.append({
                "text": text,
                "bbox": [x, y, x + w, y + h],
                "conf": conf / 100.0,  # normalize to [0,1]
                "cx": x + w / 2,
                "cy": y + h / 2,
            })

        return words

    # ── Clustering ────────────────────────────────────────────────────────────

    def _cluster_words(
        self,
        words: List[Dict[str, Any]],
        orig_w: int,
        orig_h: int,
    ) -> Dict[str, Any]:
        """Cluster words into rows and columns to form a grid."""
        if not words:
            return {"cells": [], "rows": [], "cols": [], "num_rows": 0, "num_cols": 0}

        # Sort by Y then X
        sorted_words = sorted(words, key=lambda w: (w["cy"], w["cx"]))

        # --- Group into rows ---
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
                prev_cy = (prev_cy + wd["cy"]) / 2  # rolling average

        if current_row:
            row_groups.append(current_row)

        # --- Build column boundaries from all rows ---
        all_x_centers = sorted({wd["cx"] for wd in words})
        col_boundaries: List[float] = []
        for cx in all_x_centers:
            if not col_boundaries or abs(cx - col_boundaries[-1]) > self.cluster_col_tol:
                col_boundaries.append(cx)

        n_cols = len(col_boundaries)

        # --- Assign each word to (row_idx, col_idx) ---
        def find_col(cx: float) -> int:
            best = 0
            best_d = abs(cx - col_boundaries[0])
            for i, bc in enumerate(col_boundaries):
                d = abs(cx - bc)
                if d < best_d:
                    best_d = d
                    best = i
            return best

        cells: List[Dict[str, Any]] = []
        # Aggregate words that land in the same (row, col) slot
        cell_map: Dict[Tuple[int, int], List[Dict]] = {}
        for ridx, row in enumerate(row_groups):
            for wd in sorted(row, key=lambda w: w["cx"]):
                cidx = find_col(wd["cx"])
                cell_map.setdefault((ridx, cidx), []).append(wd)

        for (ridx, cidx), slot_words in sorted(cell_map.items()):
            xs = [w["bbox"][0] for w in slot_words]
            ys = [w["bbox"][1] for w in slot_words]
            xe = [w["bbox"][2] for w in slot_words]
            ye = [w["bbox"][3] for w in slot_words]
            merged_text = " ".join(w["text"] for w in slot_words)
            avg_conf = float(np.mean([w["conf"] for w in slot_words]))
            cells.append({
                "row_idx": ridx,
                "col_idx": cidx,
                "row_span": 1,
                "col_span": 1,
                "bbox": [min(xs), min(ys), max(xe), max(ye)],
                "text": merged_text,
                "ocr_tokens": [
                    {"text": w["text"], "bbox": w["bbox"], "conf": w["conf"]}
                    for w in slot_words
                ],
                "confidence": avg_conf,
            })

        # --- Derive rows / cols summary ---
        rows_out = []
        for ridx, row in enumerate(row_groups):
            ys0 = min(w["bbox"][1] for w in row)
            xs0 = min(w["bbox"][0] for w in row)
            xs1 = max(w["bbox"][2] for w in row)
            ys1 = max(w["bbox"][3] for w in row)
            rows_out.append({"idx": ridx, "bbox": [xs0, ys0, xs1, ys1]})

        cols_out = []
        for cidx, cb_cx in enumerate(col_boundaries):
            col_words = [w for w in words if find_col(w["cx"]) == cidx]
            if col_words:
                xs0 = min(w["bbox"][0] for w in col_words)
                ys0 = min(w["bbox"][1] for w in col_words)
                xs1 = max(w["bbox"][2] for w in col_words)
                ys1 = max(w["bbox"][3] for w in col_words)
                cols_out.append({"idx": cidx, "bbox": [xs0, ys0, xs1, ys1]})

        return {
            "cells": cells,
            "rows": rows_out,
            "cols": cols_out,
            "num_rows": len(row_groups),
            "num_cols": n_cols,
        }

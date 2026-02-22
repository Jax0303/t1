"""
Table Transformer (TATR) TSR Engine Wrapper
============================================
microsoft/table-transformer-structure-recognition

Outputs
-------
raw_outputs/<image_id>_tatr_raw.json      — pre-NMS boxes + labels + scores
raw_outputs/<image_id>_tatr_post.json     — post-NMS grid cells
normalized/<image_id>_tatr.json           — standard normalized JSON

Transform Logging
-----------------
TATR's AutoImageProcessor internally resizes the image so that the longest
edge = 800 and pads to square.  We record:

  input_size : (W, H) of the original image
  proc_size  : (W, H) after processor resize (before padding)
  scale_x    : proc_W / input_W
  scale_y    : proc_H / input_H
  pad_left   : left pad in padded tensor
  pad_top    : top pad in padded tensor

Inverse mapping (padded tensor → original pixel):
  x_orig = (x_proc - pad_left) / scale_x
  y_orig = (y_proc - pad_top ) / scale_y
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TATREngine:
    """
    Table Transformer structure recognition engine.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        'cuda' or 'cpu'.
    threshold : float
        Detection confidence threshold.
    iou_threshold : float
        NMS IoU threshold for spanning-cell merging.
    """

    name: str = "tatr"

    def __init__(
        self,
        model_name: str = "microsoft/table-transformer-structure-recognition",
        device: str = "cpu",
        threshold: float = 0.2,
        iou_threshold: float = 0.3,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load HuggingFace model + processor."""
        try:
            import torch
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection

            logger.info(f"Loading TATR: {self.model_name} on {self.device}")
            self._proc = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
            self._model = self._model.to(self.device).eval()
            self._torch = torch
            logger.info("TATR loaded.")
        except Exception as exc:
            logger.error(f"TATR load failed: {exc}")
            raise

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image: np.ndarray,
        raw_out_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run TATR on a single image.

        Parameters
        ----------
        image : np.ndarray
            H×W×3 uint8 array (RGB).
        raw_out_dir : Path | None
            If given, saves raw JSON files here.

        Returns
        -------
        dict
            ``{"cells": [...], "rows": [...], "cols": [...],
               "num_rows": int, "num_cols": int,
               "transform_log": {...}}``
        """
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image.astype(np.uint8))
        orig_w, orig_h = pil_img.size  # PIL: (W, H)

        # --- Encode ---
        encoding = self._proc(images=pil_img, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Compute transform params from pixel_values shape
        _, _, enc_h, enc_w = encoding["pixel_values"].shape
        scale_x = enc_w / orig_w
        scale_y = enc_h / orig_h
        # TATR processor pads to square — find padding
        # Processor size is max(orig_w, orig_h) * scale, then padded to square
        proc_w = round(orig_w * scale_x)
        proc_h = round(orig_h * scale_y)
        pad_left = (enc_w - proc_w) // 2
        pad_top = (enc_h - proc_h) // 2

        transform_log = {
            "input_size": {"W": orig_w, "H": orig_h},
            "proc_size": {"W": proc_w, "H": proc_h},
            "enc_size": {"W": enc_w, "H": enc_h},
            "scale_x": scale_x,
            "scale_y": scale_y,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "inverse_formula": (
                "x_orig = (x_enc - pad_left) / scale_x; "
                "y_orig = (y_enc - pad_top) / scale_y"
            ),
        }

        # --- Forward ---
        with self._torch.no_grad():
            outputs = self._model(**encoding)

        # --- Pre-NMS boxes (all detections before threshold) ---
        pre_boxes = outputs.pred_boxes[0].cpu().numpy().tolist()
        pre_logits = outputs.logits[0].softmax(-1).cpu().numpy().tolist()
        raw_pre = {"boxes_cxcywh_norm": pre_boxes, "logits": pre_logits}

        # --- Post-process (apply threshold, rescale to original) ---
        target_sizes = self._torch.tensor([[orig_h, orig_w]])
        results = self._proc.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]

        boxes = results["boxes"].cpu().numpy()      # (N, 4) xyxy in orig px
        labels = results["labels"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        id2label = self._model.config.id2label
        label_strings = [id2label[int(lid)] for lid in labels]

        raw_post = [
            {"box": b.tolist(), "label": l, "score": float(s)}
            for b, l, s in zip(boxes, label_strings, scores)
        ]

        # Save raw outputs
        if raw_out_dir is not None:
            raw_out_dir = Path(raw_out_dir)
            raw_out_dir.mkdir(parents=True, exist_ok=True)

        # --- Build grid ---
        result = self._canonicalize_grid(boxes, label_strings, scores, orig_w, orig_h)
        result["transform_log"] = transform_log

        if raw_out_dir is not None:
            (raw_out_dir / "pre_nms.json").write_text(
                json.dumps({"transform_log": transform_log, "detections": raw_pre}, indent=2)
            )
            (raw_out_dir / "post_nms.json").write_text(
                json.dumps({"transform_log": transform_log, "detections": raw_post}, indent=2)
            )

        return result

    # ── Grid construction ─────────────────────────────────────────────────────

    def _canonicalize_grid(
        self,
        boxes: np.ndarray,
        labels: List[str],
        scores: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> Dict[str, Any]:
        """Assign rows/cols from TATR detections and build cell grid."""
        rows_data: List[Dict] = []
        cols_data: List[Dict] = []
        spanning_data: List[Dict] = []

        for box, label, score in zip(boxes, labels, scores):
            lo = label.lower()
            datum = {"box": box.tolist(), "score": float(score), "label": label}
            if "column" in lo:
                cols_data.append(datum)
            elif "spanning" in lo or "projected" in lo:
                spanning_data.append(datum)
            else:
                # header rows + plain rows both treated as rows
                rows_data.append(datum)

        rows_data.sort(key=lambda d: (d["box"][1] + d["box"][3]) / 2)
        cols_data.sort(key=lambda d: (d["box"][0] + d["box"][2]) / 2)

        if not rows_data or not cols_data:
            logger.warning("TATR: no rows or cols detected")
            return {"cells": [], "rows": [], "cols": [], "num_rows": 0, "num_cols": 0}

        # Structured rows / cols lists
        rows_out = [
            {"idx": i, "bbox": d["box"], "score": d["score"]}
            for i, d in enumerate(rows_data)
        ]
        cols_out = [
            {"idx": j, "bbox": d["box"], "score": d["score"]}
            for j, d in enumerate(cols_data)
        ]

        # Base grid cells (intersection of each row×col band)
        cells = []
        for i, rd in enumerate(rows_data):
            for j, cd in enumerate(cols_data):
                rb, cb = rd["box"], cd["box"]
                x0 = max(cb[0], rb[0])
                y0 = max(rb[1], cb[1])
                x1 = min(cb[2], rb[2])
                y1 = min(rb[3], cb[3])
                if x1 > x0 and y1 > y0:
                    cells.append({
                        "row_idx": i,
                        "col_idx": j,
                        "row_span": 1,
                        "col_span": 1,
                        "bbox": [x0, y0, x1, y1],
                        "text": "",
                        "ocr_tokens": [],
                        "confidence": (rd["score"] + cd["score"]) / 2,
                    })

        # Merge spanning cells
        if spanning_data:
            cells = self._apply_spanning(cells, spanning_data)

        return {
            "cells": cells,
            "rows": rows_out,
            "cols": cols_out,
            "num_rows": len(rows_data),
            "num_cols": len(cols_data),
        }

    @staticmethod
    def _bbox_iou(a: List[float], b: List[float]) -> float:
        """Axis-aligned IoU."""
        xi0 = max(a[0], b[0])
        yi0 = max(a[1], b[1])
        xi1 = min(a[2], b[2])
        yi1 = min(a[3], b[3])
        if xi1 <= xi0 or yi1 <= yi0:
            return 0.0
        inter = (xi1 - xi0) * (yi1 - yi0)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _apply_spanning(
        self,
        cells: List[Dict],
        spanning_data: List[Dict],
    ) -> List[Dict]:
        """Merge base cells covered by spanning-cell detections."""
        merged_idx: set = set()
        final = []

        for sdata in spanning_data:
            sbox = sdata["box"]
            overlaps = [
                (idx, c) for idx, c in enumerate(cells)
                if idx not in merged_idx and self._bbox_iou(sbox, c["bbox"]) > self.iou_threshold
            ]
            if overlaps:
                rs = [c["row_idx"] for _, c in overlaps]
                cs = [c["col_idx"] for _, c in overlaps]
                r0, c0 = min(rs), min(cs)
                r_span = max(rs) - r0 + 1
                c_span = max(cs) - c0 + 1
                final.append({
                    "row_idx": r0,
                    "col_idx": c0,
                    "row_span": r_span,
                    "col_span": c_span,
                    "bbox": sbox,
                    "text": "",
                    "ocr_tokens": [],
                    "confidence": sdata["score"],
                })
                for idx, _ in overlaps:
                    merged_idx.add(idx)

        for idx, cell in enumerate(cells):
            if idx not in merged_idx:
                final.append(cell)

        return final

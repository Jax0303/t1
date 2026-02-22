"""
TSR Evaluation Harness — Main CLI Entry Point
==============================================

Usage
-----
cd /home/user/t1-8
python tsr_eval/run.py --images data/images --models tatr,tesseract --out runs/run_001

See README.md for full options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tsr_eval")


# ── Engine registry ───────────────────────────────────────────────────────────

ENGINE_ALIASES = {
    "tatr": "tatr",
    "tatr_large": "tatr_large",
    "slanet": "slanet",
    "slanet_plus": "slanet_plus",
    "slanext_wired": "slanext_wired",
    "slanext_wireless": "slanext_wireless",
    "tesseract": "tesseract",
    "paddle_ocr": "paddle_ocr",
    "docling": "docling",
    "img2table": "img2table",
}


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except ImportError:
        logger.warning("PyYAML not available; using default config.")
        return {}
    except Exception as exc:
        logger.warning(f"Could not load config: {exc}")
        return {}


def _build_engine(model_key: str, cfg: Dict, device: str):
    """Instantiate the engine for the given model_key."""
    # Resolve config section
    mcfg = cfg.get("models", {}).get(model_key, {})

    if model_key in ("tatr", "tatr_large"):
        from tsr_eval.engines.tatr_engine import TATREngine
        return TATREngine(
            model_name=mcfg.get("model_name", "microsoft/table-transformer-structure-recognition"),
            device=mcfg.get("device", device),
            threshold=mcfg.get("threshold", 0.2),
            iou_threshold=mcfg.get("iou_threshold", 0.3),
        )

    elif model_key in ("slanet", "slanet_plus", "slanet_zh", "slanext_wired", "slanext_wireless"):
        from tsr_eval.engines.paddle_engine import PaddleTSREngine
        variant_map = {
            "slanet": "SLANet",
            "slanet_plus": "SLANet_plus",
            "slanet_zh": "SLANet_zh",
            "slanext_wired": "SLANeXt_wired",
            "slanext_wireless": "SLANeXt_wireless",
        }
        variant = mcfg.get("variant", variant_map.get(model_key, "SLANet_plus"))
        backend = mcfg.get("backend", "paddleocr")
        return PaddleTSREngine(
            variant=variant,
            device=mcfg.get("device", device),
            backend=backend,
        )

    elif model_key == "tesseract":
        from tsr_eval.engines.tesseract_engine import TesseractEngine
        return TesseractEngine(
            lang=mcfg.get("lang", "eng"),
            psm=mcfg.get("psm", 6),
            oem=mcfg.get("oem", 3),
            cluster_row_tol=mcfg.get("cluster_row_tol", 10),
            cluster_col_tol=mcfg.get("cluster_col_tol", 15),
        )

    elif model_key == "paddle_ocr":
        from tsr_eval.engines.paddle_ocr_engine import PaddleOCREngine
        return PaddleOCREngine(
            lang=mcfg.get("lang", "en"),
            use_angle_cls=mcfg.get("use_angle_cls", True),
            show_log=mcfg.get("show_log", False),
            device=mcfg.get("device", device),
            cluster_row_tol=mcfg.get("cluster_row_tol", 10),
            cluster_col_tol=mcfg.get("cluster_col_tol", 15),
        )

    elif model_key == "docling":
        from tsr_eval.engines.docling_engine import DoclingEngine
        return DoclingEngine()

    elif model_key == "img2table":
        from tsr_eval.engines.img2table_engine import Img2TableEngine
        return Img2TableEngine()

    raise ValueError(f"Unknown model key: {model_key}")


# ── Table export ───────────────────────────────────────────────────────────────

def _export_table(normalized: Dict[str, Any], tables_dir: Path) -> None:
    """Export normalized table to CSV and HTML."""
    image_id = normalized.get("image_id", "unknown")
    model = normalized.get("model", "unk")
    tables_dir = Path(tables_dir) / model
    tables_dir.mkdir(parents=True, exist_ok=True)

    tables = normalized.get("tables", [])
    if not tables:
        return
    cells = tables[0].get("cells", [])
    if not cells:
        return

    n_rows = max(c["row_idx"] + c["row_span"] for c in cells)
    n_cols = max(c["col_idx"] + c["col_span"] for c in cells)

    # Build grid
    grid: Dict = {}
    for c in cells:
        for dr in range(c["row_span"]):
            for dc in range(c["col_span"]):
                grid[(c["row_idx"] + dr, c["col_idx"] + dc)] = c.get("text", "")

    # CSV
    import csv, io
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    for r in range(n_rows):
        writer.writerow([grid.get((r, c), "") for c in range(n_cols)])
    (tables_dir / f"{image_id}.csv").write_text(csv_buf.getvalue(), encoding="utf-8")

    # HTML
    html_lines = ["<table border='1' cellpadding='4' cellspacing='0'>"]
    for r in range(n_rows):
        html_lines.append("<tr>")
        c = 0
        while c < n_cols:
            key = (r, c)
            cell = next((x for x in cells if x["row_idx"] == r and x["col_idx"] == c), None)
            if cell:
                rs, cs_val = cell["row_span"], cell["col_span"]
                text = cell.get("text", "")
                attrs = f' rowspan="{rs}"' if rs > 1 else ""
                attrs += f' colspan="{cs_val}"' if cs_val > 1 else ""
                html_lines.append(f"<td{attrs}>{text}</td>")
                c += cs_val
            else:
                html_lines.append(f"<td>{grid.get(key, '')}</td>")
                c += 1
        html_lines.append("</tr>")
    html_lines.append("</table>")
    (tables_dir / f"{image_id}.html").write_text("\n".join(html_lines), encoding="utf-8")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    images_dir: Path,
    gt_dir: Optional[Path],
    model_keys: List[str],
    out_dir: Path,
    cfg: Dict[str, Any],
    device: str = "cpu",
    top_n: int = 10,
) -> None:
    from PIL import Image as PILImage
    from tsr_eval.normalizer import normalize
    from tsr_eval.metrics_self import compute_self_metrics
    from tsr_eval.metrics_gt import compute_gt_metrics
    from tsr_eval.visualizer import draw_overlay, select_worst_cases, copy_worst_cases
    from tsr_eval.reporter import generate_report, save_metrics_csv

    # Collect images
    images_dir = Path(images_dir)
    image_paths = sorted(
        list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.tif"))
        + list(images_dir.glob("*.tiff"))
    )
    if not image_paths:
        logger.error(f"No images found in {images_dir}")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} image(s) in {images_dir}")

    # Create output dirs
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw_outputs"
    norm_dir = out_dir / "normalized"
    overlay_dir = out_dir / "overlays"
    tables_dir = out_dir / "tables"
    metrics_dir = out_dir / "metrics"
    for d in [raw_dir, norm_dir, overlay_dir, tables_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load engines
    engines = {}
    for key in model_keys:
        logger.info(f"Loading engine: {key}")
        try:
            engines[key] = _build_engine(key, cfg, device)
            logger.info(f"✓ {key} ready")
        except Exception as exc:
            logger.error(f"✗ {key} failed to load: {exc}")

    if not engines:
        logger.error("No engines loaded. Aborting.")
        sys.exit(1)

    # Process
    all_metrics: List[Dict] = []
    metrics_cfg = cfg.get("metrics", {})
    col_gap_thresh = metrics_cfg.get("col_gap_threshold", 5.0)
    gt_dir_path = Path(gt_dir) if gt_dir else None

    for img_path in image_paths:
        image_id = img_path.stem
        logger.info(f"Processing image: {image_id}")

        # Load image
        try:
            pil_img = PILImage.open(img_path).convert("RGB")
            image_np = np.array(pil_img, dtype=np.uint8)
            orig_h, orig_w = image_np.shape[:2]
        except Exception as exc:
            logger.error(f"Cannot load {img_path}: {exc}")
            continue

        for model_key, engine in engines.items():
            logger.info(f"  [{model_key}] running...")
            t0 = time.time()

            # Engine-specific raw_out subdir
            engine_raw_dir = raw_dir / model_key / image_id
            engine_raw_dir.mkdir(parents=True, exist_ok=True)

            try:
                raw_output = engine.predict(image_np, raw_out_dir=engine_raw_dir)
            except Exception as exc:
                logger.error(f"  [{model_key}] predict() failed: {exc}")
                raw_output = {"cells": [], "rows": [], "cols": [], "num_rows": 0, "num_cols": 0}

            elapsed = time.time() - t0

            # Normalize
            norm = normalize(
                image_id=image_id,
                image_path=str(img_path),
                image_size={"W": orig_w, "H": orig_h},
                model_name=model_key,
                engine_output=raw_output,
                out_dir=norm_dir,
            )

            # Export table CSV/HTML
            _export_table(norm, tables_dir)

            # Self-consistency metrics
            self_m = compute_self_metrics(norm, col_gap_threshold=col_gap_thresh)

            # GT metrics
            gt_m: Dict = {}
            if gt_dir_path and gt_dir_path.exists():
                gt_result = compute_gt_metrics(norm, gt_dir_path)
                if gt_result:
                    gt_m = gt_result

            # Overlay
            overlay_path = overlay_dir / model_key / f"{image_id}.png"
            try:
                draw_overlay(image_np, norm, out_path=overlay_path)
            except Exception as exc:
                logger.warning(f"  [{model_key}] overlay failed: {exc}")
                overlay_path = None

            # Aggregate metrics record
            rec = {
                "image_id": image_id,
                "model": model_key,
                "elapsed_sec": round(elapsed, 2),
                "num_rows": raw_output.get("num_rows", 0),
                "num_cols": raw_output.get("num_cols", 0),
                "num_cells": len(raw_output.get("cells", [])),
                "overlay_path": str(overlay_path) if overlay_path else "",
                **self_m,
                **gt_m,
            }
            all_metrics.append(rec)

            logger.info(
                f"  [{model_key}] done ({elapsed:.1f}s) "
                f"| score={self_m.get('summary_score', 0):.3f} "
                f"| rows={rec['num_rows']} cols={rec['num_cols']}"
            )

    # Save metrics JSON
    metrics_json_path = metrics_dir / "metrics.json"
    metrics_json_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    save_metrics_csv(all_metrics, metrics_dir / "metrics.csv")
    logger.info(f"Metrics saved to {metrics_dir}/")

    # Worst cases
    worst = select_worst_cases(all_metrics, top_n=top_n)
    worst = copy_worst_cases(worst, overlay_dir)

    # Report
    gt_used = any("teds" in m for m in all_metrics)
    generate_report(
        run_dir=out_dir,
        all_metrics=all_metrics,
        worst_cases=worst,
        models_used=list(engines.keys()),
        images_dir=images_dir,
        gt_used=gt_used,
    )
    logger.info(f"Run complete. Results in {out_dir}/")
    logger.info(f"  Report: {out_dir}/report.md")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TSR Baseline Evaluation Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images", type=Path, default=Path("data/images"),
        help="Directory containing input table images.",
    )
    parser.add_argument(
        "--gt", type=Path, default=None,
        help="Directory containing GT files (optional). If omitted, GT metrics are skipped.",
    )
    parser.add_argument(
        "--models", type=str, default="tatr,slanet_plus,tesseract,paddle_ocr",
        help="Comma-separated model key(s): tatr, slanet, slanet_plus, slanext_wired, "
             "slanext_wireless, tesseract, paddle_ocr.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("tsr_eval/config.yaml"),
        help="Config YAML path.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory. Defaults to runs/<timestamp>.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of worst-case images to highlight in report.",
    )
    parser.add_argument(
        "--create-sample", action="store_true",
        help="Create a synthetic sample table image in data/images/ for testing.",
    )

    args = parser.parse_args()

    # Optionally create a sample image
    if args.create_sample:
        _create_sample_image(args.images)

    # Resolve output dir
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = Path("runs") / f"run_{ts}"

    # Load config
    cfg = _load_config(args.config) if args.config.exists() else {}

    # Parse model keys
    model_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in model_keys if k not in ENGINE_ALIASES]
    if unknown:
        logger.warning(f"Unknown model keys (will attempt anyway): {unknown}")

    logger.info(f"Run: {args.out}")
    logger.info(f"Images: {args.images}")
    logger.info(f"Models: {model_keys}")
    logger.info(f"Device: {args.device}")

    run_pipeline(
        images_dir=args.images,
        gt_dir=args.gt,
        model_keys=model_keys,
        out_dir=args.out,
        cfg=cfg,
        device=args.device,
        top_n=args.top_n,
    )


def _create_sample_image(images_dir: Path) -> None:
    """Create a simple synthetic table image (3 rows × 3 cols) for smoke testing."""
    from PIL import Image, ImageDraw, ImageFont

    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = images_dir / "sample_3x3.png"

    if out_path.exists():
        logger.info(f"Sample image already exists: {out_path}")
        return

    W, H = 600, 300
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    headers = ["Name", "Score", "Grade"]
    rows_data = [
        ["Alice", "92", "A"],
        ["Bob", "78", "B"],
    ]
    all_rows = [headers] + rows_data
    col_w = W // 3
    row_h = H // (len(all_rows) + 1)
    y0 = 20

    for ridx, row in enumerate(all_rows):
        y = y0 + ridx * row_h
        for cidx, text in enumerate(row):
            x = cidx * col_w
            draw.rectangle([x, y, x + col_w, y + row_h], outline="black", width=2)
            draw.text((x + 8, y + 8), text, fill="black", font=font)

    img.save(str(out_path))
    logger.info(f"Sample image created: {out_path}")


if __name__ == "__main__":
    main()

# TSR Evaluation Harness

A self-contained evaluation harness for **Table Structure Recognition (TSR)** baselines.
Runs multiple models on table images, normalises outputs, computes metrics, and generates visual overlays + a full diagnostic report.

---

## Supported Models

| Key | Model | Notes |
|-----|-------|-------|
| `tatr` | Table Transformer (TATR) | `microsoft/table-transformer-structure-recognition` |
| `tatr_large` | TATR v1.1-all | Heavier checkpoint |
| `slanet` | PaddleOCR SLANet | ppstructure backend |
| `slanet_plus` | PaddleOCR SLANet_plus | Default PaddleOCR TSR |
| `slanext_wired` | PaddleX SLANeXt_wired | Requires `paddlex>=3.0` |
| `slanext_wireless` | PaddleX SLANeXt_wireless | Requires `paddlex>=3.0` |
| `tesseract` | Tesseract OCR | Word-cluster based grid |
| `paddle_ocr` | PaddleOCR OCR | Det+Rec, word-cluster grid |

---

## Directory Layout

```
/home/user/t1-8/
├── data/
│   ├── images/           ← place PNG/JPG table images here
│   └── gt/               ← optional GT files (HTML/JSON/CSV, stem = image ID)
├── runs/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── raw_outputs/       model raw output files (pre+post NMS)
│       ├── normalized/        standard JSON per (image, model)
│       ├── overlays/          row/col/cell visualisations
│       ├── tables/            CSV + HTML table renders
│       ├── metrics/           metrics.json + metrics.csv
│       └── report.md          failure analysis report
└── tsr_eval/             ← this harness
    ├── run.py            ← CLI entry-point
    ├── config.yaml       ← model & metric configuration
    └── ...
```

---

## Quick Start

### 1. Install dependencies

```bash
cd /home/user/t1-8

# Core Python packages
pip install -r tsr_eval/requirements.txt

# System Tesseract (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# PaddlePaddle (CPU)
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

### 2. Add images

Drop PNG/JPG table images into `data/images/`:
```bash
cp my_tables/*.png data/images/
```

Optionally add GT files to `data/gt/` (file stem must match image stem):
```
data/gt/table_01.html    # HTML table
data/gt/table_01.json    # {"cells": [{row, col, row_span, col_span, text}, ...]}
data/gt/table_01.csv     # Grid-layout CSV
```

### 3. Run

```bash
# All default models (TATR + SLANet_plus + Tesseract + PaddleOCR)
python -m tsr_eval.run --images data/images --out runs/run_001

# Only TATR and Tesseract, on GPU
python -m tsr_eval.run --images data/images --models tatr,tesseract --device cuda

# With GT metrics
python -m tsr_eval.run --images data/images --gt data/gt --models tatr,slanet_plus

# Quick smoke test with a synthetic sample image
python -m tsr_eval.run --create-sample --models tesseract --out runs/smoke_test
```

### 4. Inspect results

```
runs/run_001/
├── report.md             ← start here — summary + worst cases
├── metrics/
│   ├── metrics.json      ← all metrics per (image, model)
│   └── metrics.csv       ← spreadsheet-friendly
├── overlays/
│   ├── tatr/             ← per-image overlays
│   ├── tesseract/
│   └── worst_cases/      ← top-10 worst ranked
├── normalized/           ← standard JSON specs
└── tables/               ← CSV + HTML rendered tables
```

---

## Metrics Reference

### Self-Consistency (no GT required)

| Metric | Description | Higher = Better? |
|--------|-------------|:----------------:|
| `col_x_align_score` | `1/(1+std(x_left)+std(x_right))` per col, averaged | ✓ |
| `row_y_align_score` | `1/(1+std(y_top)+std(y_bottom))` per row, averaged | ✓ |
| `col_monotonicity_violations` | # adjacent col pairs with non-monotone x order | ✗ |
| `row_monotonicity_violations` | # adjacent row pairs with non-monotone y order | ✗ |
| `col_overlap_ratio` | Fraction of adjacent col pairs that overlap in X | ✗ |
| `col_gap_ratio` | Fraction of adjacent col pairs with gap > threshold | — |
| `summary_score` | Composite score [0–1] | ✓ |

### GT-Based (when GT files present)

| Metric | Description |
|--------|-------------|
| `teds` | Tree-Edit Distance Similarity on HTML |
| `cell_f1` | Cell-level F1 matching by `(row, col, row_span, col_span)` |
| `em` | Exact Match (requires text to match too) |

---

## Configuration

Edit `tsr_eval/config.yaml` to enable/disable models and tune parameters:

```yaml
models:
  tatr:
    enabled: true
    device: "cpu"          # or "cuda"
    threshold: 0.2

metrics:
  top_n_worst: 10          # worst cases highlighted in report
  col_gap_threshold: 5     # gap threshold in pixels
```

---

## Normalized JSON Format

Each `normalized/<image_id>_<model>.json` follows this schema:

```json
{
  "image_id": "table_01",
  "image_path": "data/images/table_01.png",
  "image_size": {"W": 800, "H": 600},
  "model": "tatr",
  "transform_log": {
    "input_size": {"W": 800, "H": 600},
    "scale_x": 1.0, "scale_y": 1.0,
    "pad_left": 0, "pad_top": 0,
    "inverse_formula": "x_orig = (x_enc - pad_left) / scale_x"
  },
  "tables": [{
    "table_bbox": [10, 5, 790, 595],
    "rows": [{"idx": 0, "bbox": [10, 5, 790, 55]}],
    "cols": [{"idx": 0, "bbox": [10, 5, 270, 595]}],
    "cells": [{
      "row_idx": 0, "col_idx": 0,
      "row_span": 1, "col_span": 1,
      "bbox": [10, 5, 270, 55],
      "text": "Name",
      "ocr_tokens": [{"text": "Name", "bbox": [15, 8, 260, 52], "conf": 0.98}],
      "source_model": "tatr"
    }]
  }]
}
```

---

## Failure Taxonomy

| Type | Description |
|------|-------------|
| `col_drift` | Column x-boundaries shift across rows |
| `row_drift` | Row y-boundaries shift across columns |
| `merged_cell_split` | Span cells incorrectly split |
| `ocr_outlier` | OCR bbox outside image bounds |
| `crop_mismatch` | Transform bug — boxes don't align with image |
| `monotonicity_violation` | Non-monotone row/col ordering |
| `col_overlap` | Adjacent column bboxes overlap |
| `col_gap` | Large gap between adjacent columns |

---

## Adding New Models

1. Create `tsr_eval/engines/my_model_engine.py` with a class that has `predict(image: np.ndarray, raw_out_dir: Path) -> dict`.
2. Register it in `run.py`'s `_build_engine()`.
3. Pass `--models my_model` on the CLI.

"""
Configuration file for table parsing experiment
표 파싱 실험 설정 파일
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "cache"

# Dataset configurations
DATASET_CONFIG = {
    "sample_size": 5000,  # 중간 규모: 각 데이터셋에서 5000개 샘플
    "datasets": [
        "PubTabNet",
        "FinTabNet",
        # SciTSR, ICDAR 등은 선택적으로 추가 가능
    ],
    "table_types": [
        "fully_bordered",      # 완전 테두리
        "unbordered",          # 테두리 없음
        "partially_bordered",  # 부분 테두리
        "spanning_cells",      # 병합 셀 많음
        "empty_cells",         # 빈 셀 많음
        "complex",             # 복잡한 구조
    ]
}

# Model configurations
MODEL_CONFIG = {
    "ocr_engines": {
        "tesseract": {
            "enabled": True,
            "lang": "eng+kor",  # 영어 + 한국어 지원
            "config": "--psm 6"
        },
        "easyocr": {
            "enabled": True,
            "languages": ["en", "ko"],
            "gpu": True
        }
    },
    "table_detection": {
        "model_name": "microsoft/table-transformer-detection",
        "source": "huggingface",
        "threshold": 0.7
    },
    "table_structure": {
        "model_name": "microsoft/table-transformer-structure-recognition",
        "source": "huggingface",
        "threshold": 0.7
    }
}

# Evaluation metrics
METRICS_CONFIG = {
    "compute_teds": True,
    "compute_s_teds": True,
    "compute_precision_recall": True,
    "compute_iou": True,
    "iou_threshold": 0.5
}

# Error classification (RAPTOR 기반)
ERROR_TYPES = {
    "TD": [
        "noise_detected_above",
        "noise_detected_below",
        "wrong_table_detected",
        "false_positive",
        "false_negative"
    ],
    "TSR": [
        "missing_elements",
        "wrong_header_detected",
        "segmentation_error",
        "fused_lines",
        "splitted_line",
        "implicit_header",
        "spanning_cells_error",
        "sub_tables"
    ],
    "OCR": [
        "text_recognition_error",
        "missing_text",
        "extra_text"
    ]
}

# Visualization settings
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 150,
    "style": "seaborn-v0_8-darkgrid",
    "save_format": "png"
}

# Logging
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": RESULTS_DIR / "experiment.log"
}

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

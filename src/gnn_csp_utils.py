"""
GNN-CSP Utilities for Notebook usage.
Provides easy access to the pipeline and visualization tools.
"""

import os
import sys
import torch
import logging
from typing import Optional

def setup_notebook_env(project_root: str = "/root/t1-9"):
    """Adds project paths to sys.path for notebook usage."""
    src_path = os.path.join(project_root, "src")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print(f"Environment setup complete. Project root: {project_root}")

def get_pipeline(
    rca_checkpoint: Optional[str] = None,
    gnn_checkpoint: Optional[str] = None,
    device: Optional[str] = None
):
    """Initializes and returns the GNN-CSP pipeline."""
    from hiertable_rag.gnn_csp.pipeline import GNNCSPPipeline
    
    logging.basicConfig(level=logging.INFO)
    pipeline = GNNCSPPipeline(
        rca_checkpoint=rca_checkpoint,
        gnn_checkpoint=gnn_checkpoint,
        device=device
    )
    return pipeline

def quick_parse(pipeline, image_path, ocr_data):
    """Helper for a quick parse from image path and OCR data."""
    # Placeholder for image loading logic
    # In a real notebook, users might pass a PIL image or a tensor
    pass

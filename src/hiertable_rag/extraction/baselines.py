"""
Baseline extraction strategies for table RAG.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TableBaseline:
    """Base class for table extraction baselines."""
    def extract(self, table_raw: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class OCRExtractor(TableBaseline):
    """
    B1: OCR Only baseline.
    Simulates flat text extraction where hierarchical structure is lost.
    """
    def extract(self, table_raw: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Extracting with OCR Only (Flat)")
        # Simulate flattening: take all cells and put them in a simple list
        # Missing header relationships
        rows = table_raw.get("rows", [])
        flat_text = " ".join([str(cell) for row in rows for cell in row])
        
        return {
            "strategy": "OCR_ONLY",
            "content": flat_text,
            "structure_score": 0.6,
            "metadata": {"type": "flat_text"}
        }

class TSRExtractor(TableBaseline):
    """
    B2: OCR + TSR baseline.
    Focuses on structural integrity (GraphTSR).
    """
    def extract(self, table_raw: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Extracting with OCR + TSR (Structure)")
        # Provide structural metadata but might miss deep semantic nuances
        return {
            "strategy": "OCR_TSR",
            "content": table_raw.get("rows", []),
            "hierarchy": table_raw.get("headers", []),
            "structure_score": 0.85,
            "metadata": {"type": "graph_tsr"}
        }

class VLMExtractor(TableBaseline):
    """
    B3: OCR + VLM baseline.
    Highly semantic but prone to hallucination in complex structures.
    """
    def extract(self, table_raw: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Extracting with OCR + VLM (Semantic)")
        # In a real scenario, this would call a VLM like Gemini or GPT-4o
        # Here we simulate high structural score but provide a hallucination risk flag
        return {
            "strategy": "OCR_VLM",
            "content": table_raw.get("rows", []),
            "hierarchy": table_raw.get("headers", []),
            "structure_score": 0.95,
            "hallucination_risk": True,
            "metadata": {"type": "vlm"}
        }

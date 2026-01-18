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
    Semantic Noise is high due to layout loss.
    """
    def extract(self, table_raw: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Extracting with OCR Only (Flat)")
        rows = table_raw.get("rows", [])
        
        # Simulate loss of layout: just concatenation of cell contents
        # This creates 'Semantic Noise' as related cells are far apart in text
        flat_contents = []
        for row in rows:
            for cell in row:
                content = cell.get("content", str(cell)) if isinstance(cell, dict) else str(cell)
                flat_contents.append(content)
        
        flat_text = " | ".join(flat_contents)
        
        return {
            "strategy": "OCR_ONLY",
            "content": flat_text,
            "structure_score": 0.4, # Low structure
            "semantic_noise": 0.8,  # High noise
            "metadata": {"type": "flat_text", "tta_support": False}
        }

class TSRExtractor(TableBaseline):
    """
    B2: OCR + TSR baseline.
    Focuses on structural integrity but might have OCR noise in cell text.
    Formatting Noise is minimized, but Semantic Noise remains in values.
    """
    def extract(self, table_raw: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Extracting with OCR + TSR (Structure)")
        import copy
        processed_rows = copy.deepcopy(table_raw.get("rows", []))
        
        # Simulate OCR character noise in cell contents
        # e.g., 'Revenue' -> 'Revenve', '100' -> '1O0'
        for row in processed_rows:
            for cell in row:
                if isinstance(cell, dict) and "content" in cell:
                    content = cell["content"]
                    if len(content) > 3 and np.random.random() < 0.2:
                        # Introduce a simple character swap/error
                        idx = np.random.randint(0, len(content))
                        content = content[:idx] + "X" + content[idx+1:]
                        cell["content"] = content

        return {
            "strategy": "OCR_TSR",
            "content": processed_rows,
            "hierarchy": table_raw.get("headers", []),
            "structure_score": 0.9,
            "semantic_noise": 0.3,
            "metadata": {"type": "graph_tsr", "tta_support": True}
        }

class VLMExtractor(TableBaseline):
    """
    B3: OCR + VLM baseline.
    Highly semantic but prone to structural 'Formatting Noise' (hallucination).
    """
    def extract(self, table_raw: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Extracting with OCR + VLM (Semantic)")
        import copy
        processed_rows = copy.deepcopy(table_raw.get("rows", []))
        
        # Simulate VLM structural hallucination 
        # e.g., missing a column or merging unrelated cells
        if len(processed_rows) > 1 and np.random.random() < 0.3:
            # Simulate a row being 'summarized' or skipped
            processed_rows = processed_rows[:-1] 

        return {
            "strategy": "OCR_VLM",
            "content": processed_rows,
            "hierarchy": table_raw.get("headers", []),
            "structure_score": 0.7, # Lower due to risk of structural distortion
            "semantic_noise": 0.1,  # Very low semantic noise
            "hallucination_risk": True,
            "metadata": {"type": "vlm", "tta_support": True}
        }
class RuleBasedExtractor(TableBaseline):
    """
    B4: Rule-based baseline (pdfplumber, camelot).
    Strong on clean tables but fails on complex merges and multi-page layouts.
    Common failure: flattening merged cells into the first cell.
    """
    def extract(self, table_raw: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Extracting with Rule-based Baseline (pdfplumber/camelot)")
        import copy
        processed_rows = copy.deepcopy(table_raw.get("rows", []))
        
        # Simulate 'Lattice' mode failure: losing merge cell context
        for row in processed_rows:
            for cell in row:
                if isinstance(cell, dict):
                    # Rule-based often sets span to 1 if it can't detect the border
                    if cell.get("row_span", 1) > 1 or cell.get("col_span", 1) > 1:
                        if np.random.random() < 0.4:
                            cell["row_span"] = 1
                            cell["col_span"] = 1
                            cell["content"] = cell.get("content", "") + " (Split Error)"

        return {
            "strategy": "RULE_BASED",
            "content": processed_rows,
            "structure_score": 0.6,
            "semantic_noise": 0.2, # Low noise if borders are clear
            "metadata": {"type": "rule_based", "tools": ["pdfplumber"]}
        }

class VLMDirectExtractor(TableBaseline):
    """
    B5: Vision-Language Direct Embedding (Most et al. 2025).
    Skips OCR, embeds visual patches directly.
    High semantic accuracy but low generalization to unseen layouts.
    """
    def extract(self, table_raw: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Extracting with VLM Direct Embedding")
        # Simulates a localized summary/embedding without full structural reconstruction
        return {
            "strategy": "VLM_DIRECT",
            "content": "Visual Embedding Vector [Simulated]",
            "semantic_noise": 0.05,
            "structure_score": 0.5, # Implicit structure
            "metadata": {"type": "direct_visual"}
        }

import logging
import tempfile
from typing import Dict, Any, List
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

class DoclingEngine:
    """Wrapper for IBM's Docling model"""
    def __init__(self, **kwargs):
        try:
            from docling.document_converter import DocumentConverter
            logger.info("Initializing Docling DocumentConverter...")
            self.converter = DocumentConverter()
        except ImportError:
            logger.error("docling is not installed. Run: pip install docling")
            raise

    def _empty(self):
        return {
            "html": "", "tokens": [], "bboxes": [], "cells": [],
            "num_rows": 0, "num_cols": 0, "transform_log": {}
        }

    def predict(self, image_np, **kwargs) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as temp_fw:
                PILImage.fromarray(image_np.astype('uint8')).save(temp_fw.name)
                result = self.converter.convert(temp_fw.name)
                doc = result.document
                if not doc.tables:
                    return self._empty()
                return self._parse_docling_table(doc.tables[0])
        except Exception as e:
            logger.error(f"Docling prediction failed: {e}")
            return self._empty()

    def _parse_docling_table(self, table) -> Dict[str, Any]:
        cells = []
        for row_cells in table.data.grid:
            for tcell in row_cells:
                if tcell is None: continue
                bbox_xyxy = [0, 0, 10, 10]
                if tcell.bbox is not None:
                    try:
                        bbox_xyxy = [tcell.bbox.l, tcell.bbox.t, tcell.bbox.r, tcell.bbox.b]
                    except AttributeError:
                        if hasattr(tcell.bbox, 'as_tuple'):
                            b = tcell.bbox.as_tuple()
                            bbox_xyxy = [b[0], b[1], b[2], b[3]]
                cells.append({
                    "row": tcell.start_row_offset_idx, "col": tcell.start_col_offset_idx,
                    "row_span": tcell.row_span, "col_span": tcell.col_span,
                    "bbox": bbox_xyxy, "text": tcell.text or "", "confidence": 0.95, "tokens": []
                })
        nr = max([c["row"] + c["row_span"] for c in cells]) if cells else 0
        nc = max([c["col"] + c["col_span"] for c in cells]) if cells else 0
        return {
            "html": "", "tokens": [], "bboxes": [], "cells": cells, "num_rows": nr, 
            "num_cols": nc, "transform_log": {}
        }

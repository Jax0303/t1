import logging
import tempfile
from typing import Dict, Any, List
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

class Img2TableEngine:
    """Wrapper for img2table"""
    def __init__(self, **kwargs):
        try:
            import img2table
        except ImportError:
            logger.error("img2table is not installed. Run: pip install img2table")
            raise

    def _empty(self):
        return {
            "html": "", "tokens": [], "bboxes": [], "cells": [],
            "num_rows": 0, "num_cols": 0, "transform_log": {}
        }

    def predict(self, image_np, **kwargs) -> Dict[str, Any]:
        try:
            from img2table.document import Image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as temp_fw:
                PILImage.fromarray(image_np.astype('uint8')).save(temp_fw.name)
                img_doc = Image(temp_fw.name)
                extracted_tables = img_doc.extract_tables()
                if not extracted_tables: return self._empty()
                return self._parse_img2table(extracted_tables[0])
        except Exception as e:
            logger.error(f"img2table prediction failed: {e}")
            return self._empty()

    def _parse_img2table(self, table) -> Dict[str, Any]:
        cells = []
        for row_idx, row_cells in table.content.items():
            for col_idx, tcell in enumerate(row_cells):
                if tcell is None: continue
                bx1, by1, bx2, by2 = 0, 0, 10, 10
                if tcell.bbox:
                    bx1, by1, bx2, by2 = tcell.bbox.x1, tcell.bbox.y1, tcell.bbox.x2, tcell.bbox.y2
                cells.append({
                    "row": row_idx, "col": col_idx, "row_span": 1, "col_span": 1,
                    "bbox": [bx1, by1, bx2, by2], "text": tcell.value or "",
                    "confidence": 0.8, "tokens": []
                })
        nr = max([c["row"] + 1 for c in cells]) if cells else 0
        nc = max([c["col"] + 1 for c in cells]) if cells else 0
        return {
            "html": "", "tokens": [], "bboxes": [], "cells": cells, "num_rows": nr, 
            "num_cols": nc, "transform_log": {}
        }

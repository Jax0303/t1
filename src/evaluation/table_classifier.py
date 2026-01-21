import numpy as np
from typing import Dict, Any, List

class TableClassifier:
    """
    Advanced Classifies tables into 3 levels with detailed sub-types:
    1. Macro: Global Topology (Pure-Matrix, Standard-Spanning, Heavily-Nested, Frameless-Sparse, Multi-Chunked)
    2. Structural: Header Logic (Single-Row, Multi-Deep, Cross-Tab, Section-Header, Implicit)
    3. Micro: Data Cell Nature (Numerical-Short, Multi-line-Text, Symbolic-Mixed, Empty-Dominant, Merged-Value)
    """

    def __init__(self, span_threshold: float = 0.05, nested_threshold: float = 0.2):
        self.span_threshold = span_threshold
        self.nested_threshold = nested_threshold

    def classify(self, table_data: Dict[str, Any]) -> Dict[str, str]:
        cells = table_data.get("cells", [])
        if not cells:
            return {"macro": "Unknown", "structural": "Unknown", "micro": "Unknown"}

        # Extract features
        total_cells = len(cells)
        max_row = max(c.get("row_index", 0) for c in cells)
        max_col = max(c.get("col_index", 0) for c in cells)
        spanning_cells = [c for c in cells if c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1]
        span_ratio = len(spanning_cells) / total_cells if total_cells > 0 else 0

        # 1. Macro Classification
        macro_type = self._classify_macro(cells, span_ratio, max_row, max_col)

        # 2. Structural Classification
        structural_type = self._classify_structural(cells, max_row, max_col)

        # 3. Micro Classification
        micro_type = self._classify_micro(cells, total_cells)

        return {
            "macro": macro_type,
            "structural": structural_type,
            "micro": micro_type
        }

    def _classify_macro(self, cells, span_ratio, max_row, max_col) -> str:
        if span_ratio == 0:
            return "Macro: Pure-Matrix"
        
        if span_ratio > self.nested_threshold:
            return "Macro: Heavily-Nested"
        
        # Check for multi-chunked (heuristic: large gaps in row indices or specific labels)
        row_counts = [0] * (max_row + 1)
        for c in cells: row_counts[c.get("row_index", 0)] += 1
        if any(count == 1 and max_col > 3 for count in row_counts[1:-1]): # Single cell spans whole row mid-table
            return "Macro: Multi-Chunked"

        if span_ratio < self.span_threshold:
            return "Macro: Standard-Spanning"
        
        return "Macro: Complex-Irregular"

    def _classify_structural(self, cells, max_row, max_col) -> str:
        # Check for top headers
        header_rows = 0
        for r in range(min(max_row + 1, 5)):
            row_cells = [c for c in cells if c.get("row_index", 0) == r]
            if any(c.get("col_span", 1) > 1 for c in row_cells):
                header_rows += 1
            elif r == 0:
                header_rows += 1
            else:
                break
        
        # Check for left stubs
        stub_cols = 0
        for c in range(min(max_col + 1, 2)):
            col_cells = [cell for cell in cells if cell.get("col_index", 0) == c]
            if any(cell.get("row_span", 1) > 1 for cell in col_cells):
                stub_cols += 1
        
        if header_rows > 1 and stub_cols > 0:
            return "Struct: Cross-Tab"
        if header_rows > 1:
            return "Struct: Multi-Deep-Header"
        if header_rows == 1:
            return "Struct: Single-Row-Header"
        
        return "Struct: Implicit"

    def _classify_micro(self, cells, total_cells) -> str:
        ratios = []
        is_mixed = False
        empty_slots = 0 # Placeholder for slot analysis
        
        for c in cells:
            bbox = c.get("bbox", [0, 0, 1, 1])
            h, w = bbox[2]-bbox[0], bbox[3]-bbox[1]
            if w > 0: ratios.append(h / w)
            
            content = str(c.get("content", ""))
            if any(char in content for char in "*^#$"):
                is_mixed = True
        
        avg_ratio = np.mean(ratios) if ratios else 0
        
        if avg_ratio > 1.2:
            return "Micro: Multi-line-Text"
        if is_mixed:
            return "Micro: Symbolic-Mixed"
        
        # Check for non-header merged values
        data_merged = any(c.get("row_index", 0) > 2 and (c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1) for c in cells)
        if data_merged:
            return "Micro: Merged-Value"

        return "Micro: Numerical-Short"

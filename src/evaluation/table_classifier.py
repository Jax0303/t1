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
        # Improved Header Detection
        # Check first 3 rows. If distinct features found, classify.
        
        # Heuristic: Header rows usually have spans OR are the first row
        header_rows = 0
        potential_header_rows = min(max_row + 1, 3) # Check top 3 rows
        
        for r in range(potential_header_rows):
            row_cells = [c for c in cells if c.get("row_index", 0) == r]
            if not row_cells: continue
            
            # Key feature: Spanning cells in early rows often indicate headers
            has_span = any(c.get("col_span", 1) > 1 or c.get("row_span", 1) > 1 for c in row_cells)
            
            if r == 0:
                header_rows += 1
            elif has_span:
                header_rows += 1
            else:
                # If row 1 (second row) has no spans, it might still be a header if row 0 had spans columns
                # For now, let's stop if we see a non-spanning row after row 0 to avoid false positives?
                # Actually, many headers are just 1 row.
                pass

        # Stub Column detection
        stub_cols = 0
        potential_stub_cols = min(max_col + 1, 2)
        for c in range(potential_stub_cols):
            col_cells = [cell for cell in cells if cell.get("col_index", 0) == c]
            # Key feature: Row spans in first columns often indicate stubs
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
        if not cells: return "Micro: Unknown"
        
        # Analyze content distribution
        numerical_count = 0
        text_count = 0
        mixed_count = 0
        empty_count = 0
        merged_value_count = 0
        
        for c in cells:
            content = str(c.get("content", "")).strip()
            
            # Check for merges outside headers
            is_data_cell = c.get("row_index", 0) > 1
            is_merged = c.get("row_span", 1) > 1 or c.get("col_span", 1) > 1
            if is_data_cell and is_merged:
                merged_value_count += 1

            if not content:
                empty_count += 1
                continue
                
            # Content type check
            digit_count = sum(char.isdigit() for char in content)
            alpha_count = sum(char.isalpha() for char in content)
            total_chars = len(content)
            
            if total_chars == 0: 
                empty_count += 1 # Should match if content was empty string
            elif digit_count / total_chars > 0.8:
                numerical_count += 1
            elif alpha_count / total_chars > 0.8:
                text_count += 1
            else:
                mixed_count += 1

        # Determine dominant type
        # Priority: Merged-Value -> Mixed -> Text -> Numerical -> Empty
        
        if merged_value_count / total_cells > 0.1: # Threshold for classifying as Merged-Value type
             return "Micro: Merged-Value"
             
        # Normalize counts by non-empty cells for content type
        non_empty = total_cells - empty_count
        if non_empty == 0: return "Micro: Empty-Dominant"
        
        if mixed_count / non_empty > 0.3:
            return "Micro: Symbolic-Mixed"
        if text_count / non_empty > 0.4: # If >40% is text, it's likely a text-heavy table
            return "Micro: Textual"
        if numerical_count / non_empty > 0.5:
            return "Micro: Numerical-Short"
            
        return "Micro: Mixed-General"

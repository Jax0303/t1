"""
Table Type Stratification

Classifies tables into multiple types and performs stratified sampling.
"""

import logging
import random
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np

from .utils import extract_text_content

logger = logging.getLogger(__name__)


class TableTypeClassifier:
    """
    Classifies tables into 6 complexity types (multi-label)
    """
    
    TYPE_LABELS = [
        'no_merge',              # All cells are 1x1
        'row_span_merge',        # Contains row spans
        'col_span_merge',        # Contains col spans
        'both_span_merge',       # Contains both row and col spans
        'sparse_empty',          # >30% cells empty
        'dense_text_or_small_font'  # High text density (heuristic)
    ]
    
    def classify(self, cells: List[Dict[str, Any]]) -> List[str]:
        """
        Classify table into types
        
        Args:
            cells: List of cell dicts with start_row, end_row, start_col, end_col, content
        
        Returns:
            List of applicable type labels
        """
        if not cells:
            return ['no_merge']
        
        types = set()
        
        # Analyze spans
        has_row_span = False
        has_col_span = False
        
        for cell in cells:
            row_span = cell.get('end_row', cell.get('start_row', 0)) - cell.get('start_row', 0) + 1
            col_span = cell.get('end_col', cell.get('start_col', 0)) - cell.get('start_col', 0) + 1
            
            if row_span > 1:
                has_row_span = True
            if col_span > 1:
                has_col_span = True
        
        # Assign span types
        if has_row_span and has_col_span:
            types.add('both_span_merge')
        elif has_row_span:
            types.add('row_span_merge')
        elif has_col_span:
            types.add('col_span_merge')
        else:
            types.add('no_merge')
        
        # Check sparsity
        empty_count = 0
        total_cells = len(cells)
        total_text_len = 0
        
        for cell in cells:
            content = extract_text_content(cell)
            if not content or content.strip() == '':
                empty_count += 1
            total_text_len += len(content)
        
        if total_cells > 0:
            sparsity = empty_count / total_cells
            if sparsity > 0.3:
                types.add('sparse_empty')
            
            # Dense text heuristic: average text length > 20 chars per cell
            avg_text_len = total_text_len / total_cells
            if avg_text_len > 20:
                types.add('dense_text_or_small_font')
        
        return sorted(list(types))


class FeatureExtractor:
    """
    Extracts quantitative features from table structure
    """
    
    def extract(self, cells: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract features from cells
        
        Args:
            cells: List of cell dicts
        
        Returns:
            Feature dict with keys: n_rows, n_cols, n_cells, n_spans, etc.
        """
        if not cells:
            return {
                'n_rows': 0,
                'n_cols': 0,
                'n_cells': 0,
                'n_spans': 0,
                'n_row_spans': 0,
                'n_col_spans': 0,
                'sparsity_ratio': 0.0,
                'text_density_proxy': 0.0
            }
        
        # Compute grid dimensions
        max_row = 0
        max_col = 0
        
        n_row_spans = 0
        n_col_spans = 0
        empty_count = 0
        total_text_len = 0
        
        for cell in cells:
            end_row = cell.get('end_row', cell.get('start_row', 0))
            end_col = cell.get('end_col', cell.get('start_col', 0))
            
            max_row = max(max_row, end_row + 1)
            max_col = max(max_col, end_col + 1)
            
            # Count spans
            row_span = end_row - cell.get('start_row', 0) + 1
            col_span = end_col - cell.get('start_col', 0) + 1
            
            if row_span > 1:
                n_row_spans += 1
            if col_span > 1:
                n_col_spans += 1
            
            # Analyze content
            content = extract_text_content(cell)
            if not content or content.strip() == '':
                empty_count += 1
            total_text_len += len(content)
        
        n_cells = len(cells)
        n_spans = n_row_spans + n_col_spans
        sparsity_ratio = empty_count / n_cells if n_cells > 0 else 0.0
        text_density_proxy = total_text_len / n_cells if n_cells > 0 else 0.0
        
        return {
            'n_rows': max_row,
            'n_cols': max_col,
            'n_cells': n_cells,
            'n_spans': n_spans,
            'n_row_spans': n_row_spans,
            'n_col_spans': n_col_spans,
            'sparsity_ratio': round(sparsity_ratio, 4),
            'text_density_proxy': round(text_density_proxy, 2)
        }


class StratifiedSampler:
    """
    Stratified sampling to ensure balanced representation across table types
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize sampler
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
    
    def sample(
        self,
        tables: List[Dict[str, Any]],
        per_type: int = 40,
        priority_order: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Stratified sampling by table type
        
        Args:
            tables: List of table dicts with 'type_labels' field
            per_type: Number of samples per type category
            priority_order: Optional priority order for multi-label resolution
                           Default: rarer types first
        
        Returns:
            Sampled subset of tables
        """
        if not tables:
            return []
        
        # Default priority: rarer types first
        if priority_order is None:
            # Count type frequencies
            type_counts = defaultdict(int)
            for table in tables:
                for label in table.get('type_labels', []):
                    type_counts[label] += 1
            
            # Sort by count (ascending) - rarer types have higher priority
            priority_order = sorted(type_counts.keys(), key=lambda t: type_counts[t])
        
        # Group tables by primary type (first type in priority order they belong to)
        type_to_tables = defaultdict(list)
        
        for table in tables:
            table_types = set(table.get('type_labels', []))
            
            # Assign to highest priority type
            assigned = False
            for type_label in priority_order:
                if type_label in table_types:
                    type_to_tables[type_label].append(table)
                    assigned = True
                    break
            
            # Fallback: assign to first type
            if not assigned and table_types:
                type_to_tables[list(table_types)[0]].append(table)
        
        # Sample from each type
        sampled = []
        
        for type_label in TableTypeClassifier.TYPE_LABELS:
            type_tables = type_to_tables.get(type_label, [])
            
            if not type_tables:
                logger.warning(f"No tables found for type: {type_label}")
                continue
            
            # Sample up to per_type
            n_sample = min(per_type, len(type_tables))
            type_sample = random.sample(type_tables, n_sample)
            
            logger.info(f"Type '{type_label}': sampled {n_sample}/{len(type_tables)} tables")
            sampled.extend(type_sample)
        
        # Deduplicate (in case of overlaps)
        seen_ids = set()
        unique_sampled = []
        
        for table in sampled:
            table_id = table.get('table_id')
            if table_id not in seen_ids:
                seen_ids.add(table_id)
                unique_sampled.append(table)
        
        logger.info(f"Total sampled: {len(unique_sampled)} unique tables")
        return unique_sampled

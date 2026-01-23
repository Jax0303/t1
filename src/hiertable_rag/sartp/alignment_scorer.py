"""
Header-Data Alignment Scorer for SARTP
Computes semantic consistency scores between headers and data cells.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class HeaderDataAligner:
    """
    Computes semantic alignment scores between header and data cells.
    
    Uses embeddings from SemanticEncoder to measure how well data cells
    match their column headers semantically.
    """
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize the aligner.
        
        Args:
            similarity_threshold: Minimum cosine similarity for considering
                                a header-data pair as aligned
        """
        self.similarity_threshold = similarity_threshold
    
    def score(
        self,
        cells: List[Dict[str, Any]],
        embeddings: torch.Tensor,
        header_mask: List[bool]
    ) -> Dict[str, Any]:
        """
        Compute alignment scores for a table.
        
        Args:
            cells: List of cell dicts with 'row_idx', 'col_idx', 'content'
            embeddings: Semantic embeddings [N_cells, embedding_dim]
            header_mask: Boolean mask [N_cells] indicating which cells are headers
            
        Returns:
            Dictionary containing:
                - 'col_alignment': Per-column header-data alignment scores
                - 'row_coherence': Semantic coherence within each row
                - 'global_score': Overall alignment quality
                - 'col_groups': Cell indices grouped by column
        """
        # Group cells by column
        col_groups = self._group_by_column(cells)
        
        # Compute column-wise alignment
        col_alignment = {}
        col_mean_similarity = []
        
        for col_idx, cell_indices in col_groups.items():
            # Separate header and data cells in this column
            header_indices = [i for i in cell_indices if header_mask[i]]
            data_indices = [i for i in cell_indices if not header_mask[i]]
            
            if not header_indices or not data_indices:
                # No alignment possible without both headers and data
                col_alignment[col_idx] = {
                    'mean_similarity': 0.0,
                    'max_similarity': 0.0,
                    'aligned_count': 0
                }
                continue
            
            # Get embeddings
            header_embs = embeddings[header_indices]
            data_embs = embeddings[data_indices]
            
            # Compute similarity matrix [N_headers, N_data]
            similarity = self._cosine_similarity(header_embs, data_embs)
            
            # Aggregate: use the strongest header for each data cell
            max_sim_per_data = similarity.max(dim=0).values  # [N_data]
            
            col_alignment[col_idx] = {
                'mean_similarity': max_sim_per_data.mean().item(),
                'max_similarity': max_sim_per_data.max().item(),
                'min_similarity': max_sim_per_data.min().item(),
                'aligned_count': (max_sim_per_data > self.similarity_threshold).sum().item(),
                'total_data_cells': len(data_indices)
            }
            
            col_mean_similarity.append(max_sim_per_data.mean().item())
        
        # Compute row-wise coherence (semantic consistency within rows)
        row_coherence = self._compute_row_coherence(cells, embeddings, header_mask)
        
        # Global score: weighted average of column alignments
        global_score = np.mean(col_mean_similarity) if col_mean_similarity else 0.0
        
        return {
            'col_alignment': col_alignment,
            'row_coherence': row_coherence,
            'global_score': global_score,
            'col_groups': col_groups
        }
    
    def _group_by_column(self, cells: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """Group cell indices by column."""
        col_groups = defaultdict(list)
        for i, cell in enumerate(cells):
            col_idx = cell.get('col_idx', 0)
            col_groups[col_idx].append(i)
        return dict(col_groups)
    
    def _group_by_row(self, cells: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """Group cell indices by row."""
        row_groups = defaultdict(list)
        for i, cell in enumerate(cells):
            row_idx = cell.get('row_idx', 0)
            row_groups[row_idx].append(i)
        return dict(row_groups)
    
    def _cosine_similarity(
        self, 
        embeddings1: torch.Tensor, 
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: [N, dim]
            embeddings2: [M, dim]
            
        Returns:
            similarity: [N, M]
        """
        # Normalize
        emb1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        emb2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        
        # Cosine similarity
        return torch.mm(emb1_norm, emb2_norm.t())
    
    def _compute_row_coherence(
        self,
        cells: List[Dict[str, Any]],
        embeddings: torch.Tensor,
        header_mask: List[bool]
    ) -> Dict[int, float]:
        """
        Compute semantic coherence within each row.
        
        Coherence = average pairwise similarity of non-header cells in the row.
        High coherence suggests cells are semantically related.
        
        Returns:
            Dictionary mapping row_idx to coherence score
        """
        row_groups = self._group_by_row(cells)
        row_coherence = {}
        
        for row_idx, cell_indices in row_groups.items():
            # Only look at data cells (exclude headers)
            data_indices = [i for i in cell_indices if not header_mask[i]]
            
            if len(data_indices) < 2:
                # Need at least 2 cells for coherence
                row_coherence[row_idx] = 0.0
                continue
            
            # Get embeddings for this row
            row_embs = embeddings[data_indices]
            
            # Compute pairwise similarity
            similarity = self._cosine_similarity(row_embs, row_embs)
            
            # Average of upper triangle (exclude diagonal)
            n = len(data_indices)
            triu_indices = torch.triu_indices(n, n, offset=1)
            coherence = similarity[triu_indices[0], triu_indices[1]].mean().item()
            
            row_coherence[row_idx] = coherence
        
        return row_coherence
    
    def get_misaligned_cells(
        self,
        cells: List[Dict[str, Any]],
        alignment_scores: Dict[str, Any]
    ) -> List[int]:
        """
        Identify cells that are poorly aligned with their column headers.
        
        Args:
            cells: List of cell dicts
            alignment_scores: Output from score() method
            
        Returns:
            List of cell indices that may be misaligned
        """
        misaligned = []
        col_alignment = alignment_scores['col_alignment']
        
        for col_idx, scores in col_alignment.items():
            # If column has low alignment, all its data cells are suspect
            if scores['mean_similarity'] < self.similarity_threshold:
                col_groups = alignment_scores['col_groups']
                if col_idx in col_groups:
                    misaligned.extend(col_groups[col_idx])
        
        return misaligned
    
    def suggest_column_reassignment(
        self,
        cells: List[Dict[str, Any]],
        embeddings: torch.Tensor,
        header_mask: List[bool],
        cell_idx: int
    ) -> int:
        """
        Suggest which column a misaligned cell should belong to.
        
        Args:
            cells: List of all cells
            embeddings: Cell embeddings
            header_mask: Header/data classification
            cell_idx: Index of the cell to reassign
            
        Returns:
            Suggested column index
        """
        # Get all header cells
        header_indices = [i for i, is_header in enumerate(header_mask) if is_header]
        
        if not header_indices:
            # No headers, can't reassign
            return cells[cell_idx].get('col_idx', 0)
        
        # Compute similarity to all headers
        cell_emb = embeddings[cell_idx].unsqueeze(0)
        header_embs = embeddings[header_indices]
        
        similarity = self._cosine_similarity(cell_emb, header_embs)
        
        # Find header with highest similarity
        best_header_idx = header_indices[similarity.argmax().item()]
        suggested_col = cells[best_header_idx].get('col_idx', 0)
        
        return suggested_col

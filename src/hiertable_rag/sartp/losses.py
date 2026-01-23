"""
SARTP Loss Functions for End-to-End Training
Implements multi-task loss combining visual, semantic, and structure objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class SARTPLoss(nn.Module):
    """
    Multi-task loss for SARTP training.
    
    Combines three objectives:
    1. Visual boundary loss: Binary cross-entropy for row/col predictions
    2. Semantic alignment loss: Contrastive learning for header-data pairs
    3. Structure consistency loss: TEDS-based cell matching
    """
    
    def __init__(
        self,
        w_boundary: float = 1.0,
        w_alignment: float = 0.5,
        w_structure: float = 0.3,
        margin: float = 0.3
    ):
        """
        Initialize SARTP loss.
        
        Args:
            w_boundary: Weight for visual boundary loss
            w_alignment: Weight for semantic alignment loss
            w_structure: Weight for structure consistency loss
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.w_boundary = w_boundary
        self.w_alignment = w_alignment
        self.w_structure = w_structure
        self.margin = margin
        
        logger.info(f"SARTP Loss initialized: "
                   f"w_boundary={w_boundary}, w_alignment={w_alignment}, w_structure={w_structure}")
    
    def forward(
        self,
        pred: Dict[str, Any],
        target: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SARTP loss.
        
        Args:
            pred: Model predictions with keys:
                - 'row_probs': [B, H] row boundary probabilities
                - 'col_probs': [B, W] col boundary probabilities
                - 'header_embeddings': [N_headers, D] header cell embeddings
                - 'data_embeddings': [N_data, D] data cell embeddings
                - 'cells': List of predicted cells
            target: Ground truth with keys:
                - 'row_gt': [B, H] row boundary labels
                - 'col_gt': [B, W] col boundary labels
                - 'header_data_pairs': List of (header_idx, data_idx, is_match) tuples
                - 'cells': List of ground truth cells
        
        Returns:
            Dictionary with 'total_loss' and individual loss components
        """
        losses = {}
        
        # L1: Visual boundary loss
        if 'row_probs' in pred and 'row_gt' in target:
            row_loss = F.binary_cross_entropy(
                pred['row_probs'],
                target['row_gt'],
                reduction='mean'
            )
            col_loss = F.binary_cross_entropy(
                pred['col_probs'],
                target['col_gt'],
                reduction='mean'
            )
            losses['boundary_loss'] = (row_loss + col_loss) / 2.0
        else:
            losses['boundary_loss'] = torch.tensor(0.0, device=pred['row_probs'].device)
        
        # L2: Semantic alignment loss (contrastive)
        if ('header_embeddings' in pred and 'data_embeddings' in pred and 
            'header_data_pairs' in target):
            losses['alignment_loss'] = self.contrastive_loss(
                pred['header_embeddings'],
                pred['data_embeddings'],
                target['header_data_pairs']
            )
        else:
            losses['alignment_loss'] = torch.tensor(0.0, device=pred['row_probs'].device)
        
        # L3: Structure consistency loss (TEDS-based)
        if 'cells' in pred and 'cells' in target:
            losses['structure_loss'] = self.structure_loss(
                pred['cells'],
                target['cells']
            )
        else:
            losses['structure_loss'] = torch.tensor(0.0, device=pred['row_probs'].device)
        
        # Total loss
        losses['total_loss'] = (
            self.w_boundary * losses['boundary_loss'] +
            self.w_alignment * losses['alignment_loss'] +
            self.w_structure * losses['structure_loss']
        )
        
        return losses
    
    def contrastive_loss(
        self,
        header_embeddings: torch.Tensor,
        data_embeddings: torch.Tensor,
        pairs: List[tuple]
    ) -> torch.Tensor:
        """
        Contrastive loss for header-data semantic alignment.
        
        Positive pairs (same column): minimize distance
        Negative pairs (different column): maximize distance (with margin)
        
        Args:
            header_embeddings: [N_headers, D]
            data_embeddings: [N_data, D]
            pairs: List of (header_idx, data_idx, is_match) tuples
                   is_match=1 for same column, 0 otherwise
        """
        if len(pairs) == 0:
            return torch.tensor(0.0, device=header_embeddings.device)
        
        loss = 0.0
        count = 0
        
        for header_idx, data_idx, is_match in pairs:
            # Skip if indices out of bounds
            if header_idx >= len(header_embeddings) or data_idx >= len(data_embeddings):
                continue
            
            h_emb = header_embeddings[header_idx]
            d_emb = data_embeddings[data_idx]
            
            # Cosine distance (1 - cosine_similarity)
            distance = 1.0 - F.cosine_similarity(h_emb.unsqueeze(0), d_emb.unsqueeze(0))
            
            if is_match == 1:
                # Positive pair: minimize distance
                loss += distance
            else:
                # Negative pair: maximize distance (with margin)
                loss += F.relu(self.margin - distance)
            
            count += 1
        
        return loss / count if count > 0 else torch.tensor(0.0, device=header_embeddings.device)
    
    def structure_loss(
        self,
        pred_cells: List[Dict[str, Any]],
        gt_cells: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Structure consistency loss based on cell positions.
        
        Uses soft matching: weighted by IoU of cell bboxes.
        """
        if len(pred_cells) == 0 or len(gt_cells) == 0:
            return torch.tensor(1.0)  # Max penalty for empty prediction
        
        # Compute assignment matrix
        n_pred = len(pred_cells)
        n_gt = len(gt_cells)
        
        # Simple approach: penalize count mismatch + position errors
        count_penalty = abs(n_pred - n_gt) / max(n_pred, n_gt)
        
        # Position matching: for each predicted cell, find closest GT cell
        position_errors = 0.0
        for pred_cell in pred_cells:
            pred_row = pred_cell.get('row_idx', 0)
            pred_col = pred_cell.get('col_idx', 0)
            
            # Find closest GT cell (Manhattan distance)
            min_dist = float('inf')
            for gt_cell in gt_cells:
                gt_row = gt_cell.get('row_idx', 0)
                gt_col = gt_cell.get('col_idx', 0)
                dist = abs(pred_row - gt_row) + abs(pred_col - gt_col)
                min_dist = min(min_dist, dist)
            
            position_errors += min_dist
        
        position_penalty = position_errors / n_pred if n_pred > 0 else 1.0
        
        # Normalize to [0, 1]
        structure_loss = (count_penalty + position_penalty) / 2.0
        
        return torch.tensor(structure_loss, dtype=torch.float32)


class LearnableWeights(nn.Module):
    """
    Learnable weights (α, β, γ) for SARTP fusion.
    
    Allows end-to-end learning of visual/semantic/layout balance.
    """
    
    def __init__(self, init_alpha=0.5, init_beta=0.3, init_gamma=0.2):
        """
        Initialize learnable weights.
        
        Args:
            init_alpha: Initial visual weight
            init_beta: Initial semantic weight
            init_gamma: Initial layout weight
        """
        super().__init__()
        
        # Use softmax to ensure weights sum to 1
        self.logits = nn.Parameter(torch.tensor([init_alpha, init_beta, init_gamma]))
    
    def forward(self):
        """Get normalized weights."""
        weights = F.softmax(self.logits, dim=0)
        return {
            'alpha': weights[0],
            'beta': weights[1],
            'gamma': weights[2]
        }
    
    def get_weights_dict(self):
        """Get weights as a dictionary (for logging)."""
        with torch.no_grad():
            weights = F.softmax(self.logits, dim=0)
            return {
                'alpha': weights[0].item(),
                'beta': weights[1].item(),
                'gamma': weights[2].item()
            }


def create_header_data_pairs(
    cells: List[Dict[str, Any]],
    header_mask: List[bool]
) -> List[tuple]:
    """
    Create header-data pairs for contrastive learning.
    
    Args:
        cells: List of cells
        header_mask: Boolean mask indicating headers
    
    Returns:
        List of (header_idx, data_idx, is_match) tuples
    """
    pairs = []
    
    # Get header and data indices
    header_indices = [i for i, is_header in enumerate(header_mask) if is_header]
    data_indices = [i for i, is_header in enumerate(header_mask) if not is_header]
    
    # For each header, create positive and negative pairs
    for h_idx in header_indices:
        h_col = cells[h_idx].get('col_idx', 0)
        
        for d_idx in data_indices:
            d_col = cells[d_idx].get('col_idx', 0)
            
            # Same column = positive, different = negative
            is_match = 1 if h_col == d_col else 0
            pairs.append((h_idx, d_idx, is_match))
    
    return pairs

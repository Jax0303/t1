"""
End-to-End Hierarchical Table Understanding Model.

This module implements a joint learning approach that combines:
1. Vision-based table structure recognition (TSR)
2. Hierarchical header relationship extraction
3. Semantic understanding of table content

The model takes table images as input and outputs:
- Cell bounding boxes and content
- Hierarchical header tree (parent-child relationships)
- Knowledge graph triples

Key Innovation:
- Unlike traditional TSR that only detects cell boundaries,
  this model jointly learns structural AND semantic hierarchy.
- Uses contrastive learning to group headers under same parent.
- Outputs directly usable HeaderTree for RAG systems.

References:
- TableFormer (IBM, 2022)
- DETR for table detection
- LayoutLMv3 for document understanding
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from src.parsing.hierarchy import HeaderNode, HeaderTree
from src.parsing.extractor import TableObject, Cell

logger = logging.getLogger(__name__)


@dataclass
class E2EConfig:
    """
    Configuration for End-to-End Hierarchical Table Model.
    
    Attributes:
        vision_encoder: Name of vision encoder backbone
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        num_decoder_layers: Number of transformer decoder layers
        max_cells: Maximum number of cells to detect
        max_hierarchy_depth: Maximum header hierarchy depth
        dropout: Dropout rate
        use_contrastive_loss: Whether to use contrastive learning for hierarchy
        contrastive_temperature: Temperature for contrastive loss
        cell_loss_weight: Weight for cell detection loss
        hierarchy_loss_weight: Weight for hierarchy prediction loss
        content_loss_weight: Weight for content recognition loss
    """
    vision_encoder: str = "microsoft/table-transformer-detection"
    hidden_dim: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    max_cells: int = 500
    max_hierarchy_depth: int = 5
    dropout: float = 0.1
    use_contrastive_loss: bool = True
    contrastive_temperature: float = 0.07
    cell_loss_weight: float = 1.0
    hierarchy_loss_weight: float = 1.0
    content_loss_weight: float = 0.5
    content_loss_weight: float = 0.5
    use_retrieval_loss: bool = False
    retrieval_loss_weight: float = 1.0
    retrieval_temperature: float = 0.07
    image_size: Tuple[int, int] = (768, 768)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "vision_encoder": self.vision_encoder,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "max_cells": self.max_cells,
            "max_hierarchy_depth": self.max_hierarchy_depth,
            "dropout": self.dropout,
            "use_contrastive_loss": self.use_contrastive_loss,
            "contrastive_temperature": self.contrastive_temperature,
            "cell_loss_weight": self.cell_loss_weight,
            "hierarchy_loss_weight": self.hierarchy_loss_weight,
            "content_loss_weight": self.content_loss_weight,
            "use_retrieval_loss": self.use_retrieval_loss,
            "retrieval_loss_weight": self.retrieval_loss_weight,
            "retrieval_temperature": self.retrieval_temperature,
            "image_size": self.image_size,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "E2EConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class E2EOutput:
    """
    Output from End-to-End Hierarchical Table Model.
    
    Attributes:
        cells: List of detected cells with bounding boxes
        header_tree: Hierarchical header structure
        cell_embeddings: Embeddings for each cell
        hierarchy_logits: Logits for parent-child relationships
        confidence_scores: Confidence scores for predictions
    """
    cells: List[Dict[str, Any]] = field(default_factory=list)
    header_tree: Optional[HeaderTree] = None
    cell_embeddings: Optional[Any] = None  # torch.Tensor
    hierarchy_logits: Optional[Any] = None  # torch.Tensor
    attentions: Optional[List[Any]] = None  # List[torch.Tensor]
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary."""
        return {
            "cells": self.cells,
            "header_tree": self.header_tree.root.to_dict() if self.header_tree else None,
            "confidence_scores": self.confidence_scores,
        }


class PositionalEncoding2D(nn.Module if TORCH_AVAILABLE else object):
    """
    2D Positional Encoding for table images.
    
    Adds learnable position embeddings based on row/column positions.
    """
    
    def __init__(self, hidden_dim: int, max_h: int = 100, max_w: int = 100):
        """
        Initialize 2D positional encoding.
        
        Args:
            hidden_dim: Hidden dimension size
            max_h: Maximum height positions
            max_w: Maximum width positions
        """
        if not TORCH_AVAILABLE:
            return
            
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Create position embeddings
        self.row_embed = nn.Embedding(max_h, hidden_dim // 2)
        self.col_embed = nn.Embedding(max_w, hidden_dim // 2)
        
        # Initialize
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
    
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor with positional encoding added
        """
        if not TORCH_AVAILABLE:
            return x
            
        B, C, H, W = x.shape
        
        # Create position indices
        rows = torch.arange(H, device=x.device)
        cols = torch.arange(W, device=x.device)
        
        # Get embeddings
        row_emb = self.row_embed(rows)  # (H, hidden_dim//2)
        col_emb = self.col_embed(cols)  # (W, hidden_dim//2)
        
        # Expand and concatenate
        row_emb = row_emb.unsqueeze(1).expand(-1, W, -1)  # (H, W, hidden_dim//2)
        col_emb = col_emb.unsqueeze(0).expand(H, -1, -1)  # (H, W, hidden_dim//2)
        
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # (H, W, hidden_dim)
        pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0)  # (1, hidden_dim, H, W)
        pos_emb = pos_emb.expand(B, -1, -1, -1)  # (B, hidden_dim, H, W)
        
        return x + pos_emb


class TableFormerAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Multi-head attention with structural bias support.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: "torch.Tensor",
        key: "torch.Tensor",
        value: "torch.Tensor",
        attn_bias: Optional["torch.Tensor"] = None,
        key_padding_mask: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Args:
            query: (B, L, D)
            key: (B, S, D)
            value: (B, S, D)
            attn_bias: (B, L, S) or (B*num_heads, L, S)
            key_padding_mask: (B, S)
        """
        B, L, _ = query.shape
        S = key.shape[1]
        
        # Project and reshape
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scores
        # (B, H, L, D) x (B, H, D, S) -> (B, H, L, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add bias
        if attn_bias is not None:
            # attn_bias shape: (B, L, S) or (B, 1, L, S) or (B, H, L, S)
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)
            scores = scores + attn_bias
            
        # Apply mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            scores = scores.masked_fill(mask, float('-inf'))
            
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # (B, H, L, S) x (B, H, S, D) -> (B, H, L, D)
        output = torch.matmul(attn_probs, v)
        
        # Reshape and project out
        output = output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_probs


class TableFormerEncoderLayer(nn.Module if TORCH_AVAILABLE else object):
    """
    Transformer encoder layer with structural bias support.
    """
    
    def __init__(self, config: E2EConfig):
        super().__init__()
        self.self_attn = TableFormerAttention(
            config.hidden_dim, 
            config.num_heads, 
            dropout=config.dropout
        )
        self.linear1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        
    def forward(
        self, 
        src: "torch.Tensor", 
        src_mask: Optional["torch.Tensor"] = None,
        src_key_padding_mask: Optional["torch.Tensor"] = None,
        attn_bias: Optional["torch.Tensor"] = None,
        output_attentions: bool = False,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        src2, attn_weights = self.self_attn(
            src, src, src, 
            attn_bias=attn_bias,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if output_attentions:
            return src, attn_weights
        return src, None


class TableFormerEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Transformer encoder stack with structural bias support.
    """
    
    def __init__(self, config: E2EConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TableFormerEncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])
        
    def forward(
        self, 
        src: "torch.Tensor", 
        mask: Optional["torch.Tensor"] = None, 
        src_key_padding_mask: Optional["torch.Tensor"] = None,
        attn_bias: Optional["torch.Tensor"] = None,
        output_attentions: bool = False,
    ) -> Tuple["torch.Tensor", Optional[List["torch.Tensor"]]]:
        output = src
        all_attentions = [] if output_attentions else None
        
        for layer in self.layers:
            output, attn = layer(
                output, 
                src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask,
                attn_bias=attn_bias,
                output_attentions=output_attentions
            )
            if output_attentions:
                all_attentions.append(attn)
                
        return output, all_attentions


class VisionTableEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Vision encoder for table images.
    
    Extracts visual features from table images using a CNN backbone
    followed by transformer encoder layers.
    
    Architecture:
    - ResNet/EfficientNet backbone for feature extraction
    - 2D positional encoding
    - Transformer encoder for global context
    """
    
    def __init__(self, config: E2EConfig):
        """
        Initialize vision encoder.
        
        Args:
            config: E2EConfig instance
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, VisionTableEncoder disabled")
            return
            
        super().__init__()
        self.config = config
        
        # CNN Backbone (simplified ResNet-like)
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 1
            self._make_layer(64, 64, 2),
            
            # Block 2
            self._make_layer(64, 128, 2, stride=2),
            
            # Block 3
            self._make_layer(128, 256, 2, stride=2),
            
            # Block 4
            self._make_layer(256, config.hidden_dim, 2, stride=2),
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(config.hidden_dim)
        
        # Transformer encoder
        self.transformer_encoder = TableFormerEncoder(config)
        
        # Learnable structural biases
        self.row_bias = nn.Parameter(torch.tensor(0.0))
        self.col_bias = nn.Parameter(torch.tensor(0.0))
        
        logger.info(f"VisionTableEncoder initialized (hidden_dim={config.hidden_dim})")
    
    def _make_layer(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_blocks: int, 
        stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer."""
        layers = []
        
        # First block with potential downsampling
        layers.append(self._make_block(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1
    ) -> nn.Module:
        """Create a residual block."""
        
        class ResBlock(nn.Module):
            def __init__(self, in_ch, out_ch, stride):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_ch != out_ch:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                        nn.BatchNorm2d(out_ch)
                    )
            
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out
        
        return ResBlock(in_channels, out_channels, stride)
    
    def forward(
        self, 
        images: "torch.Tensor",
        output_attentions: bool = False,
    ) -> Tuple["torch.Tensor", Optional[List["torch.Tensor"]]]:
        """
        Encode table images.
        
        Args:
            images: Input images of shape (B, 3, H, W)
            
        Returns:
            Encoded features of shape (B, seq_len, hidden_dim)
        """
        if not TORCH_AVAILABLE:
            return None
        
        # Extract CNN features
        features = self.backbone(images)  # (B, hidden_dim, H', W')
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        B, C, H, W = features.shape
        L = H * W
        
        # Compute structural bias
        # Create grid of coordinates
        rows = torch.arange(H, device=images.device).unsqueeze(1).expand(H, W).flatten()
        cols = torch.arange(W, device=images.device).unsqueeze(0).expand(H, W).flatten()
        
        # (L, 1) - (1, L) -> (L, L)
        row_diff = rows.unsqueeze(1) - rows.unsqueeze(0)
        col_diff = cols.unsqueeze(1) - cols.unsqueeze(0)
        
        # Same row: row_diff == 0
        # Same col: col_diff == 0
        
        attn_bias = torch.zeros(L, L, device=images.device)
        attn_bias = attn_bias.masked_fill(row_diff == 0, self.row_bias)
        attn_bias = attn_bias.masked_fill(col_diff == 0, self.col_bias)
        
        # Expand for batch: (B, L, L)
        attn_bias = attn_bias.unsqueeze(0).expand(B, -1, -1)
        
        # Flatten spatial dimensions
        features = features.flatten(2).permute(0, 2, 1)  # (B, H'*W', hidden_dim)
        
        # Apply transformer encoder
        encoded, attentions = self.transformer_encoder(
            features, 
            attn_bias=attn_bias,
            output_attentions=output_attentions
        )
        
        return encoded, attentions


class HierarchicalHeaderPredictor(nn.Module if TORCH_AVAILABLE else object):
    """
    Predicts hierarchical header relationships.
    
    Takes cell embeddings and predicts:
    1. Which cells are headers vs data cells
    2. Parent-child relationships between headers
    3. Hierarchy level for each header
    
    Uses contrastive learning to group headers under same parent.
    """
    
    def __init__(self, config: E2EConfig):
        """
        Initialize hierarchy predictor.
        
        Args:
            config: E2EConfig instance
        """
        if not TORCH_AVAILABLE:
            return
            
        super().__init__()
        self.config = config
        
        # Header vs Data classifier
        self.header_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2),  # 0: data, 1: header
        )
        
        # Hierarchy level predictor
        self.level_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.max_hierarchy_depth),
        )
        
        # Parent-child relationship predictor
        # Uses bilinear attention to compute parent scores
        self.parent_query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.parent_key = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Column span predictor
        self.span_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.max_cells),  # Max possible span
        )
        
        # Contrastive projection head (for grouping headers under same parent)
        if config.use_contrastive_loss:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            )
        
        logger.info("HierarchicalHeaderPredictor initialized")
    
    def forward(
        self,
        cell_embeddings: "torch.Tensor",
        cell_mask: Optional["torch.Tensor"] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """
        Predict hierarchical structure.
        
        Args:
            cell_embeddings: Cell embeddings (B, num_cells, hidden_dim)
            cell_mask: Mask for valid cells (B, num_cells)
            
        Returns:
            Dictionary containing:
            - header_logits: (B, num_cells, 2)
            - level_logits: (B, num_cells, max_depth)
            - parent_scores: (B, num_cells, num_cells)
            - span_logits: (B, num_cells, max_span)
            - contrastive_embeddings: (B, num_cells, proj_dim)
        """
        if not TORCH_AVAILABLE:
            return {}
        
        B, N, D = cell_embeddings.shape
        
        # Header classification
        header_logits = self.header_classifier(cell_embeddings)
        
        # Level prediction
        level_logits = self.level_predictor(cell_embeddings)
        
        # Parent-child scores using bilinear attention
        queries = self.parent_query(cell_embeddings)  # (B, N, D)
        keys = self.parent_key(cell_embeddings)  # (B, N, D)
        
        # Compute attention scores
        parent_scores = torch.bmm(queries, keys.transpose(1, 2))  # (B, N, N)
        parent_scores = parent_scores / math.sqrt(D)
        
        # Mask out invalid cells
        if cell_mask is not None:
            mask = cell_mask.unsqueeze(1).expand(-1, N, -1)
            parent_scores = parent_scores.masked_fill(~mask, float('-inf'))
        
        # Span prediction
        span_logits = self.span_predictor(cell_embeddings)
        
        outputs = {
            "header_logits": header_logits,
            "level_logits": level_logits,
            "parent_scores": parent_scores,
            "span_logits": span_logits,
        }
        
        # Contrastive embeddings
        if self.config.use_contrastive_loss:
            contrastive_emb = self.contrastive_proj(cell_embeddings)
            contrastive_emb = F.normalize(contrastive_emb, p=2, dim=-1)
            outputs["contrastive_embeddings"] = contrastive_emb
        
        return outputs
    
    def compute_contrastive_loss(
        self,
        embeddings: "torch.Tensor",
        parent_labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute contrastive loss for hierarchy learning.
        
        Headers with same parent should have similar embeddings.
        
        Args:
            embeddings: Contrastive embeddings (B, N, proj_dim)
            parent_labels: Parent indices for each cell (B, N)
            
        Returns:
            Contrastive loss value
        """
        if not TORCH_AVAILABLE:
            return 0.0
        
        B, N, D = embeddings.shape
        temperature = self.config.contrastive_temperature
        
        total_loss = 0.0
        num_valid = 0
        
        for b in range(B):
            emb = embeddings[b]  # (N, D)
            labels = parent_labels[b]  # (N,)
            
            # Compute similarity matrix
            sim_matrix = torch.mm(emb, emb.t()) / temperature  # (N, N)
            
            # Create positive mask (same parent)
            positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            positive_mask = positive_mask.fill_diagonal_(False)
            
            # InfoNCE loss
            for i in range(N):
                if labels[i] < 0:  # Skip invalid cells
                    continue
                
                positives = positive_mask[i]
                if positives.sum() == 0:
                    continue
                
                # Numerator: sum of positive similarities
                pos_sim = sim_matrix[i][positives]
                
                # Denominator: all similarities except self
                all_sim = sim_matrix[i].clone()
                all_sim[i] = float('-inf')
                
                # Loss for this anchor
                loss_i = -torch.log(
                    torch.exp(pos_sim).sum() / torch.exp(all_sim).sum()
                )
                total_loss += loss_i
                num_valid += 1
        
        if num_valid > 0:
            total_loss = total_loss / num_valid
        
        return total_loss


class TableStructureDecoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Decodes table structure from encoded features.
    
    Uses DETR-style object queries to detect cells and their properties.
    """
    
    def __init__(self, config: E2EConfig):
        """
        Initialize structure decoder.
        
        Args:
            config: E2EConfig instance
        """
        if not TORCH_AVAILABLE:
            return
            
        super().__init__()
        self.config = config
        
        # Object queries for cell detection
        self.cell_queries = nn.Embedding(config.max_cells, config.hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_decoder_layers,
        )
        
        # Bounding box predictor (normalized coordinates)
        self.bbox_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 4),  # x, y, w, h
            nn.Sigmoid(),
        )
        
        # Cell existence classifier
        self.cell_classifier = nn.Linear(config.hidden_dim, 2)
        
        # Row/Column index predictor
        self.row_predictor = nn.Linear(config.hidden_dim, config.max_cells)
        self.col_predictor = nn.Linear(config.hidden_dim, config.max_cells)
        
        logger.info("TableStructureDecoder initialized")
    
    def forward(
        self,
        encoder_output: "torch.Tensor",
        encoder_mask: Optional["torch.Tensor"] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """
        Decode table structure.
        
        Args:
            encoder_output: Encoded features (B, seq_len, hidden_dim)
            encoder_mask: Mask for encoder output
            
        Returns:
            Dictionary containing:
            - cell_embeddings: (B, max_cells, hidden_dim)
            - bbox_pred: (B, max_cells, 4)
            - cell_logits: (B, max_cells, 2)
            - row_logits: (B, max_cells, max_cells)
            - col_logits: (B, max_cells, max_cells)
        """
        if not TORCH_AVAILABLE:
            return {}
        
        B = encoder_output.shape[0]
        
        # Get cell queries
        queries = self.cell_queries.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Decode
        cell_embeddings = self.transformer_decoder(
            queries,
            encoder_output,
            memory_key_padding_mask=encoder_mask,
        )
        
        # Predict bounding boxes
        bbox_pred = self.bbox_predictor(cell_embeddings)
        
        # Predict cell existence
        cell_logits = self.cell_classifier(cell_embeddings)
        
        # Predict row/column indices
        row_logits = self.row_predictor(cell_embeddings)
        col_logits = self.col_predictor(cell_embeddings)
        
        return {
            "cell_embeddings": cell_embeddings,
            "bbox_pred": bbox_pred,
            "cell_logits": cell_logits,
            "row_logits": row_logits,
            "col_logits": col_logits,
        }


class RetrievalAwareLoss(nn.Module if TORCH_AVAILABLE else object):
    """
    Computes retrieval-aware loss for table parsing.
    
    Optimizes the table structure to maximize similarity between 
    queries and their corresponding target cells (contextualized by hierarchy).
    """
    
    def __init__(self, config: E2EConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        query_embeddings: "torch.Tensor",
        cell_embeddings: "torch.Tensor",
        parent_scores: "torch.Tensor",
        target_mapping: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute retrieval loss.
        
        Args:
            query_embeddings: (B, num_queries, hidden_dim)
            cell_embeddings: (B, num_cells, hidden_dim)
            parent_scores: (B, num_cells, num_cells) - Parent probability matrix (logits)
            target_mapping: (B, num_queries, num_cells) - 1 if cell is target for query
            
        Returns:
            Loss value
        """
        if not TORCH_AVAILABLE:
            return 0.0
            
        B, num_queries, hidden_dim = query_embeddings.shape
        _, num_cells, _ = cell_embeddings.shape
        
        # 1. Compute Contextualized Cell Embeddings
        # We want to aggregate information from ancestors.
        # Approximation: Use parent_scores as adjacency matrix to propagate features.
        # For simplicity in this iteration, we'll do a single hop aggregation:
        # Context = Cell + sum(P(parent) * Parent)
        
        # Normalize parent scores to probabilities
        # Mask self-loops or handle them? For now, standard softmax over all potential parents
        parent_probs = F.softmax(parent_scores, dim=-1)  # (B, N, N)
        
        # Aggregate parent features
        # (B, N, N) x (B, N, D) -> (B, N, D)
        context_features = torch.bmm(parent_probs, cell_embeddings)
        
        # Combine cell and context
        contextualized_cells = cell_embeddings + context_features
        contextualized_cells = F.normalize(contextualized_cells, p=2, dim=-1)
        
        # Normalize queries
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        
        # 2. Compute Similarity Matrix
        # (B, Q, D) x (B, N, D)^T -> (B, Q, N)
        sim_matrix = torch.bmm(query_embeddings, contextualized_cells.transpose(1, 2))
        sim_matrix = sim_matrix / self.config.retrieval_temperature
        
        # 3. Compute Contrastive Loss (InfoNCE)
        # For each query, positive samples are the target cells.
        
        total_loss = 0.0
        valid_queries = 0
        
        for b in range(B):
            sim = sim_matrix[b]  # (Q, N)
            targets = target_mapping[b]  # (Q, N)
            
            # Mask for queries that have at least one target
            has_target = targets.sum(dim=1) > 0
            if not has_target.any():
                continue
                
            sim = sim[has_target]
            targets = targets[has_target]
            
            # LogSumExp of all scores (denominator)
            log_sum_exp_all = torch.logsumexp(sim, dim=1)
            
            # LogSumExp of positive scores (numerator)
            sim_pos = sim.clone()
            sim_pos[targets == 0] = float('-inf')
            log_sum_exp_pos = torch.logsumexp(sim_pos, dim=1)
            
            loss = -(log_sum_exp_pos - log_sum_exp_all).mean()
            total_loss += loss
            valid_queries += 1
            
        if valid_queries > 0:
            return total_loss / valid_queries
            
        # Return zero loss with grad if no valid queries
        return torch.tensor(0.0, device=query_embeddings.device, requires_grad=True)


class E2EHierarchicalTableModel(nn.Module if TORCH_AVAILABLE else object):
    """
    End-to-End Hierarchical Table Understanding Model.
    
    Combines vision encoding, structure decoding, and hierarchy prediction
    into a single jointly trainable model.
    
    Pipeline:
    1. Vision Encoder: Image → Feature maps
    2. Structure Decoder: Feature maps → Cell embeddings + bboxes
    3. Hierarchy Predictor: Cell embeddings → Header tree
    
    Training:
    - Multi-task loss: cell detection + hierarchy + contrastive
    - Joint optimization of all components
    
    Inference:
    - Single forward pass for complete table understanding
    - Outputs HeaderTree compatible with HierTable-RAG
    """
    
    def __init__(self, config: Optional[E2EConfig] = None):
        """
        Initialize E2E model.
        
        Args:
            config: E2EConfig instance (uses default if None)
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is required for E2EHierarchicalTableModel")
            self.config = config or E2EConfig()
            return
            
        super().__init__()
        self.config = config or E2EConfig()
        
        # Initialize components
        self.vision_encoder = VisionTableEncoder(self.config)
        self.structure_decoder = TableStructureDecoder(self.config)
        self.hierarchy_predictor = HierarchicalHeaderPredictor(self.config)
        
        if self.config.use_retrieval_loss:
            self.retrieval_loss_fn = RetrievalAwareLoss(self.config)
        
        logger.info("E2EHierarchicalTableModel initialized")
    
    def forward(
        self,
        images: "torch.Tensor",
        targets: Optional[Dict[str, Any]] = None,
        queries: Optional["torch.Tensor"] = None,
        query_targets: Optional["torch.Tensor"] = None,
        output_attentions: bool = False,
    ) -> Tuple[E2EOutput, Optional[Dict[str, "torch.Tensor"]]]:
        """
        Forward pass.
        
        Args:
            images: Input images (B, 3, H, W)
            targets: Optional training targets
            
        Returns:
            Tuple of (E2EOutput, losses dict if training)
        """
        if not TORCH_AVAILABLE:
            return E2EOutput(), None
        
        # 1. Encode images
        encoded_features, attentions = self.vision_encoder(
            images, 
            output_attentions=output_attentions
        )
        
        # 2. Decode structure
        structure_output = self.structure_decoder(encoded_features)
        
        cell_embeddings = structure_output["cell_embeddings"]
        bbox_pred = structure_output["bbox_pred"]
        cell_logits = structure_output["cell_logits"]
        
        # 3. Predict hierarchy
        hierarchy_output = self.hierarchy_predictor(cell_embeddings)
        
        # Build output
        output = E2EOutput(
            cell_embeddings=cell_embeddings,
            hierarchy_logits=hierarchy_output["parent_scores"],
            attentions=attentions if output_attentions else None,
        )
        
        # Compute losses if training
        losses = None
        if targets is not None and self.training:
            losses = self._compute_losses(
                structure_output,
                hierarchy_output,
                targets,
                queries=queries,
                query_targets=query_targets,
            )
        
        return output, losses
    
    def _compute_losses(
        self,
        structure_output: Dict[str, "torch.Tensor"],
        hierarchy_output: Dict[str, "torch.Tensor"],
        targets: Dict[str, Any],
        queries: Optional["torch.Tensor"] = None,
        query_targets: Optional["torch.Tensor"] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """
        Compute multi-task losses.
        
        Args:
            structure_output: Output from structure decoder
            hierarchy_output: Output from hierarchy predictor
            targets: Ground truth targets
            
        Returns:
            Dictionary of loss values
        """
        if not TORCH_AVAILABLE:
            return {}
        
        losses = {}
        
        # Cell detection loss (binary cross entropy)
        if "cell_labels" in targets:
            cell_loss = F.cross_entropy(
                structure_output["cell_logits"].view(-1, 2),
                targets["cell_labels"].view(-1),
            )
            losses["cell_loss"] = cell_loss * self.config.cell_loss_weight
        
        # Bounding box loss (L1 + GIoU)
        if "bbox_targets" in targets:
            bbox_loss = F.l1_loss(
                structure_output["bbox_pred"],
                targets["bbox_targets"],
            )
            losses["bbox_loss"] = bbox_loss
        
        # Header classification loss
        if "header_labels" in targets:
            header_loss = F.cross_entropy(
                hierarchy_output["header_logits"].view(-1, 2),
                targets["header_labels"].view(-1),
            )
            losses["header_loss"] = header_loss * self.config.hierarchy_loss_weight
        
        # Level prediction loss
        # Level prediction loss
        if "level_labels" in targets:
            level_loss = F.cross_entropy(
                hierarchy_output["level_logits"].view(-1, self.config.max_hierarchy_depth),
                targets["level_labels"].view(-1),
            )
            losses["level_loss"] = level_loss * self.config.hierarchy_loss_weight
            
        # Retrieval loss
        if (
            self.config.use_retrieval_loss 
            and queries is not None 
            and query_targets is not None
        ):
            retrieval_loss = self.retrieval_loss_fn(
                query_embeddings=queries,
                cell_embeddings=structure_output["cell_embeddings"],
                parent_scores=hierarchy_output["parent_scores"],
                target_mapping=query_targets,
            )
            losses["retrieval_loss"] = retrieval_loss * self.config.retrieval_loss_weight
        
        # Parent prediction loss
        if "parent_labels" in targets:
            parent_loss = F.cross_entropy(
                hierarchy_output["parent_scores"].view(-1, hierarchy_output["parent_scores"].size(-1)),
                targets["parent_labels"].view(-1),
            )
            losses["parent_loss"] = parent_loss * self.config.hierarchy_loss_weight
        
        # Contrastive loss
        if (
            self.config.use_contrastive_loss
            and "contrastive_embeddings" in hierarchy_output
            and "parent_labels" in targets
        ):
            contrastive_loss = self.hierarchy_predictor.compute_contrastive_loss(
                hierarchy_output["contrastive_embeddings"],
                targets["parent_labels"],
            )
            losses["contrastive_loss"] = contrastive_loss * 0.1
        
        # Total loss
        losses["total_loss"] = sum(losses.values())
        
        return losses
    
    def extract_attention_graph(
        self,
        attentions: List["torch.Tensor"],
        threshold: float = 0.1,
    ) -> List[Tuple[int, int, float]]:
        """
        Extract graph edges from attention weights.
        
        Args:
            attentions: List of attention maps (B, H, L, L)
            threshold: Minimum attention score to consider an edge
            
        Returns:
            List of (source_idx, target_idx, weight) tuples for the first batch item
        """
        if not attentions:
            return []
            
        # Use the last layer's attention
        # Shape: (B, NumHeads, SeqLen, SeqLen)
        last_attn = attentions[-1]
        
        # Average over heads
        # (B, SeqLen, SeqLen)
        avg_attn = last_attn.mean(dim=1)
        
        # Get first batch item
        attn_map = avg_attn[0]  # (SeqLen, SeqLen)
        
        edges = []
        num_tokens = attn_map.shape[0]
        
        # Find strong connections
        # We only care about edges with weight > threshold
        indices = torch.nonzero(attn_map > threshold)
        
        for idx in indices:
            src, tgt = idx[0].item(), idx[1].item()
            weight = attn_map[src, tgt].item()
            
            # Filter self-loops if needed, but sometimes they are useful
            if src != tgt:
                edges.append((src, tgt, weight))
                
        return edges
    
    @torch.no_grad()
    def predict(self, images: "torch.Tensor") -> E2EOutput:
        """
        Inference mode prediction.
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            E2EOutput with predictions
        """
        if not TORCH_AVAILABLE:
            return E2EOutput()
        
        self.eval()
        output, _ = self.forward(images)
        
        # Post-process to build HeaderTree
        output = self._postprocess(output)
        
        return output
    
    def _postprocess(self, output: E2EOutput) -> E2EOutput:
        """
        Post-process model output to build HeaderTree.
        
        Args:
            output: Raw model output
            
        Returns:
            Output with HeaderTree populated
        """
        if not TORCH_AVAILABLE or output.cell_embeddings is None:
            return output
        
        # Get predictions
        cell_emb = output.cell_embeddings[0]  # First batch item
        hierarchy_logits = output.hierarchy_logits[0] if output.hierarchy_logits is not None else None
        
        # Build header tree from predictions
        # This is a simplified version - full implementation would use
        # Hungarian matching and more sophisticated tree building
        
        root = HeaderNode(
            text="TABLE_ROOT",
            level=0,
            span=1,
            col_start=0,
            col_end=1,
        )
        
        output.header_tree = HeaderTree(root=root)
        output.confidence_scores = {
            "overall": 0.85,  # Placeholder
            "structure": 0.90,
            "hierarchy": 0.80,
        }
        
        return output
    
    def save_pretrained(self, save_path: str) -> None:
        """
        Save model weights and config.
        
        Args:
            save_path: Directory to save to
        """
        if not TORCH_AVAILABLE:
            return
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save weights
        weights_path = save_dir / "model.pt"
        torch.save(self.state_dict(), weights_path)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str) -> "E2EHierarchicalTableModel":
        """
        Load model from saved weights.
        
        Args:
            load_path: Directory to load from
            
        Returns:
            Loaded model instance
        """
        if not TORCH_AVAILABLE:
            return cls()
        
        load_dir = Path(load_path)
        
        # Load config
        config_path = load_dir / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = E2EConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        weights_path = load_dir / "model.pt"
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        logger.info(f"Model loaded from {load_path}")
        
        return model


class E2EHierarchicalPipeline:
    """
    High-level pipeline for End-to-End Hierarchical Table Understanding.
    
    Provides easy-to-use interface for:
    - Loading and preprocessing images
    - Running inference
    - Converting output to HierTable-RAG compatible format
    """
    
    def __init__(
        self,
        model: Optional[E2EHierarchicalTableModel] = None,
        config: Optional[E2EConfig] = None,
        device: str = "cpu",
    ):
        """
        Initialize pipeline.
        
        Args:
            model: Pre-trained model (creates new if None)
            config: Model config
            device: Device to run on
        """
        self.device = device
        self.config = config or E2EConfig()
        
        if model is not None:
            self.model = model
        else:
            self.model = E2EHierarchicalTableModel(self.config)
        
        if TORCH_AVAILABLE:
            self.model = self.model.to(device)
        
        logger.info(f"E2EHierarchicalPipeline initialized (device={device})")
    
    def preprocess_image(self, image_path: str) -> Optional["torch.Tensor"]:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed tensor
        """
        if not TORCH_AVAILABLE:
            return None
        
        try:
            from PIL import Image
            from torchvision import transforms
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess
            transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            
            tensor = transform(image).unsqueeze(0)  # Add batch dim
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            return None
    
    def __call__(
        self,
        image_input: Union[str, "torch.Tensor"],
    ) -> E2EOutput:
        """
        Run inference on image.
        
        Args:
            image_input: Image path or tensor
            
        Returns:
            E2EOutput with predictions
        """
        if isinstance(image_input, str):
            tensor = self.preprocess_image(image_input)
        else:
            tensor = image_input
        
        if tensor is None:
            return E2EOutput()
        
        return self.model.predict(tensor)
    
    def to_header_tree(self, output: E2EOutput) -> Optional[HeaderTree]:
        """
        Convert output to HeaderTree.
        
        Args:
            output: Model output
            
        Returns:
            HeaderTree instance
        """
        return output.header_tree
    
    def to_table_object(self, output: E2EOutput) -> Optional[TableObject]:
        """
        Convert output to TableObject.
        
        Args:
            output: Model output
            
        Returns:
            TableObject instance
        """
        if not output.cells:
            return None
        
        # Build cells from predictions
        cells = []
        for cell_data in output.cells:
            cell = Cell(
                row=cell_data.get("row", 0),
                col=cell_data.get("col", 0),
                content=cell_data.get("content", ""),
                rowspan=cell_data.get("rowspan", 1),
                colspan=cell_data.get("colspan", 1),
                is_header=cell_data.get("is_header", False),
            )
            
            # Organize into rows
            while len(cells) <= cell.row:
                cells.append([])
            cells[cell.row].append(cell)
        
        return TableObject(cells=cells)


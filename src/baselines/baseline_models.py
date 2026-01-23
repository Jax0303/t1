"""
Baseline Models for SARTP Comparison
Implements Vision-Only and Lightweight VLM baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Any


class VisionOnlyTSR(nn.Module):
    """
    Vision-Only Table Structure Recognition baseline.
    Same backbone as SARTP but WITHOUT semantic refinement.
    """
    
    def __init__(self, backbone_name: str = "resnet18", embed_dim: int = 512):
        super().__init__()
        
        # Same backbone as SARTP
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        
        # Transformer for global context (same as SARTP RCA)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Row prediction head
        self.row_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.row_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # Column prediction head
        self.col_pool = nn.AdaptiveAvgPool2d((1, None))
        self.col_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass using vision only.
        
        Args:
            x: Input image [B, 3, H, W]
        
        Returns:
            Dictionary with row/col probabilities
        """
        # Extract features
        features = self.backbone(x)[-1]  # [B, 512, H/32, W/32]
        B, C, Hf, Wf = features.shape
        
        # Global attention
        x_flat = features.flatten(2).transpose(1, 2)  # [B, Hf*Wf, C]
        x_att = self.transformer(x_flat)  # [B, Hf*Wf, C]
        features = x_att.transpose(1, 2).reshape(B, C, Hf, Wf)
        
        # Predict boundaries
        row_probs = self.row_head(self.row_pool(features)).squeeze(-1).squeeze(1)  # [B, Hf]
        col_probs = self.col_head(self.col_pool(features)).squeeze(-2).squeeze(1)  # [B, Wf]
        
        return {
            'row_probs': row_probs,
            'col_probs': col_probs
        }


class LightweightVLM(nn.Module):
    """
    Lightweight Vision-Language Model baseline.
    Fuses visual and text features via simple concatenation.
    Does NOT have explicit header-data alignment like SARTP.
    """
    
    def __init__(
        self, 
        backbone_name: str = "resnet18",
        text_encoder_name: str = "roberta-base",
        embed_dim: int = 512
    ):
        super().__init__()
        
        # Visual encoder (same as SARTP)
        self.visual_encoder = timm.create_model(backbone_name, pretrained=True, features_only=True)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Text encoder (frozen RoBERTa, same as SARTP)
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Fusion layer (simple concatenation, different from SARTP!)
        text_dim = 768  # RoBERTa hidden size
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + text_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Structure prediction heads
        self.row_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.col_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        image: torch.Tensor,
        text_input_ids: torch.Tensor = None,
        text_attention_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with vision and language.
        
        Args:
            image: Input image [B, 3, H, W]
            text_input_ids: Tokenized text [B, N, seq_len]
            text_attention_mask: Attention mask [B, N, seq_len]
        
        Returns:
            Dictionary with row/col probabilities
        """
        # Visual features
        visual_feats = self.visual_encoder(image)[-1]  # [B, 512, H/32, W/32]
        B, C, Hf, Wf = visual_feats.shape
        
        # Global attention on visual features
        v_flat = visual_feats.flatten(2).transpose(1, 2)  # [B, Hf*Wf, C]
        v_att = self.transformer(v_flat)  # [B, Hf*Wf, C]
        
        # Pool to get global visual representation
        visual_global = v_att.mean(dim=1)  # [B, C]
        
        # Text features (if provided)
        if text_input_ids is not None:
            # Flatten batch and num_cells for encoding
            B_text, N_cells, seq_len = text_input_ids.shape
            text_input_flat = text_input_ids.reshape(-1, seq_len)
            text_mask_flat = text_attention_mask.reshape(-1, seq_len)
            
            # Encode text
            with torch.no_grad():
                text_outputs = self.text_encoder(
                    input_ids=text_input_flat,
                    attention_mask=text_mask_flat
                )
                text_feats = text_outputs.last_hidden_state[:, 0, :]  # [B*N, 768] CLS token
            
            # Average over cells
            text_feats = text_feats.reshape(B_text, N_cells, -1).mean(dim=1)  # [B, 768]
            
            # Fuse visual and text (simple concatenation)
            fused = torch.cat([visual_global, text_feats], dim=1)  # [B, C+768]
            fused = self.fusion(fused)  # [B, C]
        else:
            fused = visual_global
        
        # Predict row/col boundaries
        # (For simplicity, use global features to predict all rows/cols)
        # In practice, you'd want spatially-aware predictions
        row_probs = self.row_head(fused)  # [B, 1]
        col_probs = self.col_head(fused)  # [B, 1]
        
        return {
            'row_probs': row_probs,
            'col_probs': col_probs,
            'fused_features': fused
        }

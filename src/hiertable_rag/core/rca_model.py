import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Any, List, Tuple, Optional

class RCAModel(nn.Module):
    """
    Row-Column Attention (RCA) Model.
    Uses Dual-Axis Attention to identify global grid lines.
    """
    def __init__(self, backbone_name: str = "resnet18", embed_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        # Assuming the last layer of resnet18 has 512 channels
        
        # Transformer Attention for global consistency
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Row Boundary Head
        self.row_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.row_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # Column Boundary Head
        self.col_pool = nn.AdaptiveAvgPool2d((1, None))
        self.col_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary Cell Map Head
        self.cell_map_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, 3, H, W)
        features = self.backbone(x)[-1] # (B, 512, H/32, W/32)
        B, C, Hf, Wf = features.shape
        
        # Global Attention
        x_flat = features.flatten(2).transpose(1, 2) # (B, Hf*Wf, C)
        x_att = self.transformer(x_flat) # (B, Hf*Wf, C)
        features = x_att.transpose(1, 2).reshape(B, C, Hf, Wf)
        
        row_probs = self.row_head(self.row_pool(features)).squeeze(-1).squeeze(1) # (B, Hf)
        col_probs = self.col_head(self.col_pool(features)).squeeze(-2).squeeze(1) # (B, Wf)
        cell_map = torch.sigmoid(self.cell_map_head(features)) # (B, 1, Hf, Wf)
        
        # Compute confidence scores (based on entropy)
        row_confidence = self._compute_confidence(row_probs)
        col_confidence = self._compute_confidence(col_probs)
        
        return {
            "row_probs": row_probs,
            "col_probs": col_probs,
            "cell_map": cell_map,
            "row_confidence": row_confidence,
            "col_confidence": col_confidence
        }

    def _compute_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence scores based on entropy.
        Low entropy (confident predictions) -> high confidence score.
        
        Args:
            probs: Boundary probabilities [B, N]
        Returns:
            confidence: Confidence scores [B, N] in range [0, 1]
        """
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-8
        entropy = -(probs * torch.log(probs + eps) + 
                    (1 - probs) * torch.log(1 - probs + eps))
        
        # Normalize: max entropy (0.693) -> confidence 0, min entropy (0) -> confidence 1
        max_entropy = -torch.log(torch.tensor(0.5))
        confidence = 1.0 - (entropy / max_entropy)
        
        return confidence
    
    def predict_boundaries(self, x: torch.Tensor, threshold: float = 0.5, 
                          return_confidence: bool = False) -> Dict[str, List[List[float]]]:
        """
        Predict table boundaries from image.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            threshold: Probability threshold for boundary detection
            return_confidence: If True, include confidence scores in output
        
        Returns:
            Dictionary with row_boundaries, col_boundaries, and optionally confidence scores
        """
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            outputs = self.forward(x)
            row_probs = outputs["row_probs"].cpu().numpy()
            col_probs = outputs["col_probs"].cpu().numpy()
            
            batch_row_lines = []
            batch_row_conf = []
            for b in range(row_probs.shape[0]):
                lines = np.where(row_probs[b] > threshold)[0]
                # Normalize indices back to [0, 1]
                batch_row_lines.append((lines / (row_probs.shape[1] - 1)).tolist())
                
                if return_confidence:
                    row_conf = outputs["row_confidence"][b].cpu().numpy()
                    batch_row_conf.append(row_conf[lines].tolist())
                
            batch_col_lines = []
            batch_col_conf = []
            for b in range(col_probs.shape[0]):
                lines = np.where(col_probs[b] > threshold)[0]
                batch_col_lines.append((lines / (col_probs.shape[1] - 1)).tolist())
                
                if return_confidence:
                    col_conf = outputs["col_confidence"][b].cpu().numpy()
                    batch_col_conf.append(col_conf[lines].tolist())
        
        result = {
            "row_boundaries": batch_row_lines,
            "col_boundaries": batch_col_lines
        }
        
        if return_confidence:
            result["row_confidence"] = batch_row_conf
            result["col_confidence"] = batch_col_conf
            
        return result

import numpy as np

"""
Multi-resolution Vision Encoder for Table Understanding.

This module implements a vision encoder that processes images at multiple resolutions
(0.5x, 1.0x, 2.0x) to capture both global structure and fine-grained details.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import math
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet18, ResNet18_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from src.models.e2e_hierarchical import E2EConfig, PositionalEncoding2D

logger = logging.getLogger(__name__)


class MultiResVisionEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Multi-resolution Vision Encoder.
    
    Processes input image at 3 scales:
    1. 0.5x: Global structure / Layout
    2. 1.0x: Standard view
    3. 2.0x: Fine details (cell boundaries)
    
    Features are fused using a learned attention mechanism.
    """
    
    def __init__(self, config: E2EConfig):
        """
        Initialize multi-res encoder.
        
        Args:
            config: E2EConfig instance
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, MultiResVisionEncoder disabled")
            return
            
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Shared Backbone (ResNet18 for efficiency)
        # We remove the avgpool and fc layers to get feature maps
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Feature projection to hidden_dim
        # ResNet18 output channels is 512
        self.proj = nn.Conv2d(512, config.hidden_dim, kernel_size=1)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding2D(config.hidden_dim)
        
        # Multi-scale Fusion Attention
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        logger.info("MultiResVisionEncoder initialized")
        
    def forward(
        self, 
        images: "torch.Tensor"
    ) -> Tuple["torch.Tensor", Optional[List["torch.Tensor"]]]:
        """
        Forward pass with multi-resolution processing.
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            Fused features (B, SeqLen, HiddenDim)
        """
        if not TORCH_AVAILABLE:
            return None, None
            
        B, C, H, W = images.shape
        
        # 1. Multi-scale processing
        # Scale 1.0x (Original)
        feat_1x = self._extract_features(images)
        
        # Scale 0.5x (Downsampled)
        imgs_05x = F.interpolate(images, scale_factor=0.5, mode='bilinear', align_corners=False)
        feat_05x = self._extract_features(imgs_05x)
        
        # Scale 2.0x (Upsampled) - Process in patches if too large, but for now simple resize
        # Note: In production, 2x might be too heavy. We'll assume inputs are reasonable.
        # For memory efficiency, we might skip 2x if image is already huge.
        if H * W > 1024 * 1024:
            feat_2x = feat_1x # Fallback
        else:
            imgs_2x = F.interpolate(images, scale_factor=2.0, mode='bilinear', align_corners=False)
            feat_2x = self._extract_features(imgs_2x)
            
        # 2. Align feature maps to 1.0x size for fusion
        target_H, target_W = feat_1x.shape[-2:]
        
        feat_05x = F.interpolate(feat_05x, size=(target_H, target_W), mode='bilinear', align_corners=False)
        feat_2x = F.interpolate(feat_2x, size=(target_H, target_W), mode='bilinear', align_corners=False)
        
        # 3. Add positional encodings
        feat_1x = self.pos_encoding(feat_1x)
        feat_05x = self.pos_encoding(feat_05x)
        feat_2x = self.pos_encoding(feat_2x)
        
        # 4. Flatten for attention
        # (B, C, H, W) -> (B, H*W, C)
        f1 = feat_1x.flatten(2).transpose(1, 2)
        f05 = feat_05x.flatten(2).transpose(1, 2)
        f2 = feat_2x.flatten(2).transpose(1, 2)
        
        # 5. Fusion
        # We treat 1x as query, and concat([0.5x, 1x, 2x]) as key/value
        # Or simpler: Average or Learnable weights. 
        # Let's use a simple concatenation + projection for stability first, 
        # or self-attention over the stack.
        
        # Stack: (B, 3*L, C) ?? No, that's too long sequence.
        # We want (B, L, C).
        
        # Fusion Strategy: Element-wise sum + Gate? Or Cross-Attention?
        # Let's use Cross-Attention where Query=1x, Key=Value=Concat(0.5x, 2x)
        # This allows 1x to "look at" details from 2x and context from 0.5x.
        
        context = torch.cat([f05, f2], dim=1) # (B, 2*L, C)
        
        fused, _ = self.fusion_attn(
            query=f1,
            key=context,
            value=context
        )
        
        # Residual connection
        output = f1 + fused
        
        return output, None

    def _extract_features(self, x: "torch.Tensor") -> "torch.Tensor":
        """Extract features using backbone and project."""
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat

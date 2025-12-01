"""
Swin Transformer-based Multi-Resolution Vision Encoder.

Implements multi-scale feature extraction using Swin Transformer
with Feature Pyramid Network (FPN) for fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import logging

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.
    
    Fuses features from different resolutions using lateral connections
    and top-down pathway.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
    ):
        """
        Initialize FPN.
        
        Args:
            in_channels_list: List of input channels for each scale
            out_channels: Output channels for all scales
        """
        super().__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through FPN.
        
        Args:
            features: List of feature maps from different scales
                     [low_res, mid_res, high_res]
        
        Returns:
            List of fused feature maps at each scale
        """
        # Apply lateral convolutions
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway with upsampling
        results = []
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                # Highest resolution
                result = laterals[i]
            else:
                # Upsample and add
                upsampled = F.interpolate(
                    results[-1],
                    size=laterals[i].shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                result = laterals[i] + upsampled
            
            # Apply output convolution
            result = self.output_convs[i](result)
            results.insert(0, result)
        
        return results


class SwinMultiResEncoder(nn.Module):
    """
    Multi-Resolution Vision Encoder using Swin Transformer.
    
    Processes input images at multiple resolutions (0.5x, 1.0x, 2.0x)
    and fuses features using Feature Pyramid Network.
    """
    
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        hidden_dim: int = 256,
        scales: List[float] = [0.5, 1.0, 2.0],
    ):
        """
        Initialize Swin Multi-Resolution Encoder.
        
        Args:
            model_name: Swin model variant from timm
            pretrained: Whether to use pretrained weights
            hidden_dim: Hidden dimension for output features
            scales: List of resolution scales to process
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required. Install with: pip install timm")
        
        self.scales = scales
        self.hidden_dim = hidden_dim
        
        # Shared Swin Transformer backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,  # Get final features
            num_classes=0,  # Remove classification head
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            backbone_dim = dummy_output.shape[1]
        
        logger.info(f"Swin backbone output dimension: {backbone_dim}")
        
        # Projection to hidden_dim
        self.feature_proj = nn.Linear(backbone_dim, hidden_dim)
        
        # Cross-scale fusion using multi-head attention
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        logger.info(f"SwinMultiResEncoder initialized (model={model_name}, scales={scales})")
    
    def _extract_features(
        self,
        images: torch.Tensor,
        scale: float = 1.0
    ) -> torch.Tensor:
        """
        Extract features at a specific scale.
        
        Args:
            images: Input images (B, C, H, W)
            scale: Resolution scale factor
        
        Returns:
            Features (B, D)
        """
        B, C, H, W = images.shape
        
        # Resize to target scale first
        if scale != 1.0:
            target_size = (int(H * scale), int(W * scale))
            images_scaled = F.interpolate(
                images,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        else:
            images_scaled = images
        
        # Then resize to Swin's expected input size (224x224)
        images_224 = F.interpolate(
            images_scaled,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        # Extract features from Swin
        feat = self.backbone(images_224)  # (B, backbone_dim)
        
        # Project to hidden_dim
        feat = self.feature_proj(feat)  # (B, hidden_dim)
        
        return feat
    
    def forward(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through multi-resolution encoder.
        
        Args:
            images: Input images (B, C, H, W)
        
        Returns:
            - Fused features (B, N, D) where N=1 for global features
            - List of intermediate features (optional)
        """
        B, C, H, W = images.shape
        
        # Extract features at each scale
        multi_scale_features = []
        for scale in self.scales:
            feat = self._extract_features(images, scale=scale)  # (B, D)
            multi_scale_features.append(feat)
        
        # Stack features: (B, num_scales, D)
        feat_stacked = torch.stack(multi_scale_features, dim=1)
        
        # Use 1.0x scale as query
        feat_1x_idx = self.scales.index(1.0)
        feat_1x = multi_scale_features[feat_1x_idx].unsqueeze(1)  # (B, 1, D)
        
        # Cross-attention fusion
        fused, attn_weights = self.fusion_attn(
            query=feat_1x,
            key=feat_stacked,
            value=feat_stacked
        )
        
        # Residual connection
        output = feat_1x + fused  # (B, 1, D)
        
        # Final projection
        output = self.output_proj(output)
        
        return output, multi_scale_features


class SwinEncoderConfig:
    """Configuration for Swin Encoder."""
    
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        hidden_dim: int = 256,
        scales: List[float] = None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.hidden_dim = hidden_dim
        self.scales = scales or [0.5, 1.0, 2.0]


def create_swin_encoder(config: Optional[SwinEncoderConfig] = None) -> SwinMultiResEncoder:
    """
    Factory function to create Swin encoder.
    
    Args:
        config: Encoder configuration
    
    Returns:
        SwinMultiResEncoder instance
    """
    if config is None:
        config = SwinEncoderConfig()
    
    return SwinMultiResEncoder(
        model_name=config.model_name,
        pretrained=config.pretrained,
        hidden_dim=config.hidden_dim,
        scales=config.scales
    )

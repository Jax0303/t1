#!/usr/bin/env python3
"""
Test script for Swin Multi-Resolution Encoder.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
from src.models.swin_encoder import SwinMultiResEncoder, SwinEncoderConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("TestSwin")


def test_swin_encoder():
    """Test Swin encoder forward pass."""
    logger.info("=== Testing SwinMultiResEncoder ===")
    
    # Create config
    config = SwinEncoderConfig(
        model_name="swin_tiny_patch4_window7_224",
        pretrained=False,  # Use random weights for faster testing
        hidden_dim=256,
        scales=[0.5, 1.0, 2.0]
    )
    
    # Create encoder
    encoder = SwinMultiResEncoder(
        model_name=config.model_name,
        pretrained=config.pretrained,
        hidden_dim=config.hidden_dim,
        scales=config.scales
    )
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    logger.info(f"Input shape: {images.shape}")
    
    with torch.no_grad():
        output, multi_scale_features = encoder(images)
    
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Number of multi-scale features: {len(multi_scale_features)}")
    
    for i, feat in enumerate(multi_scale_features):
        logger.info(f"  Scale {config.scales[i]}x: {feat.shape}")
    
    # Verify output shape
    assert output.shape[0] == batch_size, f"Batch size mismatch: {output.shape[0]} != {batch_size}"
    assert output.shape[2] == config.hidden_dim, f"Hidden dim mismatch: {output.shape[2]} != {config.hidden_dim}"
    
    logger.info("✅ Swin Encoder Test Passed")
    
    return encoder, output


def benchmark_swin_vs_resnet():
    """Compare Swin encoder with ResNet baseline."""
    logger.info("\n=== Benchmarking Swin vs ResNet ===")
    
    from src.models.vision_encoder import MultiResVisionEncoder
    from src.models.e2e_hierarchical import E2EConfig
    
    # ResNet baseline
    resnet_config = E2EConfig(hidden_dim=256)
    resnet_encoder = MultiResVisionEncoder(resnet_config)
    
    # Swin encoder
    swin_encoder = SwinMultiResEncoder(
        model_name="swin_tiny_patch4_window7_224",
        pretrained=False,
        hidden_dim=256
    )
    
    # Test input
    images = torch.randn(1, 3, 768, 768)
    
    # ResNet forward
    with torch.no_grad():
        resnet_out, _ = resnet_encoder(images)
    
    # Swin forward
    with torch.no_grad():
        swin_out, _ = swin_encoder(images)
    
    logger.info(f"ResNet output shape: {resnet_out.shape}")
    logger.info(f"Swin output shape: {swin_out.shape}")
    
    # Count parameters
    resnet_params = sum(p.numel() for p in resnet_encoder.parameters())
    swin_params = sum(p.numel() for p in swin_encoder.parameters())
    
    logger.info(f"ResNet parameters: {resnet_params:,}")
    logger.info(f"Swin parameters: {swin_params:,}")
    logger.info(f"Parameter ratio (Swin/ResNet): {swin_params/resnet_params:.2f}x")
    
    logger.info("✅ Benchmark Complete")


if __name__ == "__main__":
    # Test Swin encoder
    encoder, output = test_swin_encoder()
    
    # Benchmark vs ResNet
    benchmark_swin_vs_resnet()
    
    print("\n✅ All tests passed!")

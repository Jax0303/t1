import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.e2e_hierarchical import E2EConfig, VisionTableEncoder

class TestTableFormerBias(unittest.TestCase):
    def test_structural_bias(self):
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda"

        print(f"Testing on {device}")

        config = E2EConfig(
            hidden_dim=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            image_size=(64, 64) # Small size for speed
        )
        encoder = VisionTableEncoder(config)
        encoder.to(device)
        encoder.eval()

        # Dummy input
        B = 1
        images = torch.randn(B, 3, 64, 64).to(device)
        
        # 1. Run forward pass with zero bias
        with torch.no_grad():
            output1 = encoder(images)
            
        # 2. Change bias and run again
        # We expect the output to change because bias is added to attention scores
        with torch.no_grad():
            encoder.row_bias.data.fill_(10.0) # Large bias
            output2 = encoder(images)
            
        # Check if outputs are different
        diff = (output1 - output2).abs().max().item()
        print(f"Max difference after changing row_bias: {diff}")
        self.assertTrue(diff > 1e-5, "Output should change when row_bias is modified")
        
        # 3. Change col bias
        with torch.no_grad():
            encoder.col_bias.data.fill_(-10.0)
            output3 = encoder(images)
            
        diff2 = (output2 - output3).abs().max().item()
        print(f"Max difference after changing col_bias: {diff2}")
        self.assertTrue(diff2 > 1e-5, "Output should change when col_bias is modified")
        
        print("Structural bias verification passed!")

if __name__ == "__main__":
    unittest.main()

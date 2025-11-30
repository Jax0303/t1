import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.e2e_hierarchical import E2EConfig, E2EHierarchicalTableModel

class TestAttentionGraph(unittest.TestCase):
    def test_attention_extraction(self):
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
            image_size=(64, 64)
        )
        model = E2EHierarchicalTableModel(config)
        model.to(device)
        model.eval()

        # Dummy input
        B = 1
        images = torch.randn(B, 3, 64, 64).to(device)
        
        # Run forward pass with output_attentions=True
        with torch.no_grad():
            output, _ = model(images, output_attentions=True)
            
        attentions = output.attentions
        self.assertIsNotNone(attentions)
        self.assertTrue(len(attentions) > 0)
        
        # Check shape: (B, NumHeads, SeqLen, SeqLen)
        # SeqLen = (64/32) * (64/32) = 2*2 = 4 (if stride 32)
        # Wait, the backbone has 5 stages. 
        # stride: 2 -> 4 -> 8 -> 16 -> 32 (maybe?)
        # Let's check the shape
        print(f"Attention shape: {attentions[0].shape}")
        
        # Extract graph
        edges = model.extract_attention_graph(attentions, threshold=0.0) # Threshold 0 to get all edges
        print(f"Extracted {len(edges)} edges")
        self.assertTrue(len(edges) > 0)
        
        # Verify edge format
        src, tgt, weight = edges[0]
        self.assertIsInstance(src, int)
        self.assertIsInstance(tgt, int)
        self.assertIsInstance(weight, float)
        
        print("Attention graph extraction verification passed!")

if __name__ == "__main__":
    unittest.main()

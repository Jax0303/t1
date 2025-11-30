import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.e2e_hierarchical import E2EConfig, E2EHierarchicalTableModel

class TestE2ERetrieval(unittest.TestCase):
    def test_retrieval_loss(self):
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
            max_cells=10,
            use_retrieval_loss=True,
            image_size=(224, 224)
        )
        model = E2EHierarchicalTableModel(config)
        model.to(device)
        model.train()

        # Dummy inputs
        B = 2
        images = torch.randn(B, 3, 224, 224).to(device)
        
        # Dummy targets
        # We need to provide targets that trigger at least one loss calculation to avoid errors if any
        # But since we are testing retrieval loss, we can just provide empty dict if the code handles it,
        # or provide minimal targets.
        # The _compute_losses method checks for keys.
        targets = {
            "cell_labels": torch.zeros(B, 10, dtype=torch.long).to(device),
        }
        
        # Retrieval inputs
        num_queries = 3
        queries = torch.randn(B, num_queries, 32).to(device)
        # Target mapping: query 0 -> cell 1, query 1 -> cell 2
        query_targets = torch.zeros(B, num_queries, 10).to(device)
        query_targets[:, 0, 1] = 1.0
        query_targets[:, 1, 2] = 1.0
        
        output, losses = model(images, targets, queries=queries, query_targets=query_targets)
        
        print("Losses keys:", losses.keys())
        self.assertIn("retrieval_loss", losses)
        print(f"Retrieval Loss: {losses['retrieval_loss'].item()}")
        self.assertTrue(losses["retrieval_loss"] > 0)
        
        # Check backward
        total_loss = sum(losses.values())
        total_loss.backward()
        print("Backward pass successful")

if __name__ == "__main__":
    unittest.main()

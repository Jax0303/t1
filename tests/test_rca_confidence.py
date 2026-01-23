"""
Unit Tests for RCA Model Confidence Scores
Tests the enhanced RCAModel with confidence score outputs.
"""

import unittest
import torch
import sys
import os
sys.path.append(os.getcwd())

from src.hiertable_rag.core.rca_model import RCAModel


class TestRCAConfidence(unittest.TestCase):
    """Test RCA model confidence score enhancements"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize model once for all tests"""
        cls.model = RCAModel(backbone_name="resnet18", embed_dim=512)
        cls.model.eval()
    
    def test_forward_with_confidence(self):
        """Test that forward pass includes confidence scores"""
        x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        
        with torch.no_grad():
            outputs = self.model(x)
        
        # Check all expected keys exist
        self.assertIn('row_probs', outputs)
        self.assertIn('col_probs', outputs)
        self.assertIn('cell_map', outputs)
        self.assertIn('row_confidence', outputs)
        self.assertIn('col_confidence', outputs)
        
        # Check confidence scores are in valid range [0, 1]
        row_conf = outputs['row_confidence']
        col_conf = outputs['col_confidence']
        
        self.assertTrue((row_conf >= 0).all() and (row_conf <= 1).all())
        self.assertTrue((col_conf >= 0).all() and (col_conf <= 1).all())
    
    def test_predict_boundaries_backward_compatible(self):
        """Test that predict_boundaries works without return_confidence flag"""
        x = torch.randn(2, 3, 224, 224)
        
        result = self.model.predict_boundaries(x, threshold=0.5)
        
        # Should have basic outputs
        self.assertIn('row_boundaries', result)
        self.assertIn('col_boundaries', result)
        
        # Should NOT have confidence by default (backward compatible)
        self.assertNotIn('row_confidence', result)
        self.assertNotIn('col_confidence', result)
    
    def test_predict_boundaries_with_confidence(self):
        """Test predict_boundaries with confidence flag enabled"""
        x = torch.randn(2, 3, 224, 224)
        
        result = self.model.predict_boundaries(x, threshold=0.5, return_confidence=True)
        
        # Should have all outputs including confidence
        self.assertIn('row_boundaries', result)
        self.assertIn('col_boundaries', result)
        self.assertIn('row_confidence', result)
        self.assertIn('col_confidence', result)
        
        # Confidence arrays should match boundary count
        for batch_idx in range(len(result['row_boundaries'])):
            self.assertEqual(
                len(result['row_boundaries'][batch_idx]),
                len(result['row_confidence'][batch_idx])
            )
    
    def test_confidence_entropy_logic(self):
        """Test that confidence scores reflect entropy correctly"""
        # High probability (confident) -> low entropy -> high confidence
        high_prob = torch.tensor([[0.95, 0.1, 0.9]])
        high_conf = self.model._compute_confidence(high_prob)
        
        # Low probability (uncertain) -> high entropy -> low confidence
        uncertain_prob = torch.tensor([[0.5, 0.5, 0.5]])
        uncertain_conf = self.model._compute_confidence(uncertain_prob)
        
        # Confident predictions should have higher confidence scores
        self.assertGreater(high_conf.mean().item(), uncertain_conf.mean().item())
    
    def test_confidence_range(self):
        """Test edge cases for confidence computation"""
        # Perfect confidence (prob=0 or prob=1)
        perfect_probs = torch.tensor([[0.0, 1.0]])
        perfect_conf = self.model._compute_confidence(perfect_probs)
        
        # Max uncertainty (prob=0.5)
        uncertain_probs = torch.tensor([[0.5, 0.5]])
        uncertain_conf = self.model._compute_confidence(uncertain_probs)
        
        # Perfect confidence should be close to 1
        self.assertGreater(perfect_conf.mean().item(), 0.95)
        
        # Max uncertainty should be close to 0
        self.assertLess(uncertain_conf.mean().item(), 0.05)


if __name__ == '__main__':
    unittest.main()

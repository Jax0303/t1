"""
Integration Tests for SARTP Pipeline
Tests end-to-end table parsing with semantic alignment.
"""

import unittest
import torch
import sys
import os
sys.path.append(os.getcwd())

from src.hiertable_rag.sartp.pipeline import SARTPPipeline


class TestSARTPPipeline(unittest.TestCase):
    """Test SARTP end-to-end pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize pipeline once for all tests"""
        print("\nInitializing SARTP Pipeline (this may take a minute)...")
        cls.pipeline = SARTPPipeline(
            rca_checkpoint=None,  # Use random init for testing
            semantic_model="roberta-base",
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            device='cpu'
        )
        print("Pipeline initialized successfully")
    
    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly"""
        self.assertIsNotNone(self.pipeline.rca_model)
        self.assertIsNotNone(self.pipeline.semantic_encoder)
        self.assertIsNotNone(self.pipeline.aligner)
        self.assertIsNotNone(self.pipeline.optimizer)
    
    def test_simple_parse(self):
        """Test parsing a simple mock table"""
        # Create dummy image
        image = torch.randn(1, 3, 224, 224)
        
        # Mock OCR boxes
        ocr_boxes = [
            {'box': [0.1, 0.1, 0.4, 0.2], 'content': 'Name'},
            {'box': [0.5, 0.1, 0.8, 0.2], 'content': 'Age'},
            {'box': [0.1, 0.3, 0.4, 0.4], 'content': 'Alice'},
            {'box': [0.5, 0.3, 0.8, 0.4], 'content': '25'}
        ]
        
        # Parse
        result = self.pipeline.parse(image, ocr_boxes, return_intermediate=True)
        
        # Check output structure
        self.assertIn('cells', result)
        self.assertIn('initial_cells', result)
        self.assertIn('alignment_scores', result)
        self.assertIn('visual_pred', result)
        
        # Check that cells were created
        self.assertGreater(len(result['cells']), 0, "Should produce at least one cell")
    
    def test_weight_update(self):
        """Test updating pipeline weights"""
        original_alpha = self.pipeline.optimizer.alpha
        
        # Update weights
        self.pipeline.set_weights(alpha=0.6, beta=0.2, gamma=0.2)
        
        # Should be normalized and different
        self.assertNotEqual(self.pipeline.optimizer.alpha, original_alpha)
        
        # Sum should be 1.0 (normalized)
        total = (self.pipeline.optimizer.alpha + 
                self.pipeline.optimizer.beta + 
                self.pipeline.optimizer.gamma)
        self.assertAlmostEqual(total, 1.0, places=5)
    
    def test_empty_ocr_handling(self):
        """Test graceful handling of empty OCR boxes"""
        image = torch.randn(1, 3, 224, 224)
        ocr_boxes = []  # No OCR boxes
        
        result = self.pipeline.parse(image, ocr_boxes, return_intermediate=False)
        
        # Should not crash, but cells should be empty
        self.assertIn('cells', result)
        self.assertEqual(len(result['cells']), 0)
    
    def test_batch_parsing(self):
        """Test batch processing"""
        images = [torch.randn(1, 3, 224, 224) for _ in range(2)]
        ocr_boxes_list = [
            [{'box': [0.1, 0.1, 0.5, 0.5], 'content': 'Cell1'}],
            [{'box': [0.2, 0.2, 0.6, 0.6], 'content': 'Cell2'}]
        ]
        
        results = self.pipeline.parse_batch(images, ocr_boxes_list)
        
        # Should return one result per image
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('cells', result)
    
    def test_intermediate_outputs(self):
        """Test that intermediate outputs are available when requested"""
        image = torch.randn(1, 3, 224, 224)
        ocr_boxes = [{'box': [0.1, 0.1, 0.5, 0.5], 'content': 'Test'}]
        
        result = self.pipeline.parse(image, ocr_boxes, return_intermediate=True)
        
        # Check all intermediate outputs
        self.assertIn('initial_cells', result)
        self.assertIn('alignment_scores', result)
        self.assertIn('visual_pred', result)
        self.assertIn('header_mask', result)
        self.assertIn('embeddings', result)
        
        # Embeddings should be a tensor
        self.assertIsInstance(result['embeddings'], torch.Tensor)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)

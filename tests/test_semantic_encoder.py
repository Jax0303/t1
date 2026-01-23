"""
Unit Tests for Semantic Encoder Module
Tests cell encoding, header detection, and similarity computation.
"""

import unittest
import torch
from src.hiertable_rag.sartp.semantic_encoder import SemanticEncoder, HeaderDetector


class TestSemanticEncoder(unittest.TestCase):
    """Test SemanticEncoder functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize encoder once for all tests"""
        cls.encoder = SemanticEncoder(model_name="roberta-base", device='cpu')
    
    def test_encode_single_cell(self):
        """Test encoding a single cell"""
        cells = [{'content': 'Revenue'}]
        embeddings = self.encoder.encode_cells(cells)
        
        self.assertEqual(embeddings.shape, (1, self.encoder.embedding_dim))
        self.assertIsInstance(embeddings, torch.Tensor)
    
    def test_encode_multiple_cells(self):
        """Test batch encoding"""
        cells = [
            {'content': 'Company'},
            {'content': 'Revenue'},
            {'content': 'Apple Inc.'},
            {'content': '$100M'}
        ]
        embeddings = self.encoder.encode_cells(cells)
        
        self.assertEqual(embeddings.shape, (4, self.encoder.embedding_dim))
    
    def test_encode_empty_cell(self):
        """Test handling empty cells"""
        cells = [{'content': ''}, {'content': None}]
        embeddings = self.encoder.encode_cells(cells)
        
        self.assertEqual(embeddings.shape, (2, self.encoder.embedding_dim))
    
    def test_similarity_computation(self):
        """Test cosine similarity computation"""
        cells1 = [{'content': 'Revenue'}, {'content': 'Profit'}]
        cells2 = [{'content': '$100M'}, {'content': 'Company'}]
        
        emb1 = self.encoder.encode_cells(cells1)
        emb2 = self.encoder.encode_cells(cells2)
        
        similarity = self.encoder.compute_similarity(emb1, emb2)
        
        self.assertEqual(similarity.shape, (2, 2))
        # Cosine similarity should be in [-1, 1]
        self.assertTrue((similarity >= -1.0).all() and (similarity <= 1.0).all())
    
    def test_semantic_similarity(self):
        """Test that semantically similar texts have higher similarity"""
        cells = [
            {'content': 'Revenue'},
            {'content': 'Income'},  # Similar to Revenue
            {'content': 'Apple'}    # Different
        ]
        
        embeddings = self.encoder.encode_cells(cells)
        similarity = self.encoder.compute_similarity(embeddings, embeddings)
        
        # Revenue-Income similarity should be higher than Revenue-Apple
        self.assertGreater(
            similarity[0, 1].item(),  # Revenue-Income
            similarity[0, 2].item()   # Revenue-Apple
        )


class TestHeaderDetector(unittest.TestCase):
    """Test HeaderDetector functionality"""
    
    def setUp(self):
        """Initialize detector for each test"""
        self.detector = HeaderDetector()
    
    def test_classify_simple_table(self):
        """Test header classification on simple table"""
        cells = [
            {'row_idx': 0, 'col_idx': 0, 'content': 'Company'},
            {'row_idx': 0, 'col_idx': 1, 'content': 'Revenue'},
            {'row_idx': 1, 'col_idx': 0, 'content': 'Apple'},
            {'row_idx': 1, 'col_idx': 1, 'content': '$100M'},
            {'row_idx': 2, 'col_idx': 0, 'content': 'Google'},
            {'row_idx': 2, 'col_idx': 1, 'content': '$200M'}
        ]
        
        is_header = self.detector.classify_cells(cells, use_semantics=False)
        
        # First row should be headers
        self.assertTrue(is_header[0])  # Company
        self.assertTrue(is_header[1])  # Revenue
        
        # Data rows should not be headers
        self.assertFalse(is_header[2])  # Apple
        self.assertFalse(is_header[3])  # $100M
    
    def test_multi_level_headers(self):
        """Test detection of hierarchical headers"""
        cells = [
            {'row_idx': 0, 'col_idx': 0, 'content': 'Financial Data'},
            {'row_idx': 1, 'col_idx': 0, 'content': 'Company'},
            {'row_idx': 1, 'col_idx': 1, 'content': 'Revenue'},
            {'row_idx': 2, 'col_idx': 0, 'content': 'Apple'},
            {'row_idx': 2, 'col_idx': 1, 'content': '$100M'}
        ]
        
        header_levels = self.detector.detect_header_levels(cells)
        
        # Should detect two header levels
        self.assertIn(0, header_levels)  # First level
        self.assertIn(1, header_levels)  # Second level
        self.assertEqual(header_levels[0], 0)
        self.assertEqual(header_levels[1], 1)
    
    def test_uppercase_header_detection(self):
        """Test that uppercase content is detected as header"""
        cells = [
            {'row_idx': 0, 'col_idx': 0, 'content': 'COMPANY'},  # Uppercase
            {'row_idx': 1, 'col_idx': 0, 'content': 'Apple Inc.'}
        ]
        
        is_header = self.detector.classify_cells(cells, use_semantics=False)
        
        self.assertTrue(is_header[0])   # COMPANY (uppercase)
        self.assertFalse(is_header[1])  # Apple Inc. (mixed case)


class TestIntegration(unittest.TestCase):
    """Integration tests for semantic encoder + header detector"""
    
    def test_end_to_end_encoding(self):
        """Test complete workflow: detect headers + encode semantically"""
        encoder = SemanticEncoder(model_name="roberta-base", device='cpu')
        detector = HeaderDetector(semantic_encoder=encoder)
        
        cells = [
            {'row_idx': 0, 'col_idx': 0, 'content': 'Product'},
            {'row_idx': 0, 'col_idx': 1, 'content': 'Price'},
            {'row_idx': 1, 'col_idx': 0, 'content': 'Laptop'},
            {'row_idx': 1, 'col_idx': 1, 'content': '$1000'}
        ]
        
        # 1. Detect headers
        is_header = detector.classify_cells(cells, use_semantics=False)
        header_cells = [c for i, c in enumerate(cells) if is_header[i]]
        data_cells = [c for i, c in enumerate(cells) if not is_header[i]]
        
        # 2. Encode both
        header_embeddings = encoder.encode_cells(header_cells)
        data_embeddings = encoder.encode_cells(data_cells)
        
        # 3. Verify dimensions
        self.assertEqual(header_embeddings.shape[0], 2)  # 2 headers
        self.assertEqual(data_embeddings.shape[0], 2)    # 2 data cells
        
        # 4. Compute cross-similarity (header-data alignment)
        similarity = encoder.compute_similarity(header_embeddings, data_embeddings)
        self.assertEqual(similarity.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()

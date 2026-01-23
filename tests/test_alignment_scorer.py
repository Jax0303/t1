"""
Unit Tests for Header-Data Alignment Scorer
Tests semantic alignment computation between headers and data cells.
"""

import unittest
import torch
from src.hiertable_rag.sartp.alignment_scorer import HeaderDataAligner


class TestHeaderDataAligner(unittest.TestCase):
    """Test HeaderDataAligner functionality"""
    
    def setUp(self):
        """Initialize aligner for each test"""
        self.aligner = HeaderDataAligner(similarity_threshold=0.5)
    
    def test_simple_table_alignment(self):
        """Test alignment scoring on a simple table"""
        # Simple 2x2 table with header row
        cells = [
            {'row_idx': 0, 'col_idx': 0, 'content': 'Name'},
            {'row_idx': 0, 'col_idx': 1, 'content': 'Age'},
            {'row_idx': 1, 'col_idx': 0, 'content': 'Alice'},
            {'row_idx': 1, 'col_idx': 1, 'content': '25'}
        ]
        
        # Mock embeddings (4 cells, 8-dim for simplicity)
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Name (header)
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Age (header)
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Alice (similar to Name)
            [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 25 (similar to Age)
        ], dtype=torch.float32)
        
        header_mask = [True, True, False, False]
        
        result = self.aligner.score(cells, embeddings, header_mask)
        
        # Check result structure
        self.assertIn('col_alignment', result)
        self.assertIn('row_coherence', result)
        self.assertIn('global_score', result)
        
        # Column 0 should have good alignment (Name-Alice)
        self.assertIn(0, result['col_alignment'])
        self.assertGreater(result['col_alignment'][0]['mean_similarity'], 0.5)
        
        # Column 1 should have good alignment (Age-25)
        self.assertIn(1, result['col_alignment'])
        self.assertGreater(result['col_alignment'][1]['mean_similarity'], 0.5)
    
    def test_misaligned_column(self):
        """Test detection of misaligned columns"""
        cells = [
            {'row_idx': 0, 'col_idx': 0, 'content': 'Product'},
            {'row_idx': 0, 'col_idx': 1, 'content': 'Price'},
            {'row_idx': 1, 'col_idx': 0, 'content': 'Laptop'},
            {'row_idx': 1, 'col_idx': 1, 'content': 'Microsoft'}  # Misaligned! Should be price
        ]
        
        # Embeddings where col 1 data is closer to col 0 header
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Product
            [0.0, 1.0, 0.0, 0.0],  # Price
            [0.9, 0.0, 0.0, 0.0],  # Laptop (aligned with Product)
            [0.8, 0.1, 0.0, 0.0]   # Microsoft (closer to Product than Price!)
        ], dtype=torch.float32)
        
        header_mask = [True, True, False, False]
        
        result = self.aligner.score(cells, embeddings, header_mask)
        
        # Column 1 should have LOW alignment
        self.assertLess(result['col_alignment'][1]['mean_similarity'], 0.5)
        
        # Global score should reflect the misalignment
        self.assertLess(result['global_score'], 0.8)
    
    def test_row_coherence(self):
        """Test row coherence computation"""
        cells = [
            {'row_idx': 0, 'col_idx': 0, 'content': 'A'},
            {'row_idx': 0, 'col_idx': 1, 'content': 'B'},
            {'row_idx': 1, 'col_idx': 0, 'content': 'C'},
            {'row_idx': 1, 'col_idx': 1, 'content': 'D'}
        ]
        
        # Row 1 has similar embeddings (coherent)
        embeddings = torch.tensor([
            [1.0, 0.0],  # Header A
            [0.0, 1.0],  # Header B
            [0.9, 0.1],  # C (similar)
            [0.8, 0.2]   # D (similar to C)
        ], dtype=torch.float32)
        
        header_mask = [True, True, False, False]
        
        result = self.aligner.score(cells, embeddings, header_mask)
        
        # Row 1 should have high coherence
        self.assertIn(1, result['row_coherence'])
        self.assertGreater(result['row_coherence'][1], 0.8)
    
    def test_suggest_column_reassignment(self):
        """Test column reassignment suggestion"""
        cells = [
            {'row_idx': 0, 'col_idx': 0, 'content': 'Name'},
            {'row_idx': 0, 'col_idx': 1, 'content': 'Age'},
            {'row_idx': 1, 'col_idx': 1, 'content': 'Alice'}  # Wrongly in col 1
        ]
        
        # Alice is more similar to Name than Age
        embeddings = torch.tensor([
            [1.0, 0.0],  # Name
            [0.0, 1.0],  # Age
            [0.95, 0.0]  # Alice (very similar to Name)
        ], dtype=torch.float32)
        
        header_mask = [True, True, False]
        
        # Suggest reassignment for cell 2 (Alice)
        suggested_col = self.aligner.suggest_column_reassignment(
            cells, embeddings, header_mask, cell_idx=2
        )
        
        # Should suggest column 0 (Name column)
        self.assertEqual(suggested_col, 0)
    
    def test_no_headers(self):
        """Test graceful handling when no headers exist"""
        cells = [
            {'row_idx': 0, 'col_idx': 0, 'content': 'Data1'},
            {'row_idx': 0, 'col_idx': 1, 'content': 'Data2'}
        ]
        
        embeddings = torch.rand(2, 4)
        header_mask = [False, False]  # No headers
        
        result = self.aligner.score(cells, embeddings, header_mask)
        
        # Should not crash, but alignment should be empty/zero
        self.assertEqual(len(result['col_alignment']), 2)
        for col_scores in result['col_alignment'].values():
            self.assertEqual(col_scores['mean_similarity'], 0.0)
    
    def test_get_misaligned_cells(self):
        """Test misaligned cell detection"""
        cells = [
            {'row_idx': 0, 'col_idx': 0, 'content': 'Good Header'},
            {'row_idx': 0, 'col_idx': 1, 'content': 'Bad Header'},
            {'row_idx': 1, 'col_idx': 0, 'content': 'Aligned Data'},
            {'row_idx': 1, 'col_idx': 1, 'content': 'Misaligned Data'}
        ]
        
        # Create alignment scores with col 1 misaligned
        alignment_scores = {
            'col_alignment': {
                0: {'mean_similarity': 0.9, 'aligned_count': 1},
                1: {'mean_similarity': 0.2, 'aligned_count': 0}  # Low!
            },
            'col_groups': {0: [0, 2], 1: [1, 3]},
            'row_coherence': {},
            'global_score': 0.5
        }
        
        misaligned = self.aligner.get_misaligned_cells(cells, alignment_scores)
        
        # Should identify cells in column 1 as misaligned
        self.assertIn(1, misaligned)  # Bad Header
        self.assertIn(3, misaligned)  # Misaligned Data


if __name__ == '__main__':
    unittest.main()

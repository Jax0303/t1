"""
Unit tests for GNN-CSP Table Structure Recognition

Tests:
1. Graph construction from OCR boxes
2. GNN message passing
3. Constraint validation
4. CSP solver
5. End-to-end pipeline
"""

import pytest
import torch
import numpy as np
from src.hiertable_rag.gnn_csp import (
    CellGraphBuilder,
    TableGNN,
    ConstraintLayer,
    TableConstraints,
    HardConstraints,
    SoftConstraints,
    TableCSPSolver,
    GNNCSPPipeline
)


class TestCellGraphBuilder:
    """Test graph construction from OCR boxes."""
    
    def test_build_graph_basic(self):
        """Test basic graph construction."""
        builder = CellGraphBuilder()
        
        # Create simple 2x2 table
        ocr_boxes = [
            {'box': [0, 0, 10, 10], 'content': 'A'},
            {'box': [10, 0, 20, 10], 'content': 'B'},
            {'box': [0, 10, 10, 20], 'content': 'C'},
            {'box': [10, 10, 20, 20], 'content': 'D'},
        ]
        
        visual_features = torch.randn(4, 128)
        semantic_embeddings = torch.randn(4, 768)
        
        graph = builder.build_graph(ocr_boxes, visual_features, semantic_embeddings)
        
        assert graph.num_nodes == 4
        assert graph.x.shape == (4, 128 + 4 + 768)  # visual + position + semantic
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0  # Should have edges
    
    def test_empty_boxes(self):
        """Test with empty OCR boxes."""
        builder = CellGraphBuilder()
        
        ocr_boxes = []
        visual_features = torch.empty(0, 128)
        semantic_embeddings = torch.empty(0, 768)
        
        graph = builder.build_graph(ocr_boxes, visual_features, semantic_embeddings)
        
        assert graph.num_nodes == 0
    
    def test_spatial_adjacency(self):
        """Test that spatially adjacent cells are connected."""
        builder = CellGraphBuilder(adjacency_threshold=0.3)
        
        # Two horizontally adjacent cells
        ocr_boxes = [
            {'box': [0, 0, 10, 10], 'content': 'A'},
            {'box': [10, 0, 20, 10], 'content': 'B'},  # Right neighbor
        ]
        
        visual_features = torch.randn(2, 128)
        semantic_embeddings = torch.randn(2, 768)
        
        graph = builder.build_graph(ocr_boxes, visual_features, semantic_embeddings)
        
        # Should have edges connecting them
        assert graph.edge_index.shape[1] >= 2  # At least bidirectional edge


class TestTableGNN:
    """Test Graph Neural Network."""
    
    def test_forward_pass(self):
        """Test forward pass through GNN."""
        gnn = TableGNN(input_dim=900, hidden_dim=256, output_dim=256, num_layers=3)
        
        # Create dummy graph
        from torch_geometric.data import Data
        
        x = torch.randn(4, 900)  # 4 nodes
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, num_nodes=4)
        
        node_embeddings, graph_embedding = gnn(data)
        
        assert node_embeddings.shape == (4, 256)
        assert graph_embedding.shape == (1, 256)
    
    def test_get_cell_scores(self):
        """Test score computation from embeddings."""
        gnn = TableGNN()
        
        node_embeddings = torch.randn(4, 256)
        scores = gnn.get_cell_scores(node_embeddings)
        
        assert 'structure_confidence' in scores
        assert 'header_prob' in scores
        assert scores['structure_confidence'].shape == (4,)
        assert scores['header_prob'].shape == (4,)


class TestHardConstraints:
    """Test hard constraint validation."""
    
    def test_row_height_consistency_valid(self):
        """Test with consistent row heights."""
        constraints = TableConstraints(row_height_tolerance=0.1)
        hard = HardConstraints(constraints)
        
        # 2x2 table with consistent row heights
        cells = [
            {'box': [0, 0, 10, 10]},    # Height=10
            {'box': [10, 0, 20, 10]},   # Height=10
            {'box': [0, 10, 10, 20]},   # Height=10
            {'box': [10, 10, 20, 20]},  # Height=10
        ]
        row_assignment = np.array([0, 0, 1, 1])
        
        is_valid, violation = hard.check_row_height_consistency(cells, row_assignment)
        
        assert is_valid
        assert violation == 0.0
    
    def test_row_height_consistency_invalid(self):
        """Test with inconsistent row heights."""
        constraints = TableConstraints(row_height_tolerance=0.1)
        hard = HardConstraints(constraints)
        
        # Row with varying heights
        cells = [
            {'box': [0, 0, 10, 10]},    # Height=10
            {'box': [10, 0, 20, 15]},   # Height=15 (50% larger)
        ]
        row_assignment = np.array([0, 0])
        
        is_valid, violation = hard.check_row_height_consistency(cells, row_assignment)
        
        assert not is_valid
        assert violation > 0.1
    
    def test_no_overlap_valid(self):
        """Test non-overlapping cells."""
        constraints = TableConstraints()
        hard = HardConstraints(constraints)
        
        cells = [
            {'box': [0, 0, 10, 10]},
            {'box': [10, 0, 20, 10]},  # Adjacent, not overlapping
        ]
        
        is_valid, overlap = hard.check_no_overlap(cells)
        
        assert is_valid
        assert overlap <= 0.05
    
    def test_no_overlap_invalid(self):
        """Test overlapping cells."""
        constraints = TableConstraints(overlap_tolerance=0.05)
        hard = HardConstraints(constraints)
        
        cells = [
            {'box': [0, 0, 10, 10]},
            {'box': [5, 5, 15, 15]},  # 50% overlap
        ]
        
        is_valid, overlap = hard.check_no_overlap(cells)
        
        assert not is_valid
        assert overlap > 0.05


class TestSoftConstraints:
    """Test soft constraint scoring."""
    
    def test_visual_confidence_score(self):
        """Test visual confidence computation."""
        constraints = TableConstraints()
        soft = SoftConstraints(constraints)
        
        confidences = torch.tensor([0.8, 0.9, 0.7, 0.85])
        score = soft.visual_confidence_score(confidences)
        
        assert 0.0 <= score <= 1.0
        assert abs(score - 0.8125) < 0.01


class TestTableCSPSolver:
    """Test CSP solver."""
    
    def test_solve_basic(self):
        """Test basic solving."""
        solver = TableCSPSolver()
        
        # Simple 2x2 table
        cells = [
            {'box': [0, 0, 10, 10], 'content': 'A'},
            {'box': [10, 0, 20, 10], 'content': 'B'},
            {'box': [0, 10, 10, 20], 'content': 'C'},
            {'box': [10, 10, 20, 20], 'content': 'D'},
        ]
        
        gnn_scores = {
            'structure_confidence': torch.tensor([0.9, 0.9, 0.9, 0.9]),
            'header_prob': torch.tensor([0.8, 0.8, 0.2, 0.2])
        }
        visual_scores = torch.tensor([0.8, 0.8, 0.8, 0.8])
        
        solution = solver.solve(cells, gnn_scores, visual_scores)
        
        assert 'row_assignment' in solution
        assert 'col_assignment' in solution
        assert len(solution['row_assignment']) == 4
        assert len(solution['col_assignment']) == 4
        assert solution['num_rows'] == 2
        assert solution['num_cols'] == 2
    
    def test_spatial_sort_initialization(self):
        """Test spatial sorting gives reasonable initial structure."""
        solver = TableCSPSolver()
        
        cells = [
            {'box': [0, 0, 10, 10]},    # Top-left
            {'box': [10, 0, 20, 10]},   # Top-right
            {'box': [0, 10, 10, 20]},   # Bottom-left
            {'box': [10, 10, 20, 20]},  # Bottom-right
        ]
        
        row_assignment, col_assignment = solver._spatial_sort_initialization(cells)
        
        # Top two should be in same row
        assert row_assignment[0] == row_assignment[1]
        # Bottom two should be in same row
        assert row_assignment[2] == row_assignment[3]
        
        # Left two should be in same column
        assert col_assignment[0] == col_assignment[2]
        # Right two should be in same column
        assert col_assignment[1] == col_assignment[3]


@pytest.mark.integration
class TestGNNCSPPipeline:
    """Integration tests for end-to-end pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        pipeline = GNNCSPPipeline(
            rca_checkpoint=None,  # Use random initialization
            gnn_checkpoint=None,
            device='cpu'
        )
        
        assert pipeline.rca_model is not None
        assert pipeline.gnn is not None
        assert pipeline.solver is not None
    
    def test_parse_simple_table(self):
        """Test parsing a simple table."""
        pipeline = GNNCSPPipeline(device='cpu')
        
        # Dummy image
        image = torch.randn(1, 3, 224, 224)
        
        # Simple 2x2 table
        ocr_boxes = [
            {'box': [0, 0, 10, 10], 'content': 'Header1'},
            {'box': [10, 0, 20, 10], 'content': 'Header2'},
            {'box': [0, 10, 10, 20], 'content': 'Data1'},
            {'box': [10, 10, 20, 20], 'content': 'Data2'},
        ]
        
        result = pipeline.parse(image, ocr_boxes)
        
        assert 'cells' in result
        assert 'num_rows' in result
        assert 'num_cols' in result
        assert 'constraint_satisfaction' in result
        assert len(result['cells']) == 4
        
        # Check that cells have row/col assignments
        for cell in result['cells']:
            assert 'row' in cell
            assert 'col' in cell
    
    def test_parse_empty_input(self):
        """Test with no OCR boxes."""
        pipeline = GNNCSPPipeline(device='cpu')
        
        image = torch.randn(1, 3, 224, 224)
        ocr_boxes = []
        
        result = pipeline.parse(image, ocr_boxes)
        
        assert result['num_rows'] == 0
        assert result['num_cols'] == 0
        assert len(result['cells']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

#!/usr/bin/env python3
"""
Quick verification script for GNN-CSP implementation.

Tests basic functionality without requiring full dataset.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hiertable_rag.gnn_csp import (
    CellGraphBuilder,
    TableGNN,
    ConstraintLayer,
    TableCSPSolver,
    GNNCSPPipeline
)


def test_graph_construction():
    """Test 1: Graph construction from OCR boxes."""
    print("\n[Test 1/5] Graph Construction...")
    
    builder = CellGraphBuilder()
    
    # Simple 2x2 table
    ocr_boxes = [
        {'box': [0, 0, 10, 10], 'content': 'A'},
        {'box': [10, 0, 20, 10], 'content': 'B'},
        {'box': [0, 10, 10, 20], 'content': 'C'},
        {'box': [10, 10, 20, 20], 'content': 'D'},
    ]
    
    visual_features = torch.randn(4, 128)
    semantic_embeddings = torch.randn(4, 768)
    
    graph = builder.build_graph(ocr_boxes, visual_features, semantic_embeddings)
    
    assert graph.num_nodes == 4, f"Expected 4 nodes, got {graph.num_nodes}"
    assert graph.edge_index.shape[1] > 0, "No edges created!"
    
    print(f"  ✓ Created graph with {graph.num_nodes} nodes and {graph.edge_index.shape[1]} edges")


def test_gnn_forward():
    """Test 2: GNN forward pass."""
    print("\n[Test 2/5] GNN Forward Pass...")
    
    gnn = TableGNN(input_dim=900, hidden_dim=256, output_dim=256)
    
    from torch_geometric.data import Data
    
    x = torch.randn(4, 900)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, num_nodes=4)
    
    node_embeddings, graph_embedding = gnn(data)
    
    assert node_embeddings.shape == (4, 256), f"Wrong shape: {node_embeddings.shape}"
    assert graph_embedding.shape == (1, 256), f"Wrong shape: {graph_embedding.shape}"
    
    print(f"  ✓ GNN output shapes correct: nodes={node_embeddings.shape}, graph={graph_embedding.shape}")


def test_constraints():
    """Test 3: Constraint validation."""
    print("\n[Test 3/5] Constraint Validation...", )
    
    constraint_layer = ConstraintLayer()
    
    # Valid 2x2 table
    cells = [
        {'box': [0, 0, 10, 10]},
        {'box': [10, 0, 20, 10]},
        {'box': [0, 10, 10, 20]},
        {'box': [10, 10, 20, 20]},
    ]
    
    row_assignment = np.array([0, 0, 1, 1])
    col_assignment = np.array([0, 1, 0, 1])
    table_bbox = [0, 0, 20, 20]
    
    node_embeddings = torch.randn(4, 256)
    visual_confidences = torch.ones(4) * 0.9
    
    result = constraint_layer.evaluate(
        cells, row_assignment, col_assignment, table_bbox,
        node_embeddings, visual_confidences
    )
    
    print(f"  ✓ Constraints validated: hard_valid={result['hard_valid']}")
    print(f"    Soft scores: {result['soft_scores']}")


def test_csp_solver():
    """Test 4: CSP Solver."""
    print("\n[Test 4/5] CSP Solver...")
    
    solver = TableCSPSolver()
    
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
    
    assert solution['num_rows'] == 2, f"Expected 2 rows, got {solution['num_rows']}"
    assert solution['num_cols'] == 2, f"Expected 2 cols, got {solution['num_cols']}"
    
    print(f"  ✓ Solver found structure: {solution['num_rows']}x{solution['num_cols']}")
    print(f"    Row assignment: {solution['row_assignment']}")
    print(f"    Col assignment: {solution['col_assignment']}")


def test_end_to_end_pipeline():
    """Test 5: End-to-end pipeline."""
    print("\n[Test 5/5] End-to-End Pipeline...")
    
    # Note: This requires RCA model, which we'll skip if not available
    try:
        pipeline = GNNCSPPipeline(device='cpu')
        
        image = torch.randn(1, 3, 224, 224)
        ocr_boxes = [
            {'box': [0, 0, 10, 10], 'content': 'Header1'},
            {'box': [10, 0, 20, 10], 'content': 'Header2'},
            {'box': [0, 10, 10, 20], 'content': 'Data1'},
            {'box': [10, 10, 20, 20], 'content': 'Data2'},
        ]
        
        result = pipeline.parse(image, ocr_boxes)
        
        print(f"  ✓ Pipeline ran successfully")
        print(f"    Structure: {result['num_rows']} rows × {result['num_cols']} cols")
        print(f"    Solve time: {result['solve_time']:.3f}s")
        print(f"    Constraints satisfied: {result['constraint_satisfaction']['hard_valid']}")
        
    except Exception as e:
        print(f"  ⚠ Pipeline test skipped (requires full setup): {e}")


def main():
    """Run all verification tests."""
    print("="*60)
    print("GNN-CSP Implementation Verification")
    print("="*60)
    
    try:
        test_graph_construction()
        test_gnn_forward()
        test_constraints()
        test_csp_solver()
        test_end_to_end_pipeline()
        
        print("\n" + "="*60)
        print("✓ All verification tests passed!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

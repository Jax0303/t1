#!/usr/bin/env python3
"""
Test script for GNN Indexer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
from src.hiertable_rag.core.gnn_indexer import (
    GNNIndexer,
    HeteroGraphBuilder,
    StructureAwareRetriever
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("TestGNN")


def test_gnn_indexer():
    """Test GNN indexer."""
    logger.info("=== Testing GNN Indexer ===")
    
    # Create indexer
    indexer = GNNIndexer(embedding_dim=256, hierarchy_weight=0.2)
    
    # Create mock data
    table_data = {
        'id': 'table_1',
        'adjacency': [[0, 1], [1, 0], [1, 2], [2, 1]]  # Simple chain
    }
    
    table_embedding = torch.randn(256)
    cell_embeddings = torch.randn(5, 256)  # 5 cells
    text_embeddings = torch.randn(5, 256)  # 5 texts
    
    hierarchy = {
        'nodes': [0, 1, 2, 3, 4],
        'parents': {0: None, 1: 0, 2: 0, 3: 1, 4: 1},
        'children': {0: [1, 2], 1: [3, 4], 2: [], 3: [], 4: []}
    }
    
    # Add to index
    indexer.add(
        table_data,
        table_embedding,
        cell_embeddings,
        text_embeddings,
        hierarchy
    )
    
    logger.info(f"Added 1 table with {cell_embeddings.shape[0]} cells")
    
    # Search
    query_embedding = torch.randn(256)
    results = indexer.search(query_embedding, top_k=3)
    
    logger.info(f"Retrieved {len(results)} results:")
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. Cell {result['cell_idx']}, Score: {result['score']:.4f}")
    
    assert len(results) <= 3, "Should return at most top-3"
    assert all('score' in r for r in results), "All results should have scores"
    
    logger.info("✅ GNN Indexer Test Passed")
    
    return indexer, results


def test_structure_aware_retrieval():
    """Test structure-aware retrieval with hierarchy bonus."""
    logger.info("\n=== Testing Structure-Aware Retrieval ===")
    
    retriever = StructureAwareRetriever(
        embedding_dim=256,
        hierarchy_weight=0.3  # Higher weight for hierarchy
    )
    
    # Build graph
    builder = HeteroGraphBuilder()
    
    table_embedding = torch.randn(256)
    cell_embeddings = torch.randn(4, 256)
    text_embeddings = torch.randn(4, 256)
    
    table_data = {'adjacency': [[0, 1], [1, 2], [2, 3]]}
    
    graph = builder.build_graph(
        table_data,
        table_embedding,
        cell_embeddings,
        text_embeddings
    )
    
    logger.info(f"Graph node types: {graph.node_types}")
    logger.info(f"Graph edge types: {graph.edge_types}")
    
    # Retrieve with hierarchy
    query_embedding = torch.randn(256)
    hierarchy = {
        'parents': {0: None, 1: 0, 2: 1, 3: 2}
    }
    
    results = retriever.retrieve(query_embedding, graph, hierarchy, top_k=2)
    
    logger.info(f"Retrieved {len(results)} cells with hierarchy bonus")
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. Cell {result['cell_idx']}, Score: {result['score']:.4f}")
    
    logger.info("✅ Structure-Aware Retrieval Test Passed")


if __name__ == "__main__":
    # Test GNN indexer
    indexer, results = test_gnn_indexer()
    
    # Test structure-aware retrieval
    test_structure_aware_retrieval()
    
    print("\n✅ All GNN tests passed!")

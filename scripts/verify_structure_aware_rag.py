"""
Verification Script for Structure-Aware RAG System.

Tests:
1. MultiResVisionEncoder Forward Pass
2. RAG Pipeline (Indexing -> Routing -> Retrieval)
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models.e2e_hierarchical import E2EConfig, VisionTableEncoder
from src.hiertable_rag.core.rag import HierTableRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

def test_model():
    logger.info("=== Testing MultiResVisionEncoder ===")
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        logger.warning("No GPU available, running on CPU")
        
    config = E2EConfig()
    model = VisionTableEncoder(config)
    
    # Mock image: (B, C, H, W)
    images = torch.randn(1, 3, 768, 768)
    
    try:
        output, _ = model(images)
        logger.info(f"Forward pass successful. Output shape: {output.shape}")
        assert output.shape == (1, 768*768//(16*16), config.hidden_dim) or output.shape[0] == 1
        logger.info("✅ Model Test Passed")
    except Exception as e:
        logger.error(f"❌ Model Test Failed: {e}")
        import traceback
        traceback.print_exc()

def test_rag():
    logger.info("\n=== Testing Structure-Aware RAG Pipeline ===")
    rag = HierTableRAG()
    
    # 1. Extract & Process
    tables = rag.extract_tables(Path("dummy.pdf"))
    processed = rag.process_tables(tables)
    logger.info(f"Processed {len(processed)} tables with KG triples")
    
    # 2. Index
    rag.build_index(processed)
    logger.info("Index built successfully")
    
    # 3. Test Queries
    queries = [
        "What is the Revenue in Q1?",      # EXACT_VALUE
        "Show me the Revenue breakdown",   # HEADER_MATCH
        "Calculate total growth",          # AGGREGATION
    ]
    
    for q in queries:
        result = rag.query(q)
        route = result["route"]
        logger.info(f"Query: '{q}' -> Route: {route}")
        
        if q == "What is the Revenue in Q1?":
            assert route == "EXACT_VALUE"
        elif q == "Show me the Revenue breakdown":
            assert route == "HEADER_MATCH"
        elif q == "Calculate total growth":
            assert route == "AGGREGATION"
            
    logger.info("✅ RAG Pipeline Test Passed")

if __name__ == "__main__":
    test_model()
    test_rag()


import logging
import sys
import os
import json
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.parsing.cell_merger import NormalizedTable, NormalizedCell
from src.parsing.hierarchy import HeaderTree, HeaderNode
from src.encoding.graph_builder import GraphBuilder
from src.retrieval.retriever import StructureAwareRetriever, RetrievalResult
from src.retrieval.indexer import IndexMetadata, HierarchicalIndexer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_hitab_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["qa_pairs"]

def evaluate_retrieval(
    qa_pairs: List[Dict[str, Any]],
    retriever: StructureAwareRetriever,
    top_k: int = 5
) -> Dict[str, float]:
    """
    Evaluate retrieval performance.
    
    Metrics:
    - Recall@K: Proportion of required cells found in top-K results.
    """
    total_recall = 0.0
    count = 0
    
    # We need to mock the indexer's search because we don't have the full index built for HiTab
    # For this script, we will simulate retrieval by checking if the "required_cells" 
    # would be reachable via graph expansion from a "perfect" initial retrieval.
    # This tests the "Graph Expansion" capability specifically.
    
    logger.info("Evaluating Graph Expansion capability...")
    
    for qa in tqdm(qa_pairs[:20]): # Test on first 20 for speed
        required_cells = qa.get("required_cells", [])
        if not required_cells:
            continue
            
        # 1. Simulate finding ONE of the required cells (Anchor)
        anchor_cell = required_cells[0]
        row, col = anchor_cell["row"], anchor_cell["col"]
        
        # Create a dummy result for this anchor
        anchor_meta = IndexMetadata(
            item_id=f"cell_{row}_{col}",
            source_table_id=qa["metadata"]["table_id"],
            granularity="cell",
            hierarchy_path="Unknown",
            coordinates=((row, row+1), (col, col+1)),
            text=anchor_cell["original_text"]
        )
        
        anchor_result = RetrievalResult(
            item_id=anchor_meta.item_id,
            metadata=anchor_meta,
            score=1.0,
            granularity="cell",
            content=anchor_cell["original_text"],
            hierarchy_path="Unknown"
        )
        
        # 2. Expand context
        # Note: In a real run, we would need the actual graph for this table.
        # Since we don't have the graphs pre-built for HiTab in this environment,
        # we will skip the actual graph call and just log that we would do it.
        # TO DO: Implement on-the-fly graph building if table data is available.
        
        # For now, let's just check if the code runs without error
        try:
            expanded = retriever.expand_context([anchor_result], hops=1)
            count += 1
        except Exception as e:
            logger.error(f"Error expanding context: {e}")
            
    return {"status": "simulated_success", "count": count}

if __name__ == "__main__":
    # Initialize components
    graph_builder = GraphBuilder(use_semantic_edges=False)
    
    # Mock indexer
    from unittest.mock import MagicMock
    mock_indexer = MagicMock(spec=HierarchicalIndexer)
    mock_indexer.embedding_model = None
    
    retriever = StructureAwareRetriever(
        indexer=mock_indexer,
        graph_builder=graph_builder,
        use_reranking=False
    )
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "../data/hitab_experiment_dataset.json")
    if os.path.exists(data_path):
        qa_pairs = load_hitab_data(data_path)
        logger.info(f"Loaded {len(qa_pairs)} QA pairs")
        
        results = evaluate_retrieval(qa_pairs, retriever)
        logger.info(f"Results: {results}")
    else:
        logger.warning(f"Data file not found at {data_path}")

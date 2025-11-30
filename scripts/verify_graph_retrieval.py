
import logging
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

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

def create_dummy_data():
    """Create dummy table and header tree."""
    # Create a simple 3x3 table with 1 header row
    # Header: Col0 | Col1 | Col2
    # Row 0:  D00  | D01  | D02
    # Row 1:  D10  | D11  | D12
    
    # Header Node
    root = HeaderNode(text="ROOT", level=0, span=3, col_end=3, row_end=3)
    h0 = HeaderNode(text="Col0", level=1, span=1, col_start=0, col_end=1, row_end=1)
    h1 = HeaderNode(text="Col1", level=1, span=1, col_start=1, col_end=2, row_end=1)
    h2 = HeaderNode(text="Col2", level=1, span=1, col_start=2, col_end=3, row_end=1)
    root.add_child(h0)
    root.add_child(h1)
    root.add_child(h2)
    
    header_tree = HeaderTree(root=root)
    
    # Normalized Table
    cells = []
    for r in range(3):
        row_cells = []
        for c in range(3):
            content = f"D{r}{c}" if r > 0 else f"Col{c}"
            cell = NormalizedCell(row=r, col=c, content=content)
            row_cells.append(cell)
        cells.append(row_cells)
        
    table = NormalizedTable(cells=cells, num_rows=3, num_cols=3)
    
    return table, header_tree

def test_graph_retrieval():
    logger.info("Starting Graph Retrieval Verification")
    
    # 1. Create Data
    table, header_tree = create_dummy_data()
    
    # 2. Build Graph
    # Disable semantic edges to avoid sentence-transformers dependency in test
    graph_builder = GraphBuilder(use_semantic_edges=False)
    graph = graph_builder.build_table_graph(table, header_tree)
    
    logger.info(f"Graph built with {graph.number_of_nodes()} nodes")
    
    # 3. Save Graph
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        graph_json = graph_builder.to_json(graph)
        f.write(graph_json)
        graph_path = f.name
        
    try:
        # 4. Initialize Retriever
        # Mock indexer
        mock_indexer = MagicMock(spec=HierarchicalIndexer)
        mock_indexer.embedding_model = None
        
        retriever = StructureAwareRetriever(
            indexer=mock_indexer,
            graph_builder=graph_builder,
            use_reranking=False
        )
        
        # 5. Load Graph
        retriever.load_graph(graph_path)
        
        if retriever.graph is None:
            logger.error("Failed to load graph!")
            return False
            
        # 6. Test expand_context
        # Create a dummy result for cell (1, 1) -> "D11"
        # We expect expansion to include neighbors like D10, D12, D01, D21
        
        initial_meta = IndexMetadata(
            item_id="test_id",
            source_table_id="table_1",
            granularity="cell",
            hierarchy_path="Table > Col1",
            coordinates=((1, 2), (1, 2)), # Row 1, Col 1
            text="D11"
        )
        
        initial_result = RetrievalResult(
            item_id="test_id",
            metadata=initial_meta,
            score=1.0,
            granularity="cell",
            content="D11",
            hierarchy_path="Table > Col1"
        )
        
        logger.info("Testing expand_context with 1 hop...")
        expanded = retriever.expand_context([initial_result], hops=1)
        
        logger.info(f"Initial results: 1")
        logger.info(f"Expanded results: {len(expanded)}")
        
        for res in expanded:
            logger.info(f" - {res.content} ({res.granularity})")
            
        # Verify we got more results
        if len(expanded) > 1:
            logger.info("SUCCESS: Context expanded!")
            return True
        else:
            logger.error("FAILURE: Context did not expand.")
            return False
            
    finally:
        if os.path.exists(graph_path):
            os.remove(graph_path)

if __name__ == "__main__":
    success = test_graph_retrieval()
    if not success:
        sys.exit(1)

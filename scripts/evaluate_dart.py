
import logging
import sys
import os
import json
import random
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.parsing.extractor import TableObject, Cell
from src.parsing.cell_merger import NormalizedTable, CellMergeHandler
from src.parsing.hierarchy import HierarchyExtractor, HeaderTree
from src.encoding.graph_builder import GraphBuilder
from src.retrieval.retriever import StructureAwareRetriever, RetrievalResult
from src.retrieval.indexer import IndexMetadata, HierarchicalIndexer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dart_tables(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def convert_to_table_object(dart_table: Dict[str, Any]) -> TableObject:
    """Convert DART JSON format to TableObject."""
    rows_data = dart_table["rows"]
    cells = []
    
    for r_idx, row_data in enumerate(rows_data):
        current_row = []
        for c_idx, cell_data in enumerate(row_data):
            # DART data has 'text' field
            text = cell_data.get("text", "")
            if text == "None": text = ""
            
            row_span = int(cell_data.get("row_span", 1))
            col_span = int(cell_data.get("col_span", 1))
            
            # Heuristic for header: first row or explicit header tag if available
            # For DART, we don't have explicit header tag in this JSON, so assume row 0 is header
            is_header = (r_idx == 0)
            
            cell = Cell(
                row=r_idx,
                col=c_idx, 
                content=str(text),
                rowspan=row_span,
                colspan=col_span,
                is_header=is_header
            )
            current_row.append(cell)
            
        cells.append(current_row)
        
    return TableObject(cells=cells)

def evaluate_dart_graph(dart_tables: List[Dict[str, Any]]):
    logger.info(f"Loaded {len(dart_tables)} DART tables")
    
    # Pick a table that looks interesting (has enough rows/cols)
    target_table = None
    for t in dart_tables:
        if t["row_count"] > 3 and t["col_count"] > 2:
            target_table = t
            break
            
    if not target_table:
        logger.warning("No suitable table found")
        return

    logger.info(f"Selected table: {target_table.get('table_id', 'Unknown')}")
    
    # 1. Convert to TableObject
    table_obj = convert_to_table_object(target_table)
    logger.info(f"Converted to TableObject: {table_obj.num_rows()}x{table_obj.num_cols()}")
    
    # 2. Extract Hierarchy
    extractor = HierarchyExtractor()
    header_tree = extractor.extract_header_tree(table_obj)
    logger.info(f"Extracted Header Tree (Max Level: {header_tree.max_level})")
    
    # 3. Normalize Table for GraphBuilder
    merge_handler = CellMergeHandler()
    norm_table = merge_handler.normalize_merged_cells(table_obj)
    logger.info(f"Normalized Table: {norm_table.num_rows}x{norm_table.num_cols}")
    
    # 4. Build Graph
    graph_builder = GraphBuilder(use_semantic_edges=False) # Disable semantic for speed
    graph = graph_builder.build_table_graph(norm_table, header_tree)
    logger.info(f"Built Graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # 5. Setup Retriever
    from unittest.mock import MagicMock
    mock_indexer = MagicMock(spec=HierarchicalIndexer)
    mock_indexer.embedding_model = None
    
    retriever = StructureAwareRetriever(
        indexer=mock_indexer,
        graph_builder=graph_builder,
        use_reranking=False
    )
    
    # Manually set the graph since we built it in memory
    retriever.graph = graph
    
    # 6. Test Expansion on a random data cell
    # Find a data cell (not header)
    data_cell = None
    # Start searching from the row after the last header level
    # max_level 1 = row 0 is header, data starts at row 1
    # max_level 2 = rows 0,1 are headers, data starts at row 2
    start_row = header_tree.max_level
    
    # Visualize table to debug
    print("Table Visualization:")
    print(merge_handler.visualize_coordinate_grid(norm_table))
    
    for r in range(start_row, norm_table.num_rows):
        for c in range(norm_table.num_cols):
            cell = norm_table.get_cell(r, c)
            if cell:
                logger.info(f"Checking cell ({r}, {c}): '{cell.content}'")
                if cell.content.strip():
                    data_cell = cell
                    break
        if data_cell: break
        
    if data_cell:
        logger.info(f"Testing expansion on cell: '{data_cell.content}' at ({data_cell.row}, {data_cell.col})")
        
        # Create dummy result
        meta = IndexMetadata(
            item_id=f"cell_{data_cell.row}_{data_cell.col}",
            source_table_id="dart_test",
            granularity="cell",
            hierarchy_path="Unknown",
            coordinates=((data_cell.row, data_cell.row+1), (data_cell.col, data_cell.col+1)),
            text=data_cell.content
        )
        
        result = RetrievalResult(
            item_id=meta.item_id,
            metadata=meta,
            score=1.0,
            granularity="cell",
            content=data_cell.content,
            hierarchy_path="Unknown"
        )
        
        expanded = retriever.expand_context([result], hops=1)
        
        logger.info(f"Expanded results: {len(expanded)}")
        for res in expanded:
            logger.info(f" - [{res.granularity}] {res.content}")
            
    else:
        logger.warning("No data cell found to test")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "../data/dart_extracted_tables.json")
    if os.path.exists(data_path):
        tables = load_dart_tables(data_path)
        evaluate_dart_graph(tables)
    else:
        logger.warning(f"Data file not found at {data_path}")

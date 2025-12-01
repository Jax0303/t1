"""
Main RAG pipeline for hierarchical table retrieval and generation.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


from src.hiertable_rag.core.indexer import MultiGranularityIndexer
from src.hiertable_rag.core.router import AdaptiveRouter
from src.hiertable_rag.core.graph import KnowledgeGraphBuilder
import numpy as np

class HierTableRAG:
    """
    Hierarchical Table-based Retrieval Augmented Generation system.
    
    Integrates:
    - Multi-granularity Indexing (Table, Text, Cell)
    - Adaptive Routing (Exact, Header, Aggregation)
    - Knowledge Graph Construction
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        vector_store_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the HierTable-RAG system.
        """
        self.config_path = config_path
        self.embedding_model = embedding_model
        self.vector_store_path = vector_store_path
        
        # Initialize components
        self.indexer = MultiGranularityIndexer()
        self.router = AdaptiveRouter()
        self.kg_builder = KnowledgeGraphBuilder()
        
        logger.info("HierTableRAG initialized with Structure-Aware components")

    def extract_tables(
        self, document_path: Path, output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Extract tables (Mock implementation for now)."""
        logger.info(f"Extracting tables from {document_path}")
        # Mock data
        return [{
            "id": "table_001",
            "title": "Quarterly Financial Report",
            "headers": [
                {"text": "Revenue", "parent": None},
                {"text": "Q1", "parent": "Revenue"},
                {"text": "Q2", "parent": "Revenue"}
            ],
            "cells": [
                {"text": "100M", "row_header": "Revenue", "col_header": "Q1"},
                {"text": "120M", "row_header": "Revenue", "col_header": "Q2"}
            ]
        }]

    def process_tables(
        self, tables: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process tables and build KG."""
        processed = []
        for table in tables:
            # Build KG
            triples = self.kg_builder.build_graph(table)
            table["triples"] = triples
            processed.append(table)
        return processed

    def build_index(
        self, processed_tables: List[Dict[str, Any]]
    ) -> None:
        """Build multi-granularity index."""
        logger.info("Building search index")
        
        for table in processed_tables:
            # 1. Table Level (Mock embedding)
            table_emb = np.random.rand(768).tolist()
            self.indexer.add_table(table, table_emb)
            
            # 2. Cell Level
            for cell in table.get("cells", []):
                cell_emb = np.random.rand(768).tolist()
                self.indexer.add_cell(cell, cell_emb)
                
            # 3. Text Level (Triples as text)
            for s, p, o in table.get("triples", []):
                text = f"{s} {p} {o}"
                text_emb = np.random.rand(768).tolist()
                self.indexer.add_text_chunk(text, table["id"], text_emb)

    def query(
        self, query_text: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query the RAG system with adaptive routing.
        """
        # 1. Route Query
        route = self.router.route(query_text)
        logger.info(f"Query: '{query_text}' -> Route: {route}")
        
        # 2. Retrieve based on route
        query_emb = np.random.rand(768).tolist() # Mock embedding
        
        if route == "EXACT_VALUE":
            results = self.indexer.search(query_emb, level="cell", top_k=top_k)
        elif route == "HEADER_MATCH":
            results = self.indexer.search(query_emb, level="text", top_k=top_k) # Search KG triples
        else: # AGGREGATION
            results = self.indexer.search(query_emb, level="table", top_k=top_k)
            
        return {
            "route": route,
            "results": results
        }

    def generate_response(
        self, query: str, retrieved_context: Dict[str, Any]
    ) -> str:
        """Generate response."""
        route = retrieved_context["route"]
        results = retrieved_context["results"]
        
        if not results:
            return "No relevant information found."
            
        # Mock generation
        return f"Based on {route} retrieval, found {len(results)} relevant items. Top result: {results[0]['item']}"



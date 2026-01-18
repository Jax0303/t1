"""
Main RAG pipeline for hierarchical table retrieval and generation.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

from src.hiertable_rag.core.indexer import MultiGranularityIndexer, MultiGranularityRetriever
from src.hiertable_rag.core.planner import RuleBasedPlanner, TrainablePlanner
from src.hiertable_rag.core.graph import TableGraph
from src.hiertable_rag.generators.agentic_generator import AgenticGenerator
from src.hiertable_rag.extraction.adaptive_extractor import AdaptiveExtractor
from src.hiertable_rag.core.evaluator import LatencyTracker

class HierTableRAG:
    """
    Agentic TableRAG system.
    """

    def __init__(
        self,
        planner_type: str = "rule_based",
        embedding_dim: int = 768
    ) -> None:
        self.indexer = MultiGranularityIndexer(embedding_dim=embedding_dim)
        self.retriever = MultiGranularityRetriever(self.indexer)
        
        if planner_type == "trainable":
            self.planner = TrainablePlanner()
        else:
            self.planner = RuleBasedPlanner()
            
        self.generator = AgenticGenerator()
        self.extractor = AdaptiveExtractor()
        self.latency_tracker = LatencyTracker()
        
        logger.info(f"HierTableRAG initialized with {planner_type} planner")

    def ingest_table(self, table_data: Dict[str, Any], force_strategy: str = None):
        """Runs adaptive extraction and indexes all levels with confidence."""
        # 1. Run Adaptive Extraction
        extraction_result = self.extractor.extract(table_data, force_strategy=force_strategy)
        confidence = extraction_result.get("tta_confidence", 1.0)
        refined_content = extraction_result.get("content", table_data.get("rows", []))
        
        table_id = table_data.get("id", "unknown")
        tg = TableGraph(table_id)
        
        # Build graph using potentially refined content
        # (Assuming build_from_dict can handle extracted structure)
        tg.build_from_dict(table_data) 
        
        # 2. Index with confidence weighting
        # Table level
        self.indexer.add_table(
            table_data, 
            np.random.rand(self.indexer.embedding_dim).tolist(),
            confidence=confidence
        )
        
        # Schema level
        for node_id, node in tg.schema_nodes.items():
            self.indexer.add_schema(
                {"id": node_id, "text": node.text}, 
                np.random.rand(self.indexer.embedding_dim).tolist(),
                confidence=confidence
            )
            
        # Cell level
        for cell_id, cell in tg.cell_nodes.items():
            context_text = tg.get_context_for_cell(cell_id)
            self.indexer.add_cell(
                {"id": cell_id, "text": context_text}, 
                np.random.rand(self.indexer.embedding_dim).tolist(),
                confidence=confidence
            )

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Runs the agentic RAG pipeline."""
        self.latency_tracker.start("total")
        
        # 1. Planning
        self.latency_tracker.start("planning")
        levels = self.planner.plan(query_text)
        self.latency_tracker.stop("planning")
        
        # 2. Retrieval
        self.latency_tracker.start("retrieval")
        query_emb = np.random.rand(self.indexer.embedding_dim).tolist() # In real app, use model.encode
        raw_results = self.retriever.retrieve(query_emb, levels=levels, top_k=top_k)
        evidence_pack = self.retriever.bundle_evidence(raw_results)
        self.latency_tracker.stop("retrieval")
        
        # 3. Generation
        self.latency_tracker.start("generation")
        response = self.generator.generate(query_text, evidence_pack)
        self.latency_tracker.stop("generation")
        
        self.latency_tracker.stop("total")
        
        return {
            "query": query_text,
            "levels_used": levels,
            "response": response,
            "evidence_count": len(evidence_pack),
            "latency_ms": self.latency_tracker.durations
        }



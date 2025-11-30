"""
Structure-aware retrieval module for hierarchical table search.

Implements adaptive retrieval strategies that consider query type and
table structure for optimal results.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

try:
    import numpy as np
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from src.retrieval.indexer import HierarchicalIndexer, HierarchicalIndex, IndexMetadata
from src.encoding.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query type classification."""
    AGGREGATE = "aggregate"  # Requires multiple cells (sum, average, comparison)
    LOOKUP = "lookup"  # Single cell lookup (specific value)
    STRUCTURAL = "structural"  # About table organization
    MULTI_TABLE = "multi_table"  # Requires joining across tables


@dataclass
class RetrievalResult:
    """
    Result from retrieval operation.
    
    Attributes:
        item_id: ID of retrieved item
        metadata: IndexMetadata for this item
        score: Relevance score
        granularity: Granularity level (table, subtable, row, cell)
        content: Cell/row content
        hierarchy_path: Full hierarchy path
    """
    item_id: str
    metadata: IndexMetadata
    score: float
    granularity: str
    content: str = ""
    hierarchy_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "score": self.score,
            "granularity": self.granularity,
            "content": self.content,
            "hierarchy_path": self.hierarchy_path,
            "metadata": self.metadata.to_dict(),
        }


@dataclass
class ContextBundle:
    """
    Bundle of retrieved context with hierarchy information.
    
    Attributes:
        retrieved_cells: List of cells with content
        hierarchy_context: Tree structure showing where cells fit
        table_metadata: Overall table info
        query: Original query
    """
    retrieved_cells: List[RetrievalResult] = field(default_factory=list)
    hierarchy_context: Dict[str, Any] = field(default_factory=dict)
    table_metadata: Dict[str, Any] = field(default_factory=dict)
    query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "retrieved_cells": [r.to_dict() for r in self.retrieved_cells],
            "hierarchy_context": self.hierarchy_context,
            "table_metadata": self.table_metadata,
        }


class StructureAwareRetriever:
    """
    Structure-aware retriever with adaptive strategies.
    
    Supports multiple retrieval strategies:
    - adaptive: Automatically select granularity based on query type
    - hierarchical: Retrieve at table level first, then drill down
    - cell_direct: Directly retrieve cells for specific value queries
    """

    def __init__(
        self,
        indexer: HierarchicalIndexer,
        graph_builder: Optional[GraphBuilder] = None,
        use_reranking: bool = True,
    ) -> None:
        """
        Initialize structure-aware retriever.
        
        Args:
            indexer: HierarchicalIndexer instance
            graph_builder: Optional GraphBuilder for context expansion
            use_reranking: Whether to use reranking
        """
        self.indexer = indexer
        self.graph_builder = graph_builder
        self.use_reranking = use_reranking
        self.graph: Optional[Any] = None  # nx.DiGraph
        
        # Get embedding model from indexer
        if hasattr(indexer, "embedding_model"):
            self.embedding_model = indexer.embedding_model
        else:
            self.embedding_model = None
        
        logger.info("StructureAwareRetriever initialized")

    def load_graph(self, path: str) -> None:
        """
        Load graph from file.
        
        Args:
            path: Path to graph file (JSON format)
        """
        if self.graph_builder is None:
            logger.warning("GraphBuilder not available, cannot load graph")
            return
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                json_str = f.read()
            self.graph = self.graph_builder.from_json(json_str)
            logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes")
        except Exception as e:
            logger.error(f"Failed to load graph from {path}: {e}")

    def classify_query_type(self, query: str) -> QueryType:
        """
        Classify query type using heuristics.
        
        Args:
            query: Query text
            
        Returns:
            QueryType enum value
        """
        query_lower = query.lower()
        
        # Aggregate queries: sum, average, total, compare, difference, etc.
        aggregate_keywords = [
            "sum", "total", "average", "mean", "compare", "difference",
            "more than", "less than", "greater", "maximum", "minimum",
            "highest", "lowest", "all", "every", "each",
        ]
        if any(keyword in query_lower for keyword in aggregate_keywords):
            return QueryType.AGGREGATE
        
        # Structural queries: about table organization
        structural_keywords = [
            "how many", "what columns", "what rows", "structure",
            "header", "organization", "layout", "format",
        ]
        if any(keyword in query_lower for keyword in structural_keywords):
            return QueryType.STRUCTURAL
        
        # Multi-table queries: multiple tables, join, across
        multi_table_keywords = [
            "across", "join", "multiple tables", "both tables",
            "all tables", "compare tables",
        ]
        if any(keyword in query_lower for keyword in multi_table_keywords):
            return QueryType.MULTI_TABLE
        
        # Default to LOOKUP for specific value queries
        return QueryType.LOOKUP

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "adaptive",
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant table segments with adaptive strategy.
        
        Strategies:
        - "adaptive": Automatically select granularity based on query type
        - "hierarchical": Retrieve at table level first, then drill down
        - "cell_direct": Directly retrieve cells (for specific value queries)
        
        Args:
            query: Query text
            top_k: Number of top results to return
            strategy: Retrieval strategy
            
        Returns:
            List of RetrievalResult objects
        """
        if self.indexer.index is None:
            raise ValueError("Index not loaded. Load index first.")
        
        logger.info(f"Retrieving with strategy '{strategy}': {query}")
        
        if strategy == "adaptive":
            return self._retrieve_adaptive(query, top_k)
        elif strategy == "hierarchical":
            return self._retrieve_hierarchical(query, top_k)
        elif strategy == "cell_direct":
            return self._retrieve_cell_direct(query, top_k)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Supported: adaptive, hierarchical, cell_direct"
            )

    def _retrieve_adaptive(
        self, query: str, top_k: int
    ) -> List[RetrievalResult]:
        """
        Adaptive retrieval based on query type.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        query_type = self.classify_query_type(query)
        
        logger.info(f"Query type: {query_type.value}")
        
        if query_type == QueryType.LOOKUP:
            # Retrieve at cell level
            return self._retrieve_cell_direct(query, top_k)
        elif query_type == QueryType.AGGREGATE:
            # Retrieve at row/subtable level first
            results = self._retrieve_at_granularity(query, "row", top_k * 2)
            if len(results) < top_k:
                # Also try subtable level
                subtable_results = self._retrieve_at_granularity(
                    query, "subtable", top_k
                )
                results.extend(subtable_results)
            return results[:top_k]
        elif query_type == QueryType.STRUCTURAL:
            # Retrieve at table level
            return self._retrieve_at_granularity(query, "table", top_k)
        else:  # MULTI_TABLE
            # Retrieve from multiple tables
            return self._retrieve_at_granularity(query, "table", top_k * 2)

    def _retrieve_hierarchical(
        self, query: str, top_k: int
    ) -> List[RetrievalResult]:
        """
        Hierarchical retrieval: start at table level, then drill down.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        # First, retrieve at table level
        table_results = self._retrieve_at_granularity(query, "table", top_k)
        
        if not table_results:
            return []
        
        # For each table result, retrieve finer granularity
        all_results = []
        for table_result in table_results[:3]:  # Top 3 tables
            table_id = table_result.metadata.source_table_id
            
            # Find subtables in this table
            subtable_results = [
                r
                for r in self._retrieve_at_granularity(query, "subtable", top_k)
                if r.metadata.source_table_id == table_id
            ]
            
            if subtable_results:
                all_results.extend(subtable_results)
            else:
                # Fall back to row level
                row_results = [
                    r
                    for r in self._retrieve_at_granularity(query, "row", top_k)
                    if r.metadata.source_table_id == table_id
                ]
                all_results.extend(row_results)
        
        # Rerank and return top_k
        if self.use_reranking:
            all_results = self.rerank(query, all_results, top_k)
        
        return all_results[:top_k]

    def _retrieve_cell_direct(
        self, query: str, top_k: int
    ) -> List[RetrievalResult]:
        """
        Direct cell-level retrieval.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        return self._retrieve_at_granularity(query, "cell", top_k)

    def _retrieve_at_granularity(
        self,
        query: str,
        granularity: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Retrieve at specific granularity level.
        
        Args:
            query: Query text
            granularity: One of "table", "subtable", "row", "cell"
            top_k: Number of results
            
        Returns:
            List of RetrievalResult objects
        """
        index = self.indexer.index
        if index is None:
            return []
        
        # Get FAISS index for this granularity
        faiss_index = index.get_index(granularity)
        if faiss_index is None or faiss_index.ntotal == 0:
            return []
        
        # Encode query
        if self.embedding_model is None:
            logger.error("Embedding model not available")
            return []
        
        query_embedding = self.embedding_model.encode([query])[0]
        query_vector = np.array([query_embedding]).astype("float32")
        
        # Search in FAISS
        k = min(top_k, faiss_index.ntotal)
        distances, indices = faiss_index.search(query_vector, k)
        
        # Build results
        results: List[RetrievalResult] = []
        
        # Find metadata for this granularity
        granularity_metadata = [
            (i, meta)
            for i, meta in enumerate(index.metadata)
            if meta.granularity == granularity
        ]
        
        # Map FAISS indices to metadata
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(granularity_metadata):
                continue
            
            meta_idx, metadata = granularity_metadata[idx]
            
            score = float(1.0 / (1.0 + dist))  # Convert distance to similarity
            
            result = RetrievalResult(
                item_id=metadata.item_id,
                metadata=metadata,
                score=score,
                granularity=granularity,
                content=metadata.text,
                hierarchy_path=metadata.hierarchy_path,
            )
            
            results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]

    def retrieve_with_hierarchy(
        self,
        query: str,
        table_id: Optional[str] = None,
        top_k: int = 5,
    ) -> ContextBundle:
        """
        Retrieve relevant cells/rows with full hierarchy path.
        
        Always includes all parent headers for retrieved cells.
        
        Args:
            query: Query text
            table_id: Optional table ID to restrict search
            top_k: Number of results
            
        Returns:
            ContextBundle with retrieved cells and hierarchy context
        """
        logger.info(f"Retrieving with hierarchy: {query}")
        
        # Retrieve initial results
        results = self.retrieve(query, top_k=top_k * 2, strategy="adaptive")
        
        # Filter by table_id if specified
        if table_id:
            results = [
                r for r in results if r.metadata.source_table_id == table_id
            ]
        
        # Expand to include full hierarchy paths
        enriched_results: List[RetrievalResult] = []
        
        for result in results[:top_k]:
            # Get full hierarchy path from metadata
            hierarchy_path = result.metadata.hierarchy_path
            
            # Find parent metadata to build tree structure
            parent_paths = []
            current_meta = result.metadata
            
            # Traverse up the hierarchy
            while current_meta.parent_ids:
                parent_id = current_meta.parent_ids[0]
                parent_meta = next(
                    (
                        m
                        for m in self.indexer.index.metadata
                        if m.item_id == parent_id
                    ),
                    None,
                )
                if parent_meta:
                    parent_paths.insert(0, parent_meta.hierarchy_path)
                    current_meta = parent_meta
                else:
                    break
            
            # Build hierarchy context
            result.hierarchy_path = " > ".join(parent_paths + [hierarchy_path])
            enriched_results.append(result)
        
        # Build hierarchy tree structure
        hierarchy_context = self._build_hierarchy_tree(enriched_results)
        
        # Get table metadata
        table_metadata = {}
        if enriched_results:
            first_result = enriched_results[0]
            table_id = first_result.metadata.source_table_id
            
            # Find table-level metadata
            table_meta = next(
                (
                    m
                    for m in self.indexer.index.metadata
                    if m.granularity == "table"
                    and m.source_table_id == table_id
                ),
                None,
            )
            if table_meta:
                table_metadata = {
                    "table_id": table_id,
                    "hierarchy_path": table_meta.hierarchy_path,
                    "coordinates": table_meta.coordinates,
                }
        
        return ContextBundle(
            retrieved_cells=enriched_results,
            hierarchy_context=hierarchy_context,
            table_metadata=table_metadata,
            query=query,
        )

    def _build_hierarchy_tree(
        self, results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        Build tree structure from results.
        
        Args:
            results: List of RetrievalResult objects
            
        Returns:
            Dictionary representing hierarchy tree
        """
        tree: Dict[str, Any] = {"root": "Table", "nodes": []}
        
        for result in results:
            path_parts = result.hierarchy_path.split(" > ")
            
            node = {
                "path": result.hierarchy_path,
                "granularity": result.granularity,
                "content": result.content,
                "score": result.score,
            }
            
            tree["nodes"].append(node)
        
        return tree

    def expand_context(
        self,
        initial_results: List[RetrievalResult],
        hops: int = 1,
    ) -> List[RetrievalResult]:
        """
        Expand context using graph structure.
        
        Given initial retrieved cells, expand to include neighboring cells
        using graph structure from GraphBuilder.
        
        Args:
            initial_results: Initial retrieval results
            hops: Number of hops to expand
            
        Returns:
            Expanded list of RetrievalResult objects
        """
        if self.graph_builder is None or self.graph is None:
            if self.graph_builder is None:
                logger.warning("GraphBuilder not available, skipping expansion")
            else:
                logger.warning("Graph not loaded, skipping expansion")
            return initial_results
        
        logger.info(f"Expanding context with {hops} hops")
        
        expanded_results = list(initial_results)
        expanded_ids = {r.item_id for r in initial_results}
        
        # Map graph nodes to metadata for reconstruction
        # This is a bit inefficient; in production, graph nodes should store metadata ID
        # For now, we'll try to match by coordinates/content if possible,
        # or just create simplified results from graph data
        
        for result in initial_results:
            # Find corresponding node in graph
            # We need a way to map RetrievalResult to graph node ID
            # Assuming metadata contains coordinates which map to graph node IDs
            
            if result.granularity != "cell":
                continue
                
            # Construct graph node ID from metadata coordinates
            # Format: data_{row}_{col}
            (row_start, _), (col_start, _) = result.metadata.coordinates
            node_id = f"data_{row_start}_{col_start}"
            
            if not self.graph.has_node(node_id):
                continue
                
            # Get context subgraph
            context_subgraph = self.graph_builder.get_cell_context(
                self.graph, node_id, hops=hops
            )
            
            # Convert subgraph nodes back to RetrievalResults
            for neighbor_id, attrs in context_subgraph.nodes(data=True):
                if attrs.get("type") != "data":
                    continue
                    
                # Skip if already in results (approximate check by content/coords)
                # Ideally we'd have the item_id in the graph
                
                # Create a new result for this neighbor
                # Note: We don't have the full metadata (like item_id) here
                # unless we stored it in the graph.
                # For this prototype, we'll create a transient result
                
                row, col = attrs.get("coordinates", (0, 0))
                
                # Check if we already have this cell
                is_new = True
                for r in expanded_results:
                    if r.metadata.coordinates == ((row, row + 1), (col, col + 1)):
                        is_new = False
                        break
                
                if is_new:
                    # Create simplified metadata
                    neighbor_meta = IndexMetadata(
                        item_id=f"graph_exp_{neighbor_id}",
                        source_table_id=result.metadata.source_table_id,
                        granularity="cell",
                        hierarchy_path=f"Graph Expansion > ({row}, {col})",
                        coordinates=((row, row + 1), (col, col + 1)),
                        text=attrs.get("text", ""),
                    )
                    
                    neighbor_result = RetrievalResult(
                        item_id=neighbor_meta.item_id,
                        metadata=neighbor_meta,
                        score=result.score * 0.8,  # Decay score for expanded nodes
                        granularity="cell",
                        content=attrs.get("text", ""),
                        hierarchy_path=neighbor_meta.hierarchy_path,
                    )
                    
                    expanded_results.append(neighbor_result)
        
        return expanded_results

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Rerank candidates considering multiple factors.
        
        Factors:
        - Semantic similarity (embeddings)
        - Structural relevance (hierarchy match)
        - Cell type (header vs data)
        
        Args:
            query: Query text
            candidates: List of candidate results
            top_k: Optional number of top results after reranking
            
        Returns:
            Reranked list of RetrievalResult objects
        """
        if not candidates:
            return []
        
        logger.info(f"Reranking {len(candidates)} candidates")
        
        # Compute query embedding if not already done
        if self.embedding_model is None:
            return candidates
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Rerank each candidate
        reranked: List[Tuple[RetrievalResult, float]] = []
        
        for candidate in candidates:
            # Base semantic score (already computed)
            semantic_score = candidate.score
            
            # Structural relevance: boost if hierarchy path matches query
            hierarchy_score = 0.0
            query_lower = query.lower()
            hierarchy_lower = candidate.hierarchy_path.lower()
            
            # Check if query mentions hierarchy path components
            path_parts = hierarchy_lower.split(" > ")
            for part in path_parts:
                if part in query_lower:
                    hierarchy_score += 0.1
            
            # Cell type score: prefer data cells for value queries
            type_score = 0.0
            if candidate.granularity == "cell":
                # Check if it's a data cell (not header)
                if "header" not in candidate.metadata.text.lower():
                    type_score = 0.05
            
            # Combined score
            combined_score = (
                semantic_score * 0.7
                + hierarchy_score * 0.2
                + type_score * 0.1
            )
            
            reranked.append((candidate, combined_score))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Update scores and return
        results = [
            RetrievalResult(
                item_id=c.item_id,
                metadata=c.metadata,
                score=score,
                granularity=c.granularity,
                content=c.content,
                hierarchy_path=c.hierarchy_path,
            )
            for c, score in reranked
        ]
        
        if top_k is not None:
            results = results[:top_k]
        
        return results

    def evaluate_recall_at_k(
        self,
        queries: List[Tuple[str, List[str]]],
        k_values: List[int] = [1, 5, 10],
        strategy: str = "adaptive",
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate Recall@k at different granularities.
        
        Args:
            queries: List of (query, relevant_item_ids) tuples
            k_values: List of k values to evaluate
            strategy: Retrieval strategy to use
            
        Returns:
            Dictionary mapping granularity to k to recall score
        """
        logger.info(f"Evaluating Recall@k for {len(queries)} queries")
        
        results_by_granularity: Dict[str, Dict[int, List[float]]] = {
            "table": {k: [] for k in k_values},
            "subtable": {k: [] for k in k_values},
            "row": {k: [] for k in k_values},
            "cell": {k: [] for k in k_values},
        }
        
        for query, relevant_ids in queries:
            # Retrieve results
            retrieved = self.retrieve(query, top_k=max(k_values), strategy=strategy)
            
            # Group by granularity
            by_granularity: Dict[str, List[RetrievalResult]] = {}
            for result in retrieved:
                gran = result.granularity
                if gran not in by_granularity:
                    by_granularity[gran] = []
                by_granularity[gran].append(result)
            
            # Compute Recall@k for each granularity
            for gran, results in by_granularity.items():
                retrieved_ids = [r.item_id for r in results]
                
                for k in k_values:
                    top_k_ids = retrieved_ids[:k]
                    relevant_retrieved = len(
                        set(top_k_ids) & set(relevant_ids)
                    )
                    recall = (
                        relevant_retrieved / len(relevant_ids)
                        if relevant_ids
                        else 0.0
                    )
                    results_by_granularity[gran][k].append(recall)
        
        # Average across queries
        final_results: Dict[str, Dict[int, float]] = {}
        for gran, k_dict in results_by_granularity.items():
            final_results[gran] = {}
            for k, recalls in k_dict.items():
                avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
                final_results[gran][k] = avg_recall
        
        return final_results


# Alias for backward compatibility
HierarchicalRetriever = StructureAwareRetriever

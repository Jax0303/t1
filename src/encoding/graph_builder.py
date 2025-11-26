"""
Graph-based representation builder for table relationships.

Creates NetworkX graphs from normalized tables and header trees,
enabling graph-based analysis and retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import logging
import json

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from src.parsing.cell_merger import NormalizedTable, NormalizedCell
from src.parsing.hierarchy import HeaderTree, HeaderNode

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Build graph representations of table structures and relationships.
    
    Creates NetworkX directed graphs where nodes represent cells and edges
    represent various relationships (hierarchical, spatial, semantic).
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        use_semantic_edges: bool = True,
    ) -> None:
        """
        Initialize graph builder.
        
        Args:
            embedding_model: Name of sentence-transformers model
                            (default: "all-MiniLM-L6-v2")
            use_semantic_edges: Whether to compute semantic edges
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "networkx is required. Install with: pip install networkx"
            )
        
        self.use_semantic_edges = use_semantic_edges
        self.embedding_model_name = (
            embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_model: Optional[Any] = None
        
        if use_semantic_edges and not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "sentence-transformers not available. "
                "Semantic edges will be disabled."
            )
            self.use_semantic_edges = False
        
        logger.info("GraphBuilder initialized")

    def build_table_graph(
        self,
        table: NormalizedTable,
        header_tree: HeaderTree,
    ) -> nx.DiGraph:
        """
        Create directed graph from normalized table and header tree.
        
        Nodes represent:
        - Header cells (from header_tree)
        - Data cells (from normalized table)
        - Column/row entities
        
        Node attributes:
        - text: Cell content
        - type: "header" or "data"
        - level: Hierarchy level (for headers)
        - coordinates: (row, col) tuple
        - cell_span: (rowspan, colspan) tuple
        
        Edges represent:
        - "parent_of": Header to sub-header relationship
        - "header_for": Header to data column relationship
        - "row_member": Cells in same row
        - "col_member": Cells in same column
        - "spatial_adjacent": Physically adjacent cells
        
        Args:
            table: NormalizedTable with cell data
            header_tree: HeaderTree with hierarchy information
            
        Returns:
            NetworkX DiGraph with nodes and edges
        """
        logger.info("Building table graph")
        
        graph = nx.DiGraph()
        
        # Add header nodes from header tree
        self._add_header_nodes(graph, header_tree)
        
        # Add data cell nodes from normalized table
        self._add_data_nodes(graph, table)
        
        # Add column/row entity nodes
        self._add_entity_nodes(graph, table)
        
        # Add hierarchical edges (parent_of, header_for)
        self._add_hierarchical_edges(graph, header_tree, table)
        
        # Add spatial edges (row_member, col_member, spatial_adjacent)
        self._add_spatial_edges(graph, table)
        
        logger.info(
            f"Built graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        
        return graph

    def _add_header_nodes(self, graph: nx.DiGraph, header_tree: HeaderTree) -> None:
        """
        Add header nodes from header tree.
        
        Args:
            graph: Graph to add nodes to
            header_tree: HeaderTree to extract nodes from
        """
        def add_node_recursive(node: HeaderNode) -> None:
            """Recursively add header nodes."""
            node_id = f"header_{node.level}_{node.col_start}_{node.col_end}"
            
            graph.add_node(
                node_id,
                text=node.text,
                type="header",
                level=node.level,
                coordinates=(node.row_start, node.col_start),
                cell_span=(node.row_span, node.span),
                col_start=node.col_start,
                col_end=node.col_end,
                row_start=node.row_start,
                row_end=node.row_end,
            )
            
            for child in node.children:
                add_node_recursive(child)
        
        add_node_recursive(header_tree.root)

    def _add_data_nodes(self, graph: nx.DiGraph, table: NormalizedTable) -> None:
        """
        Add data cell nodes from normalized table.
        
        Args:
            graph: Graph to add nodes to
            table: NormalizedTable to extract cells from
        """
        for row_idx in range(table.num_rows):
            for col_idx in range(table.num_cols):
                cell = table.get_cell(row_idx, col_idx)
                if cell is None:
                    continue
                
                # Skip if this is a merged cell reference (use root instead)
                if cell.is_merged and cell.merge_root != (row_idx, col_idx):
                    continue
                
                node_id = f"data_{row_idx}_{col_idx}"
                
                graph.add_node(
                    node_id,
                    text=cell.content,
                    type="data",
                    level=None,
                    coordinates=(row_idx, col_idx),
                    cell_span=(
                        cell.merge_span[0] if cell.merge_span else 1,
                        cell.merge_span[1] if cell.merge_span else 1,
                    ),
                    row=row_idx,
                    col=col_idx,
                    is_merged=cell.is_merged,
                )

    def _add_entity_nodes(
        self, graph: nx.DiGraph, table: NormalizedTable
    ) -> None:
        """
        Add column and row entity nodes.
        
        Args:
            graph: Graph to add nodes to
            table: NormalizedTable to extract entities from
        """
        # Add column entities
        for col_idx in range(table.num_cols):
            col_id = f"col_{col_idx}"
            graph.add_node(
                col_id,
                text=f"Column {col_idx}",
                type="column",
                level=None,
                coordinates=None,
                cell_span=None,
                col_index=col_idx,
            )
        
        # Add row entities
        for row_idx in range(table.num_rows):
            row_id = f"row_{row_idx}"
            graph.add_node(
                row_id,
                text=f"Row {row_idx}",
                type="row",
                level=None,
                coordinates=None,
                cell_span=None,
                row_index=row_idx,
            )

    def _add_hierarchical_edges(
        self,
        graph: nx.DiGraph,
        header_tree: HeaderTree,
        table: NormalizedTable,
    ) -> None:
        """
        Add hierarchical edges (parent_of, header_for).
        
        Args:
            graph: Graph to add edges to
            header_tree: HeaderTree for hierarchy
            table: NormalizedTable for data cells
        """
        def add_edges_recursive(node: HeaderNode) -> None:
            """Recursively add hierarchical edges."""
            node_id = f"header_{node.level}_{node.col_start}_{node.col_end}"
            
            # Add parent_of edges to children
            for child in node.children:
                child_id = (
                    f"header_{child.level}_{child.col_start}_{child.col_end}"
                )
                graph.add_edge(
                    node_id,
                    child_id,
                    relationship="parent_of",
                    weight=1.0,
                )
                add_edges_recursive(child)
            
            # Add header_for edges to data columns
            if node.is_leaf():
                # This header covers data columns from col_start to col_end
                for col_idx in range(node.col_start, node.col_end):
                    # Find data cells in this column (starting from data rows)
                    data_start_row = node.row_end
                    for row_idx in range(data_start_row, table.num_rows):
                        data_id = f"data_{row_idx}_{col_idx}"
                        if graph.has_node(data_id):
                            graph.add_edge(
                                node_id,
                                data_id,
                                relationship="header_for",
                                weight=1.0,
                            )
        
        add_edges_recursive(header_tree.root)

    def _add_spatial_edges(
        self, graph: nx.DiGraph, table: NormalizedTable
    ) -> None:
        """
        Add spatial edges (row_member, col_member, spatial_adjacent).
        
        Args:
            graph: Graph to add edges to
            table: NormalizedTable for spatial relationships
        """
        # Add row_member and col_member edges
        for row_idx in range(table.num_rows):
            row_id = f"row_{row_idx}"
            for col_idx in range(table.num_cols):
                col_id = f"col_{col_idx}"
                data_id = f"data_{row_idx}_{col_idx}"
                
                if graph.has_node(data_id):
                    # Row membership
                    graph.add_edge(
                        row_id,
                        data_id,
                        relationship="row_member",
                        weight=1.0,
                    )
                    
                    # Column membership
                    graph.add_edge(
                        col_id,
                        data_id,
                        relationship="col_member",
                        weight=1.0,
                    )
        
        # Add spatial_adjacent edges (4-directional)
        for row_idx in range(table.num_rows):
            for col_idx in range(table.num_cols):
                cell_id = f"data_{row_idx}_{col_idx}"
                if not graph.has_node(cell_id):
                    continue
                
                # Check adjacent cells (up, down, left, right)
                directions = [
                    (row_idx - 1, col_idx),  # up
                    (row_idx + 1, col_idx),  # down
                    (row_idx, col_idx - 1),  # left
                    (row_idx, col_idx + 1),  # right
                ]
                
                for adj_row, adj_col in directions:
                    if (
                        0 <= adj_row < table.num_rows
                        and 0 <= adj_col < table.num_cols
                    ):
                        adj_id = f"data_{adj_row}_{adj_col}"
                        if graph.has_node(adj_id):
                            graph.add_edge(
                                cell_id,
                                adj_id,
                                relationship="spatial_adjacent",
                                weight=0.5,  # Lower weight for spatial
                            )

    def add_semantic_edges(
        self,
        graph: nx.DiGraph,
        similarity_threshold: float = 0.8,
    ) -> nx.DiGraph:
        """
        Compute embeddings and add semantic similarity edges.
        
        Uses sentence-transformers to compute embeddings for cell content,
        then adds "semantically_similar" edges between cells with high similarity.
        
        Args:
            graph: Graph to add edges to
            similarity_threshold: Minimum cosine similarity for edge creation
            
        Returns:
            Updated graph with semantic edges
        """
        if not self.use_semantic_edges:
            logger.warning("Semantic edges disabled")
            return graph
        
        logger.info("Adding semantic edges")
        
        # Load embedding model if not already loaded
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name
                )
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                return graph
        
        # Collect cell texts and node IDs
        cell_texts: List[str] = []
        node_ids: List[str] = []
        
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") in ["header", "data"]:
                text = attrs.get("text", "").strip()
                if text:
                    cell_texts.append(text)
                    node_ids.append(node_id)
        
        if len(cell_texts) < 2:
            logger.warning("Not enough cells for semantic comparison")
            return graph
        
        # Compute embeddings
        try:
            embeddings = self.embedding_model.encode(
                cell_texts, show_progress_bar=False
            )
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            return graph
        
        # Compute pairwise similarities and add edges
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        similarity_matrix = cosine_similarity(embeddings)
        
        num_edges_added = 0
        for i, node_id_i in enumerate(node_ids):
            for j, node_id_j in enumerate(node_ids):
                if i >= j:  # Avoid duplicate edges
                    continue
                
                similarity = similarity_matrix[i][j]
                if similarity >= similarity_threshold:
                    graph.add_edge(
                        node_id_i,
                        node_id_j,
                        relationship="semantically_similar",
                        weight=float(similarity),
                        similarity=float(similarity),
                    )
                    num_edges_added += 1
        
        logger.info(f"Added {num_edges_added} semantic edges")
        
        return graph

    def get_cell_context(
        self,
        graph: nx.DiGraph,
        cell_id: str,
        hops: int = 2,
    ) -> nx.DiGraph:
        """
        Extract k-hop neighborhood around a cell.
        
        Returns subgraph containing relevant context for that cell,
        including all headers in the cell's hierarchy path.
        
        Args:
            graph: Full graph
            cell_id: ID of the target cell
            hops: Number of hops to include
            
        Returns:
            Subgraph containing cell context
        """
        if not graph.has_node(cell_id):
            logger.warning(f"Cell {cell_id} not found in graph")
            return nx.DiGraph()
        
        logger.info(f"Extracting {hops}-hop context for {cell_id}")
        
        # Get k-hop neighbors
        neighbors: Set[str] = {cell_id}
        current_level = {cell_id}
        
        for _ in range(hops):
            next_level: Set[str] = set()
            for node in current_level:
                # Add successors (outgoing edges)
                next_level.update(graph.successors(node))
                # Add predecessors (incoming edges)
                next_level.update(graph.predecessors(node))
            neighbors.update(next_level)
            current_level = next_level
        
        # Find all headers in hierarchy path
        cell_attrs = graph.nodes[cell_id]
        if cell_attrs.get("type") == "data":
            # Find header_for edges leading to this cell
            for pred in graph.predecessors(cell_id):
                pred_attrs = graph.nodes[pred]
                if (
                    pred_attrs.get("type") == "header"
                    and graph.has_edge(pred, cell_id)
                    and graph.edges[pred, cell_id].get("relationship")
                    == "header_for"
                ):
                    # Add this header and all its ancestors
                    self._add_header_path(graph, pred, neighbors)
        
        # Create subgraph
        subgraph = graph.subgraph(neighbors).copy()
        
        logger.info(
            f"Context subgraph: {subgraph.number_of_nodes()} nodes, "
            f"{subgraph.number_of_edges()} edges"
        )
        
        return subgraph

    def _add_header_path(
        self, graph: nx.DiGraph, header_id: str, node_set: Set[str]
    ) -> None:
        """
        Recursively add header path (all ancestors).
        
        Args:
            graph: Full graph
            header_id: Starting header node ID
            node_set: Set to add nodes to
        """
        node_set.add(header_id)
        
        # Find parent headers
        for pred in graph.predecessors(header_id):
            pred_attrs = graph.nodes[pred]
            if (
                pred_attrs.get("type") == "header"
                and graph.has_edge(pred, header_id)
                and graph.edges[pred, header_id].get("relationship")
                == "parent_of"
            ):
                self._add_header_path(graph, pred, node_set)

    def export_for_visualization(
        self,
        graph: nx.DiGraph,
        output_path: str,
        format: str = "gexf",
    ) -> None:
        """
        Export graph to formats compatible with visualization tools.
        
        Supports:
        - "gexf": Gephi format
        - "graphml": GraphML format (also for Gephi)
        - "dot": Graphviz DOT format
        - "json": JSON format for web visualization
        
        Color-codes nodes by type and hierarchy level.
        
        Args:
            graph: Graph to export
            output_path: Path to save exported file
            format: Export format ("gexf", "graphml", "dot", "json")
            
        Raises:
            ValueError: If format is not supported
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting graph to {format} format: {output_path}")
        
        # Add visualization attributes
        self._add_visualization_attributes(graph)
        
        if format == "gexf":
            nx.write_gexf(graph, output_path)
        elif format == "graphml":
            nx.write_graphml(graph, output_path)
        elif format == "dot":
            # Convert to undirected for DOT (or use DiGraph)
            nx.drawing.nx_pydot.write_dot(graph, output_path)
        elif format == "json":
            self._export_json(graph, output_path)
        else:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported: gexf, graphml, dot, json"
            )
        
        logger.info(f"Exported graph to {output_path}")

    def _add_visualization_attributes(self, graph: nx.DiGraph) -> None:
        """
        Add color and size attributes for visualization.
        
        Args:
            graph: Graph to add attributes to
        """
        # Color mapping by type
        type_colors = {
            "header": "#FF6B6B",  # Red
            "data": "#4ECDC4",  # Teal
            "column": "#95E1D3",  # Light teal
            "row": "#F38181",  # Light red
        }
        
        # Size mapping by level (headers)
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get("type", "data")
            
            # Color
            if node_type in type_colors:
                attrs["viz"] = {"color": {"r": 255, "g": 100, "b": 100}}
                # Convert hex to RGB for GEXF
                hex_color = type_colors[node_type].lstrip("#")
                r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
                attrs["viz"] = {"color": {"r": r, "g": g, "b": b}}
            
            # Size based on level
            level = attrs.get("level")
            if level is not None:
                size = 20 + (level * 5)  # Larger for higher levels
                attrs["size"] = size
            else:
                attrs["size"] = 10
            
            # Label
            text = attrs.get("text", "")
            if len(text) > 20:
                text = text[:17] + "..."
            attrs["label"] = text or node_id

    def _export_json(self, graph: nx.DiGraph, output_path: str) -> None:
        """
        Export graph to JSON format.
        
        Args:
            graph: Graph to export
            output_path: Path to save JSON file
        """
        # Use node-link format
        data = nx.node_link_data(graph)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def to_json(self, graph: nx.DiGraph) -> str:
        """
        Serialize graph to JSON string.
        
        Args:
            graph: Graph to serialize
            
        Returns:
            JSON string representation
        """
        data = nx.node_link_data(graph)
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def from_json(self, json_str: str) -> nx.DiGraph:
        """
        Deserialize graph from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            NetworkX DiGraph
        """
        data = json.loads(json_str)
        return nx.node_link_graph(data)

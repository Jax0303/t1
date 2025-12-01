"""
GNN-based Indexer for Structure-Aware Retrieval.

Implements heterogeneous graph construction and GraphSAGE-based
embedding generation for table-cell-text indexing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import logging

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, to_hetero
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    HeteroData = None

logger = logging.getLogger(__name__)


class HeteroGraphBuilder:
    """
    Heterogeneous Graph Builder.
    
    Constructs Table-Cell-Text heterogeneous graph from table data.
    """
    
    def __init__(self):
        """Initialize Heterogeneous Graph Builder."""
        logger.info("HeteroGraphBuilder initialized")
    
    def build_graph(
        self,
        table_data: Dict[str, Any],
        table_embedding: torch.Tensor,
        cell_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> HeteroData:
        """
        Build heterogeneous graph from table data.
        
        Args:
            table_data: Table metadata and structure
            table_embedding: Table-level embedding (D,)
            cell_embeddings: Cell embeddings (N_cells, D)
            text_embeddings: Text embeddings (N_texts, D)
        
        Returns:
            HeteroData graph
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required")
        
        graph = HeteroData()
        
        # Add node features
        graph['table'].x = table_embedding.unsqueeze(0)  # (1, D)
        graph['cell'].x = cell_embeddings  # (N_cells, D)
        graph['text'].x = text_embeddings  # (N_texts, D)
        
        # Add edges: table -> cells
        num_cells = cell_embeddings.shape[0]
        graph['table', 'contains', 'cell'].edge_index = torch.tensor([
            [0] * num_cells,  # All from table node 0
            list(range(num_cells))  # To each cell
        ], dtype=torch.long)
        
        # Add edges: cell -> text (assuming 1-to-1 mapping for simplicity)
        num_texts = min(num_cells, text_embeddings.shape[0])
        graph['cell', 'has_text', 'text'].edge_index = torch.tensor([
            list(range(num_texts)),
            list(range(num_texts))
        ], dtype=torch.long)
        
        # Add edges: cell -> cell (adjacency based on table structure)
        if 'adjacency' in table_data:
            adj_edges = table_data['adjacency']
            graph['cell', 'adjacent', 'cell'].edge_index = torch.tensor(
                adj_edges, dtype=torch.long
            ).t()
        else:
            # Create simple grid adjacency
            cell_adj = []
            for i in range(num_cells):
                for j in range(i + 1, min(i + 3, num_cells)):  # Connect to next 2 cells
                    cell_adj.append([i, j])
                    cell_adj.append([j, i])  # Bidirectional
            
            if cell_adj:
                graph['cell', 'adjacent', 'cell'].edge_index = torch.tensor(
                    cell_adj, dtype=torch.long
                ).t()
        
        return graph


class GraphSAGEIndexer(nn.Module):
    """
    GraphSAGE-based Indexer.
    
    Uses GNN to generate structure-aware embeddings for retrieval.
    """
    
    def __init__(
        self,
        hidden_channels: int = 256,
        num_layers: int = 2,
    ):
        """
        Initialize GraphSAGE Indexer.
        
        Args:
            hidden_channels: Hidden dimension
            num_layers: Number of GNN layers
        """
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required")
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Create a simple GNN model
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv(hidden_channels, hidden_channels)
            self.convs.append(conv)
        
        logger.info(f"GraphSAGEIndexer initialized (layers={num_layers})")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GraphSAGE.
        
        Args:
            x: Node features (N, D)
            edge_index: Edge indices (2, E)
        
        Returns:
            Updated node embeddings (N, D)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        return x


class StructureAwareRetriever(nn.Module):
    """
    Structure-Aware Retriever.
    
    Retrieves relevant cells using GNN embeddings and hierarchy bonus.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hierarchy_weight: float = 0.2,
    ):
        """
        Initialize Structure-Aware Retriever.
        
        Args:
            embedding_dim: Embedding dimension
            hierarchy_weight: Weight for hierarchy bonus
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hierarchy_weight = hierarchy_weight
        
        # GNN indexer
        self.gnn_indexer = GraphSAGEIndexer(
            hidden_channels=embedding_dim,
            num_layers=2
        )
        
        # Graph builder
        self.graph_builder = HeteroGraphBuilder()
        
        logger.info(f"StructureAwareRetriever initialized (α={hierarchy_weight})")
    
    def _compute_hierarchy_bonus(
        self,
        cell_idx: int,
        hierarchy: Dict[str, Any],
        query_embedding: torch.Tensor,
        cell_embeddings: torch.Tensor
    ) -> float:
        """
        Compute hierarchy bonus for a cell.
        
        Args:
            cell_idx: Cell index
            hierarchy: Hierarchy dictionary
            query_embedding: Query embedding
            cell_embeddings: All cell embeddings
        
        Returns:
            Hierarchy bonus score
        """
        if 'parents' not in hierarchy:
            return 0.0
        
        # Get ancestors
        ancestors = []
        current = cell_idx
        while hierarchy['parents'].get(current) is not None:
            parent = hierarchy['parents'][current]
            ancestors.append(parent)
            current = parent
        
        if not ancestors:
            return 0.0
        
        # Compute relevance of ancestors
        ancestor_embeddings = cell_embeddings[ancestors]
        ancestor_scores = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            ancestor_embeddings,
            dim=1
        )
        
        # Return average ancestor relevance
        return ancestor_scores.mean().item()
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        graph: HeteroData,
        hierarchy: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k cells using structure-aware scoring.
        
        Args:
            query_embedding: Query embedding (D,)
            graph: Heterogeneous graph
            hierarchy: Optional hierarchy information
            top_k: Number of results to return
        
        Returns:
            List of retrieval results with scores
        """
        # Get cell embeddings from graph
        cell_embeddings = graph['cell'].x
        
        # Apply GNN to get structure-aware embeddings
        if ('cell', 'adjacent', 'cell') in graph.edge_types:
            edge_index = graph['cell', 'adjacent', 'cell'].edge_index
            cell_embeddings_gnn = self.gnn_indexer(cell_embeddings, edge_index)
        else:
            cell_embeddings_gnn = cell_embeddings
        
        # Compute base similarity scores
        base_scores = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            cell_embeddings_gnn,
            dim=1
        )
        
        # Add hierarchy bonus if available
        if hierarchy is not None:
            hierarchy_bonuses = torch.tensor([
                self._compute_hierarchy_bonus(
                    i, hierarchy, query_embedding, cell_embeddings_gnn
                )
                for i in range(cell_embeddings_gnn.shape[0])
            ], device=base_scores.device)
            
            final_scores = base_scores + self.hierarchy_weight * hierarchy_bonuses
        else:
            final_scores = base_scores
        
        # Get top-k
        top_k = min(top_k, final_scores.shape[0])
        top_scores, top_indices = torch.topk(final_scores, top_k)
        
        # Format results
        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            results.append({
                'cell_idx': idx,
                'score': score,
                'embedding': cell_embeddings_gnn[idx].detach()
            })
        
        return results
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        table_data: Dict[str, Any],
        table_embedding: torch.Tensor,
        cell_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        hierarchy: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Forward pass: build graph and retrieve.
        
        Args:
            query_embedding: Query embedding
            table_data: Table metadata
            table_embedding: Table embedding
            cell_embeddings: Cell embeddings
            text_embeddings: Text embeddings
            hierarchy: Optional hierarchy
            top_k: Number of results
        
        Returns:
            Retrieval results
        """
        # Build graph
        graph = self.graph_builder.build_graph(
            table_data,
            table_embedding,
            cell_embeddings,
            text_embeddings
        )
        
        # Retrieve
        results = self.retrieve(query_embedding, graph, hierarchy, top_k)
        
        return results


class GNNIndexer:
    """
    GNN-based Indexer (wrapper for compatibility with existing code).
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hierarchy_weight: float = 0.2
    ):
        """
        Initialize GNN Indexer.
        
        Args:
            embedding_dim: Embedding dimension
            hierarchy_weight: Hierarchy bonus weight
        """
        self.retriever = StructureAwareRetriever(
            embedding_dim=embedding_dim,
            hierarchy_weight=hierarchy_weight
        )
        
        # Storage for indexed data
        self.graphs = []
        self.hierarchies = []
        
        logger.info("GNNIndexer initialized")
    
    def add(
        self,
        table_data: Dict[str, Any],
        table_embedding: torch.Tensor,
        cell_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        hierarchy: Optional[Dict[str, Any]] = None
    ):
        """
        Add table to index.
        
        Args:
            table_data: Table metadata
            table_embedding: Table embedding
            cell_embeddings: Cell embeddings
            text_embeddings: Text embeddings
            hierarchy: Optional hierarchy
        """
        graph = self.retriever.graph_builder.build_graph(
            table_data,
            table_embedding,
            cell_embeddings,
            text_embeddings
        )
        
        self.graphs.append(graph)
        self.hierarchies.append(hierarchy)
    
    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search across all indexed tables.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results
        
        Returns:
            Retrieval results
        """
        all_results = []
        
        for graph_idx, (graph, hierarchy) in enumerate(zip(self.graphs, self.hierarchies)):
            results = self.retriever.retrieve(
                query_embedding,
                graph,
                hierarchy,
                top_k=top_k
            )
            
            # Add table index to results
            for result in results:
                result['table_idx'] = graph_idx
            
            all_results.extend(results)
        
        # Sort by score and return top-k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]

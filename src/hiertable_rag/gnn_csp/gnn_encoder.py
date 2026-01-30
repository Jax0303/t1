"""
GNN Encoder for Table Structure Recognition

Implements:
1. CellGraphBuilder: Constructs spatial graph from OCR boxes
2. TableGNN: Graph Neural Network for learning cell embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import logging

logger = logging.getLogger(__name__)


class CellGraphBuilder:
    """
    Builds a spatial graph from OCR boxes where:
    - Nodes = OCR boxes (cells)
    - Edges = Spatial adjacency (4-connectivity: up, down, left, right)
    """
    
    def __init__(self, adjacency_threshold: float = 0.3):
        """
        Args:
            adjacency_threshold: IoU threshold for considering cells as neighbors
        """
        self.threshold = adjacency_threshold
    
    def build_graph(
        self,
        ocr_boxes: List[Dict[str, Any]],
        visual_features: torch.Tensor,
        semantic_embeddings: torch.Tensor
    ) -> Data:
        """
        Build graph from OCR boxes and features.
        
        Args:
            ocr_boxes: List of dicts with 'box' [x1, y1, x2, y2] and 'content' (text)
            visual_features: Tensor of shape [N, D_visual] from RCA
            semantic_embeddings: Tensor of shape [N, D_semantic] from RoBERTa
            
        Returns:
            PyTorch Geometric Data object with:
                - x: node features [N, D_visual + 4 + D_semantic]
                - edge_index: [2, E] edge connectivity
                - pos: [N, 4] bounding box coordinates
        """
        num_cells = len(ocr_boxes)
        
        # Extract position features
        positions = torch.zeros((num_cells, 4))  # [x, y, w, h]
        for i, box_data in enumerate(ocr_boxes):
            box = box_data['box']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            positions[i] = torch.tensor([
                (x1 + x2) / 2,  # center x
                (y1 + y2) / 2,  # center y
                x2 - x1,        # width
                y2 - y1         # height
            ])
        
        # Normalize positions to [0, 1]
        if num_cells > 0:
            max_x = positions[:, 0].max()
            max_y = positions[:, 1].max()
            max_w = positions[:, 2].max()
            max_h = positions[:, 3].max()
            
            if max_x > 0:
                positions[:, 0] /= max_x
            if max_y > 0:
                positions[:, 1] /= max_y
            if max_w > 0:
                positions[:, 2] /= max_w
            if max_h > 0:
                positions[:, 3] /= max_h
        
        # Concatenate all features
        node_features = torch.cat([
            visual_features,      # [N, D_visual]
            positions,            # [N, 4]
            semantic_embeddings   # [N, D_semantic]
        ], dim=-1)
        
        # Build edges based on spatial adjacency
        edge_index, _ = self._build_edges(ocr_boxes)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            pos=positions,
            num_nodes=num_cells
        )
        
        logger.info(f"Built graph with {num_cells} nodes and {edge_index.shape[1]} edges")
        
        return data
    
    def _build_edges(self, ocr_boxes: List[Dict[str, Any]], return_labels: bool = False, 
                     gt_structure: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build edge connectivity based on 4-connectivity (up, down, left, right).
        
        Two cells are connected if they are spatially adjacent.
        """
        num_cells = len(ocr_boxes)
        edges = []
        edge_labels = [] # 0: None, 1: Same-Row, 2: Same-Col
        
        for i in range(num_cells):
            box_i = ocr_boxes[i]['box']
            x1_i, y1_i, x2_i, y2_i = box_i
            cx_i, cy_i = (x1_i + x2_i) / 2, (y1_i + y2_i) / 2
            w_i, h_i = x2_i - x1_i, y2_i - y1_i
            
            for j in range(i + 1, num_cells):
                box_j = ocr_boxes[j]['box']
                x1_j, y1_j, x2_j, y2_j = box_j
                cx_j, cy_j = (x1_j + x2_j) / 2, (y1_j + y2_j) / 2
                
                # Check horizontal adjacency
                y_overlap = min(y2_i, y2_j) - max(y1_i, y1_j)
                y_union = max(y2_i, y2_j) - min(y1_i, y1_j)
                y_iou = y_overlap / y_union if y_union > 0 else 0
                
                x_distance = abs(cx_i - cx_j)
                horizontal_neighbor = (y_iou > self.threshold and x_distance < (w_i + (x2_j - x1_j)) * 0.6)
                
                # Check vertical adjacency
                x_overlap = min(x2_i, x2_j) - max(x1_i, x1_j)
                x_union = max(x2_i, x2_j) - min(x1_i, x1_j)
                x_iou = x_overlap / x_union if x_union > 0 else 0
                
                y_distance = abs(cy_i - cy_j)
                vertical_neighbor = (x_iou > self.threshold and y_distance < (h_i + (y2_j - y1_j)) * 0.6)
                
                if horizontal_neighbor or vertical_neighbor:
                    edges.extend([[i, j], [j, i]])
                    
                    if return_labels and gt_structure:
                        c_i, c_j = gt_structure['cells'][i], gt_structure['cells'][j]
                        label = 0
                        if c_i.get('row') == c_j.get('row'): label = 1
                        elif c_i.get('col') == c_j.get('col'): label = 2
                        edge_labels.extend([label, label])
        
        if len(edges) == 0:
            edges = [[i, i] for i in range(num_cells)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        labels = torch.tensor(edge_labels, dtype=torch.long) if (return_labels and edge_labels) else None
        
        return edge_index, labels


class TableGNN(nn.Module):
    """
    Graph Neural Network for table structure understanding.
    
    Uses 3 layers of graph convolutions to aggregate spatial and semantic
    information from neighboring cells.
    """
    
    def __init__(
        self,
        input_dim: int = 900,  # 128 (visual) + 4 (position) + 768 (semantic)
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output node embeddings
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        
        # First layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # Hidden layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_layers - 2)
        ])
        
        # Final layer
        self.conv_final = GCNConv(hidden_dim, output_dim)
        
        # Batch normalization for stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_layers - 1)
        ])
        self.batch_norm_final = nn.BatchNorm1d(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Edge relation predictor (Pairwise classification)
        # Concatenate two node embeddings -> relation logits [None, Same-Row, Same-Col]
        self.edge_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3) 
        )
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            node_embeddings: [N, output_dim] refined node features
            graph_embedding: [1, output_dim] global graph representation
        """
        x, edge_index = data.x, data.edge_index
        
        # First layer
        x = self.conv1(x, edge_index)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Hidden layers
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.batch_norms[i + 1](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            
            # Residual connection
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        
        # Final layer
        x = self.conv_final(x, edge_index)
        x = self.batch_norm_final(x)
        node_embeddings = F.relu(x)
        
        # Global graph embedding (mean pooling)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)
        graph_embedding = global_mean_pool(node_embeddings, batch)
        
        return node_embeddings, graph_embedding
    
    def predict_edges(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict edge relations (Same-Row/Same-Col) for given edges.
        
        Args:
            node_embeddings: [N, D]
            edge_index: [2, E]
            
        Returns:
            logits: [E, 3] (0:None, 1:Row, 2:Col)
        """
        src, dst = edge_index
        src_emb = node_embeddings[src]
        dst_emb = node_embeddings[dst]
        pair_feat = torch.cat([src_emb, dst_emb], dim=-1)
        return self.edge_predictor(pair_feat)
    
    def get_cell_scores(self, node_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute various scores from node embeddings for constraint solving.
        
        Args:
            node_embeddings: [N, output_dim]
            
        Returns:
            Dictionary of scores:
                - 'structure_confidence': [N] how confident about structure
                - 'header_prob': [N] probability of being a header cell
        """
        # Simple linear projections for now
        # In practice, these would be learned
        structure_confidence = torch.sigmoid(node_embeddings.mean(dim=-1))
        header_prob = torch.sigmoid(node_embeddings[:, 0])  # Use first dim as proxy
        
        return {
            'structure_confidence': structure_confidence,
            'header_prob': header_prob
        }

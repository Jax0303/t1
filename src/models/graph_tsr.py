"""
GraphTSR: Graph-based Table Structure Recognition.

Implements cell detection, edge classification, and hierarchy building
for table structure understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import logging

try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class CellDetector(nn.Module):
    """
    Cell Detector using Faster R-CNN.
    
    Detects cell bounding boxes in table images.
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # background + cell
        pretrained: bool = True,
    ):
        """
        Initialize Cell Detector.
        
        Args:
            num_classes: Number of classes (including background)
            pretrained: Whether to use pretrained backbone
        """
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required. Install with: pip install torchvision")
        
        # Load pretrained Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        
        # Replace the classifier head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        logger.info(f"CellDetector initialized (num_classes={num_classes})")
    
    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Forward pass through cell detector.
        
        Args:
            images: List of images (each is C x H x W)
            targets: Optional list of target dictionaries for training
        
        Returns:
            - List of predictions (boxes, labels, scores)
            - Loss dictionary (if targets provided)
        """
        if self.training and targets is None:
            raise ValueError("Targets must be provided during training")
        
        if self.training:
            loss_dict = self.model(images, targets)
            return [], loss_dict
        else:
            predictions = self.model(images)
            return predictions, {}
    
    def detect_cells(
        self,
        image: torch.Tensor,
        score_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Detect cells in a single image.
        
        Args:
            image: Input image (C x H x W)
            score_threshold: Minimum confidence score
        
        Returns:
            Dictionary with 'boxes', 'scores', 'labels'
        """
        self.eval()
        with torch.no_grad():
            predictions = self.model([image])[0]
        
        # Filter by score
        keep = predictions['scores'] > score_threshold
        
        return {
            'boxes': predictions['boxes'][keep],
            'scores': predictions['scores'][keep],
            'labels': predictions['labels'][keep]
        }


class EdgeClassifier(nn.Module):
    """
    Edge Classifier using GNN.
    
    Classifies relationships between detected cells:
    - same-row
    - same-col
    - parent-child
    - none
    """
    
    def __init__(
        self,
        node_dim: int = 256,
        edge_types: int = 4,  # same-row, same-col, parent-child, none
        hidden_dim: int = 128,
    ):
        """
        Initialize Edge Classifier.
        
        Args:
            node_dim: Node feature dimension
            edge_types: Number of edge types
            hidden_dim: Hidden dimension for GNN
        """
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("torch_geometric not available. Using simple MLP.")
            self.use_gnn = False
        else:
            self.use_gnn = True
        
        self.node_dim = node_dim
        self.edge_types = edge_types
        
        if self.use_gnn:
            # GNN layers
            self.gcn1 = GCNConv(node_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Edge classifier MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 8, hidden_dim),  # +8 for spatial features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, edge_types)
        )
        
        logger.info(f"EdgeClassifier initialized (use_gnn={self.use_gnn})")
    
    def _compute_spatial_features(
        self,
        boxes_i: torch.Tensor,
        boxes_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial features between two boxes.
        
        Args:
            boxes_i: Boxes for node i (N, 4) [x1, y1, x2, y2]
            boxes_j: Boxes for node j (N, 4)
        
        Returns:
            Spatial features (N, 8)
        """
        # Center points
        center_i = (boxes_i[:, :2] + boxes_i[:, 2:]) / 2
        center_j = (boxes_j[:, :2] + boxes_j[:, 2:]) / 2
        
        # Relative position
        rel_pos = center_j - center_i
        
        # Distance
        dist = torch.norm(rel_pos, dim=1, keepdim=True)
        
        # Angle
        angle = torch.atan2(rel_pos[:, 1:2], rel_pos[:, 0:1])
        
        # Size ratio
        size_i = (boxes_i[:, 2:] - boxes_i[:, :2]).prod(dim=1, keepdim=True)
        size_j = (boxes_j[:, 2:] - boxes_j[:, :2]).prod(dim=1, keepdim=True)
        size_ratio = size_j / (size_i + 1e-6)
        
        # IoU
        x1 = torch.max(boxes_i[:, 0], boxes_j[:, 0])
        y1 = torch.max(boxes_i[:, 1], boxes_j[:, 1])
        x2 = torch.min(boxes_i[:, 2], boxes_j[:, 2])
        y2 = torch.min(boxes_i[:, 3], boxes_j[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        union = size_i + size_j - intersection
        iou = (intersection / (union + 1e-6)).unsqueeze(1)
        
        # Concatenate all features
        spatial_features = torch.cat([
            rel_pos,  # 2
            dist,  # 1
            angle,  # 1
            size_ratio,  # 1
            iou,  # 1
            torch.sin(angle),  # 1
            torch.cos(angle),  # 1
        ], dim=-1)  # Use dim=-1 to concatenate along last dimension
        
        return spatial_features
    
    def forward(
        self,
        node_features: torch.Tensor,
        boxes: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through edge classifier.
        
        Args:
            node_features: Node features (N, node_dim)
            boxes: Bounding boxes (N, 4)
            edge_index: Edge indices (2, E) for GNN (optional)
        
        Returns:
            Edge logits (E, edge_types) or (N, N, edge_types)
        """
        N = node_features.shape[0]
        
        if self.use_gnn and edge_index is not None:
            # Apply GNN
            h = F.relu(self.gcn1(node_features, edge_index))
            h = self.gcn2(h, edge_index)
        else:
            # Use node features directly
            h = node_features
        
        # Compute pairwise edge features
        if edge_index is not None:
            # Only compute for specified edges
            i, j = edge_index
            h_i = h[i]
            h_j = h[j]
            boxes_i = boxes[i]
            boxes_j = boxes[j]
        else:
            # Compute for all pairs
            h_i = h.unsqueeze(1).expand(N, N, -1).reshape(N * N, -1)
            h_j = h.unsqueeze(0).expand(N, N, -1).reshape(N * N, -1)
            boxes_i = boxes.unsqueeze(1).expand(N, N, 4).reshape(N * N, 4)
            boxes_j = boxes.unsqueeze(0).expand(N, N, 4).reshape(N * N, 4)
        
        # Compute spatial features
        spatial_features = self._compute_spatial_features(boxes_i, boxes_j)
        
        # Concatenate node and spatial features
        edge_features = torch.cat([h_i, h_j, spatial_features], dim=1)
        
        # Classify edges
        edge_logits = self.edge_mlp(edge_features)
        
        if edge_index is None:
            # Reshape to (N, N, edge_types)
            edge_logits = edge_logits.view(N, N, self.edge_types)
        
        return edge_logits


class HierarchyBuilder(nn.Module):
    """
    Hierarchy Builder.
    
    Constructs hierarchy tree from cell detections and edge classifications.
    """
    
    def __init__(self):
        """Initialize Hierarchy Builder."""
        super().__init__()
        logger.info("HierarchyBuilder initialized")
    
    def build_hierarchy(
        self,
        boxes: torch.Tensor,
        edge_logits: torch.Tensor,
        edge_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Build hierarchy from edge predictions.
        
        Args:
            boxes: Cell bounding boxes (N, 4)
            edge_logits: Edge classification logits (N, N, 4)
            edge_threshold: Threshold for edge confidence
        
        Returns:
            Hierarchy dictionary with parent-child relationships
        """
        N = boxes.shape[0]
        
        # Get edge probabilities
        edge_probs = F.softmax(edge_logits, dim=-1)
        
        # Extract parent-child edges (index 2)
        parent_child_probs = edge_probs[:, :, 2]
        
        # Threshold
        parent_child_edges = (parent_child_probs > edge_threshold).nonzero(as_tuple=False)
        
        # Build hierarchy dictionary
        hierarchy = {
            'nodes': list(range(N)),
            'edges': parent_child_edges.tolist(),
            'boxes': boxes.tolist(),
        }
        
        # Build parent-child mapping
        children = {i: [] for i in range(N)}
        parents = {i: None for i in range(N)}
        
        for parent, child in parent_child_edges.tolist():
            children[parent].append(child)
            parents[child] = parent
        
        hierarchy['children'] = children
        hierarchy['parents'] = parents
        
        return hierarchy
    
    def forward(
        self,
        boxes: torch.Tensor,
        edge_logits: torch.Tensor
    ) -> Dict[str, Any]:
        """Forward pass (alias for build_hierarchy)."""
        return self.build_hierarchy(boxes, edge_logits)


class GraphTSR(nn.Module):
    """
    Complete GraphTSR module.
    
    Combines cell detection, edge classification, and hierarchy building.
    """
    
    def __init__(
        self,
        node_dim: int = 256,
        pretrained_detector: bool = True,
    ):
        """
        Initialize GraphTSR.
        
        Args:
            node_dim: Node feature dimension
            pretrained_detector: Whether to use pretrained cell detector
        """
        super().__init__()
        
        self.cell_detector = CellDetector(pretrained=pretrained_detector)
        self.edge_classifier = EdgeClassifier(node_dim=node_dim)
        self.hierarchy_builder = HierarchyBuilder()
        
        logger.info("GraphTSR initialized")
    
    def forward(
        self,
        image: torch.Tensor,
        node_features: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through GraphTSR.
        
        Args:
            image: Input image (C, H, W)
            node_features: Optional precomputed node features
        
        Returns:
            Dictionary with cells, edges, and hierarchy
        """
        # Detect cells
        detections = self.cell_detector.detect_cells(image)
        boxes = detections['boxes']
        
        if boxes.shape[0] == 0:
            return {'cells': [], 'edges': [], 'hierarchy': {}}
        
        # Use dummy node features if not provided
        if node_features is None:
            node_features = torch.randn(boxes.shape[0], 256, device=boxes.device)
        
        # Classify edges
        edge_logits = self.edge_classifier(node_features, boxes)
        
        # Build hierarchy
        hierarchy = self.hierarchy_builder(boxes, edge_logits)
        
        return {
            'cells': boxes,
            'edge_logits': edge_logits,
            'hierarchy': hierarchy
        }

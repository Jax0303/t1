#!/usr/bin/env python3
"""
Training script for GNN Indexer with contrastive learning.

Trains structure-aware embeddings using:
- Contrastive loss (same-header cells should be close)
- Triplet loss (anchor-positive-negative)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from src.hiertable_rag.core.gnn_indexer import (
    GNNIndexer,
    StructureAwareRetriever,
    GraphSAGEIndexer
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("TrainGNN")


@dataclass
class GNNTrainingConfig:
    """GNN training configuration."""
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Contrastive learning
    temperature: float = 0.07
    margin: float = 0.5
    
    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for structure-aware embeddings.
    
    Cells under the same header should have similar embeddings.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Node embeddings (N, D)
            labels: Header labels (N,) - cells with same label should be close
        
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create positive mask (same label)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float()
        
        # Remove self-similarity
        positive_mask.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(similarity)
        
        # For each anchor, sum over positives and all negatives
        loss = 0.0
        num_positives = positive_mask.sum(dim=1)
        
        for i in range(embeddings.shape[0]):
            if num_positives[i] > 0:
                # Positive similarities
                pos_sim = (exp_sim[i] * positive_mask[i]).sum()
                
                # All similarities (excluding self)
                all_sim = exp_sim[i].sum() - exp_sim[i, i]
                
                # Contrastive loss for this anchor
                loss += -torch.log(pos_sim / all_sim)
        
        # Average over anchors with positives
        num_valid = (num_positives > 0).sum()
        if num_valid > 0:
            loss = loss / num_valid
        
        return loss


class TripletLoss(nn.Module):
    """Triplet loss for metric learning."""
    
    def __init__(self, margin: float = 0.5):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings (N, D)
            positive: Positive embeddings (N, D)
            negative: Negative embeddings (N, D)
        
        Returns:
            Triplet loss
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class TableGraphDataset(Dataset):
    """Dataset for GNN training with graph structure."""
    
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Initialize dataset.
        
        Args:
            data_dir: Data directory
            split: Dataset split
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Mock data
        self.samples = self._create_mock_samples(50)
        
        logger.info(f"Loaded {len(self.samples)} {split} graph samples")
    
    def _create_mock_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Create mock graph samples."""
        samples = []
        for i in range(num_samples):
            num_cells = torch.randint(5, 15, (1,)).item()
            
            samples.append({
                'cell_embeddings': torch.randn(num_cells, 256),
                'adjacency': self._create_adjacency(num_cells),
                'header_labels': torch.randint(0, 3, (num_cells,)),  # 3 different headers
            })
        return samples
    
    def _create_adjacency(self, num_cells: int) -> List[List[int]]:
        """Create mock adjacency list."""
        adj = []
        for i in range(num_cells):
            for j in range(i + 1, min(i + 3, num_cells)):
                adj.append([i, j])
                adj.append([j, i])
        return adj
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class GNNTrainer:
    """Trainer for GNN indexer."""
    
    def __init__(self, config: GNNTrainingConfig):
        """
        Initialize GNN trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize GNN model
        self.gnn = GraphSAGEIndexer(
            hidden_channels=256,
            num_layers=2
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.gnn.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss functions
        self.contrastive_loss = ContrastiveLoss(temperature=config.temperature)
        self.triplet_loss = TripletLoss(margin=config.margin)
        
        logger.info(f"GNNTrainer initialized (device={self.device})")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.gnn.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # batch contains lists of tensors (not stacked)
            cell_embeddings_list = batch['cell_embeddings']
            header_labels_list = batch['header_labels']
            adjacency_list = batch['adjacency']
            
            batch_loss = 0.0
            
            # Process each graph in batch
            for i in range(len(cell_embeddings_list)):
                embeddings = cell_embeddings_list[i].to(self.device)
                labels = header_labels_list[i].to(self.device)
                
                # Create edge index from adjacency
                if adjacency_list[i] and len(adjacency_list[i]) > 0:
                    edge_index = torch.tensor(
                        adjacency_list[i],
                        dtype=torch.long,
                        device=self.device
                    ).t()
                else:
                    # Fully connected if no adjacency
                    num_nodes = embeddings.shape[0]
                    edge_index = torch.combinations(
                        torch.arange(num_nodes, device=self.device),
                        r=2
                    ).t()
                    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                
                # Forward pass through GNN
                gnn_embeddings = self.gnn(embeddings, edge_index)
                
                # Contrastive loss
                loss = self.contrastive_loss(gnn_embeddings, labels)
                batch_loss += loss
            
            # Average over batch
            batch_size = len(cell_embeddings_list)
            batch_loss = batch_loss / batch_size
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {batch_loss.item():.4f}"
                )
        
        return epoch_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model."""
        self.gnn.eval()
        
        val_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                cell_embeddings_list = batch['cell_embeddings']
                header_labels_list = batch['header_labels']
                adjacency_list = batch['adjacency']
                
                for i in range(len(cell_embeddings_list)):
                    embeddings = cell_embeddings_list[i].to(self.device)
                    labels = header_labels_list[i].to(self.device)
                    
                    # Create edge index
                    if adjacency_list[i] and len(adjacency_list[i]) > 0:
                        edge_index = torch.tensor(
                            adjacency_list[i],
                            dtype=torch.long,
                            device=self.device
                        ).t()
                    else:
                        num_nodes = embeddings.shape[0]
                        edge_index = torch.combinations(
                            torch.arange(num_nodes, device=self.device),
                            r=2
                        ).t()
                        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                    
                    # Forward
                    gnn_embeddings = self.gnn(embeddings, edge_index)
                    
                    # Loss
                    loss = self.contrastive_loss(gnn_embeddings, labels)
                    val_loss += loss.item()
                    num_samples += 1
        
        return val_loss / num_samples if num_samples > 0 else 0.0
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        logger.info("Starting GNN training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch + 1)
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, val_loss)
                logger.info(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        logger.info("\n🎉 GNN training complete!")
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'gnn': self.gnn.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }
        
        checkpoint_path = checkpoint_dir / f"gnn_indexer_epoch{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Custom collate function for variable-size graphs.
    
    Args:
        batch: List of samples
    
    Returns:
        Batched data (as lists, not stacked tensors)
    """
    return {
        'cell_embeddings': [item['cell_embeddings'] for item in batch],
        'adjacency': [item['adjacency'] for item in batch],
        'header_labels': [item['header_labels'] for item in batch],
    }


def main():
    """Main training function."""
    # Configuration
    config = GNNTrainingConfig(
        batch_size=4,
        num_epochs=5,
        learning_rate=1e-3,
    )
    
    # Create datasets
    train_dataset = TableGraphDataset(config.data_dir, split="train")
    val_dataset = TableGraphDataset(config.data_dir, split="val")
    
    # Create data loaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate  # Use custom collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate  # Use custom collate
    )
    
    # Create trainer
    trainer = GNNTrainer(config)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

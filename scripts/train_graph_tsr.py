#!/usr/bin/env python3
"""
Training script for GraphTSR model.

Multi-task training with:
- Cell detection loss
- Edge classification loss
- Hierarchy prediction loss
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Any, Tuple
import json
from dataclasses import dataclass

from src.models.swin_encoder import SwinMultiResEncoder
from src.models.graph_tsr import GraphTSR, CellDetector, EdgeClassifier, HierarchyBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("TrainGraphTSR")


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Loss weights
    lambda_detection: float = 1.0
    lambda_edge: float = 0.5
    lambda_hierarchy: float = 0.3
    
    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TableDataset(Dataset):
    """
    Dataset for table structure recognition.
    
    Mock implementation - replace with actual data loading.
    """
    
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Initialize dataset.
        
        Args:
            data_dir: Data directory
            split: Dataset split (train/val/test)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Mock data - replace with actual loading
        self.samples = self._create_mock_samples(100)
        
        logger.info(f"Loaded {len(self.samples)} {split} samples")
    
    def _create_mock_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Create mock samples for testing."""
        samples = []
        for i in range(num_samples):
            samples.append({
                'image': torch.randn(3, 224, 224),
                'cells': torch.randint(0, 200, (10, 4)).float(),  # 10 cells, (x1,y1,x2,y2)
                'edge_labels': torch.randint(0, 4, (10, 10)),  # Edge type matrix
                'hierarchy': {
                    'parents': {j: (j-1) if j > 0 else None for j in range(10)}
                }
            })
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class GraphTSRTrainer:
    """Trainer for GraphTSR model."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.vision_encoder = SwinMultiResEncoder(
            model_name="swin_tiny_patch4_window7_224",
            pretrained=True,
            hidden_dim=256
        ).to(self.device)
        
        self.graph_tsr = GraphTSR(
            node_dim=256,
            pretrained_detector=True
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.vision_encoder.parameters()) + 
            list(self.graph_tsr.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss functions
        self.detection_loss = nn.SmoothL1Loss()
        self.edge_loss = nn.CrossEntropyLoss()
        self.hierarchy_loss = nn.BCEWithLogitsLoss()
        
        logger.info(f"GraphTSRTrainer initialized (device={self.device})")
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            - Total loss
            - Loss components dictionary
        """
        losses = {}
        
        # Detection loss (if available)
        if 'cells' in predictions and 'cells' in targets:
            losses['detection'] = self.detection_loss(
                predictions['cells'],
                targets['cells']
            )
        else:
            losses['detection'] = torch.tensor(0.0, device=self.device)
        
        # Edge classification loss
        if 'edge_logits' in predictions and 'edge_labels' in targets:
            edge_logits = predictions['edge_logits']
            edge_labels = targets['edge_labels']
            
            # Flatten for cross-entropy
            N = edge_logits.shape[0]
            edge_logits_flat = edge_logits.view(N * N, -1)
            edge_labels_flat = edge_labels.view(N * N)
            
            losses['edge'] = self.edge_loss(edge_logits_flat, edge_labels_flat)
        else:
            losses['edge'] = torch.tensor(0.0, device=self.device)
        
        # Hierarchy loss (simplified)
        if 'hierarchy' in predictions and 'hierarchy' in targets:
            # Mock hierarchy loss - replace with actual implementation
            losses['hierarchy'] = torch.tensor(0.0, device=self.device)
        else:
            losses['hierarchy'] = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = (
            self.config.lambda_detection * losses['detection'] +
            self.config.lambda_edge * losses['edge'] +
            self.config.lambda_hierarchy * losses['hierarchy']
        )
        
        # Convert to float for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Average losses for the epoch
        """
        self.vision_encoder.train()
        self.graph_tsr.train()
        
        epoch_losses = {'total': 0.0, 'detection': 0.0, 'edge': 0.0, 'hierarchy': 0.0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            images = batch['image'].to(self.device)
            
            # Forward pass through vision encoder
            features, _ = self.vision_encoder(images)
            
            # For each image in batch (simplified - actual would be more complex)
            batch_loss = torch.tensor(0.0, device=self.device)
            batch_loss_dict = {'total': 0.0, 'detection': 0.0, 'edge': 0.0, 'hierarchy': 0.0}
            
            for i in range(images.shape[0]):
                # Mock: use random node features
                node_features = torch.randn(10, 256, device=self.device)
                boxes = batch['cells'][i].to(self.device)
                
                # Forward through GraphTSR components
                edge_logits = self.graph_tsr.edge_classifier(node_features, boxes)
                
                # Compute loss
                predictions = {
                    'edge_logits': edge_logits,
                }
                targets = {
                    'edge_labels': batch['edge_labels'][i].to(self.device),
                }
                
                loss, loss_dict = self.compute_loss(predictions, targets)
                batch_loss += loss
                
                for k, v in loss_dict.items():
                    batch_loss_dict[k] += v
            
            # Average over batch
            batch_loss = batch_loss / images.shape[0]
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            for k in epoch_losses.keys():
                epoch_losses[k] += batch_loss_dict[k] / images.shape[0]
            
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {batch_loss.item():.4f}"
                )
        
        # Average over all batches
        for k in epoch_losses.keys():
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Validation metrics
        """
        self.vision_encoder.eval()
        self.graph_tsr.eval()
        
        val_losses = {'total': 0.0, 'detection': 0.0, 'edge': 0.0, 'hierarchy': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                
                # Forward pass
                features, _ = self.vision_encoder(images)
                
                # Simplified validation
                for i in range(images.shape[0]):
                    node_features = torch.randn(10, 256, device=self.device)
                    boxes = batch['cells'][i].to(self.device)
                    
                    edge_logits = self.graph_tsr.edge_classifier(node_features, boxes)
                    
                    predictions = {'edge_logits': edge_logits}
                    targets = {'edge_labels': batch['edge_labels'][i].to(self.device)}
                    
                    _, loss_dict = self.compute_loss(predictions, targets)
                    
                    for k, v in loss_dict.items():
                        val_losses[k] += v
                
                num_batches += images.shape[0]
        
        # Average
        for k in val_losses.keys():
            val_losses[k] /= num_batches
        
        return val_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            
            # Train
            train_losses = self.train_epoch(train_loader, epoch + 1)
            logger.info(
                f"Train Loss: {train_losses['total']:.4f} "
                f"(det: {train_losses['detection']:.4f}, "
                f"edge: {train_losses['edge']:.4f}, "
                f"hier: {train_losses['hierarchy']:.4f})"
            )
            
            # Validate
            val_losses = self.validate(val_loader)
            logger.info(
                f"Val Loss: {val_losses['total']:.4f} "
                f"(det: {val_losses['detection']:.4f}, "
                f"edge: {val_losses['edge']:.4f}, "
                f"hier: {val_losses['hierarchy']:.4f})"
            )
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.save_checkpoint(epoch + 1, val_losses['total'])
                logger.info(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        logger.info("\n🎉 Training complete!")
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'vision_encoder': self.vision_encoder.state_dict(),
            'graph_tsr': self.graph_tsr.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }
        
        checkpoint_path = checkpoint_dir / f"graph_tsr_epoch{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main training function."""
    # Configuration
    config = TrainingConfig(
        batch_size=2,  # Small batch for testing
        num_epochs=3,  # Few epochs for testing
        learning_rate=1e-4,
    )
    
    # Create datasets
    train_dataset = TableDataset(config.data_dir, split="train")
    val_dataset = TableDataset(config.data_dir, split="val")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create trainer
    trainer = GraphTSRTrainer(config)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

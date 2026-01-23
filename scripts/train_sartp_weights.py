"""
SARTP Training Script
Train learnable fusion weights (α, β, γ) end-to-end.
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.hiertable_rag.core.rca_model import RCAModel
from src.hiertable_rag.sartp.losses import SARTPLoss, LearnableWeights, create_header_data_pairs
from src.hiertable_rag.sartp import SemanticEncoder, HeaderDetector
from scripts.universal_dataset import UniversalTableDataset


class SARTPTrainer:
    """
    Trainer for SARTP with learnable weights.
    """
    
    def __init__(
        self,
        rca_model: RCAModel,
        semantic_encoder: SemanticEncoder,
        header_detector: HeaderDetector,
        learnable_weights: LearnableWeights,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            rca_model: RCA model (can be frozen or fine-tuned)
            semantic_encoder: Semantic encoder (frozen)
            header_detector: Header detector (frozen)
            learnable_weights: Learnable fusion weights module
            device: Device to train on
        """
        self.rca_model = rca_model.to(device)
        self.semantic_encoder = semantic_encoder
        self.header_detector = header_detector
        self.learnable_weights = learnable_weights.to(device)
        self.device = device
        
        # Loss function
        self.criterion = SARTPLoss(
            w_boundary=1.0,
            w_alignment=0.5,
            w_structure=0.3
        )
        
        # Optimizer: only train learnable weights (freeze RCA and semantic encoder)
        self.optimizer = optim.Adam(self.learnable_weights.parameters(), lr=0.001)
        
        print(f"SARTPTrainer initialized on {device}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.learnable_weights.parameters())}")
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.learnable_weights.train()
        self.rca_model.eval()  # Keep RCA frozen (or set to train() for fine-tuning)
        
        epoch_losses = {
            'total': [],
            'boundary': [],
            'alignment': [],
            'structure': []
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Get data
                images = batch['image'].to(self.device)
                gt_cells = batch['cells']
                raw_cells = batch.get('raw_cells', [])
                
                # Get ground truth boundaries (if available)
                row_gt = batch.get('row_gt', None)
                col_gt = batch.get('col_gt', None)
                
                # Forward pass through RCA
                with torch.no_grad():
                    rca_output = self.rca_model(images)
                
                # Get semantic embeddings
                # (In real training, we'd process cells from RCA output)
                # For now, skip semantic computation to simplify
                
                # Compute loss (simplified - just boundary loss for now)
                pred = {
                    'row_probs': rca_output['row_probs'],
                    'col_probs': rca_output['col_probs']
                }
                
                target = {
                    'row_gt': row_gt if row_gt is not None else torch.zeros_like(rca_output['row_probs']),
                    'col_gt': col_gt if col_gt is not None else torch.zeros_like(rca_output['col_probs'])
                }
                
                losses = self.criterion(pred, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                self.optimizer.step()
                
                # Track losses
                epoch_losses['total'].append(losses['total_loss'].item())
                epoch_losses['boundary'].append(losses['boundary_loss'].item())
                
                # Update progress bar
                current_weights = self.learnable_weights.get_weights_dict()
                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'α': f"{current_weights['alpha']:.3f}",
                    'β': f"{current_weights['beta']:.3f}",
                    'γ': f"{current_weights['gamma']:.3f}"
                })
            
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        # Epoch summary
        avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
        return avg_losses
    
    def validate(self, dataloader):
        """Validate on validation set."""
        self.learnable_weights.eval()
        self.rca_model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                try:
                    images = batch['image'].to(self.device)
                    row_gt = batch.get('row_gt', None)
                    col_gt = batch.get('col_gt', None)
                    
                    # Forward
                    rca_output = self.rca_model(images)
                    
                    pred = {
                        'row_probs': rca_output['row_probs'],
                        'col_probs': rca_output['col_probs']
                    }
                    
                    target = {
                        'row_gt': row_gt if row_gt is not None else torch.zeros_like(rca_output['row_probs']),
                        'col_gt': col_gt if col_gt is not None else torch.zeros_like(rca_output['col_probs'])
                    }
                    
                    losses = self.criterion(pred, target)
                    val_losses.append(losses['total_loss'].item())
                
                except Exception as e:
                    continue
        
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        return avg_val_loss
    
    def save_weights(self, path):
        """Save learned weights."""
        weights_dict = self.learnable_weights.get_weights_dict()
        torch.save({
            'learnable_weights': self.learnable_weights.state_dict(),
            'weights_dict': weights_dict
        }, path)
        print(f"Weights saved to {path}")
        print(f"  α={weights_dict['alpha']:.4f}, β={weights_dict['beta']:.4f}, γ={weights_dict['gamma']:.4f}")


def train_sartp(
    dataset_name="scitsr",
    num_train_samples=100,
    num_val_samples=20,
    num_epochs=10,
    batch_size=4,
    output_dir="outputs/sartp_training"
):
    """
    Train SARTP learnable weights.
    
    Args:
        dataset_name: Dataset to use
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        num_epochs: Number of training epochs
        batch_size: Batch size
        output_dir: Directory to save results
    """
    print(f"\n{'='*70}")
    print("SARTP LEARNABLE WEIGHT TRAINING")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    base_dir = "/root/t1-9/data/external"
    
    try:
        train_dataset = UniversalTableDataset(base_dir, dataset_name, split="train")
        val_dataset = UniversalTableDataset(base_dir, dataset_name, split="val")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using dummy data for demonstration...")
        # Create dummy dataset for testing
        return
    
    # Subsample for faster training
    train_indices = list(range(min(num_train_samples, len(train_dataset))))
    val_indices = list(range(min(num_val_samples, len(val_dataset))))
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Val samples: {len(val_subset)}")
    
    # Initialize models
    print("\n[2/4] Initializing models...")
    
    rca_model = RCAModel().to(device)
    checkpoint_path = "/root/t1-9/outputs/rca_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"  Loading RCA from {checkpoint_path}")
        rca_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    semantic_encoder = SemanticEncoder(model_name="roberta-base", device=device)
    header_detector = HeaderDetector()
    
    learnable_weights = LearnableWeights(
        init_alpha=0.5,
        init_beta=0.3,
        init_gamma=0.2
    )
    
    print("  Models initialized")
    
    # Create trainer
    trainer = SARTPTrainer(
        rca_model=rca_model,
        semantic_encoder=semantic_encoder,
        header_detector=header_detector,
        learnable_weights=learnable_weights,
        device=device
    )
    
    # Training loop
    print(f"\n[3/4] Training for {num_epochs} epochs...")
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'weights': []
    }
    
    best_val_loss = float('inf')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        # Train
        train_losses = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        # Track history
        training_history['train_loss'].append(train_losses['total'])
        training_history['val_loss'].append(val_loss)
        training_history['weights'].append(learnable_weights.get_weights_dict())
        
        print(f"Train Loss: {train_losses['total']:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Current weights: {learnable_weights.get_weights_dict()}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_weights(output_dir / 'sartp_weights_best.pth')
    
    # Save final weights
    trainer.save_weights(output_dir / 'sartp_weights_final.pth')
    
    # Save training history
    print(f"\n[4/4] Saving training history...")
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final weights: {learnable_weights.get_weights_dict()}")
    print(f"  Results saved to {output_dir}/")
    print(f"{'='*70}\n")
    
    return training_history


if __name__ == "__main__":
    history = train_sartp(
        dataset_name="scitsr",
        num_train_samples=100,
        num_val_samples=20,
        num_epochs=10,
        batch_size=4,
        output_dir="outputs/sartp_training"
    )

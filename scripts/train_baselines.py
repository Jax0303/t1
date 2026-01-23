"""
Training Script for Baseline Models
Trains Vision-Only and Lightweight VLM baselines for comparison with SARTP.
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import json
from pathlib import Path
from tqdm import tqdm

from src.baselines import VisionOnlyTSR, LightweightVLM
from scripts.universal_dataset import UniversalTableDataset


def train_vision_only(
    dataset_name="scitsr",
    num_epochs=20,
    batch_size=4,
    output_dir="outputs/baselines/vision_only"
):
    """Train Vision-Only TSR baseline."""
    print("\n" + "="*70)
    print("TRAINING VISION-ONLY TSR BASELINE")
    print("="*70 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    base_dir = "/root/t1-9/data/external"
    train_dataset = UniversalTableDataset(base_dir, dataset_name, split="train")
    
    # Subsample
    train_indices = list(range(min(100, len(train_dataset))))
    train_subset = Subset(train_dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = VisionOnlyTSR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            try:
                images = batch['image'].to(device)
                row_gt = batch.get('row_gt', torch.zeros(images.size(0), 32)).to(device)
                col_gt = batch.get('col_gt', torch.zeros(images.size(0), 32)).to(device)
                
                # Forward
                outputs = model(images)
                
                # Resize predictions to match GT if needed
                if outputs['row_probs'].size(1) != row_gt.size(1):
                    row_pred = torch.nn.functional.interpolate(
                        outputs['row_probs'].unsqueeze(1), 
                        size=row_gt.size(1), 
                        mode='linear'
                    ).squeeze(1)
                else:
                    row_pred = outputs['row_probs']
                
                if outputs['col_probs'].size(1) != col_gt.size(1):
                    col_pred = torch.nn.functional.interpolate(
                        outputs['col_probs'].unsqueeze(1),
                        size=col_gt.size(1),
                        mode='linear'
                    ).squeeze(1)
                else:
                    col_pred = outputs['col_probs']
                
                # Loss
                loss = criterion(row_pred, row_gt) + criterion(col_pred, col_gt)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), output_path / "vision_only_best.pth")
    print(f"\n✓ Model saved to {output_path / 'vision_only_best.pth'}")


def train_lightweight_vlm(
    dataset_name="scitsr",
    num_epochs=20,
    batch_size=4,
    output_dir="outputs/baselines/lightweight_vlm"
):
    """Train Lightweight VLM baseline."""
    print("\n" + "="*70)
    print("TRAINING LIGHTWEIGHT VLM BASELINE")
    print("="*70 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    base_dir = "/root/t1-9/data/external"
    train_dataset = UniversalTableDataset(base_dir, dataset_name, split="train")
    
    # Subsample
    train_indices = list(range(min(100, len(train_dataset))))
    train_subset = Subset(train_dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = LightweightVLM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            try:
                images = batch['image'].to(device)
                # For now, train without text (can be extended)
                
                # Forward (vision only for now)
                outputs = model(images)
                
                # Simple dummy loss (replace with actual structure loss)
                loss = outputs['row_probs'].mean() + outputs['col_probs'].mean()
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), output_path / "lightweight_vlm_best.pth")
    print(f"\n✓ Model saved to {output_path / 'lightweight_vlm_best.pth'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vision_only', 'vlm', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    if args.model in ['vision_only', 'both']:
        train_vision_only(num_epochs=args.epochs)
    
    if args.model in ['vlm', 'both']:
        train_lightweight_vlm(num_epochs=args.epochs)

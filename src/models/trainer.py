"""
Training utilities for End-to-End Hierarchical Table Model.

Provides:
- Dataset classes for table images with hierarchy annotations
- Training loop with multi-task loss
- Evaluation metrics
- Checkpointing and logging
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import time

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from src.models.e2e_hierarchical import (
    E2EHierarchicalTableModel,
    E2EConfig,
    E2EOutput,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Training configuration.
    
    Attributes:
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps
        gradient_clip: Gradient clipping value
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        log_steps: Log every N steps
        output_dir: Directory to save outputs
        resume_from: Path to resume training from
    """
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    log_steps: int = 100
    output_dir: str = "outputs/e2e_training"
    resume_from: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "gradient_clip": self.gradient_clip,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "log_steps": self.log_steps,
            "output_dir": self.output_dir,
            "resume_from": self.resume_from,
        }


class HierarchicalTableDataset(Dataset if TORCH_AVAILABLE else object):
    """
    Dataset for hierarchical table images.
    
    Expected annotation format:
    {
        "image_path": "path/to/image.png",
        "cells": [
            {
                "bbox": [x1, y1, x2, y2],
                "content": "cell text",
                "row": 0,
                "col": 0,
                "rowspan": 1,
                "colspan": 2,
                "is_header": true,
                "level": 1,
                "parent_idx": -1
            },
            ...
        ],
        "hierarchy": {
            "root": {...},
            "max_level": 3
        }
    }
    """
    
    def __init__(
        self,
        annotations_path: str,
        image_dir: str,
        config: E2EConfig,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            annotations_path: Path to annotations JSON file
            image_dir: Directory containing images
            config: Model config
            transform: Optional image transform
        """
        if not TORCH_AVAILABLE:
            return
        
        self.config = config
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load annotations
        with open(annotations_path, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        
        logger.info(f"Loaded {len(self.annotations)} annotations")
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.annotations) if hasattr(self, 'annotations') else 0
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary with image tensor and targets
        """
        if not TORCH_AVAILABLE:
            return {}
        
        from PIL import Image
        from torchvision import transforms
        
        ann = self.annotations[idx]
        
        # Load image
        image_path = self.image_dir / ann["image_path"]
        image = Image.open(image_path).convert("RGB")
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            image = transform(image)
        
        # Process cells
        cells = ann.get("cells", [])
        num_cells = len(cells)
        
        # Pad to max_cells
        max_cells = self.config.max_cells
        
        # Initialize tensors
        cell_labels = torch.zeros(max_cells, dtype=torch.long)
        bbox_targets = torch.zeros(max_cells, 4)
        header_labels = torch.zeros(max_cells, dtype=torch.long)
        level_labels = torch.zeros(max_cells, dtype=torch.long)
        parent_labels = torch.full((max_cells,), -1, dtype=torch.long)
        row_labels = torch.zeros(max_cells, dtype=torch.long)
        col_labels = torch.zeros(max_cells, dtype=torch.long)
        
        # Fill in values
        for i, cell in enumerate(cells[:max_cells]):
            cell_labels[i] = 1  # Cell exists
            
            # Normalize bbox
            bbox = cell.get("bbox", [0, 0, 0, 0])
            bbox_targets[i] = torch.tensor([
                bbox[0] / self.config.image_size[1],  # x1
                bbox[1] / self.config.image_size[0],  # y1
                bbox[2] / self.config.image_size[1],  # x2
                bbox[3] / self.config.image_size[0],  # y2
            ])
            
            header_labels[i] = 1 if cell.get("is_header", False) else 0
            level_labels[i] = min(cell.get("level", 0), self.config.max_hierarchy_depth - 1)
            
            parent_idx = cell.get("parent_idx", -1)
            if parent_idx >= 0 and parent_idx < max_cells:
                parent_labels[i] = parent_idx
            
            row_labels[i] = cell.get("row", 0)
            col_labels[i] = cell.get("col", 0)
        
        return {
            "image": image,
            "cell_labels": cell_labels,
            "bbox_targets": bbox_targets,
            "header_labels": header_labels,
            "level_labels": level_labels,
            "parent_labels": parent_labels,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "num_cells": num_cells,
        }


class E2ETrainer:
    """
    Trainer for End-to-End Hierarchical Table Model.
    
    Handles:
    - Multi-task training with balanced losses
    - Gradient accumulation
    - Mixed precision training
    - Checkpointing
    - Logging and visualization
    """
    
    def __init__(
        self,
        model: E2EHierarchicalTableModel,
        train_config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for training")
            return
        
        self.model = model
        self.config = train_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup output directory
        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.num_epochs,
        )
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Resume if specified
        if train_config.resume_from:
            self._load_checkpoint(train_config.resume_from)
        
        logger.info(f"Trainer initialized (device={self.device})")
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Returns:
            Training history dictionary
        """
        if not TORCH_AVAILABLE or self.train_dataset is None:
            return {}
        
        logger.info("Starting training...")
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if TQDM_AVAILABLE else train_loader
            
            for batch in iterator:
                # Move to device
                images = batch["image"].to(self.device)
                targets = {
                    k: v.to(self.device) 
                    for k, v in batch.items() 
                    if k != "image"
                }
                
                # Forward pass
                output, losses = self.model(images, targets)
                
                if losses is None:
                    continue
                
                loss = losses["total_loss"]
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
                
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        f"Step {self.global_step}: loss={avg_loss:.4f}, "
                        f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    history["eval_loss"].append(eval_loss)
                    
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self._save_checkpoint("best")
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_epoch_loss)
            history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            self.scheduler.step()
            
            logger.info(f"Epoch {epoch+1} completed: avg_loss={avg_epoch_loss:.4f}")
        
        # Save final model
        self._save_checkpoint("final")
        
        # Save history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training completed!")
        
        return history
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Run evaluation.
        
        Returns:
            Average evaluation loss
        """
        if not TORCH_AVAILABLE or self.eval_dataset is None:
            return float('inf')
        
        self.model.eval()
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in eval_loader:
            images = batch["image"].to(self.device)
            targets = {
                k: v.to(self.device) 
                for k, v in batch.items() 
                if k != "image"
            }
            
            output, losses = self.model(images, targets)
            
            if losses is not None:
                total_loss += losses["total_loss"].item()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Evaluation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _save_checkpoint(self, name: str) -> None:
        """Save checkpoint."""
        if not TORCH_AVAILABLE:
            return
        
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config.to_dict(),
        }
        
        path = checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        if not TORCH_AVAILABLE:
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        logger.info(f"Checkpoint loaded: {path}")


@dataclass
class EvaluationMetrics:
    """
    Evaluation metrics for E2E model.
    
    Attributes:
        cell_precision: Precision for cell detection
        cell_recall: Recall for cell detection
        cell_f1: F1 score for cell detection
        header_accuracy: Accuracy for header classification
        level_accuracy: Accuracy for level prediction
        parent_accuracy: Accuracy for parent prediction
        hierarchy_tree_accuracy: Accuracy for complete tree structure
    """
    cell_precision: float = 0.0
    cell_recall: float = 0.0
    cell_f1: float = 0.0
    header_accuracy: float = 0.0
    level_accuracy: float = 0.0
    parent_accuracy: float = 0.0
    hierarchy_tree_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "cell_precision": self.cell_precision,
            "cell_recall": self.cell_recall,
            "cell_f1": self.cell_f1,
            "header_accuracy": self.header_accuracy,
            "level_accuracy": self.level_accuracy,
            "parent_accuracy": self.parent_accuracy,
            "hierarchy_tree_accuracy": self.hierarchy_tree_accuracy,
        }


class E2EEvaluator:
    """
    Evaluator for End-to-End Hierarchical Table Model.
    
    Computes comprehensive metrics for:
    - Cell detection (precision, recall, F1)
    - Header classification
    - Hierarchy structure (level, parent)
    - Complete tree accuracy
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize evaluator.
        
        Args:
            iou_threshold: IoU threshold for cell matching
        """
        self.iou_threshold = iou_threshold
    
    def evaluate(
        self,
        model: E2EHierarchicalTableModel,
        dataset: Dataset,
        device: str = "cpu",
    ) -> EvaluationMetrics:
        """
        Evaluate model on dataset.
        
        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            device: Device to run on
            
        Returns:
            EvaluationMetrics instance
        """
        if not TORCH_AVAILABLE:
            return EvaluationMetrics()
        
        model.eval()
        model = model.to(device)
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        all_metrics = []
        
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                targets = {
                    k: v.to(device) 
                    for k, v in batch.items() 
                    if k != "image"
                }
                
                output, _ = model(images, targets)
                
                metrics = self._compute_sample_metrics(output, targets)
                all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = self._aggregate_metrics(all_metrics)
        
        return avg_metrics
    
    def _compute_sample_metrics(
        self,
        output: E2EOutput,
        targets: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute metrics for a single sample."""
        # Placeholder implementation
        return {
            "cell_precision": 0.0,
            "cell_recall": 0.0,
            "header_accuracy": 0.0,
            "level_accuracy": 0.0,
            "parent_accuracy": 0.0,
        }
    
    def _aggregate_metrics(
        self,
        all_metrics: List[Dict[str, float]],
    ) -> EvaluationMetrics:
        """Aggregate metrics across samples."""
        if not all_metrics:
            return EvaluationMetrics()
        
        avg = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg[key] = sum(values) / len(values)
        
        return EvaluationMetrics(
            cell_precision=avg.get("cell_precision", 0.0),
            cell_recall=avg.get("cell_recall", 0.0),
            cell_f1=2 * avg.get("cell_precision", 0.0) * avg.get("cell_recall", 0.0) / 
                   max(avg.get("cell_precision", 0.0) + avg.get("cell_recall", 0.0), 1e-6),
            header_accuracy=avg.get("header_accuracy", 0.0),
            level_accuracy=avg.get("level_accuracy", 0.0),
            parent_accuracy=avg.get("parent_accuracy", 0.0),
        )


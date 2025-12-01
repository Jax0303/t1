"""
Experiment script for TableBank and DocLayNet integration.

This script:
1. Loads converted datasets
2. Initializes E2EHierarchicalTableModel
3. Runs training with E2ETrainer
4. Reports metrics
"""
import os
import sys
from pathlib import Path
import logging
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.e2e_hierarchical import E2EHierarchicalTableModel, E2EConfig
from src.models.trainer import (
    E2ETrainer,
    TrainingConfig,
    HierarchicalTableDataset,
    E2EEvaluator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiment(
    train_annotations: str,
    val_annotations: str,
    image_dir: str,
    output_dir: str = "outputs/e2e_training",
    num_epochs: int = 5,
    batch_size: int = 4,
    limit_samples: int = None
):
    """
    Run training experiment.
    
    Args:
        train_annotations: Path to training annotations JSON
        val_annotations: Path to validation annotations JSON
        image_dir: Directory containing images
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        limit_samples: Limit number of samples (for quick testing)
    """
    logger.info("=" * 60)
    logger.info("Starting E2E Hierarchical Table Model Experiment")
    logger.info("=" * 60)
    
    # Check if PyTorch is available
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
    except ImportError:
        logger.error("PyTorch not available! Please install PyTorch to run training.")
        return
    
    # Initialize model config
    config = E2EConfig(
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=3,  # Reduced for faster training
        num_decoder_layers=3,
        max_cells=200,  # Reduced for synthetic data
        max_hierarchy_depth=5,
        dropout=0.1,
        use_contrastive_loss=True,
        image_size=(512, 512),  # Smaller for faster training
    )
    
    logger.info(f"Model config: {config.to_dict()}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = E2EHierarchicalTableModel(config)
    
    # Initialize training config
    train_config = TrainingConfig(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        gradient_clip=1.0,
        save_steps=500,
        eval_steps=250,
        log_steps=50,
        output_dir=output_dir,
    )
    
    logger.info(f"Training config: {train_config.to_dict()}")
    
    # Create datasets
    logger.info(f"Loading training data from {train_annotations}")
    logger.info(f"Loading validation data from {val_annotations}")
    
    try:
        train_dataset = HierarchicalTableDataset(
            annotations_path=train_annotations,
            image_dir=image_dir,
            config=config,
        )
        
        val_dataset = HierarchicalTableDataset(
            annotations_path=val_annotations,
            image_dir=image_dir,
            config=config,
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")
        
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        logger.info("Creating mock datasets for demonstration...")
        
        # Create mock datasets if real data fails
        train_dataset = None
        val_dataset = None
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = E2ETrainer(
        model=model,
        train_config=train_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Run training
    if train_dataset is not None and len(train_dataset) > 0:
        logger.info("Starting training...")
        try:
            history = trainer.train()
            
            logger.info("Training completed!")
            logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
            if history['eval_loss']:
                logger.info(f"Final eval loss: {history['eval_loss'][-1]:.4f}")
            
            # Save results
            results_file = Path(output_dir) / "experiment_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "config": config.to_dict(),
                    "training_config": train_config.to_dict(),
                    "history": history,
                }, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("No training data available. Skipping training.")
        logger.info("To run actual training, provide valid annotations and images.")
    
    logger.info("=" * 60)
    logger.info("Experiment completed!")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run E2E table model experiment")
    parser.add_argument(
        "--train-annotations",
        type=str,
        default="data/processed/synthetic_train.json",
        help="Path to training annotations JSON"
    )
    parser.add_argument(
        "--val-annotations",
        type=str,
        default="data/processed/synthetic_val.json",
        help="Path to validation annotations JSON"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/processed/images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/e2e_training",
        help="Output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited samples"
    )
    
    args = parser.parse_args()
    
    run_experiment(
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        limit_samples=10 if args.debug else None,
    )


if __name__ == "__main__":
    main()

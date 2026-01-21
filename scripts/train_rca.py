import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import logging

from src.hiertable_rag.core.rca_model import RCAModel
from scripts.universal_dataset import UniversalTableDataset
from src.evaluation.tsr_evaluator import TSREvaluator
from src.models.baselines.wrapper import RCAWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RCA_Training")

def rca_collate(batch):
    """Custom collate to skip non-tensor fields like 'cells'."""
    from torch.utils.data.dataloader import default_collate
    # Filter out keys that cause collation errors
    new_batch = []
    for sample in batch:
        new_sample = {k: v for k, v in sample.items() if k not in ["cells", "row_lines", "col_lines"]}
        new_batch.append(new_sample)
    return default_collate(new_batch)

def dice_loss(pred, target, smooth=1e-6):
    """Compute Dice loss for boundary probability maps."""
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=1)
    return 1 - ((2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth))

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        img = batch["image"].to(device)
        row_gt = batch["row_gt"].to(device) # (B, grid_size)
        col_gt = batch["col_gt"].to(device) # (B, grid_size)
        
        optimizer.zero_grad()
        outputs = model(img)
        
        # Loss computation
        loss_row = F.binary_cross_entropy(outputs["row_probs"], row_gt) + dice_loss(outputs["row_probs"], row_gt).mean()
        loss_col = F.binary_cross_entropy(outputs["col_probs"], col_gt) + dice_loss(outputs["col_probs"], col_gt).mean()
        loss_cell = F.binary_cross_entropy(outputs["cell_map"], batch.get("cell_map", torch.zeros_like(outputs["cell_map"])).to(device))
        
        loss = loss_row + loss_col + 0.1 * loss_cell
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    # Config
    base_dir = "/home/user/t1-4/data/external"
    batch_size = 8
    epochs = 10
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    train_dataset = UniversalTableDataset(base_dir, "scitsr", split="train")
    val_dataset = UniversalTableDataset(base_dir, "scitsr", split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=rca_collate)
    
    # Model
    model = RCAModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_teds = 0.0
    
    for epoch in range(epochs): 
        logger.info(f"Epoch {epoch+1}/{epochs}")
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        logger.info(f"Average Training Loss: {avg_loss:.4f}")
        
        # Periodic Evaluation
        if (epoch + 1) % 2 == 0:
            from src.hiertable_rag.core.grid_restorer import IntersectionRestorer
            restorer = IntersectionRestorer()
            rca_wrapper = RCAWrapper(model, restorer)
            
            evaluator = TSREvaluator(val_dataset, {"Proposed_RCA": rca_wrapper})
            evaluator.run_evaluation(num_samples=20)
            
            if not evaluator.results:
                logger.error("Evaluation failed: No results returned.")
                continue

            results = pd.DataFrame(evaluator.results)
            if "Proposed_RCA_TEDS" not in results.columns:
                logger.error("Evaluation failed: 'Proposed_RCA_TEDS' key missing.")
                continue
                
            current_teds = results["Proposed_RCA_TEDS"].mean()
            logger.info(f"Validation TEDS: {current_teds:.4f}")
            
            if current_teds > best_teds:
                best_teds = current_teds
                torch.save(model.state_dict(), "outputs/rca_best.pth")
                logger.info("Saved best model.")

if __name__ == "__main__":
    import torch.nn.functional as F
    import pandas as pd
    os.makedirs("outputs", exist_ok=True)
    main()

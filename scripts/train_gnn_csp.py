import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from hiertable_rag.gnn_csp import GNNCSPPipeline, CellGraphBuilder
import json
import os
import argparse
from glob import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.files = glob(os.path.join(root_dir, '*.json'))
        self.graph_builder = CellGraphBuilder()
        logger.info(f"Found {len(self.files)} files in {root_dir}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx]) as f:
            data = json.load(f)
        
        ocr_boxes = [{'box': c.get('box', [0,0,0,0]), 'content': c.get('text', '')} for c in data['cells']]
        
        # Build GT Graph
        # Note: In real training, we need visual/semantic features from RCA/RoBERTa
        # Using simplified random features for demo executability/speed
        N = len(ocr_boxes)
        v_feat = torch.randn(N, 128)
        s_feat = torch.randn(N, 768)
        
        # We need the graph builder to return labels
        data_obj, labels = self.graph_builder.build_graph(ocr_boxes, return_labels=True, gt_structure=data)
        
        # Manually assemble node features as pipeline does
        # Position features
        positions = torch.zeros((N, 4))
        for i, b in enumerate(ocr_boxes):
            x1, y1, x2, y2 = b['box']
            positions[i] = torch.tensor([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
            
        data_obj.x = torch.cat([v_feat, positions, s_feat], dim=-1)
        data_obj.y_edge = labels
        
        return data_obj

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize pipeline model (only using GNN part)
    # We create a pipeline just to get the initialized GNN easily
    pipeline = GNNCSPPipeline(device=device, semantic_model="roberta-base")
    model = pipeline.gnn
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    dataset = TableDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    logger.info(f"Starting training on {len(dataset)} samples for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        batches = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            node_emb, _ = model(batch)
            edge_logits = model.predict_edges(node_emb, batch.edge_index)
            
            if batch.y_edge is not None:
                # Class weights to handle imbalance (optional but good)
                loss = nn.CrossEntropyLoss()(edge_logits, batch.y_edge)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1
        
        avg_loss = total_loss / batches if batches > 0 else 0
        logger.info(f"Epoch {epoch+1}/{args.epochs}: Loss = {avg_loss:.4f}")
    
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/gnn_csp_trained.pth'
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    train(args)

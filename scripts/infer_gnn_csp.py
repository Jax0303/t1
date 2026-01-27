import torch
import json
import argparse
import logging
from hiertable_rag.gnn_csp import GNNCSPPipeline

logging.basicConfig(level=logging.INFO)

def infer(json_path, checkpoint=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pipeline = GNNCSPPipeline(
        gnn_checkpoint=checkpoint,
        device=device
    )
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Format input for pipeline
    ocr_boxes = []
    for c in data['cells']:
        box = c.get('box', [0,0,0,0])
        # Validate box
        if not box or len(box) != 4: box = [0,0,0,0]
        ocr_boxes.append({'box': box, 'content': c.get('text', '')})
        
    # Dummy image (pipeline handles extraction if needed, but we provide boxes directly)
    # The pipeline expects an image tensor for RCA, but we can bypass if we rely on provided boxes
    image = torch.zeros(3, 1000, 1000).to(device)
    
    # Run pipeline
    result = pipeline.parse(image, ocr_boxes, use_csp=True, use_semantic=False)
    
    print("-" * 50)
    print(f"File: {json_path}")
    print(f"Recovered Structure: {result['num_rows']} Rows x {result['num_cols']} Cols")
    print(f"Constraint Satisfaction: {result['constraint_satisfaction']}")
    print("-" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="Path to input JSON file")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to GNN checkpoint")
    args = parser.parse_args()
    
    infer(args.file, args.checkpoint)

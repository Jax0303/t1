"""
Create synthetic table images for training.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json

def create_synthetic_images(annotations_path: str, output_dir: str):
    """Create synthetic table images matching the annotations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"Creating {len(annotations)} synthetic images...")
    
    for ann in annotations:
        image_path = ann['image_path']
        cells = ann['cells']
        
        # Determine image size from cell bboxes
        max_x = max(cell['bbox'][2] for cell in cells)
        max_y = max(cell['bbox'][3] for cell in cells)
        
        # Create image
        img = Image.new('RGB', (int(max_x) + 20, int(max_y) + 20), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw cells
        for cell in cells:
            x1, y1, x2, y2 = cell['bbox']
            
            # Draw cell border
            color = 'blue' if cell['is_header'] else 'black'
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw cell content
            if cell['content']:
                try:
                    draw.text((x1 + 5, y1 + 5), cell['content'], fill='black')
                except:
                    pass
        
        # Save image
        full_path = output_path / image_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(full_path)
    
    print(f"Created images in {output_dir}")

if __name__ == "__main__":
    create_synthetic_images("data/processed/synthetic_train.json", "data/processed/images")
    create_synthetic_images("data/processed/synthetic_val.json", "data/processed/images")
    print("Synthetic images created!")

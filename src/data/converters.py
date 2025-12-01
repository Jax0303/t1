"""
Data converters for TableBank and DocLayNet to HierarchicalTableDataset format.

This module converts COCO-format table annotations to the format expected by
the E2EHierarchicalTableModel trainer.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class BaseConverter:
    """Base class for dataset converters."""
    
    def convert(self, input_path: str, output_path: str, limit: Optional[int] = None):
        """
        Convert dataset to HierarchicalTableDataset format.
        
        Args:
            input_path: Path to input dataset
            output_path: Path to save converted annotations
            limit: Maximum number of samples to convert (for testing)
        """
        raise NotImplementedError


class DocLayNetConverter(BaseConverter):
    """
    Converts DocLayNet dataset to HierarchicalTableDataset format.
    
    DocLayNet has COCO-format annotations with 11 categories including 'Table'.
    We filter for pages containing tables and extract table bounding boxes.
    """
    
    def convert(self, input_path: str, output_path: str, limit: Optional[int] = None):
        """
        Convert DocLayNet COCO annotations to training format.
        
        Expected input structure:
        - input_path/COCO/train.json (or val.json, test.json)
        - input_path/PNG/*.png
        
        Output format:
        [
            {
                "image_path": "relative/path/to/image.png",
                "cells": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "content": "",  # DocLayNet doesn't have OCR text
                        "row": 0,
                        "col": 0,
                        "rowspan": 1,
                        "colspan": 1,
                        "is_header": false,
                        "level": 0,
                        "parent_idx": -1
                    },
                    ...
                ]
            },
            ...
        ]
        """
        input_dir = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to find COCO annotations
        coco_files = list(input_dir.glob("**/COCO/*.json"))
        if not coco_files:
            logger.warning(f"No COCO annotations found in {input_dir}")
            # Create empty output
            with open(output_file, 'w') as f:
                json.dump([], f, indent=2)
            return
        
        # Use first COCO file found
        coco_file = coco_files[0]
        logger.info(f"Converting DocLayNet from {coco_file}")
        
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        images = {img['id']: img for img in coco_data['images']}
        annotations = coco_data['annotations']
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image
        image_annotations = defaultdict(list)
        for ann in annotations:
            image_annotations[ann['image_id']].append(ann)
        
        converted_data = []
        count = 0
        
        for image_id, img_info in images.items():
            if limit and count >= limit:
                break
            
            # Filter for tables only (category_id 8 is 'Table' in DocLayNet)
            table_anns = [
                ann for ann in image_annotations[image_id]
                if categories.get(ann['category_id']) == 'Table'
            ]
            
            if not table_anns:
                continue
            
            # For each table, create a training sample
            # Note: DocLayNet gives table bounding boxes, not cell-level annotations
            # We'll create a simplified representation
            cells = []
            for idx, ann in enumerate(table_anns):
                bbox = ann['bbox']  # [x, y, width, height] in COCO format
                # Convert to [x1, y1, x2, y2]
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
                cells.append({
                    "bbox": [x1, y1, x2, y2],
                    "content": "",  # No OCR text available
                    "row": 0,
                    "col": idx,
                    "rowspan": 1,
                    "colspan": 1,
                    "is_header": False,
                    "level": 0,
                    "parent_idx": -1
                })
            
            converted_data.append({
                "image_path": img_info['file_name'],
                "cells": cells,
                "hierarchy": {
                    "root": None,
                    "max_level": 1
                }
            })
            
            count += 1
        
        logger.info(f"Converted {len(converted_data)} samples from DocLayNet")
        
        with open(output_file, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        logger.info(f"Saved to {output_file}")


class TableBankConverter(BaseConverter):
    """
    Converts TableBank dataset to HierarchicalTableDataset format.
    
    TableBank has two tasks:
    1. Table Detection: Bounding boxes for tables in documents
    2. Table Structure Recognition: HTML/LaTeX structure of tables
    
    We focus on structure recognition data which has cell-level information.
    """
    
    def convert(self, input_path: str, output_path: str, limit: Optional[int] = None):
        """
        Convert TableBank annotations to training format.
        
        TableBank structure recognition data typically includes:
        - Images of tables
        - HTML or LaTeX markup
        
        We need to parse the HTML/LaTeX to extract cell positions.
        """
        input_dir = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting TableBank from {input_dir}")
        
        # Look for annotation files
        json_files = list(input_dir.glob("**/*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {input_dir}")
            # Create empty output
            with open(output_file, 'w') as f:
                json.dump([], f, indent=2)
            return
        
        converted_data = []
        count = 0
        
        for json_file in json_files:
            if limit and count >= limit:
                break
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # TableBank format varies, handle different structures
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict) and 'annotations' in data:
                    items = data['annotations']
                else:
                    items = [data]
                
                for item in items:
                    if limit and count >= limit:
                        break
                    
                    # Extract image path and cells
                    # This is a simplified version - actual implementation
                    # would need to parse HTML/LaTeX
                    
                    image_path = item.get('filename', item.get('image', 'unknown.png'))
                    
                    # Placeholder cell extraction
                    # In reality, we'd parse HTML/LaTeX structure
                    cells = []
                    
                    if 'cells' in item:
                        cells = item['cells']
                    elif 'html' in item:
                        # Would parse HTML here
                        cells = self._parse_html_to_cells(item['html'])
                    
                    if cells:
                        converted_data.append({
                            "image_path": image_path,
                            "cells": cells,
                            "hierarchy": {
                                "root": None,
                                "max_level": 1
                            }
                        })
                        count += 1
                        
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        logger.info(f"Converted {len(converted_data)} samples from TableBank")
        
        with open(output_file, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        logger.info(f"Saved to {output_file}")
    
    def _parse_html_to_cells(self, html: str) -> List[Dict[str, Any]]:
        """
        Parse HTML table structure to extract cells.
        
        This is a placeholder - full implementation would use BeautifulSoup
        or similar to parse HTML and extract cell positions.
        """
        # Placeholder implementation
        return []


def create_synthetic_dataset(output_path: str, num_samples: int = 50):
    """
    Create a synthetic dataset for testing the training pipeline.
    
    Args:
        output_path: Path to save synthetic annotations
        num_samples: Number of synthetic samples to create
    """
    import numpy as np
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating {num_samples} synthetic samples")
    
    data = []
    
    for i in range(num_samples):
        # Create a synthetic table with random cells
        num_rows = np.random.randint(3, 8)
        num_cols = np.random.randint(3, 6)
        
        cells = []
        cell_width = 100
        cell_height = 40
        
        for row in range(num_rows):
            for col in range(num_cols):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                is_header = (row == 0)
                
                cells.append({
                    "bbox": [x1, y1, x2, y2],
                    "content": f"Cell_{row}_{col}",
                    "row": row,
                    "col": col,
                    "rowspan": 1,
                    "colspan": 1,
                    "is_header": is_header,
                    "level": 1 if is_header else 0,
                    "parent_idx": -1
                })
        
        data.append({
            "image_path": f"synthetic/table_{i:04d}.png",
            "cells": cells,
            "hierarchy": {
                "root": None,
                "max_level": 2
            }
        })
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Created synthetic dataset: {output_file}")
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic dataset for testing
    create_synthetic_dataset("data/processed/synthetic_train.json", num_samples=100)
    create_synthetic_dataset("data/processed/synthetic_val.json", num_samples=20)
    
    print("Synthetic datasets created successfully!")

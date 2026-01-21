
import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class UniversalTableDataset(Dataset):
    """
    Unified dataset loader for RCA benchmark datasets (PubTables-1M, FinTabNet.c, SciTSR, cTDaR).
    Normalizes diverse annotation formats into a standard structure for boundary regression.
    """
    def __init__(
        self,
        base_dir: str,
        dataset_type: str,  # "pubtables", "fintabnet", "scitsr", "ctdar", "pubtabnet"
        split: str = "train",
        transform: Optional[Any] = None,
        img_size: int = 448
    ):
        self.base_dir = Path(base_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.grid_size = img_size // 32 # 448 // 32 = 14
        self.samples = self._load_sample_list()

    def _load_sample_list(self) -> List[Dict[str, Any]]:
        samples = []
        if self.dataset_type == "fintabnet":
            # XML format in structure/val, structure/test, structure/train
            data_dir = self.base_dir / "fintabnet_c" / "extracted" / "FinTabNet.c-Structure" / self.split
            if not data_dir.exists():
                print(f"Warning: {data_dir} does not exist")
                return []
            for xml_file in data_dir.glob("*.xml"):
                img_path = self.base_dir / "fintabnet_c" / "extracted" / "FinTabNet.c-Structure" / "images" / (xml_file.stem + ".jpg")
                if img_path.exists():
                    samples.append({
                        "xml_path": xml_file,
                        "img_path": img_path
                    })
        elif self.dataset_type == "pubtables":
            # XML format in extracted/train, extracted/val, extracted/test
            data_dir = self.base_dir / "pubtables-1m" / "extracted" / self.split
            if not data_dir.exists():
                print(f"Warning: {data_dir} does not exist")
                return []
            # PubTables has images and xml in the same dir or parallel
            # Based on list_dir, they are in the same dir
            for xml_file in data_dir.glob("*.xml"):
                img_path = xml_file.with_suffix(".jpg")
                if img_path.exists():
                    samples.append({
                        "xml_path": xml_file,
                        "img_path": img_path
                    })
        elif self.dataset_type == "scitsr":
            # JSON format in SciTsr_Logical/train/gt or test/gt
            scitsr_dir = "test" if self.split == "val" else self.split
            data_dir = self.base_dir / "scitsr" / "extracted" / "SciTsr_Logical" / scitsr_dir
            if not data_dir.exists():
                print(f"Warning: {data_dir} does not exist")
                return []
            gt_dir = data_dir / "gt"
            img_dir = data_dir / "images"
            for json_file in gt_dir.glob("*.json"):
                img_path = img_dir / (json_file.stem + ".png")
                if img_path.exists():
                    samples.append({
                        "json_path": json_file,
                        "img_path": img_path
                    })
        elif self.dataset_type == "ctdar":
            # Parquet format
            parquet_path = self.base_dir / "ctdar2019" / "data" / f"{self.split}-00000-of-00001.parquet"
            if parquet_path.exists():
                import pandas as pd
                df = pd.read_parquet(parquet_path)
                for idx, row in df.iterrows():
                    samples.append({
                        "parquet_row": row,
                        "dataset": "ctdar"
                    })
        elif self.dataset_type == "pubtabnet":
            # JSONL format
            jsonl_path = self.base_dir / "pubtabnet" / f"PubTabNet_2.0.0_{self.split}.jsonl"
            if jsonl_path.exists():
                import json
                with open(jsonl_path, "r") as f:
                    for line in f:
                        samples.append(json.loads(line))
            else:
                 print(f"Warning: {jsonl_path} does not exist")
        return samples

    def _parse_voc_xml(self, xml_path: Path) -> Dict[str, Any]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        
        rows = []
        cols = []
        
        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            
            # Normalize to [0, 1]
            box = [xmin / width, ymin / height, xmax / width, ymax / height]
            
            if name == "table row":
                rows.append(box)
            elif name == "table column":
                cols.append(box)
                
        # Sort rows by top coordinate, cols by left coordinate
        rows.sort(key=lambda x: x[1])
        cols.sort(key=lambda x: x[0])
        
        return {"rows": rows, "cols": cols, "width": width, "height": height}

    def _parse_scitsr_json(self, json_path: Path) -> Dict[str, Any]:
        import json
        with open(json_path, "r") as f:
            cells = json.load(f)
            
        # Infer row/col lines from cell boxes and logical coords
        # SciTSR box: [xmin, ymin, xmax, ymax]
        # logical_coords: [row_start, row_end, col_start, col_end]
        
        # We need image size to normalize, but we'll do it later or get it from img
        rows = {} # row_idx -> [ymin_list, ymax_list]
        cols = {} # col_idx -> [xmin_list, xmax_list]
        
        for cell in cells:
            r_start, r_end, c_start, c_end = cell["logical_coords"]
            box = cell["box"]
            
            for r in range(r_start, r_end + 1):
                if r not in rows: rows[r] = [[], []]
                rows[r][0].append(box[1])
                rows[r][1].append(box[3])
                
            for c in range(c_start, c_end + 1):
                if c not in cols: cols[c] = [[], []]
                cols[c][0].append(box[0])
                cols[c][1].append(box[2])
                
        # Row boundaries: min of ymin for each row
        row_bounds = []
        for r in sorted(rows.keys()):
            row_bounds.append(min(rows[r][0]))
        if rows:
            last_r = max(rows.keys())
            row_bounds.append(max(rows[last_r][1]))
            
        col_bounds = []
        for c in sorted(cols.keys()):
            col_bounds.append(min(cols[c][0]))
        if cols:
            last_c = max(cols.keys())
            col_bounds.append(max(cols[last_c][1]))

        # Prepare cells for TEDS
        final_cells = []
        for cell in cells:
            r_start, r_end, c_start, c_end = cell["logical_coords"]
            final_cells.append({
                "row_idx": r_start,
                "col_idx": c_start,
                "row_span": r_end - r_start + 1,
                "col_span": c_end - c_start + 1,
                "content": cell.get("content", "")
            })

        return {"row_lines": row_bounds, "col_lines": col_bounds, "cells": final_cells, "raw_cells": cells}

    def _generate_prob_map(self, lines: List[float]) -> torch.Tensor:
        """
        Converts normalized coordinates [0, 1] to a probability map.
        Uses a Gaussian-like smoothing around the GT lines.
        """
        prob_map = torch.zeros(self.grid_size)
        for line in lines:
            idx = int(line * (self.grid_size - 1))
            idx = max(0, min(self.grid_size - 1, idx))
            prob_map[idx] = 1.0
            # Optional: Add small smoothing
            if idx > 0: prob_map[idx-1] = max(prob_map[idx-1], 0.5)
            if idx < self.grid_size - 1: prob_map[idx+1] = max(prob_map[idx+1], 0.5)
        return prob_map

    def _parse_ctdar_parquet(self, row: Any) -> Dict[str, Any]:
        # cTDaR parquet has 'image' {'bytes': ...} and 'objects' {'bbox': [[...]], 'categories': [...]}
        # Categories: 0: table, 1: row, 2: column... (Need to verify)
        # Based on typical cTDaR/HF: 
        # objects = {'bbox': [[xmin, ymin, xmax, ymax], ...], 'categories': [0, 1, 2...]}
        # We'll need to check category mapping
        bboxes = row["objects"]["bbox"]
        cats = row["objects"]["categories"]
        
        row_lines = []
        col_lines = []
        
        # This is a bit complex as we don't know the mapping for sure without checking
        # But let's assume standard VOC-like categories if it was converted from VOC
        for bbox, cat in zip(bboxes, cats):
            # Normalization needs image size
            pass
        return {}

    def _generate_cell_map(self, cells: List[Dict], width: int, height: int) -> torch.Tensor:
        """Generates a 2D binary map where cells are 1.0."""
        mask = torch.zeros((self.grid_size, self.grid_size))
        for cell in cells:
            if "box" in cell:
                # box: [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = cell["box"]
                x1 = int((xmin / width) * (self.grid_size - 1))
                y1 = int((ymin / height) * (self.grid_size - 1))
                x2 = int((xmax / width) * (self.grid_size - 1))
                y2 = int((ymax / height) * (self.grid_size - 1))
                
                # Clip
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.grid_size - 1, x2), min(self.grid_size - 1, y2)
                
                mask[y1:y2+1, x1:x2+1] = 1.0
        return mask.unsqueeze(0) # (1, H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, retry_count=0):
        if retry_count > 10:
            # Return a dummy sample to avoid infinite recursion
            return {
                "image": torch.zeros((3, self.img_size, self.img_size)),
                "row_lines": torch.tensor([0.0, 1.0]),
                "col_lines": torch.tensor([0.0, 1.0]),
                "dataset": "dummy"
            }
        
        sample = self.samples[idx]
        
        if self.dataset_type in ["fintabnet", "pubtables", "scitsr", "pubtabnet"]:
            if self.dataset_type == "pubtabnet":
                img_path = self.base_dir / "pubtabnet" / self.split / sample["filename"]
            else:
                img_path = sample["img_path"]
            
            try:
                img = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img.size
                
                # Resize image
                img_resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
                
                if self.dataset_type == "pubtabnet":
                    html_tokens = sample['html']['structure']['tokens']
                    true_html = "".join(html_tokens)
                    return {
                        "image": img_tensor,
                        "html": true_html,
                        "dataset": self.dataset_type
                    }
                elif self.dataset_type == "scitsr":
                    data = self._parse_scitsr_json(sample["json_path"])
                    row_lines = [y / orig_h for y in data["row_lines"]]
                    col_lines = [x / orig_w for x in data["col_lines"]]
                    return {
                        "image": img_tensor,
                        "row_gt": self._generate_prob_map(row_lines),
                        "col_gt": self._generate_prob_map(col_lines),
                        "cells": data["cells"],
                        "cell_map": self._generate_cell_map(data.get("raw_cells", []), orig_w, orig_h),
                        "dataset": self.dataset_type
                    }
                else:
                    # XML-based
                    data = self._parse_voc_xml(sample["xml_path"])
                    row_lines = []
                    for r in data["rows"]:
                        row_lines.append(r[1]) # top
                    if data["rows"]:
                        row_lines.append(data["rows"][-1][3]) # bottom
                    
                    col_lines = []
                    for c in data["cols"]:
                        col_lines.append(c[0]) # left
                    if data["cols"]:
                        col_lines.append(data["cols"][-1][2]) # right
                    
                return {
                    "image": img_tensor,
                    "row_gt": self._generate_prob_map(row_lines),
                    "col_gt": self._generate_prob_map(col_lines),
                    "row_lines": torch.tensor(row_lines), # Raw coordinates for eval
                    "col_lines": torch.tensor(col_lines),
                    "dataset": self.dataset_type
                }
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                return self.__getitem__(np.random.randint(0, len(self)), retry_count + 1)
        
        elif self.dataset_type == "ctdar":
            import io
            row = sample["parquet_row"]
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            orig_w, orig_h = img.size
            
            img_resized = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
            
            # Need cTDaR category mapping. Usually 0=table, 1=row, 2=col or similar.
            # Let's iterate and try to find rows/cols by aspect ratio if unknown? 
            # Better to check categories.
            # For now, let's keep it minimal or check a few samples.
            return {"image": img_tensor, "dataset": "ctdar"}
        
        return {} # WIP for others

if __name__ == "__main__":
    # Test loader for different datasets
    base_dir = "/home/user/t1-4/data/external"
    
    for dtype in ["fintabnet", "scitsr"]:
        print(f"\n--- Testing {dtype} ---")
        try:
            split = "val"
            ds = UniversalTableDataset(base_dir, dtype, split=split)
            print(f"Loaded {len(ds)} samples from {dtype} {split}")
            if len(ds) > 0:
                batch = ds[0]
                print(f"Image shape: {batch['image'].shape}")
                print(f"Row lines: {batch['row_lines']}")
                print(f"Col lines: {batch['col_lines']}")
        except Exception as e:
            print(f"Failed to load {dtype}: {e}")

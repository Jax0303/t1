#!/usr/bin/env python3
"""
TATR GSR Fine-tuning on PubTables-1M
=====================================
Hugging Face TATR(microsoft/table-transformer-structure-recognition-v1.1-all)을
PubTables-1M 데이터에서 튜닝하여, Baseline과 GSR(SnapGT)의 차이를 검증합니다.

사용법:
  # Baseline 학습
  python experiments/train_tatr_gsr.py --mode baseline --epochs 20 --batch_size 2
  
  # GSR 학습
  python experiments/train_tatr_gsr.py --mode gsr --epochs 20 --batch_size 2
"""
import os
import random
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from gsr_hooks.snap_gt_transform import _extract_separators_from_gt, _snap_bbox

ID2LABEL = {
    0: "table",
    1: "table column",
    2: "table row",
    3: "table column header",
    4: "table projected row header",
    5: "table spanning cell"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

def xyxy_to_cxcywh_normalized(bbox, w, h):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [cx, cy, bw, bh]

class PubTablesTATRDataset(Dataset):
    def __init__(self, img_dir, ann_dir, processor, mode="baseline", max_samples=None, seed=42):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.processor = processor
        self.mode = mode
        
        all_stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png"))
            and os.path.exists(os.path.join(ann_dir, os.path.splitext(f)[0] + ".xml"))
        ])
        if max_samples and max_samples < len(all_stems):
            rng = random.Random(seed)
            self.stems = sorted(rng.sample(all_stems, max_samples))
        else:
            self.stems = all_stems
            
        print(f"[{mode.upper()}] Dataset loaded: {len(self.stems)} samples")
        
    def __len__(self):
        return len(self.stems)
        
    def __getitem__(self, idx):
        stem = self.stems[idx]
        img_path = self._find_img(stem)
        xml_path = os.path.join(self.ann_dir, stem + ".xml")
        
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        boxes_xyxy, labels = self._parse_xml(xml_path)
        
        if self.mode == "gsr":
            if len(boxes_xyxy) > 0:
                boxes_np = np.array(boxes_xyxy, dtype=np.float32)
                labels_np = np.array(labels, dtype=np.int64)
                row_seps, col_seps = _extract_separators_from_gt(boxes_np, labels_np, {2, 4}, {1, 3})
                if len(row_seps) > 0 and len(col_seps) > 0:
                    for i, (b, lbl) in enumerate(zip(boxes_np, labels_np)):
                        if lbl == 5: # spanning cell
                            boxes_xyxy[i] = _snap_bbox(b, row_seps, col_seps).tolist()

        boxes_normalized = [xyxy_to_cxcywh_normalized(b, w, h) for b in boxes_xyxy]
        
        pt_target = {
            "boxes": torch.tensor(boxes_normalized, dtype=torch.float32),
            "class_labels": torch.tensor(labels, dtype=torch.long)
        }
        
        return image, pt_target

    def _find_img(self, stem):
        for ext in [".jpg", ".png"]:
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(stem)

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip()
            if name not in LABEL2ID:
                continue
            bb = obj.find("bndbox")
            if bb is None: continue
            xmin = float(bb.findtext("xmin", 0))
            ymin = float(bb.findtext("ymin", 0))
            xmax = float(bb.findtext("xmax", 0))
            ymax = float(bb.findtext("ymax", 0))
            if xmax <= xmin or ymax <= ymin: continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(LABEL2ID[name])
        return boxes, labels

class TATRCollator:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, batch):
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        batch_encoded = self.processor(images=images, return_tensors="pt")
        return {"pixel_values": batch_encoded["pixel_values"], "labels": labels}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model
    model_id = "microsoft/table-transformer-structure-recognition-v1.1-all"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = TableTransformerForObjectDetection.from_pretrained(
        model_id,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # Dataset
    train_img = os.path.join(args.pubtables_root, "train", "images")
    train_ann = os.path.join(args.pubtables_root, "train", "annotations")
    
    # For dev, fallback to test if train missing
    if not os.path.exists(train_img):
        print(f"WARNING: Train dir {train_img} not found. Using test set for dry run.")
        train_img = os.path.join(args.pubtables_root, "test", "images")
        train_ann = os.path.join(args.pubtables_root, "test", "annotations")

    train_ds = PubTablesTATRDataset(
        img_dir=train_img,
        ann_dir=train_ann,
        processor=processor,
        mode=args.mode,
        max_samples=args.max_train_samples
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=TATRCollator(processor)
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad], 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': 1e-5}
    ], weight_decay=1e-4)
    
    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch_pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            outputs = model(pixel_values=batch_pixel_values, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if args.dry_run and n_batches >= 2:
                print("Dry run completed 2 batches. Exiting.")
                return

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
        
        model.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch}"))
        processor.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["baseline", "gsr"], required=True)
    parser.add_argument("--pubtables_root", default="./data/pubtables-1m")
    parser.add_argument("--output_dir", default="./checkpoints/tatr_gsr")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)
    train(args)

#!/usr/bin/env python3
"""
TATR GSR 모델 평가 스크립트
==============================
Baseline과 GSR 모드로 튜닝된 TATR 체크포인트를 평가합니다.

사용법:
  python experiments/eval_tatr_gsr.py --model_dir checkpoints/tatr_gsr/baseline/epoch_20
  python experiments/eval_tatr_gsr.py --model_dir checkpoints/tatr_gsr/gsr/epoch_20
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import xml.etree.ElementTree as ET

ID2LABEL = {
    0: "table",
    1: "table column",
    2: "table row",
    3: "table column header",
    4: "table projected row header",
    5: "table spanning cell"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = TableTransformerForObjectDetection.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    test_img = os.path.join(args.pubtables_root, "test", "images")
    test_ann = os.path.join(args.pubtables_root, "test", "annotations")
    
    stems = sorted([os.path.splitext(f)[0] for f in os.listdir(test_img) if f.endswith(('.jpg', '.png'))])
    
    if args.max_samples:
        stems = stems[:args.max_samples]
        
    print(f"Evaluating {len(stems)} samples...")
    
    all_results = []
    for stem in tqdm(stems):
        img_p = os.path.join(test_img, stem + ".jpg")
        
        image = Image.open(img_p).convert("RGB")
        w, h = image.size
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]]).to(device) # (h, w)
        results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        all_results.append({
            "stem": stem,
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist()
        })
        
    out_file = os.path.join(args.output_dir, "eval_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f)
    print(f"Saved predictions to {out_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--pubtables_root", default="./data/pubtables-1m")
    parser.add_argument("--output_dir", default="./experiments/results_gsr_eval")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    # modify output dir based on model path base (e.g. baseline or gsr)
    parent_dir = os.path.basename(os.path.dirname(args.model_dir.rstrip("/")))
    args.output_dir = os.path.join(args.output_dir, parent_dir)
    run_eval(args)

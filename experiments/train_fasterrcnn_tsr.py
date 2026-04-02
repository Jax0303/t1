#!/usr/bin/env python3
"""
Faster R-CNN TSR Baseline Training on PubTables-1M
===================================================
목적: detection-based TSR의 boundary precision 문제가 TATR(DETR 계열) 특이적이 아님을
     Faster R-CNN (anchor-based, RPN 기반 — 아키텍처 완전히 다름) 로 입증.

아키텍처 비교:
  TATR: DETR 계열 — Transformer encoder/decoder, bipartite matching loss, no anchors
  FRCNN: anchor-based — RPN (Region Proposal), RoI Pooling/Align, classification + bbox regression

학습 후 boundary precision analysis는 experiments/fasterrcnn_boundary.py 로 실행.

사용법:
  # 1. 학습
  python train_fasterrcnn_tsr.py \
      --pubtables_root /home/ugh/t1/data/pubtables-1m \
      --output_dir     ./checkpoints/fasterrcnn_tsr \
      --epochs 12 \
      --batch_size 4 \
      --lr 0.005

  # 2. 경계 정밀도 분석 (학습 완료 후)
  python fasterrcnn_boundary.py \
      --checkpoint ./checkpoints/fasterrcnn_tsr/best.pth \
      --pubtables_root /home/ugh/t1/data/pubtables-1m \
      --output_dir ./results/fasterrcnn_boundary

NOTE: 학습에 GPU 권장 (CPU: 수 일, RTX3080: 약 6-8시간 @ 12 epochs)
      val mAP 목표: row/col ≥ 0.90, span ≥ 0.75 (논문 수준 baseline)
"""

import os
import json
import random
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────
# Label Map
# ─────────────────────────────────────────────
# PubTables-1M 6 categories + background (0)
LABEL_MAP = {
    "table":                     1,
    "table row":                 2,
    "table column":              3,
    "table spanning cell":       4,
    "table column header":       5,
    "table projected row header": 6,
}
NUM_CLASSES = len(LABEL_MAP) + 1   # +1 for background

# boundary analysis에 필요한 label IDs
LABEL_ROW  = LABEL_MAP["table row"]
LABEL_COL  = LABEL_MAP["table column"]
LABEL_SPAN = LABEL_MAP["table spanning cell"]


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class PubTablesDataset(torch.utils.data.Dataset):
    """
    PubTables-1M Pascal VOC XML annotation loader for Faster R-CNN.

    __getitem__ returns:
        image : PIL.Image converted to torch.FloatTensor [3, H, W]  (0~1)
        target: dict with keys
            boxes  : FloatTensor [N, 4] — xyxy format
            labels : Int64Tensor [N]
            image_id: IntTensor [1]
            area   : FloatTensor [N]
            iscrowd: Int64Tensor [N]  (all 0)
    """

    def __init__(self, img_dir, ann_dir, transforms=None, max_samples=None, seed=42):
        self.img_dir    = img_dir
        self.ann_dir    = ann_dir
        self.transforms = transforms

        all_stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png"))
            and os.path.exists(os.path.join(ann_dir, os.path.splitext(f)[0] + ".xml"))
        ])

        if max_samples and max_samples < len(all_stems):
            rng = random.Random(seed)
            all_stems = sorted(rng.sample(all_stems, max_samples))

        self.stems = all_stems
        print(f"  PubTablesDataset: {len(self.stems)} samples ({img_dir})")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem     = self.stems[idx]
        img_path = self._find_img(stem)
        xml_path = os.path.join(self.ann_dir, stem + ".xml")

        image = Image.open(img_path).convert("RGB")
        w, h  = image.size

        boxes, labels = self._parse_xml(xml_path)

        # PIL → tensor
        img_t = torchvision.transforms.functional.to_tensor(image)

        target = {
            "boxes":    torch.as_tensor(boxes,  dtype=torch.float32),
            "labels":   torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area":     torch.as_tensor(
                [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32
            ) if boxes else torch.zeros(0, dtype=torch.float32),
            "iscrowd":  torch.zeros(len(boxes), dtype=torch.int64),
        }

        if self.transforms:
            img_t, target = self.transforms(img_t, target)

        return img_t, target

    def _find_img(self, stem):
        for ext in [".jpg", ".jpeg", ".png"]:
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
            if name not in LABEL_MAP:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = float(bb.findtext("xmin", 0))
            ymin = float(bb.findtext("ymin", 0))
            xmax = float(bb.findtext("xmax", 0))
            ymax = float(bb.findtext("ymax", 0))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(LABEL_MAP[name])
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
def build_model(num_classes, pretrained_backbone=True):
    """
    torchvision Faster R-CNN ResNet-50 FPN.
    COCO pretrained weights for backbone + FPN, 새 head만 학습.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ─────────────────────────────────────────────
# Validation mAP (simplified — per-class recall@IoU0.5)
# ─────────────────────────────────────────────
def compute_val_recall(model, data_loader, device, iou_thresh=0.5):
    """
    빠른 validation: per-class recall@IoU0.5 (Hungarian matching).
    full COCO mAP보다 빠르고 boundary precision 목적에 충분.
    """
    from torchvision.ops import box_iou as tv_box_iou
    from scipy.optimize import linear_sum_assignment

    model.eval()
    stats = defaultdict(lambda: {"tp": 0, "n_gt": 0})

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="  [val]", leave=False):
            images  = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes  = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)

                pred_boxes  = output["boxes"]
                pred_labels = output["labels"]
                pred_scores = output["scores"]

                # conf > 0.5 필터
                keep = pred_scores >= 0.5
                pred_boxes  = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                for cls_id in [LABEL_ROW, LABEL_COL, LABEL_SPAN]:
                    gt_mask   = (gt_labels == cls_id)
                    pred_mask = (pred_labels == cls_id)
                    n_gt = int(gt_mask.sum())
                    stats[cls_id]["n_gt"] += n_gt
                    if n_gt == 0:
                        continue
                    gb = gt_boxes[gt_mask]
                    pb = pred_boxes[pred_mask]
                    if len(pb) == 0:
                        continue
                    iou_m = tv_box_iou(gb, pb).cpu().numpy()
                    gi, pi = linear_sum_assignment(1.0 - iou_m)
                    tp = sum(1 for g, p in zip(gi, pi) if iou_m[g, p] >= iou_thresh)
                    stats[cls_id]["tp"] += tp

    recall = {}
    name_map = {LABEL_ROW: "row", LABEL_COL: "col", LABEL_SPAN: "span"}
    for cls_id, name in name_map.items():
        n_gt = stats[cls_id]["n_gt"]
        tp   = stats[cls_id]["tp"]
        recall[name] = tp / n_gt if n_gt > 0 else 0.0
    return recall


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────
    train_img = os.path.join(args.pubtables_root, "train", "images")
    train_ann = os.path.join(args.pubtables_root, "train", "annotations")
    val_img   = os.path.join(args.pubtables_root, "val",   "images")
    val_ann   = os.path.join(args.pubtables_root, "val",   "annotations")

    train_ds = PubTablesDataset(train_img, train_ann,
                                max_samples=args.max_train_samples)
    val_ds   = PubTablesDataset(val_img,   val_ann,
                                max_samples=args.max_val_samples)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(NUM_CLASSES, pretrained_backbone=True)
    model.to(device)

    # ── Optimizer ─────────────────────────────────────────────────────
    # Backbone frozen for first 3 epochs (warm-up), then fine-tune all
    params_head = [p for n, p in model.named_parameters()
                   if "backbone" not in n and p.requires_grad]
    params_backbone = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]

    optimizer = torch.optim.SGD(
        [
            {"params": params_head,     "lr": args.lr},
            {"params": params_backbone, "lr": args.lr * 0.1},
        ],
        momentum=0.9,
        weight_decay=1e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.6), int(args.epochs * 0.85)],
        gamma=0.1,
    )

    # ── Training ──────────────────────────────────────────────────────
    best_span_recall = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Backbone unfreeze after epoch 3
        if epoch == 4:
            for p in params_backbone:
                p.requires_grad_(True)
            print("  -> Backbone unfrozen")

        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += losses.item()
            n_batches  += 1

        lr_scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        recall = compute_val_recall(model, val_loader, device)
        span_recall = recall["span"]

        log = {
            "epoch":        epoch,
            "train_loss":   avg_loss,
            "val_row_rec":  recall["row"],
            "val_col_rec":  recall["col"],
            "val_span_rec": span_recall,
        }
        history.append(log)
        print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"row={recall['row']:.3f}  col={recall['col']:.3f}  span={recall['span']:.3f}")

        # Save best
        ckpt = {
            "epoch":         epoch,
            "model_state":   model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "recall":        recall,
            "label_map":     LABEL_MAP,
            "num_classes":   NUM_CLASSES,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pth"))
        if span_recall > best_span_recall:
            best_span_recall = span_recall
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
            print(f"  ** Best span recall updated: {best_span_recall:.3f}")

    # Save history
    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best span recall: {best_span_recall:.3f}")
    print(f"Checkpoint: {args.output_dir}/best.pth")


# ─────────────────────────────────────────────
# Post-training: Boundary Precision Analysis
# ─────────────────────────────────────────────
"""
학습 완료 후 boundary precision analysis는 아래 스크립트로:
  experiments/fasterrcnn_boundary.py

핵심 차이:
  TATR inference: AutoModelForObjectDetection (HuggingFace)
  FRCNN inference: torchvision model.eval() + post_process (confidence filter)

분석 로직은 boundary_precision_analysis.py와 동일:
  GT span ↔ pred span (label=LABEL_SPAN=4) Hungarian matching
  → analyze_boundary() → IoU bin vs grid error rate

예상 결과 (TATR와 동일 패턴 확인 목표):
  Recall@0.5 ≈ 0.75-0.85
  Recall@0.9 ≈ 0.35-0.50  (35pp+ drop expected)
  Grid error monotonic decrease with IoU
  → regression target 문제가 아키텍처 무관함을 입증
"""


def build_inference_fn(checkpoint_path, device):
    """
    학습된 Faster R-CNN checkpoint를 로드해서
    boundary_precision_analysis.py 호환 inference 함수 반환.

    사용법:
        infer_fn = build_inference_fn("best.pth", device)
        pred_spans = infer_fn(image)  # list of [x1,y1,x2,y2]
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get("num_classes", NUM_CLASSES)

    model = build_model(num_classes, pretrained_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    @torch.no_grad()
    def infer_fn(image_pil, conf_threshold=0.5):
        """PIL Image → pred spans list [[x1,y1,x2,y2], ...]"""
        img_t = torchvision.transforms.functional.to_tensor(image_pil).to(device)
        output = model([img_t])[0]

        boxes  = output["boxes"].cpu()
        labels = output["labels"].cpu()
        scores = output["scores"].cpu()

        keep = scores >= conf_threshold
        boxes  = boxes[keep]
        labels = labels[keep]

        pred_spans = [boxes[i].tolist()
                      for i in range(len(labels)) if labels[i] == LABEL_SPAN]
        pred_rows  = [boxes[i].tolist()
                      for i in range(len(labels)) if labels[i] == LABEL_ROW]
        pred_cols  = [boxes[i].tolist()
                      for i in range(len(labels)) if labels[i] == LABEL_COL]
        return {"spans": pred_spans, "rows": pred_rows, "cols": pred_cols}

    return infer_fn


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pubtables_root",     default="/home/ugh/t1/data/pubtables-1m")
    p.add_argument("--output_dir",         default="./checkpoints/fasterrcnn_tsr")
    p.add_argument("--epochs",             type=int,   default=12)
    p.add_argument("--batch_size",         type=int,   default=4)
    p.add_argument("--lr",                 type=float, default=0.005)
    p.add_argument("--num_workers",        type=int,   default=4)
    p.add_argument("--max_train_samples",  type=int,   default=None,
                   help="학습 샘플 수 제한 (None=전체, 빠른 테스트용으로 10000 등)")
    p.add_argument("--max_val_samples",    type=int,   default=2000,
                   help="validation 샘플 수 제한")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

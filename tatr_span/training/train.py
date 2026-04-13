"""
TATR-Span Training Script
===========================
DETRSpan (TATR + ordinal regression for rowspan/colspan) 학습.

실행 예시 (단일 GPU):
  python tatr_span/training/train.py --config tatr_span/training/config.yaml

실행 예시 (4 GPU DDP):
  torchrun --nproc_per_node=4 tatr_span/training/train.py \
      --config tatr_span/training/config.yaml

주요 플래그:
  --config       YAML 설정 파일 경로
  --resume       체크포인트 경로 (재개)
  --eval-only    학습 없이 평가만
  --dataset      데이터셋 override (pubtables1m | fintabnet_c | scitsr)
  --output-dir   출력 디렉토리 override
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

# TATR base 경로 추가
REPO_ROOT = Path(__file__).parents[2]
TATR_BASE = REPO_ROOT / "tatr_base" / "detr"
sys.path.insert(0, str(TATR_BASE))

from util.misc import get_rank, get_world_size, init_distributed_mode, is_main_process

# 내부 모듈
sys.path.insert(0, str(REPO_ROOT))
from tatr_span.models.detr_span import build_detr_span, load_pretrained_weights
from tatr_span.datasets.transforms import make_train_transforms, make_val_transforms


# ─────────────────────────────────────────────────────────────
# Config Dataclass
# ─────────────────────────────────────────────────────────────

class Args:
    """YAML dict → argparse.Namespace 호환 Args 객체."""

    def __init__(self, cfg: dict):
        m  = cfg["model"]
        tr = cfg["training"]
        ls = cfg["loss"]
        ds = cfg["dataset"]
        pt = cfg.get("pretrained", {})
        ou = cfg["output"]
        ev = cfg.get("eval", {})
        di = cfg.get("dist", {})

        # Model
        self.num_classes      = m["num_classes"]
        self.num_queries      = m["num_queries"]
        self.aux_loss         = m["aux_loss"]
        self.backbone         = m["backbone"]
        self.dilation         = m.get("dilation", False)
        self.position_embedding = m.get("position_embedding", "sine")
        self.hidden_dim       = m.get("hidden_dim", 256)
        self.dropout          = m.get("dropout", 0.1)
        self.nheads           = m.get("nheads", 8)
        self.dim_feedforward  = m.get("dim_feedforward", 2048)
        self.enc_layers       = m.get("enc_layers", 6)
        self.dec_layers       = m.get("dec_layers", 6)
        self.pre_norm         = m.get("pre_norm", False)
        self.ordinal_k        = m.get("ordinal_k", 9)
        self.masks            = False
        self.frozen_weights   = None

        # Pretrained
        self.pretrained_path  = pt.get("path")
        self.pretrained_strict = pt.get("strict", False)

        # Training
        self.epochs           = tr["epochs"]
        self.batch_size       = tr["batch_size"]
        self.num_workers      = tr.get("num_workers", 4)
        self.lr               = tr["lr"]
        self.lr_backbone      = tr["lr_backbone"]
        self.weight_decay     = tr["weight_decay"]
        self.clip_max_norm    = tr.get("clip_max_norm", 0.1)
        self.lr_drop          = tr.get("lr_drop", 15)
        self.min_size         = tr.get("min_size", 800)
        self.max_size         = tr.get("max_size", 1333)

        # Loss
        self.ce_loss_coef     = ls.get("ce_loss_coef", 1.0)
        self.bbox_loss_coef   = ls.get("bbox_loss_coef", 5.0)
        self.giou_loss_coef   = ls.get("giou_loss_coef", 2.0)
        self.eos_coef         = ls.get("eos_coef", 0.1)
        self.span_loss_coef   = ls.get("span_loss_coef", 1.0)
        ew_raw                = ls.get("emphasized_weights", {})
        self.emphasized_weights = {int(k): float(v) for k, v in ew_raw.items()}

        # Dataset
        self.dataset_primary  = ds.get("primary", "pubtables1m")
        self.ds_pubtables     = ds.get("pubtables1m", {})
        self.ds_fintabnet     = ds.get("fintabnet_c", {})
        self.ds_scitsr        = ds.get("scitsr", {})

        # Output
        self.output_dir       = ou.get("output_dir", "outputs/tatr_span")
        self.save_freq        = ou.get("save_freq", 5)
        self.log_freq         = ou.get("log_freq", 50)
        self.eval_freq        = ou.get("eval_freq", 1)

        # Eval
        self.eval_threshold   = ev.get("threshold", 0.5)

        # Dist
        self.dist_backend     = di.get("backend", "nccl")
        self.dist_url         = di.get("dist_url", "env://")
        self.device           = "cuda" if torch.cuda.is_available() else "cpu"

        # Runtime (overridable by CLI)
        self.resume           = None
        self.eval_only        = False
        self.dataset_file     = "structure"   # TATR compat


# ─────────────────────────────────────────────────────────────
# Dataset Factory
# ─────────────────────────────────────────────────────────────

def build_dataset(args: Args, split: str):
    """config에 따라 데이터셋 생성."""
    is_train = split == "train"
    tf = make_train_transforms(args.min_size, args.max_size) if is_train \
         else make_val_transforms(args.min_size, args.max_size)

    ds_name = args.dataset_primary

    if ds_name == "pubtables1m":
        from tatr_span.datasets.pubtables1m import PubTables1MDataset
        cfg = args.ds_pubtables
        return PubTables1MDataset(
            data_root=cfg["data_root"],
            split=split,
            transforms=tf,
            use_span_attr=cfg.get("use_span_attr", True),
            complex_only=cfg.get("complex_only", False),
        )

    elif ds_name == "fintabnet_c":
        from tatr_span.datasets.fintabnet_c import FinTabNetCDataset
        cfg = args.ds_fintabnet
        split_map = cfg.get("split_map", {"train": "train", "val": "validation", "test": "test"})
        hf_split = split_map.get(split, split)
        return FinTabNetCDataset(
            split=hf_split,
            cache_dir=cfg.get("cache_dir"),
            transforms=tf,
            use_span_attr=cfg.get("use_span_attr", True),
            complex_only=cfg.get("complex_only", False),
        )

    elif ds_name == "scitsr":
        from tatr_span.datasets.scitsr import SciTSRDataset
        cfg = args.ds_scitsr
        return SciTSRDataset(
            data_root=cfg["data_root"],
            split=split,
            transforms=tf,
            use_span_attr=cfg.get("use_span_attr", True),
            complex_only=cfg.get("complex_only", False),
        )

    else:
        raise ValueError(f"Unknown dataset: {ds_name}")


# ─────────────────────────────────────────────────────────────
# Training / Evaluation
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, data_loader, optimizer,
                    device, epoch, max_norm, log_freq):
    model.train()
    criterion.train()

    metric_dict: Dict[str, float] = {}
    n_steps = 0

    for step, (samples, targets) in enumerate(data_loader):
        from util.misc import nested_tensor_from_tensor_list
        samples = nested_tensor_from_tensor_list(samples)
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # Accumulate for logging
        for k, v in loss_dict.items():
            metric_dict[k] = metric_dict.get(k, 0.0) + v.item()
        n_steps += 1

        if is_main_process() and (step + 1) % log_freq == 0:
            avg = {k: v / n_steps for k, v in metric_dict.items()}
            parts = [f"Epoch {epoch} [{step+1}/{len(data_loader)}]"]
            parts += [f"{k}: {v:.4f}" for k, v in sorted(avg.items())
                      if not k.startswith("cardinality")]
            print("  ".join(parts))

    return {k: v / max(n_steps, 1) for k, v in metric_dict.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    metric_dict: Dict[str, float] = {}
    n_steps = 0

    for samples, targets in data_loader:
        from util.misc import nested_tensor_from_tensor_list
        samples = nested_tensor_from_tensor_list(samples)
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        for k, v in loss_dict.items():
            metric_dict[k] = metric_dict.get(k, 0.0) + v.item()
        n_steps += 1

    return {k: v / max(n_steps, 1) for k, v in metric_dict.items()}


def save_checkpoint(args, epoch, model, optimizer, lr_scheduler, metrics):
    if not is_main_process():
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":        epoch,
        "model":        model.module.state_dict() if hasattr(model, "module")
                        else model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "args":         vars(args) if hasattr(args, "__dict__") else {},
        "metrics":      metrics,
    }
    path = out / f"checkpoint_{epoch:03d}.pth"
    torch.save(state, path)
    # latest symlink
    latest = out / "checkpoint_latest.pth"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    print(f"  [ckpt] saved → {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("TATR-Span training")
    parser.add_argument("--config",     required=True,
                        help="YAML 설정 파일 경로")
    parser.add_argument("--resume",     default=None,
                        help="체크포인트 재개 경로")
    parser.add_argument("--eval-only",  action="store_true",
                        help="평가만 실행")
    parser.add_argument("--dataset",    default=None,
                        help="데이터셋 override (pubtables1m|fintabnet_c|scitsr)")
    parser.add_argument("--output-dir", default=None,
                        help="출력 디렉토리 override")
    parser.add_argument("--local_rank", type=int, default=0)
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    args = Args(cfg)
    if cli.resume:
        args.resume = cli.resume
    if cli.eval_only:
        args.eval_only = True
    if cli.dataset:
        args.dataset_primary = cli.dataset
    if cli.output_dir:
        args.output_dir = cli.output_dir

    # Distributed init
    init_distributed_mode(args)
    device = torch.device(args.device)

    # Model
    model, criterion, _ = build_detr_span(args)

    # Pretrained weights
    if args.pretrained_path and Path(args.pretrained_path).exists():
        load_pretrained_weights(model, args.pretrained_path, strict=args.pretrained_strict)

    model.to(device)

    # DDP
    model_without_ddp = model
    if getattr(args, "distributed", False):
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    if is_main_process():
        print(f"[model] trainable params: {n_params:,}")

    # Optimizer — separate LR for backbone
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    # Datasets
    dataset_train = build_dataset(args, "train")
    dataset_val   = build_dataset(args, "val")

    if getattr(args, "distributed", False):
        sampler_train = DistributedSampler(dataset_train)
        sampler_val   = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val   = SequentialSampler(dataset_val)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        collate_fn=dataset_train.collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory if hasattr(args, "pin_memory") else True,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        sampler=sampler_val,
        collate_fn=dataset_val.collate_fn,
        num_workers=args.num_workers,
    )

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if is_main_process():
            print(f"[resume] from epoch {start_epoch}")

    if args.eval_only:
        metrics = evaluate(model, criterion, loader_val, device)
        if is_main_process():
            print("[eval]", {k: f"{v:.4f}" for k, v in sorted(metrics.items())})
        return

    # Training loop
    if is_main_process():
        print(f"[train] start — epochs {start_epoch}~{args.epochs-1}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = math.inf
    for epoch in range(start_epoch, args.epochs):
        if getattr(args, "distributed", False):
            sampler_train.set_epoch(epoch)

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, criterion, loader_train, optimizer,
            device, epoch, args.clip_max_norm, args.log_freq)
        lr_scheduler.step()
        t1 = time.time()

        val_metrics = {}
        if (epoch + 1) % args.eval_freq == 0:
            val_metrics = evaluate(model, criterion, loader_val, device)

        if is_main_process():
            elapsed = t1 - t0
            total_loss = train_metrics.get("loss_ce", 0) + \
                         train_metrics.get("loss_bbox", 0) + \
                         train_metrics.get("loss_giou", 0)
            span_loss = train_metrics.get("loss_rowspan", 0) + \
                        train_metrics.get("loss_colspan", 0)
            print(f"Epoch {epoch:3d} | {elapsed:.0f}s | "
                  f"total={total_loss:.4f} span={span_loss:.4f}")
            if val_metrics:
                print(f"  val: " + "  ".join(
                    f"{k}={v:.4f}" for k, v in sorted(val_metrics.items())
                    if not k.startswith("cardinality")))

            # Log to JSON
            log_entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            with open(output_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            all_m = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            save_checkpoint(args, epoch, model_without_ddp,
                            optimizer, lr_scheduler, all_m)

    if is_main_process():
        print("[train] done.")


if __name__ == "__main__":
    main()

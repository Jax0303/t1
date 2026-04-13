"""
TSRDet Training Script
========================
Cascade R-CNN + Single-Label Regularization 학습.

실행:
  python baselines/tsrdet/train.py --config baselines/tsrdet/config.yaml

실행 (4 GPU):
  torchrun --nproc_per_node=4 baselines/tsrdet/train.py \
      --config baselines/tsrdet/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baselines.tsrdet.cascade_rcnn import build_tsrdet
from tatr_span.datasets.transforms import make_train_transforms, make_val_transforms


def build_dataset_pubtables(split: str, cfg: dict):
    from tatr_span.datasets.pubtables1m import PubTables1MDataset
    is_train = split == "train"
    tf = make_train_transforms(cfg.get("min_size", 800), cfg.get("max_size", 1333)) \
         if is_train else make_val_transforms(cfg.get("min_size", 800), cfg.get("max_size", 1333))
    return PubTables1MDataset(
        data_root=cfg["data_root"],
        split=split,
        transforms=tf,
        use_span_attr=False,
    )


def _collate_fn(batch):
    """torchvision detection 형식 collate."""
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def train_one_epoch(model, optimizer, loader, device, epoch, log_freq):
    model.train()
    for step, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        if (step + 1) % log_freq == 0:
            print(f"Epoch {epoch} [{step+1}/{len(loader)}] "
                  f"loss={losses.item():.4f}  "
                  + "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items()))


def main():
    parser = argparse.ArgumentParser("TSRDet training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    tr  = cfg["training"]
    ds  = cfg["dataset"]
    m   = cfg["model"]
    out = cfg["output"]

    dist_flag = "LOCAL_RANK" in __import__("os").environ
    if dist_flag:
        import torch.distributed as td
        td.init_process_group("nccl")
        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_tsrdet(
        num_classes=m.get("num_classes", 5),
        pretrained_backbone=m.get("pretrained_backbone", True),
        slr_weight=m.get("slr_weight", 0.1),
    ).to(device)

    if dist_flag:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)

    ds_train = build_dataset_pubtables("train", ds)
    ds_val   = build_dataset_pubtables("val",   ds)

    sampler_train = DistributedSampler(ds_train) if dist_flag else RandomSampler(ds_train)
    loader_train  = DataLoader(ds_train, batch_size=tr["batch_size"],
                               sampler=sampler_train,
                               collate_fn=_collate_fn,
                               num_workers=tr.get("num_workers", 4),
                               drop_last=True)
    loader_val    = DataLoader(ds_val, batch_size=1,
                               sampler=SequentialSampler(ds_val),
                               collate_fn=_collate_fn,
                               num_workers=tr.get("num_workers", 4))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=tr.get("lr", 0.02),
        momentum=tr.get("momentum", 0.9),
        weight_decay=tr.get("weight_decay", 1e-4),
    )
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=tr.get("lr_milestones", [12, 16]),
        gamma=tr.get("lr_gamma", 0.1),
    )

    output_dir = Path(out["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if cli.resume:
        ckpt = torch.load(cli.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer"    in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt: lr_sched.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1

    for epoch in range(start_epoch, tr["epochs"]):
        if dist_flag:
            sampler_train.set_epoch(epoch)

        t0 = time.time()
        train_one_epoch(model, optimizer, loader_train, device, epoch,
                        out.get("log_freq", 50))
        lr_sched.step()

        if not dist_flag or local_rank == 0:
            print(f"Epoch {epoch:3d} | {time.time()-t0:.0f}s")
            if (epoch + 1) % out.get("save_freq", 5) == 0:
                ckpt = {
                    "epoch":        epoch,
                    "model":        (model.module if dist_flag else model).state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "lr_scheduler": lr_sched.state_dict(),
                }
                torch.save(ckpt, output_dir / f"checkpoint_{epoch:03d}.pth")

    if not dist_flag or local_rank == 0:
        print("[train] done.")


if __name__ == "__main__":
    main()

"""
Deformable-DETR Fine-tuning on PubTables-1M
=============================================
fundamentalvision/Deformable-DETR를 TATR 6-class 구조 인식에 파인튜닝.

실행 (단일 GPU):
  python baselines/deformable_detr/finetune.py --config baselines/deformable_detr/config.yaml

실행 (4 GPU):
  torchrun --nproc_per_node=4 baselines/deformable_detr/finetune.py \
      --config baselines/deformable_detr/config.yaml

Deformable-DETR src/ 필요 — setup.py 먼저 실행할 것.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

# Deformable-DETR src 경로
SRC_DIR = Path(__file__).parent / "src"
if not SRC_DIR.exists():
    sys.exit("[error] Deformable-DETR src/ 없음. setup.py 먼저 실행하세요.")
sys.path.insert(0, str(SRC_DIR))

# TATR datasets
REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tatr_span.datasets.transforms import make_train_transforms, make_val_transforms


def build_deformable_detr(args_dict: dict, device):
    """Deformable-DETR src/ build() 호출."""
    import argparse as _ap
    a = _ap.Namespace(**args_dict)
    from models import build_model
    model, criterion, postprocessors = build_model(a)
    model.to(device)
    return model, criterion, postprocessors


def build_dataset_pubtables(split: str, cfg: dict):
    from tatr_span.datasets.pubtables1m import PubTables1MDataset
    is_train = split == "train"
    tf = make_train_transforms(cfg.get("min_size", 800), cfg.get("max_size", 1333)) \
         if is_train else make_val_transforms(cfg.get("min_size", 800), cfg.get("max_size", 1333))
    return PubTables1MDataset(
        data_root=cfg["data_root"],
        split=split,
        transforms=tf,
        use_span_attr=False,   # Deformable-DETR baseline: span attr 미사용
    )


def main():
    parser = argparse.ArgumentParser("Deformable-DETR finetune")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    # Distributed init (간소화)
    dist = "LOCAL_RANK" in __import__("os").environ
    if dist:
        import torch.distributed as td
        td.init_process_group("nccl")
        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m   = cfg["model"]
    tr  = cfg["training"]
    ls  = cfg["loss"]
    ds  = cfg["dataset"]
    out = cfg["output"]

    # Build model args for Deformable-DETR
    model_args = {
        "backbone":              m["backbone"],
        "num_classes":           m["num_classes"],
        "num_queries":           m["num_queries"],
        "two_stage":             m.get("two_stage", False),
        "with_box_refine":       m.get("with_box_refine", True),
        "num_feature_levels":    m.get("num_feature_levels", 4),
        "dec_n_points":          m.get("dec_n_points", 4),
        "enc_n_points":          m.get("enc_n_points", 4),
        "hidden_dim":            m.get("hidden_dim", 256),
        "nheads":                m.get("nheads", 8),
        "enc_layers":            m.get("enc_layers", 6),
        "dec_layers":            m.get("dec_layers", 6),
        "dim_feedforward":       m.get("dim_feedforward", 1024),
        "dropout":               m.get("dropout", 0.1),
        "aux_loss":              m.get("aux_loss", True),
        "masks":                 False,
        "frozen_weights":        None,
        "device":                str(device),
        # loss
        "cls_loss_coef":         ls.get("cls_loss_coef", 2.0),
        "bbox_loss_coef":        ls.get("bbox_loss_coef", 5.0),
        "giou_loss_coef":        ls.get("giou_loss_coef", 2.0),
        "focal_alpha":           ls.get("focal_alpha", 0.25),
        "eos_coef":              0.1,
        "set_cost_class":        2.0,
        "set_cost_bbox":         5.0,
        "set_cost_giou":         2.0,
        "position_embedding":    "sine",
        "dilation":              False,
        "dataset_file":          "coco",
    }

    model, criterion, _ = build_deformable_detr(model_args, device)

    if dist:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    model_without_ddp = model.module if dist else model

    # Datasets
    ds_train = build_dataset_pubtables("train", ds)
    ds_val   = build_dataset_pubtables("val",   ds)

    sampler_train = DistributedSampler(ds_train) if dist else RandomSampler(ds_train)
    sampler_val   = SequentialSampler(ds_val)

    loader_train = DataLoader(ds_train, batch_size=tr["batch_size"],
                              sampler=sampler_train,
                              collate_fn=ds_train.collate_fn,
                              num_workers=tr.get("num_workers", 4),
                              drop_last=True)
    loader_val   = DataLoader(ds_val, batch_size=1,
                              sampler=sampler_val,
                              collate_fn=ds_val.collate_fn,
                              num_workers=tr.get("num_workers", 4))

    # Optimizer
    lr_linear_proj_names = ["reference_points", "sampling_offsets"]
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if "backbone" not in n
                    and not any(nd in n for nd in lr_linear_proj_names)
                    and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if "backbone" not in n
                    and any(nd in n for nd in lr_linear_proj_names)
                    and p.requires_grad],
         "lr": tr["lr"] * m.get("lr_linear_proj_mult", 0.1)},
        {"params": [p for n, p in model_without_ddp.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": tr["lr_backbone"]},
    ]
    optimizer   = torch.optim.AdamW(param_dicts, lr=tr["lr"],
                                    weight_decay=tr["weight_decay"])
    lr_sched    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tr.get("lr_drop", 15))

    output_dir  = Path(out["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if cli.resume:
        ckpt = torch.load(cli.resume, map_location="cpu")
        model_without_ddp.load_state_dict(ckpt["model"])
        if "optimizer"    in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt: lr_sched.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1

    if cli.eval_only:
        model.eval()
        print("[eval-only] done.")
        return

    for epoch in range(start_epoch, tr["epochs"]):
        if dist:
            sampler_train.set_epoch(epoch)

        model.train()
        criterion.train()
        t0 = time.time()

        for step, (samples, targets) in enumerate(loader_train):
            from util.misc import nested_tensor_from_tensor_list
            samples = nested_tensor_from_tensor_list(samples).to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]
            out_dict = model(samples)
            loss_dict = criterion(out_dict, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k]
                         for k in loss_dict if k in weight_dict)
            optimizer.zero_grad()
            losses.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tr.get("clip_max_norm", 0.1))
            optimizer.step()

            if (not dist or local_rank == 0) and (step + 1) % out.get("log_freq", 50) == 0:
                print(f"Epoch {epoch} [{step+1}/{len(loader_train)}] loss={losses.item():.4f}")

        lr_sched.step()
        elapsed = time.time() - t0

        if not dist or local_rank == 0:
            print(f"Epoch {epoch:3d} | {elapsed:.0f}s")
            if (epoch + 1) % out.get("save_freq", 5) == 0:
                ckpt = {
                    "epoch":        epoch,
                    "model":        model_without_ddp.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "lr_scheduler": lr_sched.state_dict(),
                }
                torch.save(ckpt, output_dir / f"checkpoint_{epoch:03d}.pth")

    if not dist or local_rank == 0:
        print("[train] done.")


if __name__ == "__main__":
    main()

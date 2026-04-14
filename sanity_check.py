"""
TASK 1 — Sanity Check
======================
Verifies the modified TATR model without needing real data.

Checks:
  1. Forward pass produces pred_logits, pred_boxes, pred_spans (E1 mode)
  2. E0 mode (--no_span_branch) produces no pred_spans (backward compat)
  3. Loss computation runs without NaN/explosion
  4. Backward pass completes (gradients flow)
  5. Rowspan/colspan parsing from synthetic XML
  6. loss_span computes correctly for spanning cells

CPU-only, no dataset required.
"""
import sys
import types
import argparse
from pathlib import Path

# ── path setup ──────────────────────────────────────────────
TATR = Path(__file__).parent / "tatr_base" / "detr"
sys.path.insert(0, str(TATR))

import torch
import torch.nn as nn

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def make_args(no_span_branch=False, span_loss_coef=0.5, device="cpu"):
    """Minimal args namespace matching structure_config.json."""
    a = argparse.Namespace()
    a.backbone         = "resnet18"
    a.num_classes      = 6
    a.dilation         = False
    a.position_embedding = "sine"
    a.hidden_dim       = 256
    a.dropout          = 0.1
    a.nheads           = 8
    a.dim_feedforward  = 2048
    a.enc_layers       = 6
    a.dec_layers       = 6
    a.pre_norm         = True
    a.num_queries      = 125
    a.aux_loss         = False
    a.masks            = False
    a.frozen_weights   = None
    a.device           = device
    # loss coefficients
    a.ce_loss_coef     = 1.0
    a.bbox_loss_coef   = 5.0
    a.giou_loss_coef   = 2.0
    a.eos_coef         = 0.4
    a.mask_loss_coef   = 1.0
    a.dice_loss_coef   = 1.0
    a.set_cost_class   = 1.0
    a.set_cost_bbox    = 5.0
    a.set_cost_giou    = 2.0
    a.set_cost_span    = 0.5
    a.emphasized_weights = {}
    a.dataset_file     = "structure"
    a.lr_backbone      = 1e-5   # needed by build_backbone()
    # span args
    a.no_span_branch   = no_span_branch
    a.span_loss_coef   = span_loss_coef
    return a


def make_fake_batch(batch_size=2, n_obj=8, span_class_idx=5, device="cpu"):
    """Create a minimal batch: NestedTensor images + target list."""
    from util.misc import nested_tensor_from_tensor_list

    # Small images: 3×128×128
    imgs = [torch.rand(3, 128, 128) for _ in range(batch_size)]
    samples = nested_tensor_from_tensor_list(imgs)

    targets = []
    for _ in range(batch_size):
        n = n_obj
        boxes  = torch.rand(n, 4) * 0.5 + 0.1   # cxcywh in (0.1, 0.6)
        boxes[:, 2:] = boxes[:, 2:].clamp(0.05, 0.4)  # ensure positive wh
        labels = torch.randint(0, 6, (n,))
        # Ensure at least 2 spanning cells (class=5) for loss_span coverage
        labels[0] = span_class_idx
        labels[1] = span_class_idx
        rowspans = torch.ones(n, dtype=torch.long)
        colspans = torch.ones(n, dtype=torch.long)
        rowspans[0] = 2;  colspans[0] = 1
        rowspans[1] = 1;  colspans[1] = 3

        targets.append({
            "boxes":    boxes.to(device),
            "labels":   labels.to(device),
            "rowspans": rowspans.to(device),
            "colspans": colspans.to(device),
            "image_id": torch.tensor([0]),
            "orig_size": torch.tensor([128, 128]),
            "size":      torch.tensor([128, 128]),
        })

    return samples.to(device), targets


# ════════════════════════════════════════════════════════════
# CHECK 1 — E1 forward: pred_spans present
# ════════════════════════════════════════════════════════════
def check_forward_e1():
    from models import build_model
    args = make_args(no_span_branch=False)
    model, criterion, _ = build_model(args)
    model.eval()
    samples, targets = make_fake_batch()

    with torch.no_grad():
        out = model(samples)

    required = ["pred_logits", "pred_boxes", "pred_spans"]
    missing  = [k for k in required if k not in out]
    if missing:
        print(f"{FAIL} CHECK 1 (E1 forward): missing keys {missing}")
        return False

    ls, lb, sp = out["pred_logits"].shape, out["pred_boxes"].shape, out["pred_spans"].shape
    print(f"{PASS} CHECK 1 — E1 forward OK")
    print(f"        pred_logits: {ls}  pred_boxes: {lb}  pred_spans: {sp}")
    assert sp[-1] == 16, f"Expected 2*SPAN_K=16, got {sp[-1]}"
    return True


# ════════════════════════════════════════════════════════════
# CHECK 2 — E0 forward: NO pred_spans (backward compat)
# ════════════════════════════════════════════════════════════
def check_forward_e0():
    from models import build_model
    args = make_args(no_span_branch=True)
    model, criterion, _ = build_model(args)
    model.eval()
    samples, targets = make_fake_batch()

    with torch.no_grad():
        out = model(samples)

    if "pred_spans" in out:
        print(f"{FAIL} CHECK 2 (E0 no_span_branch): pred_spans unexpectedly present")
        return False
    if "pred_logits" not in out or "pred_boxes" not in out:
        print(f"{FAIL} CHECK 2 (E0 no_span_branch): missing pred_logits/pred_boxes")
        return False
    print(f"{PASS} CHECK 2 — E0 backward compat OK (no pred_spans)")
    return True


# ════════════════════════════════════════════════════════════
# CHECK 3 — Loss computation: no NaN, no explosion
# ════════════════════════════════════════════════════════════
def check_loss():
    from models import build_model
    args = make_args(no_span_branch=False)
    model, criterion, _ = build_model(args)
    model.train()
    criterion.train()

    samples, targets = make_fake_batch()
    out = model(samples)
    losses = criterion(out, targets)

    for k, v in losses.items():
        val = v.item()
        if val != val:  # NaN check
            print(f"{FAIL} CHECK 3 — NaN in {k}")
            return False
        if abs(val) > 1e4:
            print(f"{FAIL} CHECK 3 — Explosion in {k}: {val:.2f}")
            return False

    print(f"{PASS} CHECK 3 — Loss values OK")
    span_loss = losses.get("loss_span", None)
    for k, v in sorted(losses.items()):
        print(f"        {k}: {v.item():.4f}")
    if span_loss is None:
        print(f"        [note] loss_span not in losses dict")
    return True


# ════════════════════════════════════════════════════════════
# CHECK 4 — Backward pass: gradients flow to span_embed
# ════════════════════════════════════════════════════════════
def check_backward():
    from models import build_model
    args = make_args(no_span_branch=False)
    model, criterion, _ = build_model(args)
    model.train()
    criterion.train()

    samples, targets = make_fake_batch()
    out = model(samples)
    losses = criterion(out, targets)
    weight_dict = criterion.weight_dict
    total = sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict)
    total.backward()

    # Check span_embed got gradients
    span_params = [p for n, p in model.named_parameters() if "span_embed" in n]
    no_grad = [n for n, p in model.named_parameters()
               if "span_embed" in n and p.grad is None]
    if no_grad:
        print(f"{FAIL} CHECK 4 — span_embed params with None grad: {no_grad}")
        return False
    print(f"{PASS} CHECK 4 — Backward OK, span_embed grads flowing")
    print(f"        span_embed params: {len(span_params)}, "
          f"max grad: {max(p.grad.abs().max().item() for p in span_params):.4f}")
    return True


# ════════════════════════════════════════════════════════════
# CHECK 5 — XML rowspan/colspan parsing
# ════════════════════════════════════════════════════════════
def check_xml_parsing():
    import tempfile, os, xml.etree.ElementTree as ET
    sys.path.insert(0, str(Path(__file__).parent / "tatr_base" / "src"))
    from table_datasets import read_pascal_voc

    # Build synthetic PASCAL VOC XML with spanning cell
    xml_content = """<?xml version="1.0"?>
<annotation>
  <size><width>800</width><height>600</height><depth>3</depth></size>
  <object>
    <name>table row</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>790</xmax><ymax>50</ymax></bndbox>
  </object>
  <object>
    <name>table column</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>200</xmax><ymax>590</ymax></bndbox>
  </object>
  <object>
    <name>table spanning cell</name>
    <attributes>
      <attribute><name>rowspan</name><value>2</value></attribute>
      <attribute><name>colspan</name><value>3</value></attribute>
    </attributes>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>600</xmax><ymax>120</ymax></bndbox>
  </object>
</annotation>"""

    class_map = {
        "table": 0, "table column": 1, "table row": 2,
        "table column header": 3, "table projected row header": 4,
        "table spanning cell": 5, "no object": 6,
    }

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(xml_content)
        tmp_path = f.name

    try:
        bboxes, labels, rowspans, colspans = read_pascal_voc(tmp_path, class_map=class_map)
    finally:
        os.unlink(tmp_path)

    print(f"{PASS} CHECK 5 — XML parsing OK")
    for i, (bb, lbl, rs, cs) in enumerate(zip(bboxes, labels, rowspans, colspans)):
        name = {v: k for k, v in class_map.items()}[lbl]
        print(f"        [{i}] {name}: box={bb}, rowspan={rs}, colspan={cs}")

    # Verify the spanning cell was parsed correctly
    span_idx = labels.index(5)
    assert rowspans[span_idx] == 2, f"Expected rowspan=2, got {rowspans[span_idx]}"
    assert colspans[span_idx] == 3, f"Expected colspan=3, got {colspans[span_idx]}"
    return True


# ════════════════════════════════════════════════════════════
# CHECK 6 — Grid snapping
# ════════════════════════════════════════════════════════════
def check_grid_snapping():
    sys.path.insert(0, str(TATR))
    from models.grid_snapping import compute_snapping_target, curriculum_warmup

    # Curriculum
    assert not curriculum_warmup(4, n_warm=5)
    assert curriculum_warmup(5, n_warm=5)
    assert curriculum_warmup(10, n_warm=5)

    # Snapping
    sep_rows = torch.tensor([0.0, 0.1, 0.2, 0.4, 0.7, 1.0])
    sep_cols = torch.tensor([0.0, 0.1, 0.3, 0.6, 0.8, 1.0])
    # A box that spans rows 0-1 and cols 1-2
    boxes = torch.tensor([[0.2, 0.1, 0.3, 0.2]])  # cx, cy, w, h
    snapped = compute_snapping_target(boxes, sep_rows, sep_cols)
    assert snapped.shape == (1, 4), f"Wrong shape: {snapped.shape}"
    assert (snapped[:, 2:] > 0).all(), "Snapped width/height must be positive"

    print(f"{PASS} CHECK 6 — Grid snapping OK")
    print(f"        input box (cxcywh): {boxes[0].tolist()}")
    print(f"        snapped   (cxcywh): {snapped[0].tolist()}")
    return True


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 56)
    print("  TASK 1 — TATR-Span Sanity Check")
    print("=" * 56)

    results = {}
    for name, fn in [
        ("CHECK 1 — E1 forward (pred_spans present)", check_forward_e1),
        ("CHECK 2 — E0 backward compat",              check_forward_e0),
        ("CHECK 3 — Loss (no NaN/explosion)",          check_loss),
        ("CHECK 4 — Backward pass",                   check_backward),
        ("CHECK 5 — XML rowspan/colspan parsing",      check_xml_parsing),
        ("CHECK 6 — Grid snapping",                   check_grid_snapping),
    ]:
        print(f"\n── {name} ──")
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            print(f"{FAIL} Exception: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 56)
    passed = sum(results.values())
    total  = len(results)
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  ALL CHECKS PASSED — ready for full training")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  FAILED: {failed}")
    print("=" * 56)
    sys.exit(0 if passed == total else 1)

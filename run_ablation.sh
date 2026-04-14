#!/usr/bin/env bash
# run_ablation.sh — 4-experiment ablation for TATR-Span
# =======================================================
# E0: baseline TATR (no span branch, no grid-snapping)
# E1: + span loss (ordinal regression head, no grid-snapping)
# E2: + span loss + hard grid-snapping (n_warm=5)
# E3: + span loss + soft grid-snapping (n_warm=5)
#
# Each experiment: 3 seeds × 20 epochs, AdamW lr=1e-4, BS=2, ResNet-18
#
# Usage:
#   bash run_ablation.sh [DATA_ROOT] [OUTPUT_DIR] [CONFIG]
#
# Defaults:
#   DATA_ROOT  = /mnt/d/pubtables1m
#   OUTPUT_DIR = outputs/ablation
#   CONFIG     = tatr_base/src/structure_config.json

set -euo pipefail

DATA_ROOT="${1:-/mnt/d/pubtables1m}"
OUTPUT_DIR="${2:-outputs/ablation}"
CONFIG="${3:-tatr_base/src/structure_config.json}"
TRAIN_SCRIPT="tatr_base/src/main.py"

SEEDS=(42 43 44)
EPOCHS=20
LR=1e-4
BATCH=2
BACKBONE=resnet18

# Common args shared across all experiments
COMMON_ARGS=(
    --data_root_dir "${DATA_ROOT}"
    --config_file   "${CONFIG}"
    --backbone      "${BACKBONE}"
    --data_type     structure
    --mode          train
    --epochs        "${EPOCHS}"
    --lr            "${LR}"
    --batch_size    "${BATCH}"
)

run_experiment() {
    local exp_id="$1"
    local seed="$2"
    shift 2
    local extra_args=("$@")

    local out_dir="${OUTPUT_DIR}/${exp_id}/seed${seed}"
    mkdir -p "${out_dir}"
    local log_file="${out_dir}/train.log"

    echo "========================================"
    echo "  ${exp_id}  seed=${seed}"
    echo "  output → ${out_dir}"
    echo "========================================"

    python "${TRAIN_SCRIPT}" \
        "${COMMON_ARGS[@]}" \
        --model_save_dir "${out_dir}" \
        "${extra_args[@]}" \
        2>&1 | tee "${log_file}"

    echo "[done] ${exp_id} seed=${seed}"
}

# ── E0: Baseline TATR ────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    run_experiment "E0_baseline" "${seed}" \
        --no_span_branch \
        --no_grid_snapping
done

# ── E1: + Span Loss ──────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    run_experiment "E1_span" "${seed}" \
        --span_loss_coef 0.5 \
        --no_grid_snapping
done

# ── E2: + Span Loss + Hard Grid-Snapping ─────────────────────
for seed in "${SEEDS[@]}"; do
    run_experiment "E2_snap_hard" "${seed}" \
        --span_loss_coef 0.5 \
        --grid_snapping hard \
        --n_warm 5
done

# ── E3: + Span Loss + Soft Grid-Snapping ─────────────────────
for seed in "${SEEDS[@]}"; do
    run_experiment "E3_snap_soft" "${seed}" \
        --span_loss_coef 0.5 \
        --grid_snapping soft \
        --n_warm 5
done

echo "========================================"
echo "All ablation experiments complete."
echo "Results in: ${OUTPUT_DIR}"
echo "========================================"

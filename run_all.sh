#!/usr/bin/env bash
# run_all.sh — Full 12-run ablation (E0~E3, seeds 42/43/44, 20 epochs each)
# =============================================================================
# Runs sequentially by default. Set PARALLEL=1 to attempt GPU-parallel runs
# (only if multiple GPUs available).
#
# Usage:
#   bash run_all.sh [DATA_ROOT] [OUTPUT_DIR]
#
# Defaults:
#   DATA_ROOT  = /mnt/d/pubtables1m/PubTables-1M-Structure
#   OUTPUT_DIR = outputs/ablation

set -euo pipefail

DATA_ROOT="${1:-/mnt/d/pubtables1m/PubTables-1M-Structure}"
OUTPUT_DIR="${2:-outputs/ablation}"
CONFIG="tatr_base/src/structure_config.json"
TRAIN_SCRIPT="tatr_base/src/main.py"
EVAL_SCRIPT="tatr_base/src/eval_by_complexity.py"

SEEDS=(42 43 44)
EPOCHS=20
BACKBONE=resnet18

# ── GPU check ─────────────────────────────────────────────────
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "$N_GPUS" -eq 0 ]; then
    DEVICE="cpu"
    echo "[warn] No GPU found — running on CPU (training will be slow)"
else
    DEVICE="cuda"
    echo "[info] ${N_GPUS} GPU(s) detected"
fi

# ── Helpers ───────────────────────────────────────────────────
run_one() {
    local exp_id="$1"
    local seed="$2"
    shift 2
    local extra_args=("$@")

    local out_dir="${OUTPUT_DIR}/${exp_id}/seed${seed}"
    mkdir -p "${out_dir}"
    local log_file="${out_dir}/run.log"
    local metrics_file="${out_dir}/metrics.json"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  EXP: ${exp_id}  SEED: ${seed}"
    echo "  OUT: ${out_dir}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python "${TRAIN_SCRIPT}" \
        --data_root_dir  "${DATA_ROOT}" \
        --config_file    "${CONFIG}" \
        --backbone       "${BACKBONE}" \
        --data_type      structure \
        --mode           train \
        --epochs         "${EPOCHS}" \
        --model_save_dir "${out_dir}" \
        --metrics_save_filepath "${metrics_file}" \
        --device         "${DEVICE}" \
        --seed           "${seed}" \
        --num_workers    2 \
        "${extra_args[@]}" \
        2>&1 | tee "${log_file}"

    echo "[done] ${exp_id} seed=${seed} → ${log_file}"
}

# ── E0: Baseline TATR ────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    run_one "E0_baseline" "${seed}" \
        --no_span_branch \
        --no_grid_snapping
done

# ── E1: + Span Attribute Branch ───────────────────────────────
for seed in "${SEEDS[@]}"; do
    run_one "E1_span" "${seed}" \
        --span_loss_coef 0.5 \
        --no_grid_snapping
done

# ── E2: + Span Loss + Hard Grid-Snapping ─────────────────────
for seed in "${SEEDS[@]}"; do
    run_one "E2_snap_hard" "${seed}" \
        --span_loss_coef 0.5 \
        --grid_snapping  hard \
        --n_warm         5
done

# ── E3: + Span Loss + Soft Grid-Snapping ─────────────────────
for seed in "${SEEDS[@]}"; do
    run_one "E3_snap_soft" "${seed}" \
        --span_loss_coef 0.5 \
        --grid_snapping  soft \
        --n_warm         5
done

# ── Aggregate results ─────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Aggregating results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python tatr_base/src/eval_by_complexity.py \
    --results_dir "${OUTPUT_DIR}" \
    --data_root   "${DATA_ROOT}" \
    --xml_subdir  test \
    --output_csv  "${OUTPUT_DIR}/results_by_complexity.csv"

python aggregate_results.py \
    --results_dir "${OUTPUT_DIR}" \
    --output_csv  "${OUTPUT_DIR}/summary_results.csv"

echo ""
echo "All done. Results in: ${OUTPUT_DIR}/"

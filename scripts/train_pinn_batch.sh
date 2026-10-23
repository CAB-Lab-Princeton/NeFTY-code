#!/bin/bash
# Batch training + evaluation + aggregation of the PINN baseline on the benchmark specimens.
#
#   bash scripts/train_pinn_batch.sh                                   # PINN baseline (GradNorm)
#   CONFIG=configs/pinn.txt bash scripts/train_pinn_batch.sh           # fixed-loss-weight recipe
#
# Environment: CONFIG (configs/pinn_gradnorm.txt), DATA_DIR (./data/synthetic_1000), SEED (0),
# TRAIN_ARGS (extra arguments for train_pinn.py), plus the shared options in
# scripts/_common.sh (GPUS, MAX_PER_GPU, OUTPUT_ROOT, INDICES, SETTINGS, DATASET_SIZE,
# DRY_RUN, PYTHON). The script stops at the first failing stage.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

CONFIG="${CONFIG:-configs/pinn_gradnorm.txt}"
DATA_DIR="${DATA_DIR:-./data/synthetic_1000}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logs/output_pinn_$(date +%Y%m%d_%H%M%S)}"

echo "Config: ${CONFIG} | Data: ${DATA_DIR} | GPUs: ${GPUS} (max ${MAX_PER_GPU} per GPU) | Output: ${OUTPUT_ROOT}"
START_TIME=$(date +%s)

${PYTHON} train_batch.py --script train_pinn.py \
  --config "${CONFIG}" --data_dir "${DATA_DIR}" --output_root "${OUTPUT_ROOT}" --seed "${SEED}" \
  --extra_args "${TRAIN_ARGS:-}" \
  "${POOL_ARGS[@]}" ${SELECT_ARGS[@]+"${SELECT_ARGS[@]}"} ${DRY_ARGS[@]+"${DRY_ARGS[@]}"}

${PYTHON} evaluate_pinn_batch.py --output_root "${OUTPUT_ROOT}" \
  "${POOL_ARGS[@]}" ${SELECT_ARGS[@]+"${SELECT_ARGS[@]}"} ${DRY_ARGS[@]+"${DRY_ARGS[@]}"}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run: nothing was executed."
else
  echo "Done in $(( $(date +%s) - START_TIME ))s. Summary: ${OUTPUT_ROOT}/metrics_report.txt"
fi

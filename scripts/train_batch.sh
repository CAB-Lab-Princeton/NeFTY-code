#!/bin/bash
# Batch training + evaluation + aggregation of NeFTY (or Grid Opt. with
# CONFIG=configs/grid_opt.txt) on the benchmark specimens.
#
#   bash scripts/train_batch.sh
#   CONFIG=configs/grid_opt.txt bash scripts/train_batch.sh
#   CONFIG=configs/quick/nefty.txt DATA_DIR=./data/synthetic_test GPUS=0 MAX_PER_GPU=1 \
#       INDICES="0 7" DATASET_SIZE=8 bash scripts/train_batch.sh
#
# Environment: CONFIG (configs/nefty.txt), DATA_DIR (./data/synthetic_1000), SEED (0),
# TRAIN_ARGS (extra arguments for train.py, e.g. "--N_iters 100"), plus the shared
# options in scripts/_common.sh (GPUS, MAX_PER_GPU, OUTPUT_ROOT, INDICES, SETTINGS,
# DATASET_SIZE, DRY_RUN, PYTHON). The script stops at the first failing stage.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

CONFIG="${CONFIG:-configs/nefty.txt}"
DATA_DIR="${DATA_DIR:-./data/synthetic_1000}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logs/output_batch_$(date +%Y%m%d_%H%M%S)}"

echo "Config: ${CONFIG} | Data: ${DATA_DIR} | GPUs: ${GPUS} (max ${MAX_PER_GPU} per GPU) | Output: ${OUTPUT_ROOT}"
START_TIME=$(date +%s)

${PYTHON} train_batch.py --script train.py \
  --config "${CONFIG}" --data_dir "${DATA_DIR}" --output_root "${OUTPUT_ROOT}" --seed "${SEED}" \
  --extra_args "${TRAIN_ARGS:-}" \
  "${POOL_ARGS[@]}" ${SELECT_ARGS[@]+"${SELECT_ARGS[@]}"} ${DRY_ARGS[@]+"${DRY_ARGS[@]}"}

${PYTHON} evaluate_batch.py --output_root "${OUTPUT_ROOT}" \
  "${POOL_ARGS[@]}" ${SELECT_ARGS[@]+"${SELECT_ARGS[@]}"} ${DRY_ARGS[@]+"${DRY_ARGS[@]}"}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run: nothing was executed."
else
  echo "Done in $(( $(date +%s) - START_TIME ))s. Summary: ${OUTPUT_ROOT}/metrics_report.txt"
fi

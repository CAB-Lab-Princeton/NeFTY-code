#!/bin/bash
# Train the Sound-Only U-Net on the defect-free dataset, then evaluate it on the
# defect benchmark specimens.
#
#   bash scripts/train_unet_sound_only.sh
#
# Environment: CONFIG (configs/unet_sound_only.txt), TRAIN_DATA_DIR
# (./data/synthetic_1000_nodefect), EVAL_DATA_DIR (./data/synthetic_1000), SEED (0),
# GPU (training GPU, default: first entry of GPUS), EXPNAME (unet_sound_only),
# BASEDIR (./logs), TRAIN_ARGS (extra arguments for train_unet.py), plus the shared
# options in scripts/_common.sh (GPUS, MAX_PER_GPU, OUTPUT_ROOT, INDICES, SETTINGS,
# DATASET_SIZE, DRY_RUN, PYTHON) for the evaluation stage.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

CONFIG="${CONFIG:-configs/unet_sound_only.txt}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-./data/synthetic_1000_nodefect}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-./data/synthetic_1000}"
SEED="${SEED:-0}"
GPU="${GPU:-${GPUS%%,*}}"
EXPNAME="${EXPNAME:-unet_sound_only}"
BASEDIR="${BASEDIR:-./logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logs/output_unet_batch_$(date +%Y%m%d_%H%M%S)}"

read -r -a TRAIN_EXTRA <<< "${TRAIN_ARGS:-}"
TRAIN_CMD=(${PYTHON} train_unet.py --config "${CONFIG}" --datadir "${TRAIN_DATA_DIR}"
           --expname "${EXPNAME}" --basedir "${BASEDIR}" --seed "${SEED}" --gpu "${GPU}"
           ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"})

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "${TRAIN_CMD[*]}"
else
  "${TRAIN_CMD[@]}"
fi

${PYTHON} evaluate_unet_batch.py --checkpoint "${BASEDIR}/${EXPNAME}" --datadir "${EVAL_DATA_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  "${POOL_ARGS[@]}" ${SELECT_ARGS[@]+"${SELECT_ARGS[@]}"} ${DRY_ARGS[@]+"${DRY_ARGS[@]}"}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run: nothing was executed."
else
  echo "Summary: ${OUTPUT_ROOT}/metrics_report.txt"
fi

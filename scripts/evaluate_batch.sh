#!/bin/bash
# Re-run the per-specimen evaluation of an existing batch and aggregate the metrics.
#
#   OUTPUT_ROOT=./logs/output_batch_<timestamp> bash scripts/evaluate_batch.sh
#   OUTPUT_ROOT=... METHOD=pinn bash scripts/evaluate_batch.sh
#
# Environment: OUTPUT_ROOT (required), METHOD (nefty | grid_opt | pinn; default nefty),
# METRICS_ONLY=1 (only re-aggregate existing metrics.txt files), plus GPUS, MAX_PER_GPU,
# DRY_RUN, PYTHON from scripts/_common.sh. The tasks are those recorded by the training
# stage (OUTPUT_ROOT/tasks.json), unless INDICES / SETTINGS / DATASET_SIZE select them or
# ALL_FOUND=1 evaluates every run directory found under OUTPUT_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

METHOD="${METHOD:-nefty}"
if [[ -z "${OUTPUT_ROOT:-}" || ! -d "${OUTPUT_ROOT}" ]]; then
  echo "Set OUTPUT_ROOT to an existing batch directory." >&2
  exit 1
fi

case "${METHOD}" in
  nefty|grid_opt) EVAL_SCRIPT=evaluate_batch.py ;;
  pinn) EVAL_SCRIPT=evaluate_pinn_batch.py ;;
  *) echo "Unknown METHOD: ${METHOD}" >&2; exit 1 ;;
esac

EXTRA=()
if [[ "${METRICS_ONLY:-0}" == "1" ]]; then EXTRA+=(--metrics_only); fi
if [[ "${ALL_FOUND:-0}" == "1" ]]; then EXTRA+=(--all_found); fi
if [[ -n "${INDICES:-}${SETTINGS:-}${DATASET_SIZE:-}" ]]; then EXTRA+=(${SELECT_ARGS[@]+"${SELECT_ARGS[@]}"}); fi

${PYTHON} "${EVAL_SCRIPT}" --output_root "${OUTPUT_ROOT}" \
  "${POOL_ARGS[@]}" ${DRY_ARGS[@]+"${DRY_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}

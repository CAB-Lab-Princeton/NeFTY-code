#!/bin/bash
# Generate the synthetic benchmark (homogeneous + layered, with defects) and the
# defect-free dataset used to train the Sound-Only U-Net.
#
#   bash scripts/generate_data.sh
#
# Environment: NUM_SAMPLES (1000), SEED (42), DATA_ROOT (./data), PYTHON (python),
# GEN_ARGS (extra arguments for dataset/synthetic.py, e.g. "--no_vis"),
# DEFECT_FREE_ARGS (extra arguments for dataset/generate_defect_free.py).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -gt 0 ]]; then
  echo "This script takes no arguments; use GEN_ARGS / DEFECT_FREE_ARGS (see its header)." >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-./data}"
read -r -a GEN_EXTRA <<< "${GEN_ARGS:-}"
read -r -a DF_EXTRA <<< "${DEFECT_FREE_ARGS:-}"

${PYTHON} -m dataset.synthetic \
  --out_dir "${DATA_ROOT}/synthetic_${NUM_SAMPLES}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" ${GEN_EXTRA[@]+"${GEN_EXTRA[@]}"}

${PYTHON} -m dataset.generate_defect_free \
  --out_dir "${DATA_ROOT}/synthetic_${NUM_SAMPLES}_nodefect" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" ${DF_EXTRA[@]+"${DF_EXTRA[@]}"}

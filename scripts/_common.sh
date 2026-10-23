# Shared settings for the batch wrappers (sourced, not executed).
# All options are environment variables; positional arguments are rejected so that
# every stage (training, evaluation, aggregation) sees the same settings.
#
#   GPUS          comma-separated GPU ids                    (default 0,1)
#   MAX_PER_GPU   concurrent runs per GPU                    (default 20)
#   OUTPUT_ROOT   batch output directory                     (default ./logs/<name>_<timestamp>)
#   INDICES       space-separated sample indices             (default: all samples)
#   SETTINGS      space-separated categories, e.g. "1_defect 3_layer"
#   DATASET_SIZE  number of samples in the dataset           (default: the size recorded in
#                 OUTPUT_ROOT/tasks.json, else 1000)
#   DRY_RUN=1     print the commands of every stage without running or writing anything
#   PYTHON        python executable                          (default python)

if [[ $# -gt 0 ]]; then
  echo "This script takes no arguments; configure it with environment variables" >&2
  echo "(see the header of $0)." >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1}"
MAX_PER_GPU="${MAX_PER_GPU:-20}"

# Task selection, passed unchanged to every stage that takes it. --dataset_size is only
# forwarded when DATASET_SIZE is set, so a reused OUTPUT_ROOT keeps its recorded size.
SELECT_ARGS=()
if [[ -n "${DATASET_SIZE:-}" ]]; then SELECT_ARGS+=(--dataset_size "${DATASET_SIZE}"); fi
if [[ -n "${INDICES:-}" ]]; then read -r -a _idx <<< "${INDICES}"; SELECT_ARGS+=(--indices "${_idx[@]}"); fi
if [[ -n "${SETTINGS:-}" ]]; then read -r -a _set <<< "${SETTINGS}"; SELECT_ARGS+=(--settings "${_set[@]}"); fi

POOL_ARGS=(--gpus "${GPUS}" --max_per_gpu "${MAX_PER_GPU}")
DRY_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then DRY_ARGS=(--dry_run); fi

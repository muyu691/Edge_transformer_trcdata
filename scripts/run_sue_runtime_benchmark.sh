#!/bin/bash
# Slurm array for SUE runtime benchmarking on the test split.
#
# Default usage from the project root:
#   sbatch scripts/run_sue_runtime_benchmark.sh
#
# For a faster smoke test:
#   sbatch --export=ALL,NUM_TEST_GRAPHS=20 scripts/run_sue_runtime_benchmark.sh
#
# Checkpointing is enabled by default. If a long Anaheim run times out, submit
# the same command again and it will continue from the existing per-graph CSV.

#SBATCH -J sue_runtime
#SBATCH -o logs/sue_runtime_%A_%a.out
#SBATCH -e logs/sue_runtime_%A_%a.err
#SBATCH -t 4-00:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -A NA
#SBATCH -p cpu
#SBATCH --array=0-2

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
if [[ ! -f "${PROJECT_ROOT}/scripts/benchmark_sue_runtime.py" ]]; then
  PROJECT_ROOT="/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main"
fi
if [[ ! -f "${PROJECT_ROOT}/scripts/benchmark_sue_runtime.py" ]]; then
  echo "Could not find scripts/benchmark_sue_runtime.py under PROJECT_ROOT=${PROJECT_ROOT}" >&2
  echo "Submit this script from the project root or set PROJECT_ROOT explicitly." >&2
  exit 1
fi

DATASETS=(siouxfalls ema anaheim)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET_NAME="${DATASETS[$TASK_ID]}"

PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/sue_runtime}"
NUM_TEST_GRAPHS="${NUM_TEST_GRAPHS:-0}"
MAX_ITER="${MAX_ITER:-120}"
CONVERGENCE_THRESHOLD="${CONVERGENCE_THRESHOLD:-1e-5}"
THETA="${THETA:-0.8}"
VALUE_ITER="${VALUE_ITER:-250}"
VALUE_TOL="${VALUE_TOL:-1e-8}"
FLOW_ITER="${FLOW_ITER:-500}"
FLOW_TOL="${FLOW_TOL:-1e-9}"
RETRY_FLOW_ITER="${RETRY_FLOW_ITER:-2000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
RESUME="${RESUME:-1}"

if [[ "${DATASET_NAME}" == "siouxfalls" ]]; then
  PAIRS_DIR="${PROCESSED_ROOT}/siouxfalls_pairs_newpolicy_lhs"
elif [[ "${DATASET_NAME}" == "ema" ]]; then
  PAIRS_DIR="${PROCESSED_ROOT}/ema_pairs_newpolicy_lhs"
elif [[ "${DATASET_NAME}" == "anaheim" ]]; then
  PAIRS_DIR="${PROCESSED_ROOT}/anaheim_pairs_newpolicy_lhs"
else
  echo "Unsupported DATASET_NAME=${DATASET_NAME}" >&2
  exit 1
fi
INPUT_PKL="${INPUT_PKL:-${PAIRS_DIR}/network_pairs_dataset.pkl}"

if [[ ! -f "${INPUT_PKL}" ]]; then
  shopt -s nullglob
  CHECKPOINT_CANDIDATES=(
    "${PAIRS_DIR}/pairs_completed.pkl.resume_10000.pkl"
    "${PAIRS_DIR}"/pairs_completed.pkl.resume_*.pkl
  )
  shopt -u nullglob
  for candidate in "${CHECKPOINT_CANDIDATES[@]}"; do
    if [[ -s "${candidate}" ]]; then
      INPUT_PKL="${candidate}"
      echo "network_pairs_dataset.pkl not found; using completed checkpoint: ${INPUT_PKL}"
      break
    fi
  done
fi

RUN_NAME="${RUN_NAME:-${DATASET_NAME}_sue_runtime_test}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

echo "==================== SUE Runtime Benchmark ===================="
echo "SLURM_JOB_ID        : ${SLURM_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "HOSTNAME            : $(hostname)"
echo "PROJECT_ROOT        : ${PROJECT_ROOT}"
echo "DATASET_NAME        : ${DATASET_NAME}"
echo "INPUT_PKL           : ${INPUT_PKL}"
echo "OUTPUT_ROOT         : ${OUTPUT_ROOT}"
echo "NUM_TEST_GRAPHS     : ${NUM_TEST_GRAPHS}"
echo "MAX_ITER            : ${MAX_ITER}"
echo "CONV_THRESHOLD      : ${CONVERGENCE_THRESHOLD}"
echo "CHECKPOINT_EVERY    : ${CHECKPOINT_EVERY}"
echo "RESUME              : ${RESUME}"
echo "START_TIME          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================================="

echo "Python binary       : $(which python)"
python -V
python -c "import numpy, scipy, networkx; print('core imports ok')"

EXTRA_RUNTIME_ARGS=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "True" ]]; then
  EXTRA_RUNTIME_ARGS+=(--resume)
fi

srun python scripts/benchmark_sue_runtime.py \
  --network_name "${DATASET_NAME}" \
  --input_pkl "${INPUT_PKL}" \
  --output_dir "${OUTPUT_ROOT}" \
  --run_name "${RUN_NAME}" \
  --num_test_graphs "${NUM_TEST_GRAPHS}" \
  --max_iter "${MAX_ITER}" \
  --convergence_threshold "${CONVERGENCE_THRESHOLD}" \
  --theta "${THETA}" \
  --value_iter "${VALUE_ITER}" \
  --value_tol "${VALUE_TOL}" \
  --flow_iter "${FLOW_ITER}" \
  --flow_tol "${FLOW_TOL}" \
  --retry_flow_iter "${RETRY_FLOW_ITER}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  "${EXTRA_RUNTIME_ARGS[@]}" \
  "$@"

echo "END_TIME            : $(date '+%Y-%m-%d %H:%M:%S')"

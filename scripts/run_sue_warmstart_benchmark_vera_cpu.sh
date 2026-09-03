#!/usr/bin/env bash

#SBATCH -J sue_ws_cpu
#SBATCH -o logs/sue_ws_cpu_%A_%a.out
#SBATCH -e logs/sue_ws_cpu_%A_%a.err
#SBATCH -t 0-06:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A NA
#SBATCH -p cpu
#SBATCH --array=0-2%1

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"
mkdir -p logs results/sue_warmstart

KEYS=(siouxfalls ema anaheim)
NETWORKS=(SiouxFalls EMA Anaheim)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
KEY="${KEYS[$TASK_ID]}"
NETWORK="${NETWORKS[$TASK_ID]}"

PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/sue_warmstart}"
PAIRS_DIR="${PROCESSED_ROOT}/${KEY}_pairs_newpolicy_lhs"
INPUT_PKL="${INPUT_PKL:-${PAIRS_DIR}/network_pairs_dataset.pkl}"
PREDICTION_DIR="${RESULTS_ROOT}/${KEY}/prediction"
OUTPUT_DIR="${RESULTS_ROOT}/${KEY}/benchmark"
NUM_TEST_GRAPHS="${NUM_TEST_GRAPHS:-0}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
MAX_ITER="${MAX_ITER:-120}"
CONVERGENCE_THRESHOLD="${CONVERGENCE_THRESHOLD:-1e-5}"
THETA="${THETA:-0.8}"
VALUE_ITER="${VALUE_ITER:-250}"
VALUE_TOL="${VALUE_TOL:-1e-8}"
FLOW_ITER="${FLOW_ITER:-500}"
FLOW_TOL="${FLOW_TOL:-1e-9}"
RETRY_FLOW_ITER="${RETRY_FLOW_ITER:-2000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
if [[ -z "${SUE_LOADING_PROTOCOL:-}" ]]; then
  if [[ "${KEY}" == "anaheim" ]]; then
    SUE_LOADING_PROTOCOL="reasonable_links"
  else
    SUE_LOADING_PROTOCOL="legacy_unrestricted"
  fi
fi

for required in \
  "${INPUT_PKL}" \
  "${PREDICTION_DIR}/test_predictions.npz" \
  "${PREDICTION_DIR}/prediction_metadata.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done
mkdir -p "${OUTPUT_DIR}"

echo "================ Paired SUE Warm-start Benchmark ================="
echo "JOB/TASK       : ${SLURM_JOB_ID:-N/A}/${TASK_ID}"
echo "NETWORK        : ${NETWORK}"
echo "INPUT_PKL      : ${INPUT_PKL}"
echo "OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "NUM_GRAPHS     : ${NUM_TEST_GRAPHS} (0=all exported test graphs)"
echo "NUM_WORKERS    : ${NUM_WORKERS}"
echo "MAX_ITER       : ${MAX_ITER}"
echo "THRESHOLD      : ${CONVERGENCE_THRESHOLD}"
echo "SUE_PROTOCOL   : ${SUE_LOADING_PROTOCOL}"
echo "=================================================================="

srun python "${PROJECT_ROOT}/scripts/benchmark_sue_warmstart.py" \
  --network "${NETWORK}" \
  --input-pkl "${INPUT_PKL}" \
  --predictions "${PREDICTION_DIR}/test_predictions.npz" \
  --prediction-metadata "${PREDICTION_DIR}/prediction_metadata.json" \
  --output-dir "${OUTPUT_DIR}" \
  --num-test-graphs "${NUM_TEST_GRAPHS}" \
  --num-workers "${NUM_WORKERS}" \
  --max-iter "${MAX_ITER}" \
  --convergence-threshold "${CONVERGENCE_THRESHOLD}" \
  --theta "${THETA}" \
  --value-iter "${VALUE_ITER}" \
  --value-tol "${VALUE_TOL}" \
  --flow-iter "${FLOW_ITER}" \
  --flow-tol "${FLOW_TOL}" \
  --retry-flow-iter "${RETRY_FLOW_ITER}" \
  --sue-loading-protocol "${SUE_LOADING_PROTOCOL}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --resume

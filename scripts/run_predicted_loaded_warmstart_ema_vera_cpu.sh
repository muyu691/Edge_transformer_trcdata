#!/usr/bin/env bash
#SBATCH --job-name=ema_ws_legacy
#SBATCH --output=logs/ema_ws_legacy_%j.out
#SBATCH --error=logs/ema_ws_legacy_%j.err
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=NA
#SBATCH --partition=cpu

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

PROJECT_ROOT="${PROJECT_ROOT:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
RAW_PKL="${RAW_PKL:-${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pairs_reconstructed_from_pyg/network_pairs_dataset.pkl}"
RAW_VALIDATION_MARKER="${RAW_VALIDATION_MARKER:-${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pairs_reconstructed_from_pyg/RAW_PAIRS_MATCH_PYG.ok}"
PREDICTION_DIR="${PREDICTION_DIR:-${PROJECT_ROOT}/results/sue_warmstart_ema_reconstructed/ema/prediction}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/sue_warmstart_ema_legacy_prior_preserving}"
OUTPUT_DIR="${RESULTS_ROOT}/ema/benchmark"

NUM_TEST_GRAPHS="${NUM_TEST_GRAPHS:-0}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
MAX_ITER="${MAX_ITER:-300}"
# legacy_unrestricted has an empirical residual floor near 1e-4 on EMA;
# 1e-3 is the pre-declared practical tolerance for this matched-solver study.
CONVERGENCE_THRESHOLD="${CONVERGENCE_THRESHOLD:-1e-3}"
VALUE_ITER="${VALUE_ITER:-250}"
VALUE_TOL="${VALUE_TOL:-1e-8}"
FLOW_ITER="${FLOW_ITER:-500}"
FLOW_TOL="${FLOW_TOL:-1e-9}"
RETRY_FLOW_ITER="${RETRY_FLOW_ITER:-2000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
RESIDUAL_STEP_EXPONENT="${RESIDUAL_STEP_EXPONENT:-0.5}"
RESIDUAL_STEP_MIN="${RESIDUAL_STEP_MIN:-0.02}"
RESIDUAL_STEP_MAX="${RESIDUAL_STEP_MAX:-0.80}"
STEP_WARMUP_ITERS="${STEP_WARMUP_ITERS:-5}"
STEP_WARMUP_MAX="${STEP_WARMUP_MAX:-0.10}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"
mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

for required in \
  "${RAW_PKL}" \
  "${RAW_VALIDATION_MARKER}" \
  "${PREDICTION_DIR}/test_predictions.npz" \
  "${PREDICTION_DIR}/prediction_metadata.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

echo "============= EMA Legacy-SUE Prior-preserving Warm Start ==========="
echo "SLURM_JOB_ID       : ${SLURM_JOB_ID:-interactive}"
echo "RAW_PKL            : ${RAW_PKL}"
echo "PREDICTION_DIR     : ${PREDICTION_DIR}"
echo "OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "NUM_TEST_GRAPHS    : ${NUM_TEST_GRAPHS} (0=all 2000 test graphs)"
echo "NUM_WORKERS        : ${NUM_WORKERS}"
echo "MAX_ITER           : ${MAX_ITER}"
echo "THRESHOLD          : ${CONVERGENCE_THRESHOLD}"
echo "LOADING_PROTOCOL   : legacy_unrestricted (matches training labels)"
echo "INITIAL_LOADING    : legacy_unrestricted"
echo "STEP_RULE          : msa_sr (same outer rule as label generation)"
echo "PRIOR WARMUP       : first ${STEP_WARMUP_ITERS} updates capped at ${STEP_WARMUP_MAX}"
echo "RESIDUAL_STEP      : exponent=${RESIDUAL_STEP_EXPONENT}, min=${RESIDUAL_STEP_MIN}, max=${RESIDUAL_STEP_MAX}"
echo "START_TIME         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "====================================================================="

srun python "${PROJECT_ROOT}/scripts/benchmark_sue_warmstart.py" \
  --network EMA \
  --input-pkl "${RAW_PKL}" \
  --predictions "${PREDICTION_DIR}/test_predictions.npz" \
  --prediction-metadata "${PREDICTION_DIR}/prediction_metadata.json" \
  --output-dir "${OUTPUT_DIR}" \
  --num-test-graphs "${NUM_TEST_GRAPHS}" \
  --num-workers "${NUM_WORKERS}" \
  --max-iter "${MAX_ITER}" \
  --convergence-threshold "${CONVERGENCE_THRESHOLD}" \
  --theta 0.8 \
  --value-iter "${VALUE_ITER}" \
  --value-tol "${VALUE_TOL}" \
  --flow-iter "${FLOW_ITER}" \
  --flow-tol "${FLOW_TOL}" \
  --retry-flow-iter "${RETRY_FLOW_ITER}" \
  --sue-loading-protocol legacy_unrestricted \
  --initial-loading-protocol legacy_unrestricted \
  --step-rule msa_sr \
  --residual-step-exponent "${RESIDUAL_STEP_EXPONENT}" \
  --residual-step-min "${RESIDUAL_STEP_MIN}" \
  --residual-step-max "${RESIDUAL_STEP_MAX}" \
  --step-warmup-iters "${STEP_WARMUP_ITERS}" \
  --step-warmup-max "${STEP_WARMUP_MAX}" \
  --stop-before-update \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --resume

echo "====================================================================="
echo "END_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
echo "SUMMARY : ${OUTPUT_DIR}/summary.json"
echo "DETAILS : ${OUTPUT_DIR}/per_graph.csv"
echo "====================================================================="

#!/usr/bin/env bash

#SBATCH -J braess_verify
#SBATCH -o logs/braess_verify_%A_%a.out
#SBATCH -e logs/braess_verify_%A_%a.err
#SBATCH -t 0-02:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A NA
#SBATCH -p cpu
#SBATCH --array=0-9%5

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
RAW_DIR="${RAW_DIR:-${PROCESSED_ROOT}/ema_pairs_reconstructed_from_pyg}"
INPUT_PKL="${INPUT_PKL:-${RAW_DIR}/network_pairs_dataset.pkl}"
VALIDATION_MARKER="${VALIDATION_MARKER:-${RAW_DIR}/RAW_PAIRS_MATCH_PYG.ok}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/braessian_edges/ema}"
MODEL_DIR="${MODEL_DIR:-${RESULTS_ROOT}/model}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/verification}"
RANKING_CSV="${RANKING_CSV:-${MODEL_DIR}/model_rankings.csv}"
RANKING_METADATA="${RANKING_METADATA:-${MODEL_DIR}/ranking_metadata.json}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:-${SHARD_ID:-0}}"
NUM_SHARDS="${NUM_SHARDS:-10}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
AUDIT_SCENARIOS="${AUDIT_SCENARIOS:-100}"
AUDIT_SEED="${AUDIT_SEED:-2027}"
FINAL_CHECK_K="${FINAL_CHECK_K:-10}"
MAX_ITER="${MAX_ITER:-120}"
CONVERGENCE_THRESHOLD="${CONVERGENCE_THRESHOLD:-1e-5}"
THETA="${THETA:-0.8}"
VALUE_ITER="${VALUE_ITER:-250}"
VALUE_TOL="${VALUE_TOL:-1e-8}"
FLOW_ITER="${FLOW_ITER:-500}"
FLOW_TOL="${FLOW_TOL:-1e-9}"
RETRY_FLOW_ITER="${RETRY_FLOW_ITER:-2000}"
SUE_LOADING_PROTOCOL="${SUE_LOADING_PROTOCOL:-legacy_unrestricted}"
STEP_RULE="${STEP_RULE:-msa_sr}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
for required in \
  "${INPUT_PKL}" \
  "${VALIDATION_MARKER}" \
  "${RAW_DIR}/raw_pairs_validation.json" \
  "${RANKING_CSV}" \
  "${RANKING_METADATA}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

echo "============= EMA Braessian Hybrid SUE Verification ==========="
echo "JOB/TASK       : ${SLURM_JOB_ID:-interactive}/${SHARD_ID}"
echo "INPUT_PKL      : ${INPUT_PKL}"
echo "RANKING_CSV    : ${RANKING_CSV}"
echo "SHARD          : ${SHARD_ID}/${NUM_SHARDS}"
echo "NUM_WORKERS    : ${NUM_WORKERS}"
echo "AUDIT_SCENARIOS: ${AUDIT_SCENARIOS} (exhaustive SUE)"
echo "AUDIT_SEED     : ${AUDIT_SEED}"
echo "FINAL_CHECK_K  : ${FINAL_CHECK_K} (non-audit scenarios)"
echo "SUE_PROTOCOL   : ${SUE_LOADING_PROTOCOL}"
echo "MAX_ITER/TOL   : ${MAX_ITER}/${CONVERGENCE_THRESHOLD}"
echo "OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "==============================================================="

srun python "${PROJECT_ROOT}/scripts/braessian_edges.py" oracle \
  --input-pkl "${INPUT_PKL}" \
  --validation-marker "${VALIDATION_MARKER}" \
  --ranking-csv "${RANKING_CSV}" \
  --ranking-metadata "${RANKING_METADATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --num-workers "${NUM_WORKERS}" \
  --audit-scenarios "${AUDIT_SCENARIOS}" \
  --audit-seed "${AUDIT_SEED}" \
  --final-check-k "${FINAL_CHECK_K}" \
  --max-iter "${MAX_ITER}" \
  --convergence-threshold "${CONVERGENCE_THRESHOLD}" \
  --theta "${THETA}" \
  --value-iter "${VALUE_ITER}" \
  --value-tol "${VALUE_TOL}" \
  --flow-iter "${FLOW_ITER}" \
  --flow-tol "${FLOW_TOL}" \
  --retry-flow-iter "${RETRY_FLOW_ITER}" \
  --sue-loading-protocol "${SUE_LOADING_PROTOCOL}" \
  --step-rule "${STEP_RULE}" \
  --resume

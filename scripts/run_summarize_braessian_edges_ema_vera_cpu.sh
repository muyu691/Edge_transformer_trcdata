#!/usr/bin/env bash

#SBATCH -J braess_sum
#SBATCH -o logs/braess_sum_%j.out
#SBATCH -e logs/braess_sum_%j.err
#SBATCH -t 0-00:30:00
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -A NA
#SBATCH -p cpu

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

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/braessian_edges/ema}"
MODEL_DIR="${MODEL_DIR:-${RESULTS_ROOT}/model}"
VERIFICATION_DIR="${VERIFICATION_DIR:-${RESULTS_ROOT}/verification}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/summary}"
TOP_K_VALUES="${TOP_K_VALUES:-1,3,5,10}"
PRIMARY_BRAESS_THRESHOLD="${PRIMARY_BRAESS_THRESHOLD:-0.001}"
BRAESS_THRESHOLDS="${BRAESS_THRESHOLDS:-0,0.0005,0.001,0.005}"
NUM_BOOTSTRAP="${NUM_BOOTSTRAP:-2000}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
for required in \
  "${MODEL_DIR}/model_rankings.csv" \
  "${MODEL_DIR}/ranking_metadata.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done
if ! compgen -G "${VERIFICATION_DIR}/oracle_shard_*.csv" >/dev/null; then
  echo "No hybrid SUE verification shard CSV files found under ${VERIFICATION_DIR}" >&2
  exit 2
fi

echo "============= EMA Braessian Hybrid Screening Summary ========="
echo "MODEL_DIR       : ${MODEL_DIR}"
echo "VERIFICATION_DIR: ${VERIFICATION_DIR}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "TOP_K_VALUES    : ${TOP_K_VALUES}"
echo "PRIMARY_THRESH  : ${PRIMARY_BRAESS_THRESHOLD}"
echo "BOOTSTRAP       : ${NUM_BOOTSTRAP} scenario-cluster resamples"
echo "==============================================================="

srun python "${PROJECT_ROOT}/scripts/braessian_edges.py" summarize \
  --ranking-csv "${MODEL_DIR}/model_rankings.csv" \
  --ranking-metadata "${MODEL_DIR}/ranking_metadata.json" \
  --oracle-dir "${VERIFICATION_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k "${TOP_K_VALUES}" \
  --primary-braess-threshold "${PRIMARY_BRAESS_THRESHOLD}" \
  --braess-thresholds "${BRAESS_THRESHOLDS}" \
  --num-bootstrap "${NUM_BOOTSTRAP}"

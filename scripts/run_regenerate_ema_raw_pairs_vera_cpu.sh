#!/usr/bin/env bash
#SBATCH --job-name=ema_raw_10k
#SBATCH --output=logs/ema_raw_10k_%j.out
#SBATCH --error=logs/ema_raw_10k_%j.err
#SBATCH --time=0-08:00:00
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
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pairs_baseline_perturb_recovered}"
PYG_DIR="${PYG_DIR:-${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pyg_baseline_perturb}"
NETWORK_FILE="${NETWORK_FILE:-${PROJECT_ROOT}/ema_data/EMA_net.tntp}"
OD_FILE="${OD_FILE:-${PROJECT_ROOT}/ema_data/EMA_trips.tntp}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-16}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
SUE_LOADING_PROTOCOL="${SUE_LOADING_PROTOCOL:-reasonable_links}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
CLEAN_CHECKPOINTS="${CLEAN_CHECKPOINTS:-1}"

RAW_PKL="${OUTPUT_DIR}/network_pairs_dataset.pkl"
REPORT_JSON="${OUTPUT_DIR}/raw_pairs_validation.json"
SUCCESS_MARKER="${OUTPUT_DIR}/RAW_PAIRS_MATCH_PYG.ok"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

echo "================ EMA 10000 Raw-Pair Recovery (Vera CPU) ================"
echo "SLURM_JOB_ID       : ${SLURM_JOB_ID:-interactive}"
echo "HOSTNAME           : $(hostname)"
echo "PROJECT_ROOT       : ${PROJECT_ROOT}"
echo "NETWORK_FILE       : ${NETWORK_FILE}"
echo "OD_FILE            : ${OD_FILE}"
echo "OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "EXISTING_PYG_DIR   : ${PYG_DIR}"
echo "NUM_SAMPLES        : ${NUM_SAMPLES}"
echo "SEED               : ${SEED}"
echo "NUM_WORKERS        : ${NUM_WORKERS}"
echo "CHECKPOINT_INTERVAL: ${CHECKPOINT_INTERVAL}"
echo "SUE_PROTOCOL       : ${SUE_LOADING_PROTOCOL}"
echo "FORCE_REGENERATE   : ${FORCE_REGENERATE}"
echo "START_TIME         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================="

for required in "${NETWORK_FILE}" "${PYG_DIR}/split_indices.npz"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 2
  fi
done

if [[ ! -f "${OD_FILE}" ]]; then
  echo "Missing required EMA OD file: ${OD_FILE}" >&2
  exit 2
fi

python --version
python - <<'PY'
import networkx
import numpy
import scipy
import sklearn
import torch
import torch_geometric
print("Core imports OK")
PY

if [[ -f "${RAW_PKL}" && "${FORCE_REGENERATE}" != "1" ]]; then
  echo "Existing recovered raw pickle found; generation is skipped."
  echo "Set FORCE_REGENERATE=1 to overwrite it with a fresh run."
else
  rm -f "${SUCCESS_MARKER}"
  srun python create_sioux_data/solve_network_pairs.py \
    --network_name EMA \
    --dataset_root "${PROJECT_ROOT}/ema_data" \
    --network_file "${NETWORK_FILE}" \
    --od_file "${OD_FILE}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_iter 120 \
    --convergence_threshold 1e-5 \
    --theta 0.8 \
    --value_iter 250 \
    --value_tol 1e-8 \
    --flow_iter 500 \
    --flow_tol 1e-9 \
    --retry_flow_iter 2000 \
    --sue_loading_protocol "${SUE_LOADING_PROTOCOL}" \
    --num_workers "${NUM_WORKERS}" \
    --checkpoint \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}"
fi

echo "Validating regenerated raw pairs against all existing EMA PyG samples..."
srun python scripts/validate_recovered_raw_pairs.py \
  --input-pkl "${RAW_PKL}" \
  --pyg-dir "${PYG_DIR}" \
  --expected-network EMA \
  --expected-pairs "${NUM_SAMPLES}" \
  --output-json "${REPORT_JSON}" \
  --success-marker "${SUCCESS_MARKER}"

if [[ "${CLEAN_CHECKPOINTS}" == "1" ]]; then
  echo "Validation passed; deleting redundant checkpoint pickles to release quota."
  rm -f "${OUTPUT_DIR}"/pairs_completed.pkl.ckpt_*.pkl
fi

ls -lh \
  "${RAW_PKL}" \
  "${OUTPUT_DIR}/base_scenarios.npz" \
  "${OUTPUT_DIR}/flows_old.npy" \
  "${REPORT_JSON}" \
  "${SUCCESS_MARKER}"

echo "========================================================================="
echo "EMA raw-pair recovery and full PyG consistency validation completed."
echo "RAW_PKL : ${RAW_PKL}"
echo "END_TIME: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================="

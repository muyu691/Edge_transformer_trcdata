#!/bin/bash
# Resume Anaheim data generation from the latest second-SUE checkpoint.
#
# Use this when anaheim_generation_fast_vera.sh stalls in Step 4 and no new
# checkpoint is produced for a long time. It reconstructs the deterministic
# scenarios, keeps the checkpointed pairs, solves only missing samples with
# per-sample timeouts, then builds the PyG dataset.

#SBATCH -J ana_resume
#SBATCH -o logs/ana_resume_%j.out
#SBATCH -e logs/ana_resume_%j.err
#SBATCH -t 4-00:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH -A NA
#SBATCH -p cpu

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}}"
if [[ ! -f "${PROJECT_ROOT}/create_sioux_data/resume_network_pairs_from_checkpoint.py" ]]; then
  PROJECT_ROOT="/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main"
fi
if [[ ! -f "${PROJECT_ROOT}/create_sioux_data/resume_network_pairs_from_checkpoint.py" ]]; then
  echo "Could not find resume script under PROJECT_ROOT=${PROJECT_ROOT}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/anaheim_data}"
NETWORK_FILE="${NETWORK_FILE:-${DATASET_ROOT}/Anaheim_net.tntp}"
OD_FILE="${OD_FILE:-${DATASET_ROOT}/Anaheim_trips.tntp}"

PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
PAIRS_DIR="${PAIRS_DIR:-${PROCESSED_ROOT}/anaheim_pairs_baseline_perturb}"
PYG_DIR="${PYG_DIR:-${PROCESSED_ROOT}/anaheim_pyg_baseline_perturb}"

NUM_SAMPLES="${NUM_SAMPLES:-10000}"
SEED="${SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.6}"
VAL_RATIO="${VAL_RATIO:-0.2}"

MAX_ITER="${MAX_ITER:-120}"
CONVERGENCE_THRESHOLD="${CONVERGENCE_THRESHOLD:-1e-5}"
THETA="${THETA:-0.8}"
VALUE_ITER="${VALUE_ITER:-250}"
VALUE_TOL="${VALUE_TOL:-1e-8}"
FLOW_ITER="${FLOW_ITER:-500}"
FLOW_TOL="${FLOW_TOL:-1e-9}"
RETRY_FLOW_ITER="${RETRY_FLOW_ITER:-2000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"

if [[ -z "${CKPT_PATH:-}" ]]; then
  CKPT_PATH="$(ls -1v "${PAIRS_DIR}"/pairs_completed.pkl.ckpt_*.pkl "${PAIRS_DIR}"/pairs_completed.pkl.resume_*.pkl 2>/dev/null | tail -n 1 || true)"
fi
if [[ -z "${CKPT_PATH}" || ! -f "${CKPT_PATH}" ]]; then
  echo "Could not find checkpoint under ${PAIRS_DIR}. Set CKPT_PATH explicitly." >&2
  exit 1
fi

if [[ ! -f "${NETWORK_FILE}" || ! -f "${OD_FILE}" ]]; then
  echo "Missing Anaheim TNTP input: NETWORK_FILE=${NETWORK_FILE}, OD_FILE=${OD_FILE}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${PAIRS_DIR}" "${PYG_DIR}"

echo "================ Anaheim Resume From Checkpoint (Vera CPU) ================"
echo "SLURM_JOB_ID   : ${SLURM_JOB_ID:-N/A}"
echo "HOSTNAME       : $(hostname)"
echo "PWD            : $(pwd)"
echo "CKPT_PATH      : ${CKPT_PATH}"
echo "NETWORK_FILE   : ${NETWORK_FILE}"
echo "OD_FILE        : ${OD_FILE}"
echo "NUM_SAMPLES    : ${NUM_SAMPLES}"
echo "NUM_WORKERS    : ${NUM_WORKERS}"
echo "TIMEOUT_SEC    : ${TIMEOUT_SEC}"
echo "PAIRS_DIR      : ${PAIRS_DIR}"
echo "PYG_DIR        : ${PYG_DIR}"
echo "START_TIME     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"

python -V
python -c "import numpy, scipy, networkx, torch, torch_geometric; print('core imports ok')"

srun --cpu-bind=cores python create_sioux_data/resume_network_pairs_from_checkpoint.py \
  --network_name Anaheim \
  --dataset_root "${DATASET_ROOT}" \
  --network_file "${NETWORK_FILE}" \
  --od_file "${OD_FILE}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --output_dir "${PAIRS_DIR}" \
  --checkpoint_path "${CKPT_PATH}" \
  --max_iter "${MAX_ITER}" \
  --convergence_threshold "${CONVERGENCE_THRESHOLD}" \
  --theta "${THETA}" \
  --value_iter "${VALUE_ITER}" \
  --value_tol "${VALUE_TOL}" \
  --flow_iter "${FLOW_ITER}" \
  --flow_tol "${FLOW_TOL}" \
  --retry_flow_iter "${RETRY_FLOW_ITER}" \
  --num_workers "${NUM_WORKERS}" \
  --timeout_sec "${TIMEOUT_SEC}" \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"

srun --cpu-bind=cores python create_sioux_data/build_network_pairs_dataset.py \
  --input_pkl "${PAIRS_DIR}/network_pairs_dataset.pkl" \
  --output_dir "${PYG_DIR}" \
  --train_ratio "${TRAIN_RATIO}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}"

PYG_DIR="${PYG_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

meta_path = Path(os.environ["PYG_DIR"]) / "dataset_meta.json"
if meta_path.exists():
    print(meta_path.read_text(encoding="utf-8"))
PY

echo "END_TIME       : $(date '+%Y-%m-%d %H:%M:%S')"

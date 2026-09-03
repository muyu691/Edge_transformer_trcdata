#!/bin/bash
#SBATCH -J ema_data
#SBATCH -o logs/ema_data_%j.out
#SBATCH -e logs/ema_data_%j.err
#SBATCH -t 0-02:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A NA
#SBATCH -p cpu

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/ema_data}"
NETWORK_FILE="${NETWORK_FILE:-${DATASET_ROOT}/EMA_net.tntp}"
OD_FILE="${OD_FILE:-${DATASET_ROOT}/EMA_trips.tntp}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
PAIRS_DIR="${PAIRS_DIR:-${PROCESSED_ROOT}/ema_pairs_baseline_perturb}"
PYG_DIR="${PYG_DIR:-${PROCESSED_ROOT}/ema_pyg_baseline_perturb}"
NUM_SAMPLES="${NUM_SAMPLES:-4000}"
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
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

if [[ ! -f "${NETWORK_FILE}" || ! -f "${OD_FILE}" ]]; then
  echo "Missing EMA TNTP input: NETWORK_FILE=${NETWORK_FILE}, OD_FILE=${OD_FILE}" >&2
  exit 1
fi

mkdir -p \
  "${PROJECT_ROOT}/logs" \
  "${PAIRS_DIR}" \
  "${PYG_DIR}"

srun python create_sioux_data/solve_network_pairs.py \
  --network_name EMA \
  --dataset_root "${DATASET_ROOT}" \
  --network_file "${NETWORK_FILE}" \
  --od_file "${OD_FILE}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --output_dir "${PAIRS_DIR}" \
  --max_iter "${MAX_ITER}" \
  --convergence_threshold "${CONVERGENCE_THRESHOLD}" \
  --theta "${THETA}" \
  --value_iter "${VALUE_ITER}" \
  --value_tol "${VALUE_TOL}" \
  --flow_iter "${FLOW_ITER}" \
  --flow_tol "${FLOW_TOL}" \
  --retry_flow_iter "${RETRY_FLOW_ITER}" \
  --num_workers "${NUM_WORKERS}"

srun python create_sioux_data/build_network_pairs_dataset.py \
  --input_pkl "${PAIRS_DIR}/network_pairs_dataset.pkl" \
  --output_dir "${PYG_DIR}" \
  --train_ratio "${TRAIN_RATIO}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}"

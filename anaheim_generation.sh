#!/bin/bash
# CPU Slurm launcher for generating Anaheim network-pair data with the new policy.
#
# Outputs:
#   create_sioux_data/processed_data/anaheim_pairs_newpolicy_lhs/network_pairs_dataset.pkl
#   create_sioux_data/processed_data/anaheim_pyg_newpolicy_lhs/{train,val,test}_dataset.pt
#
# Typical usage on Vera:
#   sbatch anaheim_generation.sh
#
# Useful overrides:
#   NUM_SAMPLES=200 SEED=42 sbatch anaheim_generation.sh
#   DEMAND_SOURCE=trips OD_FILE=/path/to/Anaheim_trips.tntp sbatch anaheim_generation.sh

#SBATCH -J anaheim_data
#SBATCH -o logs/anaheim_data_%j.out
#SBATCH -e logs/anaheim_data_%j.err
#SBATCH -t 7-00:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -A NA
#SBATCH -p cpu

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}}"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/anaheim_data}"
NETWORK_FILE="${NETWORK_FILE:-${DATASET_ROOT}/Anaheim_net.tntp}"
OD_FILE="${OD_FILE:-${DATASET_ROOT}/Anaheim_trips.tntp}"
DEMAND_SOURCE="${DEMAND_SOURCE:-lhs}"

PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
PAIRS_DIR="${PAIRS_DIR:-${PROCESSED_ROOT}/anaheim_pairs_newpolicy_lhs}"
PYG_DIR="${PYG_DIR:-${PROCESSED_ROOT}/anaheim_pyg_newpolicy_lhs}"

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
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

if [[ ! -f "${NETWORK_FILE}" ]]; then
  echo "Missing Anaheim network file: ${NETWORK_FILE}" >&2
  exit 1
fi

OD_ARGS=()
if [[ -f "${OD_FILE}" ]]; then
  OD_ARGS=(--od_file "${OD_FILE}")
elif [[ "${DEMAND_SOURCE}" == "trips" ]]; then
  echo "DEMAND_SOURCE=trips requires OD_FILE to exist: ${OD_FILE}" >&2
  exit 1
fi

SOLVE_EXTRA_ARGS=()
if [[ "${SKIP_FIRST_SOLVE:-0}" == "1" ]]; then
  SOLVE_EXTRA_ARGS+=(--skip_first_solve)
fi
if [[ "${CHECKPOINT:-1}" == "1" ]]; then
  SOLVE_EXTRA_ARGS+=(--checkpoint --checkpoint_interval "${CHECKPOINT_INTERVAL}")
fi
if [[ -n "${CENTROID_NODES:-}" ]]; then
  SOLVE_EXTRA_ARGS+=(--centroid_nodes "${CENTROID_NODES}")
fi

mkdir -p "${PROJECT_ROOT}/logs" "${PAIRS_DIR}" "${PYG_DIR}"

echo "================ Anaheim Data Generation (Vera CPU) ================"
echo "SLURM_JOB_ID   : ${SLURM_JOB_ID:-N/A}"
echo "HOSTNAME       : $(hostname)"
echo "PWD            : $(pwd)"
echo "DATASET_ROOT   : ${DATASET_ROOT}"
echo "NETWORK_FILE   : ${NETWORK_FILE}"
echo "OD_FILE        : ${OD_FILE:-<none>}"
echo "DEMAND_SOURCE  : ${DEMAND_SOURCE}"
echo "NUM_SAMPLES    : ${NUM_SAMPLES}"
echo "NUM_WORKERS    : ${NUM_WORKERS}"
echo "PAIRS_DIR      : ${PAIRS_DIR}"
echo "PYG_DIR        : ${PYG_DIR}"
echo "START_TIME     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "====================================================================="

python -V
python -c "import numpy, scipy, networkx, torch, torch_geometric; print('core imports ok')"

srun python create_sioux_data/solve_network_pairs.py \
  --network_name Anaheim \
  --dataset_root "${DATASET_ROOT}" \
  --network_file "${NETWORK_FILE}" \
  "${OD_ARGS[@]}" \
  --demand_source "${DEMAND_SOURCE}" \
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
  --num_workers "${NUM_WORKERS}" \
  "${SOLVE_EXTRA_ARGS[@]}"

srun python create_sioux_data/build_network_pairs_dataset.py \
  --input_pkl "${PAIRS_DIR}/network_pairs_dataset.pkl" \
  --output_dir "${PYG_DIR}" \
  --train_ratio "${TRAIN_RATIO}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}"

echo "END_TIME       : $(date '+%Y-%m-%d %H:%M:%S')"

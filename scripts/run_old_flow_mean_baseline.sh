#!/bin/bash
# Slurm array for the old-flow/mean-flow baseline.
#
# Default usage from the project root on Vera:
#   sbatch scripts/run_old_flow_mean_baseline.sh
#
# Single dataset:
#   sbatch --array=0 scripts/run_old_flow_mean_baseline.sh  # Sioux Falls
#   sbatch --array=1 scripts/run_old_flow_mean_baseline.sh  # EMA
#   sbatch --array=2 scripts/run_old_flow_mean_baseline.sh  # Anaheim

#SBATCH -J old_mean_base
#SBATCH -o logs/old_mean_base_%A_%a.out
#SBATCH -e logs/old_mean_base_%A_%a.err
#SBATCH -t 0-02:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -A NA
#SBATCH -p cpu
#SBATCH --array=0-2

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJECT_ROOT="${PROJECT_ROOT:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/baselines/old_flow_mean}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE_VALUE="${DEVICE_VALUE:-cpu}"
SEED="${SEED:-0}"

DATASETS=(siouxfalls ema anaheim)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET_NAME="${DATASETS[$TASK_ID]}"

case "${DATASET_NAME}" in
  siouxfalls)
    NETWORK_NAME="SiouxFalls"
    DATASET_DIR="${PROCESSED_ROOT}/siouxfalls_pyg_newpolicy_lhs"
    RUN_DIR="${OUTPUT_ROOT}/old_flow_mean_siouxfalls_10000"
    ;;
  ema)
    NETWORK_NAME="EMA"
    DATASET_DIR="${PROCESSED_ROOT}/ema_pyg_newpolicy_lhs"
    RUN_DIR="${OUTPUT_ROOT}/old_flow_mean_ema_10000"
    ;;
  anaheim)
    NETWORK_NAME="Anaheim"
    DATASET_DIR="${PROCESSED_ROOT}/anaheim_pyg_newpolicy_lhs"
    RUN_DIR="${OUTPUT_ROOT}/old_flow_mean_anaheim_10000"
    ;;
  *)
    echo "Unsupported DATASET_NAME=${DATASET_NAME}" >&2
    exit 1
    ;;
esac

mkdir -p "${PROJECT_ROOT}/logs" "${RUN_DIR}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

echo "================ Old-flow/mean-flow baseline ================"
echo "SLURM_JOB_ID        : ${SLURM_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "HOSTNAME            : $(hostname)"
echo "PROJECT_ROOT        : ${PROJECT_ROOT}"
echo "DATASET_NAME        : ${DATASET_NAME}"
echo "NETWORK_NAME        : ${NETWORK_NAME}"
echo "DATASET_DIR         : ${DATASET_DIR}"
echo "RUN_DIR             : ${RUN_DIR}"
echo "DEVICE_VALUE        : ${DEVICE_VALUE}"
echo "BATCH_SIZE          : ${BATCH_SIZE}"
echo "START_TIME          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================="

python -V
python -c "import torch, torch_geometric; print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available()); print('torch_geometric ok')"

srun python baseline/old_flow_mean_baseline.py \
  --dataset_name "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --processed_root "${PROCESSED_ROOT}" \
  --output_dir "${RUN_DIR}" \
  --seed "${SEED}" \
  --device "${DEVICE_VALUE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

echo "END_TIME            : $(date '+%Y-%m-%d %H:%M:%S')"

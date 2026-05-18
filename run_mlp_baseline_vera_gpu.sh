#!/bin/bash
# GPU Slurm launcher for the standalone MLP baseline on Vera.
#
# Default usage:
#   sbatch run_mlp_baseline_vera_gpu.sh
#
# Switch dataset:
#   DATASET_NAME=siouxfalls sbatch run_mlp_baseline_vera_gpu.sh
#
# Use an explicit processed dataset directory:
#   DATASET_DIR=/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main/create_sioux_data/processed_data/ema_pyg_newpolicy_lhs \
#   sbatch run_mlp_baseline_vera_gpu.sh

#SBATCH -J base_mlp
#SBATCH -o logs/base_mlp_%j.out
#SBATCH -e logs/base_mlp_%j.err
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A NA
#SBATCH -p gpu
#SBATCH --gres=gpu:A40:1

set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}}"
DEVICE_VALUE="${DEVICE_VALUE:-cuda}"
DATASET_NAME="${DATASET_NAME:-ema}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR_VALUE="${LR_VALUE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/baseline/results}"
RUN_TAG="${RUN_TAG:-mlp_baseline_${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}}"

cd "${PROJECT_ROOT}"

echo "================ MLP Baseline (Vera GPU) ================="
echo "SLURM_JOB_ID   : ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NAME : ${SLURM_JOB_NAME:-N/A}"
echo "HOSTNAME       : $(hostname)"
echo "PWD            : $(pwd)"
echo "DATASET_NAME   : ${DATASET_NAME}"
echo "DATASET_DIR    : ${DATASET_DIR:-<auto>}"
echo "OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "HIDDEN_DIM     : ${HIDDEN_DIM}"
echo "EPOCHS         : ${EPOCHS}"
echo "BATCH_SIZE     : ${BATCH_SIZE}"
echo "START_TIME     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}" "${OUTPUT_DIR}"

echo "Python binary   : $(which python)"
python -V
echo "CUDA visibility : ${CUDA_VISIBLE_DEVICES:-N/A}"
nvidia-smi || true
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"
python -c "import torch_scatter, torch_geometric, baseline; print('core imports ok')"

DATASET_ARGS=(--dataset_name "${DATASET_NAME}" --processed_root "${PROCESSED_ROOT}")
if [[ -n "${DATASET_DIR:-}" ]]; then
  DATASET_ARGS=(--dataset_dir "${DATASET_DIR}")
fi

srun python -m baseline.run \
  --model mlp_baseline \
  "${DATASET_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR_VALUE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_layers_old 3 \
  --num_layers_new 3 \
  --dropout 0.1 \
  --loss_name l1 \
  --lambda_old 1.0 \
  --lambda_new_start 1.0 \
  --lambda_new_final 1.0 \
  --device "${DEVICE_VALUE}" \
  "$@"

echo "END_TIME       : $(date '+%Y-%m-%d %H:%M:%S')"

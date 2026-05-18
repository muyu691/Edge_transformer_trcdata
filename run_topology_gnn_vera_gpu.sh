#!/bin/bash
# GPU Slurm launcher for the full topology_gnn model on Vera.

#SBATCH -J ours_topo
#SBATCH -o logs/ours_topo_%j.out
#SBATCH -e logs/ours_topo_%j.err
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
source "${PROJECT_ROOT}/run_topology_common.sh"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/GatedGCN/network-pairs-topology.yaml}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
DATASET_NAME="${DATASET_NAME:-ema}"
DEVICE_VALUE="${DEVICE_VALUE:-cuda}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-200}"
LR_VALUE="${LR_VALUE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/ours}"
RUN_TAG="${RUN_TAG:-topology_gnn_${DATASET_NAME}}"

RESOLVED_DATASET_DIR="$(resolve_topology_dataset_dir)"
NETWORK_NAME="$(resolve_topology_network_name)"

cd "${PROJECT_ROOT}"

echo "============== Topology GNN (Vera GPU) =============="
echo "SLURM_JOB_ID   : ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NAME : ${SLURM_JOB_NAME:-N/A}"
echo "HOSTNAME       : $(hostname)"
echo "PWD            : $(pwd)"
echo "DATASET_NAME   : ${DATASET_NAME}"
echo "NETWORK_NAME   : ${NETWORK_NAME}"
echo "DATASET_DIR    : ${RESOLVED_DATASET_DIR}"
echo "RUN_TAG        : ${RUN_TAG}"
echo "START_TIME     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "====================================================="

source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}"

echo "Python binary   : $(which python)"
python -V
echo "CUDA visibility : ${CUDA_VISIBLE_DEVICES:-N/A}"
nvidia-smi || true
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"
python -c "import torch_scatter, torch_geometric, graphgps; print('core imports ok')"

srun python main.py \
  --cfg "${CONFIG_PATH}" \
  --repeat 1 \
  out_dir "${OUTPUT_ROOT}" \
  name_tag "${RUN_TAG}" \
  metric_best "rmse_norm" \
  metric_agg "argmin" \
  train.ckpt_best True \
  train.ckpt_clean False \
  dataset.network_name "${NETWORK_NAME}" \
  dataset.dir "${RESOLVED_DATASET_DIR}" \
  dataset.processed_root "${PROCESSED_ROOT}" \
  accelerator "${DEVICE_VALUE}" \
  topology_gnn.hidden_dim "${HIDDEN_DIM}" \
  train.batch_size "${BATCH_SIZE}" \
  optim.max_epoch "${EPOCHS}" \
  optim.base_lr "${LR_VALUE}" \
  optim.weight_decay "${WEIGHT_DECAY}" \
  "$@"

echo "END_TIME       : $(date '+%Y-%m-%d %H:%M:%S')"

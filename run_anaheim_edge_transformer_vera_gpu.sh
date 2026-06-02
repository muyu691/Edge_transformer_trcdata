#!/bin/bash
# GPU Slurm launcher for training the Edge Transformer model on Anaheim.
#
# Expected dataset:
#   create_sioux_data/processed_data/anaheim_pyg_newpolicy_lhs
#
# Typical usage on Vera:
#   sbatch run_anaheim_edge_transformer_vera_gpu.sh

#SBATCH -J edge_ana
#SBATCH -o logs/edge_ana_%j.out
#SBATCH -e logs/edge_ana_%j.err
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -A NA
#SBATCH -p gpu
#SBATCH --gres=gpu:A40:1

set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}}"
if [[ ! -f "${PROJECT_ROOT}/run_topology_common.sh" ]]; then
  PROJECT_ROOT="/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main"
fi
if [[ ! -f "${PROJECT_ROOT}/run_topology_common.sh" ]]; then
  echo "Could not find run_topology_common.sh under PROJECT_ROOT=${PROJECT_ROOT}" >&2
  exit 1
fi

source "${PROJECT_ROOT}/run_topology_common.sh"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/GatedGCN/network-pairs-topology.yaml}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
DATASET_NAME="${DATASET_NAME:-anaheim}"
NETWORK_NAME="${NETWORK_NAME:-Anaheim}"
DATASET_DIR="${DATASET_DIR:-${PROCESSED_ROOT}/anaheim_pyg_newpolicy_lhs}"

DEVICE_VALUE="${DEVICE_VALUE:-cuda}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-200}"
LR_VALUE="${LR_VALUE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/ours}"
RUN_TAG="${RUN_TAG:-edge_transformer_anaheim_newpolicy_lhs}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${DATASET_DIR}/dataset_meta.json" ]]; then
  echo "Missing processed Anaheim dataset: ${DATASET_DIR}/dataset_meta.json" >&2
  echo "Run anaheim_generation.sh first, or set DATASET_DIR to the processed PyG directory." >&2
  exit 1
fi

echo "================ Edge Transformer Anaheim (Vera GPU) ================"
echo "SLURM_JOB_ID   : ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NAME : ${SLURM_JOB_NAME:-N/A}"
echo "HOSTNAME       : $(hostname)"
echo "PWD            : $(pwd)"
echo "NETWORK_NAME   : ${NETWORK_NAME}"
echo "DATASET_DIR    : ${DATASET_DIR}"
echo "RUN_TAG        : ${RUN_TAG}"
echo "START_TIME     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "====================================================================="

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
  dataset.dir "${DATASET_DIR}" \
  dataset.processed_root "${PROCESSED_ROOT}" \
  accelerator "${DEVICE_VALUE}" \
  topology_gnn.hidden_dim "${HIDDEN_DIM}" \
  topology_gnn.num_diffusion_steps 4 \
  topology_gnn.attention_every_k_steps 2 \
  topology_gnn.dropout 0.1 \
  topology_gnn.residual True \
  topology_gnn.num_heads 4 \
  topology_gnn.local_backbone "edge_transformer" \
  topology_gnn.num_edge_transformer_layers 1 \
  topology_gnn.ffn_type "swiglu" \
  topology_gnn.ffn_mult "8/3" \
  topology_gnn.norm_type "rmsnorm" \
  topology_gnn.norm_position "pre" \
  topology_gnn.edge_to_node_agg "mean" \
  topology_gnn.edge_endpoint_mode "fusion" \
  topology_gnn.enable_global_attn False \
  topology_gnn.init_scheme "default" \
  topology_gnn.inject_rho_to_edges True \
  topology_gnn.inject_flow_to_edges True \
  topology_gnn.inject_rho_to_nodes True \
  topology_gnn.initial_flow_mode "old_flow_warm_start" \
  topology_gnn.initial_pressure_mode "from_initial_flow" \
  topology_gnn.pressure_update_mode "lwr" \
  topology_gnn.share_diffusion_cell True \
  topology_gnn.alignment_mode "full" \
  train.batch_size "${BATCH_SIZE}" \
  optim.max_epoch "${EPOCHS}" \
  optim.base_lr "${LR_VALUE}" \
  optim.weight_decay "${WEIGHT_DECAY}" \
  model.lambda_con_final_epoch "${EPOCHS}" \
  "$@"

echo "END_TIME       : $(date '+%Y-%m-%d %H:%M:%S')"

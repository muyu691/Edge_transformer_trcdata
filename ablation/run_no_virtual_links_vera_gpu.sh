#!/bin/bash
# GPU Slurm launcher for the no-virtual-links ablation on Vera.

#SBATCH -J abl_novl
#SBATCH -o logs/abl_novl_%j.out
#SBATCH -e logs/abl_novl_%j.err
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:A40:1

set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}}"
if [[ ! -f "${PROJECT_ROOT}/ablation/ablation_common.sh" ]]; then
  PROJECT_ROOT="/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main"
fi
if [[ ! -f "${PROJECT_ROOT}/ablation/ablation_common.sh" ]]; then
  echo "Could not find ablation_common.sh under PROJECT_ROOT=${PROJECT_ROOT}" >&2
  exit 1
fi
source "${PROJECT_ROOT}/ablation/ablation_common.sh"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/GatedGCN/network-pairs-topology.yaml}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
DATASET_NAME="${DATASET_NAME:-ema}"
DEVICE_VALUE="${DEVICE_VALUE:-cuda}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-200}"
LR_VALUE="${LR_VALUE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/ablation}"
RUN_TAG="${RUN_TAG:-no_virtual_links_${DATASET_NAME}}"

set_ablation_main_hparams
RESOLVED_DATASET_DIR="$(resolve_ablation_dataset_dir)"
NETWORK_NAME="$(resolve_ablation_network_name)"

cd "${PROJECT_ROOT}"

echo "=============== No Virtual Links (Vera GPU) ================"
echo "DATASET_NAME   : ${DATASET_NAME}"
echo "NETWORK_NAME   : ${NETWORK_NAME}"
echo "DATASET_DIR    : ${RESOLVED_DATASET_DIR}"
echo "LAMBDA_NEW     : ${LAMBDA_NEW_FINAL}"
echo "LAMBDA_CON     : ${LAMBDA_CON}"
echo "RUN_TAG        : ${RUN_TAG}"
echo "START_TIME     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}"

python -V
nvidia-smi || true
python -c "import torch_scatter, torch_geometric, graphgps; print('core imports ok')"

srun python main.py \
  --cfg "${CONFIG_PATH}" \
  --repeat 1 \
  out_dir "${OUTPUT_ROOT}" \
  name_tag "${RUN_TAG}" \
  metric_best "rmse_norm" \
  metric_agg "argmin" \
  train.ckpt_best True \
  train.ckpt_clean True \
  dataset.network_name "${NETWORK_NAME}" \
  dataset.dir "${RESOLVED_DATASET_DIR}" \
  dataset.processed_root "${PROCESSED_ROOT}" \
  accelerator "${DEVICE_VALUE}" \
  topology_gnn.hidden_dim "${HIDDEN_DIM}" \
  topology_gnn.num_diffusion_steps 4 \
  topology_gnn.attention_every_k_steps 1 \
  topology_gnn.dropout 0.1 \
  topology_gnn.residual True \
  topology_gnn.num_heads 4 \
  topology_gnn.local_backbone "gatedgcn" \
  topology_gnn.ffn_type "swiglu" \
  topology_gnn.ffn_mult "8/3" \
  topology_gnn.norm_type "rmsnorm" \
  topology_gnn.norm_position "pre" \
  topology_gnn.edge_to_node_agg "mean" \
  topology_gnn.edge_endpoint_mode "fusion" \
  train.batch_size "${BATCH_SIZE}" \
  optim.max_epoch "${EPOCHS}" \
  optim.base_lr "${LR_VALUE}" \
  optim.weight_decay "${WEIGHT_DECAY}" \
  topology_gnn.enable_global_attn False \
  topology_gnn.init_scheme "default" \
  topology_gnn.init_residual_scale 0.1 \
  topology_gnn.init_delta_scale 0.1 \
  topology_gnn.inject_rho_to_edges True \
  topology_gnn.inject_flow_to_edges True \
  topology_gnn.inject_rho_to_nodes True \
  topology_gnn.initial_flow_mode "old_flow_warm_start" \
  topology_gnn.initial_pressure_mode "from_initial_flow" \
  topology_gnn.pressure_update_mode "lwr" \
  topology_gnn.share_diffusion_cell True \
  topology_gnn.alignment_mode "full" \
  model.lambda_new_final "${LAMBDA_NEW_FINAL}" \
  model.lambda_con "${LAMBDA_CON}" \
  model.lambda_con_schedule "${LAMBDA_CON_SCHEDULE}" \
  model.lambda_con_mid "${LAMBDA_CON_MID}" \
  model.lambda_con_zero_epochs 50 \
  model.lambda_con_mid_epoch 120 \
  model.lambda_con_final_epoch "${EPOCHS}" \
  "$@"

echo "END_TIME       : $(date '+%Y-%m-%d %H:%M:%S')"

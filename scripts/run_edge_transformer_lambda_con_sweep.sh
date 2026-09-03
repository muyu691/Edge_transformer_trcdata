#!/bin/bash
# Slurm array for the Edge Transformer lambda_con sweep on Vera.
#
# Tasks:
#   0-4:   SiouxFalls lambda_con in {0.0, 0.01, 0.05, 0.1, 0.2}
#   5-9:   EMA        lambda_con in {0.0, 0.01, 0.05, 0.1, 0.2}
#   10-14: Anaheim    lambda_con in {0.0, 0.01, 0.05, 0.1, 0.2}

#SBATCH -J et_lcon
#SBATCH -o logs/et_lcon_%A_%a.out
#SBATCH -e logs/et_lcon_%A_%a.err
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A NA
#SBATCH -p gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --array=0-14

set -euo pipefail

export PYTHONUNBUFFERED=1

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/edge_transformer_lambda_con_sweep}"

DEVICE_VALUE="${DEVICE_VALUE:-cuda}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-200}"
LR_VALUE="${LR_VALUE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"

DATASETS=(
  siouxfalls siouxfalls siouxfalls siouxfalls siouxfalls
  ema ema ema ema ema
  anaheim anaheim anaheim anaheim anaheim
)
LAMBDA_CON_VALUES=(
  0.0 0.01 0.05 0.1 0.2
  0.0 0.01 0.05 0.1 0.2
  0.0 0.01 0.05 0.1 0.2
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#DATASETS[@]}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-$((${#DATASETS[@]} - 1))" >&2
  exit 1
fi

DATASET_NAME="${DATASETS[$TASK_ID]}"
LAMBDA_CON="${LAMBDA_CON_VALUES[$TASK_ID]}"
SAFE_LAMBDA="${LAMBDA_CON//./p}"

set_topology_main_hparams
LAMBDA_CON_MID="$(
awk -v value="${LAMBDA_CON}" 'BEGIN { mid = (value < 0.01 ? value : 0.01); printf "%.6f\n", mid }'
)"

RESOLVED_DATASET_DIR="$(resolve_topology_dataset_dir)"
NETWORK_NAME="$(resolve_topology_network_name)"
RUN_TAG="${RUN_TAG:-edge_transformer_${DATASET_NAME}_lambda_con_${SAFE_LAMBDA}_10000}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${RESOLVED_DATASET_DIR}/dataset_meta.json" ]]; then
  echo "Missing processed dataset: ${RESOLVED_DATASET_DIR}/dataset_meta.json" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_ROOT}"

echo "============== Edge Transformer lambda_con sweep =============="
echo "SLURM_JOB_ID        : ${SLURM_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "DATASET_NAME        : ${DATASET_NAME}"
echo "NETWORK_NAME        : ${NETWORK_NAME}"
echo "DATASET_DIR         : ${RESOLVED_DATASET_DIR}"
echo "LAMBDA_NEW_FINAL    : ${LAMBDA_NEW_FINAL}"
echo "LAMBDA_CON          : ${LAMBDA_CON}"
echo "LAMBDA_CON_MID      : ${LAMBDA_CON_MID}"
echo "LAMBDA_CON_SCHEDULE : ${LAMBDA_CON_SCHEDULE}"
echo "RUN_TAG             : ${RUN_TAG}"
echo "START_TIME          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================================="

source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

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
  topology_gnn.num_diffusion_steps 4 \
  topology_gnn.attention_every_k_steps 1 \
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
  train.batch_size "${BATCH_SIZE}" \
  optim.max_epoch "${EPOCHS}" \
  optim.base_lr "${LR_VALUE}" \
  optim.weight_decay "${WEIGHT_DECAY}" \
  "$@"

echo "END_TIME            : $(date '+%Y-%m-%d %H:%M:%S')"

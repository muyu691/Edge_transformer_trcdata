#!/bin/bash
# Edge Transformer old-flow robustness evaluation on Vera.
#
# Clean 0% rows are reused from the completed full-model summaries. This array
# runs only the four new conditions per network:
#   missing       in {15%, 30%}
#   Gaussian noise in {5%, 15%}
#
# Array mapping:
#   0-3   SiouxFalls
#   4-7   EMA
#   8-11  Anaheim

#SBATCH -J et_flow_rob
#SBATCH -o logs/et_flow_rob_%A_%a.out
#SBATCH -e logs/et_flow_rob_%A_%a.err
#SBATCH -t 0-02:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A NA
#SBATCH -p gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --array=0-11%3

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
if [[ ! -f "${PROJECT_ROOT}/scripts/evaluate_old_flow_robustness.py" ]]; then
  PROJECT_ROOT="/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main"
fi
if [[ ! -f "${PROJECT_ROOT}/scripts/evaluate_old_flow_robustness.py" ]]; then
  echo "Could not find scripts/evaluate_old_flow_robustness.py under PROJECT_ROOT=${PROJECT_ROOT}" >&2
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/GatedGCN/network-pairs-topology.yaml}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
OURS_ROOT="${OURS_ROOT:-${PROJECT_ROOT}/results/ours}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/old_flow_robustness}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
DEVICE_VALUE="${DEVICE_VALUE:-cuda}"

DATASETS=(
  siouxfalls siouxfalls siouxfalls siouxfalls
  ema ema ema ema
  anaheim anaheim anaheim anaheim
)
MODES=(
  missing missing gaussian_noise gaussian_noise
  missing missing gaussian_noise gaussian_noise
  missing missing gaussian_noise gaussian_noise
)
LEVELS=(
  0.15 0.30 0.05 0.15
  0.15 0.30 0.05 0.15
  0.15 0.30 0.05 0.15
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#DATASETS[@]}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-$((${#DATASETS[@]} - 1))" >&2
  exit 1
fi

DATASET_NAME="${DATASETS[$TASK_ID]}"
PERTURBATION_MODE="${MODES[$TASK_ID]}"
PERTURBATION_LEVEL="${LEVELS[$TASK_ID]}"
LEVEL_PERCENT="$(awk -v value="${PERTURBATION_LEVEL}" 'BEGIN { printf "%d", value * 100 + 0.5 }')"

case "${DATASET_NAME}" in
  siouxfalls)
    NETWORK_NAME="SiouxFalls"
    DATASET_DIR="${PROCESSED_ROOT}/siouxfalls_pyg_newpolicy_lhs"
    MODEL_RUN_DIR="${OURS_ROOT}/network-pairs-topology-ours_edge_transformer_siouxfalls_10000/0"
    LAMBDA_NEW_FINAL="1.0"
    LAMBDA_CON="0.1"
    ;;
  ema)
    NETWORK_NAME="EMA"
    DATASET_DIR="${PROCESSED_ROOT}/ema_pyg_newpolicy_lhs"
    MODEL_RUN_DIR="${OURS_ROOT}/network-pairs-topology-ours_edge_transformer_ema_10000/0"
    LAMBDA_NEW_FINAL="1.5"
    LAMBDA_CON="0.05"
    ;;
  anaheim)
    NETWORK_NAME="Anaheim"
    DATASET_DIR="${PROCESSED_ROOT}/anaheim_pyg_newpolicy_lhs"
    MODEL_RUN_DIR="${OURS_ROOT}/network-pairs-topology-ours_edge_transformer_anaheim_10000/0"
    LAMBDA_NEW_FINAL="1.0"
    LAMBDA_CON="0.05"
    ;;
  *)
    echo "Unsupported DATASET_NAME=${DATASET_NAME}" >&2
    exit 1
    ;;
esac

SUMMARY_PATH="${MODEL_RUN_DIR}/summary.json"
RUN_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${PERTURBATION_MODE}_${LEVEL_PERCENT}pct"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_DIR}/dataset_meta.json" ]]; then
  echo "Missing processed dataset: ${DATASET_DIR}/dataset_meta.json" >&2
  exit 1
fi
if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "Missing completed our-model summary: ${SUMMARY_PATH}" >&2
  exit 1
fi

BEST_EPOCH="$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["best_epoch"]))' "${SUMMARY_PATH}")"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${MODEL_RUN_DIR}/ckpt/${BEST_EPOCH}.ckpt}"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Missing best checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${RUN_DIR}"

echo "================ Edge Transformer old-flow robustness ================"
echo "SLURM_JOB_ID        : ${SLURM_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "HOSTNAME            : $(hostname)"
echo "NETWORK_NAME        : ${NETWORK_NAME}"
echo "DATASET_DIR         : ${DATASET_DIR}"
echo "CHECKPOINT_PATH     : ${CHECKPOINT_PATH}"
echo "PERTURBATION_MODE   : ${PERTURBATION_MODE}"
echo "PERTURBATION_LEVEL  : ${PERTURBATION_LEVEL}"
echo "RUN_DIR             : ${RUN_DIR}"
echo "BATCH_SIZE          : ${BATCH_SIZE}"
echo "SEED                : ${SEED}"
echo "START_TIME          : $(date '+%Y-%m-%d %H:%M:%S')"
echo "====================================================================="

python -V
nvidia-smi || true
python -c "import torch, torch_geometric, graphgps; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available()); print('core imports ok')"

MAX_BATCH_ARGS=()
if [[ -n "${MAX_BATCHES:-}" ]]; then
  MAX_BATCH_ARGS=(--max-batches "${MAX_BATCHES}")
fi

srun python scripts/evaluate_old_flow_robustness.py \
  --cfg "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --network "${NETWORK_NAME}" \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${RUN_DIR}" \
  --mode "${PERTURBATION_MODE}" \
  --level "${PERTURBATION_LEVEL}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE_VALUE}" \
  --seed "${SEED}" \
  "${MAX_BATCH_ARGS[@]}" \
  topology_gnn.attention_every_k_steps 1 \
  topology_gnn.enable_global_attn False \
  model.lambda_new_final "${LAMBDA_NEW_FINAL}" \
  model.lambda_con "${LAMBDA_CON}" \
  "$@"

echo "END_TIME            : $(date '+%Y-%m-%d %H:%M:%S')"

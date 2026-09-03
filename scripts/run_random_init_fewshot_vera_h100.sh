#!/usr/bin/env bash
# Random-initialization control, paired with the completed LOSO few-shot run.

#SBATCH -J et_rand_fs
#SBATCH -o logs/et_rand_fs_%A_%a.out
#SBATCH -e logs/et_rand_fs_%A_%a.err
#SBATCH -t 0-06:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -A NA
#SBATCH -p vera
#SBATCH --gpus-per-node=H100:1
#SBATCH --array=0-2%1

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
if [[ ! -f "${PROJECT_ROOT}/scripts/run_random_init_fewshot.py" ]]; then
  PROJECT_ROOT="/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main"
fi
if [[ ! -f "${PROJECT_ROOT}/scripts/run_random_init_fewshot.py" ]]; then
  echo "Could not find random-init script under PROJECT_ROOT=${PROJECT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p logs results/random_init_fewshot
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

TARGETS=(SiouxFalls EMA Anaheim)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#TARGETS[@]}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-2" >&2
  exit 1
fi
TARGET="${TARGETS[$TASK_ID]}"

CFG="${CFG:-${PROJECT_ROOT}/configs/GatedGCN/network-pairs-topology.yaml}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${PROJECT_ROOT}/results/multisource_loso_fewshot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/random_init_fewshot}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
ADAPT_LR="${ADAPT_LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
SEED="${SEED:-42}"

echo "================ Random-init Few-shot Control (Vera H100) ================"
echo "SLURM_JOB_ID    : ${SLURM_JOB_ID:-interactive}"
echo "ARRAY_TASK_ID   : ${TASK_ID}"
echo "HOSTNAME        : $(hostname)"
echo "TARGET          : ${TARGET}"
echo "TRANSFER_ROOT   : ${TRANSFER_ROOT}"
echo "OUTPUT_ROOT     : ${OUTPUT_ROOT}"
echo "ADAPT_EPOCHS    : ${ADAPT_EPOCHS}"
echo "BATCH_SIZE      : ${BATCH_SIZE}"
echo "ADAPT_LR        : ${ADAPT_LR}"
echo "WEIGHT_DECAY    : ${WEIGHT_DECAY}"
echo "SEED            : ${SEED}"
echo "K_VALUES        : 0 50 100 250 500 1000 2000 4000"
echo "START_TIME      : $(date '+%F %T')"
echo "============================================================================"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the H100 allocation")
print("GPU:", torch.cuda.get_device_name(0))
PY

python "${PROJECT_ROOT}/scripts/run_random_init_fewshot.py" \
  --cfg "${CFG}" \
  --data-root "${DATA_ROOT}" \
  --transfer-root "${TRANSFER_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --target "${TARGET}" \
  --k-values 0 50 100 250 500 1000 2000 4000 \
  --adapt-epochs "${ADAPT_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --adapt-lr "${ADAPT_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --device cuda

echo "END_TIME        : $(date '+%F %T')"

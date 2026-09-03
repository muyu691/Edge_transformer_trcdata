#!/usr/bin/env bash
# Multi-source LOSO + few-shot adaptation, one held-out target per H100 task.

#SBATCH -J et_loso_fs
#SBATCH -o logs/et_loso_fs_%A_%a.out
#SBATCH -e logs/et_loso_fs_%A_%a.err
#SBATCH -t 0-08:00:00
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
if [[ ! -f "${PROJECT_ROOT}/scripts/run_multisource_loso_fewshot.py" ]]; then
  PROJECT_ROOT="/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main"
fi
if [[ ! -f "${PROJECT_ROOT}/scripts/run_multisource_loso_fewshot.py" ]]; then
  echo "Could not find transfer script under PROJECT_ROOT=${PROJECT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p logs results/multisource_loso_fewshot
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/multisource_loso_fewshot}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-200}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
SOURCE_LR="${SOURCE_LR:-0.001}"
ADAPT_LR="${ADAPT_LR:-0.0001}"
SEED="${SEED:-42}"

echo "================ Multi-source LOSO + Few-shot (Vera H100) ================"
echo "SLURM_JOB_ID    : ${SLURM_JOB_ID:-interactive}"
echo "ARRAY_TASK_ID   : ${TASK_ID}"
echo "HOSTNAME        : $(hostname)"
echo "TARGET          : ${TARGET}"
echo "CFG             : ${CFG}"
echo "DATA_ROOT       : ${DATA_ROOT}"
echo "OUTPUT_ROOT     : ${OUTPUT_ROOT}"
echo "SOURCE_EPOCHS   : ${SOURCE_EPOCHS}"
echo "ADAPT_EPOCHS    : ${ADAPT_EPOCHS}"
echo "BATCH_SIZE      : ${BATCH_SIZE}"
echo "K_VALUES        : 0 50 100 250 500 1000 2000 4000"
echo "START_TIME      : $(date '+%F %T')"
echo "============================================================================"

python - <<'PY'
import torch
print("Python/PyTorch import ok")
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the H100 allocation")
print("GPU:", torch.cuda.get_device_name(0))
PY

python "${PROJECT_ROOT}/scripts/run_multisource_loso_fewshot.py" \
  --cfg "${CFG}" \
  --data-root "${DATA_ROOT}" \
  --target "${TARGET}" \
  --output-root "${OUTPUT_ROOT}" \
  --k-values 0 50 100 250 500 1000 2000 4000 \
  --source-epochs "${SOURCE_EPOCHS}" \
  --adapt-epochs "${ADAPT_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --source-lr "${SOURCE_LR}" \
  --adapt-lr "${ADAPT_LR}" \
  --seed "${SEED}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --device cuda

echo "END_TIME        : $(date '+%F %T')"

#!/usr/bin/env bash

#SBATCH -J sue_ws_pred
#SBATCH -o logs/sue_ws_pred_%A_%a.out
#SBATCH -e logs/sue_ws_pred_%A_%a.err
#SBATCH -t 0-00:30:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"
mkdir -p logs results/sue_warmstart

KEYS=(siouxfalls ema anaheim)
NETWORKS=(SiouxFalls EMA Anaheim)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
KEY="${KEYS[$TASK_ID]}"
NETWORK="${NETWORKS[$TASK_ID]}"

PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/sue_warmstart}"
CFG="${CFG:-${PROJECT_ROOT}/configs/GatedGCN/network-pairs-topology.yaml}"
DATASET_DIR="${PROCESSED_ROOT}/${KEY}_pyg_newpolicy_lhs"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/results/ours/network-pairs-topology-ours_edge_transformer_${KEY}_10000/0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_GRAPHS="${NUM_TEST_GRAPHS:-0}"

for required in \
  "${DATASET_DIR}/test_dataset.pt" \
  "${DATASET_DIR}/split_indices.npz" \
  "${RUN_DIR}/summary.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 1
  fi
done

OUTPUT_DIR="${RESULTS_ROOT}/${KEY}/prediction"
mkdir -p "${OUTPUT_DIR}"

echo "================ SUE Warm-start Prediction Export ================"
echo "JOB/TASK       : ${SLURM_JOB_ID:-N/A}/${TASK_ID}"
echo "NETWORK        : ${NETWORK}"
echo "DATASET_DIR    : ${DATASET_DIR}"
echo "RUN_DIR        : ${RUN_DIR}"
echo "OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "MAX_GRAPHS     : ${MAX_GRAPHS}"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "=================================================================="

srun python "${PROJECT_ROOT}/scripts/export_sue_warmstart_predictions.py" \
  --cfg "${CFG}" \
  --network "${NETWORK}" \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --max-graphs "${MAX_GRAPHS}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --device cuda

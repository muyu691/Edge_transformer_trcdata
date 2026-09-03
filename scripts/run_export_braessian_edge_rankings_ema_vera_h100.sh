#!/usr/bin/env bash

#SBATCH -J braess_rank
#SBATCH -o logs/braess_rank_%j.out
#SBATCH -e logs/braess_rank_%j.err
#SBATCH -t 0-04:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A NA
#SBATCH -p vera
#SBATCH --gpus-per-node=H100:1

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
RAW_DIR="${RAW_DIR:-${PROCESSED_ROOT}/ema_pairs_reconstructed_from_pyg}"
INPUT_PKL="${INPUT_PKL:-${RAW_DIR}/network_pairs_dataset.pkl}"
VALIDATION_MARKER="${VALIDATION_MARKER:-${RAW_DIR}/RAW_PAIRS_MATCH_PYG.ok}"
DATASET_DIR="${DATASET_DIR:-${PROCESSED_ROOT}/ema_pyg_newpolicy_lhs}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/results/ours/network-pairs-topology-ours_edge_transformer_ema_10000/0}"
CFG="${CFG:-${PROJECT_ROOT}/configs/GatedGCN/network-pairs-topology.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/braessian_edges/ema}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/model}"
NUM_SCENARIOS="${NUM_SCENARIOS:-500}"
SELECTION_SEED="${SELECTION_SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-236}"
PREDICTION_FLOW_POLICY="${PREDICTION_FLOW_POLICY:-clip_zero}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
for required in \
  "${INPUT_PKL}" \
  "${VALIDATION_MARKER}" \
  "${RAW_DIR}/raw_pairs_validation.json" \
  "${DATASET_DIR}/dataset_meta.json" \
  "${DATASET_DIR}/split_indices.npz" \
  "${DATASET_DIR}/scalers/attr_scaler.pkl" \
  "${DATASET_DIR}/scalers/flow_scaler.pkl" \
  "${RUN_DIR}/summary.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

echo "================ EMA Braessian Model Ranking ================="
echo "JOB             : ${SLURM_JOB_ID:-interactive}"
echo "INPUT_PKL       : ${INPUT_PKL}"
echo "DATASET_DIR     : ${DATASET_DIR}"
echo "RUN_DIR         : ${RUN_DIR}"
echo "NUM_SCENARIOS   : ${NUM_SCENARIOS}"
echo "SELECTION_SEED  : ${SELECTION_SEED}"
echo "BATCH_SIZE      : ${BATCH_SIZE}"
echo "FLOW_POLICY     : ${PREDICTION_FLOW_POLICY}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "GPU             : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "==============================================================="

srun python "${PROJECT_ROOT}/scripts/braessian_edges.py" score \
  --input-pkl "${INPUT_PKL}" \
  --validation-marker "${VALIDATION_MARKER}" \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}" \
  --cfg "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-scenarios "${NUM_SCENARIOS}" \
  --selection-seed "${SELECTION_SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-16}" \
  --prediction-flow-policy "${PREDICTION_FLOW_POLICY}" \
  --device cuda \
  --resume

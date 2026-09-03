#!/usr/bin/env bash
#SBATCH --job-name=ema_raw_rebuild
#SBATCH --output=logs/ema_raw_rebuild_%j.out
#SBATCH --error=logs/ema_raw_rebuild_%j.err
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=NA
#SBATCH --partition=cpu

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_ROOT="${PROJECT_ROOT:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
PYG_DIR="${PYG_DIR:-${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pyg_newpolicy_lhs}"
SCENARIOS_NPZ="${SCENARIOS_NPZ:-${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pairs_newpolicy_lhs_recovered/base_scenarios.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pairs_reconstructed_from_pyg}"
RAW_PKL="${OUTPUT_DIR}/network_pairs_dataset.pkl"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

echo "================ EMA Raw-Pair Reconstruction From PyG ================="
echo "SLURM_JOB_ID : ${SLURM_JOB_ID:-interactive}"
echo "PYG_DIR      : ${PYG_DIR}"
echo "SCENARIOS    : ${SCENARIOS_NPZ}"
echo "OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "START_TIME   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

srun python scripts/reconstruct_raw_pairs_from_pyg.py \
  --pyg-dir "${PYG_DIR}" \
  --scenarios-npz "${SCENARIOS_NPZ}" \
  --output-pkl "${RAW_PKL}" \
  --expected-network EMA \
  --expected-pairs 10000 \
  --overwrite

srun python scripts/validate_recovered_raw_pairs.py \
  --input-pkl "${RAW_PKL}" \
  --pyg-dir "${PYG_DIR}" \
  --expected-network EMA \
  --expected-pairs 10000 \
  --output-json "${OUTPUT_DIR}/raw_pairs_validation.json" \
  --success-marker "${OUTPUT_DIR}/RAW_PAIRS_MATCH_PYG.ok"

ls -lh \
  "${RAW_PKL}" \
  "${OUTPUT_DIR}/raw_pairs_validation.json" \
  "${OUTPUT_DIR}/RAW_PAIRS_MATCH_PYG.ok"

echo "END_TIME: $(date '+%Y-%m-%d %H:%M:%S')"

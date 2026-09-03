#!/bin/bash
# Build the robustness CSV and figures after all GPU array tasks finish.

#SBATCH -J flow_rob_sum
#SBATCH -o logs/flow_rob_sum_%j.out
#SBATCH -e logs/flow_rob_sum_%j.err
#SBATCH -t 0-00:30:00
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -A NA
#SBATCH -p cpu

set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/old_flow_robustness}"
OURS_ROOT="${OURS_ROOT:-${PROJECT_ROOT}/results/ours}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/summary}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"
mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

python scripts/summarize_old_flow_robustness.py \
  --results-root "${RESULTS_ROOT}" \
  --ours-root "${OURS_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

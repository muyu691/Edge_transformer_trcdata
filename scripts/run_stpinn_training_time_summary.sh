#!/bin/bash
# Summarize ST-PINN GatedGCN training time from completed final runs.
#
# Run from the project root on Vera:
#   bash scripts/run_stpinn_training_time_summary.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/training_time}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

python scripts/summarize_stpinn_training_time.py \
  --project-root "${PROJECT_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

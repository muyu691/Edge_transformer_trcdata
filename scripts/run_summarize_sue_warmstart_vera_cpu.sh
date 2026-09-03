#!/usr/bin/env bash

#SBATCH -J sue_ws_sum
#SBATCH -o logs/sue_ws_sum_%j.out
#SBATCH -e logs/sue_ws_sum_%j.err
#SBATCH -t 0-00:20:00
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -A NA
#SBATCH -p cpu

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/sue_warmstart}"

srun python "${PROJECT_ROOT}/scripts/summarize_sue_warmstart.py" \
  --results-root "${RESULTS_ROOT}"

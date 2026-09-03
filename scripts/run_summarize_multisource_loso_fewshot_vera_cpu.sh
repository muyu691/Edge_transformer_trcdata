#!/usr/bin/env bash

#SBATCH -J et_loso_sum
#SBATCH -o logs/et_loso_sum_%j.out
#SBATCH -e logs/et_loso_sum_%j.err
#SBATCH -t 0-00:20:00
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -A NA

set -euo pipefail

DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
cd "${PROJECT_ROOT}"
mkdir -p logs
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/multisource_loso_fewshot}"
python "${PROJECT_ROOT}/scripts/summarize_multisource_loso_fewshot.py" \
  --results-root "${RESULTS_ROOT}" \
  --k-values 0 50 100 250 500 1000 2000 4000

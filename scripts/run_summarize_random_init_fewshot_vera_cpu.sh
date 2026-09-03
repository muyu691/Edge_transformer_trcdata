#!/usr/bin/env bash

#SBATCH -J et_rand_sum
#SBATCH -o logs/et_rand_sum_%j.out
#SBATCH -e logs/et_rand_sum_%j.err
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

RANDOM_ROOT="${RANDOM_ROOT:-${PROJECT_ROOT}/results/random_init_fewshot}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${PROJECT_ROOT}/results/multisource_loso_fewshot}"
python "${PROJECT_ROOT}/scripts/summarize_random_init_fewshot.py" \
  --random-root "${RANDOM_ROOT}" \
  --transfer-root "${TRANSFER_ROOT}" \
  --k-values 0 50 100 250 500 1000 2000 4000

#!/bin/bash

#SBATCH -J plot_conv
#SBATCH -o logs/plot_conv_%j.out
#SBATCH -e logs/plot_conv_%j.err
#SBATCH -t 0-01:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -A NA
#SBATCH -p shared

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

mkdir -p "${PROJECT_ROOT}/logs"

srun python "${PROJECT_ROOT}/convergence_curves/plot_suite.py" "$@"

#!/bin/bash

#SBATCH -J cv_ours
#SBATCH -o logs/cv_ours_%j.out
#SBATCH -e logs/cv_ours_%j.err
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A NA
#SBATCH -p gpu
#SBATCH --gres=gpu:A40:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/constraint_violation}"

exec bash "${PROJECT_ROOT}/run_topology_gnn_vera_gpu.sh" "$@"

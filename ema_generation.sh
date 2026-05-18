#!/bin/bash
#SBATCH -J ema_data
#SBATCH -o logs/ema_data_%j.out
#SBATCH -e logs/ema_data_%j.err
#SBATCH -t 0-02:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -A NA
#SBATCH -p cpu

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

mkdir -p \
  "${PROJECT_ROOT}/logs" \
  "${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pairs_new" \
  "${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pyg_new"

srun python create_sioux_data/solve_network_pairs.py \
  --network_name EMA \
  --dataset_root /cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main/ema_data \
  --network_file /cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main/ema_data/EMA_net.tntp \
  --demand_source lhs \
  --num_samples 4000 \
  --seed 42 \
  --output_dir "${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pairs_new"

srun python create_sioux_data/build_network_pairs_dataset.py \
  --input_pkl "${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pairs_new/network_pairs_dataset.pkl" \
  --output_dir "${PROJECT_ROOT}/create_sioux_data/processed_data/ema_pyg_new" \
  --train_ratio 0.6 \
  --val_ratio 0.2 \
  --seed 42

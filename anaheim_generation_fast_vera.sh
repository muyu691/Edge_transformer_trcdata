#!/bin/bash
# Fast Vera CPU launcher for generating Anaheim network-pair data.
#
# This runs the realism-preserving data-generation pipeline:
#   1. baseline-preserving OD and network perturbations
#   2. first SUE solve on G -> flows_old
#   3. one sampled closure, capacity change, or new-link event -> G'
#   4. second SUE solve on G' -> flows_new
#   5. PyG train/val/test build with 0.6/0.2/0.2 split
#
# Outputs:
#   create_sioux_data/processed_data/anaheim_pairs_baseline_perturb/network_pairs_dataset.pkl
#   create_sioux_data/processed_data/anaheim_pyg_baseline_perturb/{train,val,test}_dataset.pt
#
# Typical usage on Vera:
#   sbatch anaheim_generation_fast_vera.sh
#
# Useful overrides:
#   NUM_SAMPLES=4000 sbatch anaheim_generation_fast_vera.sh
#   NUM_WORKERS=16 sbatch --cpus-per-task=16 anaheim_generation_fast_vera.sh
#   FORCE_FIRST_SOLVE=1 sbatch anaheim_generation_fast_vera.sh

#SBATCH -J ana_fast
#SBATCH -o logs/ana_fast_%j.out
#SBATCH -e logs/ana_fast_%j.err
#SBATCH -t 4-00:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -A NA
#SBATCH -p cpu

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main}}"
if [[ ! -f "${PROJECT_ROOT}/create_sioux_data/solve_network_pairs.py" ]]; then
  PROJECT_ROOT="/cephyr/users/wuxin/Vera/Physics-Informed_Diffusion_Model-main/Network_reconfiguration-main"
fi
if [[ ! -f "${PROJECT_ROOT}/create_sioux_data/solve_network_pairs.py" ]]; then
  echo "Could not find project root. Set PROJECT_ROOT explicitly." >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/anaheim_data}"
NETWORK_FILE="${NETWORK_FILE:-${DATASET_ROOT}/Anaheim_net.tntp}"
OD_FILE="${OD_FILE:-${DATASET_ROOT}/Anaheim_trips.tntp}"

PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/create_sioux_data/processed_data}"
PAIRS_DIR="${PAIRS_DIR:-${PROCESSED_ROOT}/anaheim_pairs_baseline_perturb}"
PYG_DIR="${PYG_DIR:-${PROCESSED_ROOT}/anaheim_pyg_baseline_perturb}"

NUM_SAMPLES="${NUM_SAMPLES:-10000}"
SEED="${SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.6}"
VAL_RATIO="${VAL_RATIO:-0.2}"

MAX_ITER="${MAX_ITER:-120}"
CONVERGENCE_THRESHOLD="${CONVERGENCE_THRESHOLD:-1e-5}"
THETA="${THETA:-0.8}"
VALUE_ITER="${VALUE_ITER:-250}"
VALUE_TOL="${VALUE_TOL:-1e-8}"
FLOW_ITER="${FLOW_ITER:-500}"
FLOW_TOL="${FLOW_TOL:-1e-9}"
RETRY_FLOW_ITER="${RETRY_FLOW_ITER:-2000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"

if [[ ! -f "${NETWORK_FILE}" ]]; then
  echo "Missing Anaheim network file: ${NETWORK_FILE}" >&2
  exit 1
fi

if [[ ! -f "${OD_FILE}" ]]; then
  echo "Missing Anaheim OD file: ${OD_FILE}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${PAIRS_DIR}" "${PYG_DIR}"

SCENARIOS_PATH="${PAIRS_DIR}/base_scenarios.npz"
FLOWS_OLD_PATH="${PAIRS_DIR}/flows_old.npy"

SOLVE_EXTRA_ARGS=()
if [[ "${FORCE_FIRST_SOLVE:-0}" != "1" && -f "${SCENARIOS_PATH}" && -f "${FLOWS_OLD_PATH}" ]]; then
  if SCENARIOS_PATH="${SCENARIOS_PATH}" FLOWS_OLD_PATH="${FLOWS_OLD_PATH}" NUM_SAMPLES="${NUM_SAMPLES}" python - <<'PY'
import os
import sys
import numpy as np

num_samples = int(os.environ["NUM_SAMPLES"])
try:
    scenarios = np.load(os.environ["SCENARIOS_PATH"])
    flows_old = np.load(os.environ["FLOWS_OLD_PATH"])
    ok = (
        scenarios["od_matrices"].shape[0] == num_samples
        and scenarios["capacities"].shape[0] == num_samples
        and scenarios["speeds"].shape[0] == num_samples
        and flows_old.shape[0] == num_samples
    )
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
  then
    SOLVE_EXTRA_ARGS+=(--skip_first_solve)
    echo "Detected reusable base_scenarios.npz and flows_old.npy; enabling --skip_first_solve."
  else
    echo "Existing first-stage files do not match NUM_SAMPLES=${NUM_SAMPLES}; recomputing first SUE solve."
  fi
fi

if [[ "${CHECKPOINT:-1}" == "1" ]]; then
  SOLVE_EXTRA_ARGS+=(--checkpoint --checkpoint_interval "${CHECKPOINT_INTERVAL}")
fi
if [[ -n "${CENTROID_NODES:-}" ]]; then
  SOLVE_EXTRA_ARGS+=(--centroid_nodes "${CENTROID_NODES}")
fi

echo "================ Anaheim Data Generation Fast (Vera CPU) ================"
echo "SLURM_JOB_ID   : ${SLURM_JOB_ID:-N/A}"
echo "HOSTNAME       : $(hostname)"
echo "PWD            : $(pwd)"
echo "NETWORK_FILE   : ${NETWORK_FILE}"
echo "OD_FILE        : ${OD_FILE}"
echo "NUM_SAMPLES    : ${NUM_SAMPLES}"
echo "NUM_WORKERS    : ${NUM_WORKERS}"
echo "THREADS        : OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}"
echo "PAIRS_DIR      : ${PAIRS_DIR}"
echo "PYG_DIR        : ${PYG_DIR}"
echo "EXTRA_ARGS     : ${SOLVE_EXTRA_ARGS[*]:-<none>}"
echo "START_TIME     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================="

python -V
python -c "import numpy, scipy, networkx, torch, torch_geometric; print('core imports ok')"

srun --cpu-bind=cores python create_sioux_data/solve_network_pairs.py \
  --network_name Anaheim \
  --dataset_root "${DATASET_ROOT}" \
  --network_file "${NETWORK_FILE}" \
  --od_file "${OD_FILE}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --output_dir "${PAIRS_DIR}" \
  --max_iter "${MAX_ITER}" \
  --convergence_threshold "${CONVERGENCE_THRESHOLD}" \
  --theta "${THETA}" \
  --value_iter "${VALUE_ITER}" \
  --value_tol "${VALUE_TOL}" \
  --flow_iter "${FLOW_ITER}" \
  --flow_tol "${FLOW_TOL}" \
  --retry_flow_iter "${RETRY_FLOW_ITER}" \
  --num_workers "${NUM_WORKERS}" \
  "${SOLVE_EXTRA_ARGS[@]}"

srun --cpu-bind=cores python create_sioux_data/build_network_pairs_dataset.py \
  --input_pkl "${PAIRS_DIR}/network_pairs_dataset.pkl" \
  --output_dir "${PYG_DIR}" \
  --train_ratio "${TRAIN_RATIO}" \
  --val_ratio "${VAL_RATIO}" \
  --seed "${SEED}"

PYG_DIR="${PYG_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

meta_path = Path(os.environ["PYG_DIR"]) / "dataset_meta.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print("Generated dataset metadata:")
    print(json.dumps({
        "network_name": meta.get("network_name"),
        "num_nodes": meta.get("num_nodes"),
        "num_edges_old": meta.get("num_edges_old"),
        "num_edges_new": meta.get("num_edges_new"),
        "od_dim": meta.get("od_dim"),
        "centroid_count": meta.get("centroid_count"),
        "splits": meta.get("splits"),
    }, indent=2))
PY

echo "END_TIME       : $(date '+%Y-%m-%d %H:%M:%S')"

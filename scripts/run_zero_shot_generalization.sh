#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

source "${PROJECT_ROOT}/scripts/activate_venv_cuda.sh"

CFG="${CFG:-configs/GatedGCN/network-pairs-topology.yaml}"
DATA_ROOT="${DATA_ROOT:-create_sioux_data/processed_data}"
OUT_DIR="${OUT_DIR:-results/zero_shot_generalization}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-auto}"
SPLITS="${SPLITS:-test}"

SIOUX_CKPT="${SIOUX_CKPT:-results/convergence_curves/network-pairs-topology-conv_ours_siouxfalls/0/ckpt/175.ckpt}"
EMA_CKPT="${EMA_CKPT:-results/convergence_curves/network-pairs-topology-conv_ours_ema/0/ckpt/185.ckpt}"

if [[ "${SMOKE:-0}" == "1" && -z "${MAX_BATCHES:-}" ]]; then
  MAX_BATCHES=1
fi

MAX_BATCHES_ARGS=()
if [[ -n "${MAX_BATCHES:-}" ]]; then
  MAX_BATCHES_ARGS=(--max-batches "$MAX_BATCHES")
fi

for required in "$CFG" "$DATA_ROOT" "$SIOUX_CKPT" "$EMA_CKPT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

run_pair() {
  local source_network="$1"
  local target_network="$2"
  local checkpoint="$3"
  local run_dir="$OUT_DIR/${source_network}_to_${target_network}"

  python scripts/evaluate_zero_shot.py \
    --cfg "$CFG" \
    --checkpoint "$checkpoint" \
    --source-network "$source_network" \
    --target-network "$target_network" \
    --target-processed-root "$DATA_ROOT" \
    --out-dir "$run_dir" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --splits $SPLITS \
    "${MAX_BATCHES_ARGS[@]}"
}

run_pair "SiouxFalls" "EMA" "$SIOUX_CKPT"
run_pair "EMA" "SiouxFalls" "$EMA_CKPT"

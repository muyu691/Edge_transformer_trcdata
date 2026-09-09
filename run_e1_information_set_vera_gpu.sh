#!/usr/bin/env bash
# One job = one network x one mode x one seed. Activate your CUDA environment first.
#SBATCH --job-name=e1_info
#SBATCH --output=logs/e1_%j.out
#SBATCH --error=logs/e1_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
cd "$PROJECT_ROOT"
: "${NETWORK:?Set NETWORK=siouxfalls|ema|anaheim}"
: "${MODE:?Set MODE=od_only|old_state|hybrid|persistence}"
SEED="${SEED:-42}"
args=(--network "$NETWORK" --mode "$MODE" --seed "$SEED" --device "${DEVICE:-cuda}")
[[ -z "${BATCH_SIZE:-}" ]] || args+=(--batch-size "$BATCH_SIZE")
[[ -z "${DATASET_DIR:-}" ]] || args+=(--dataset-dir "$DATASET_DIR")
[[ -z "${OUTPUT_ROOT:-}" ]] || args+=(--output-root "$OUTPUT_ROOT")
exec "${PYTHON_EXE:-python}" -B scripts/e1_information_set.py "${args[@]}" "$@"

#!/bin/bash

normalize_dataset_key() {
  local dataset_key="${DATASET_NAME:-ema}"
  printf '%s' "${dataset_key}" | tr '[:upper:]' '[:lower:]'
}

resolve_ablation_dataset_dir() {
  if [[ -n "${DATASET_DIR:-}" ]]; then
    printf '%s\n' "${DATASET_DIR}"
    return 0
  fi

  local dataset_key
  dataset_key="$(normalize_dataset_key)"

  case "${dataset_key}" in
    ema)
      printf '%s\n' "${PROCESSED_ROOT}/ema_pyg_baseline_perturb"
      ;;
    siouxfalls|sioux_falls|sioux-falls)
      printf '%s\n' "${PROCESSED_ROOT}/siouxfalls_pyg_baseline_perturb"
      ;;
    anaheim)
      printf '%s\n' "${PROCESSED_ROOT}/anaheim_pyg_baseline_perturb"
      ;;
    *)
      printf '%s\n' "${PROCESSED_ROOT}"
      ;;
  esac
}


resolve_ablation_network_name() {
  if [[ -n "${NETWORK_NAME:-}" ]]; then
    printf '%s\n' "${NETWORK_NAME}"
    return 0
  fi

  if [[ -n "${DATASET_DIR:-}" && -f "${DATASET_DIR}/dataset_meta.json" ]]; then
    local meta_network_name
    meta_network_name="$(python - <<'PY'
import json
import os

dataset_dir = os.environ.get("DATASET_DIR", "")
meta_path = os.path.join(dataset_dir, "dataset_meta.json")
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
print(meta.get("network_name", ""))
PY
)"
    if [[ -n "${meta_network_name}" ]]; then
      printf '%s\n' "${meta_network_name}"
      return 0
    fi
  fi

  if [[ -n "${DATASET_DIR:-}" ]]; then
    echo "Could not infer NETWORK_NAME from DATASET_DIR=${DATASET_DIR}. Set NETWORK_NAME explicitly." >&2
    return 1
  fi

  local dataset_key
  dataset_key="$(normalize_dataset_key)"

  case "${dataset_key}" in
    ema)
      printf '%s\n' "EMA"
      ;;
    siouxfalls|sioux_falls|sioux-falls)
      printf '%s\n' "SiouxFalls"
      ;;
    anaheim)
      printf '%s\n' "Anaheim"
      ;;
    *)
      echo "Unsupported DATASET_NAME=${DATASET_NAME}. Set NETWORK_NAME explicitly for custom networks." >&2
      return 1
      ;;
  esac
}

set_ablation_main_hparams() {
  local dataset_key
  dataset_key="$(normalize_dataset_key)"

  case "${dataset_key}" in
    siouxfalls|sioux_falls|sioux-falls)
      LAMBDA_NEW_FINAL="${LAMBDA_NEW_FINAL:-1.0}"
      LAMBDA_CON="${LAMBDA_CON:-0.1}"
      LAMBDA_CON_SCHEDULE="${LAMBDA_CON_SCHEDULE:-staged_linear}"
      LAMBDA_CON_MID="${LAMBDA_CON_MID:-0.01}"
      ;;
    ema)
      LAMBDA_NEW_FINAL="${LAMBDA_NEW_FINAL:-1.5}"
      LAMBDA_CON="${LAMBDA_CON:-0.05}"
      LAMBDA_CON_SCHEDULE="${LAMBDA_CON_SCHEDULE:-staged_linear}"
      LAMBDA_CON_MID="${LAMBDA_CON_MID:-0.01}"
      ;;
    *)
      LAMBDA_NEW_FINAL="${LAMBDA_NEW_FINAL:-1.0}"
      LAMBDA_CON="${LAMBDA_CON:-0.05}"
      LAMBDA_CON_SCHEDULE="${LAMBDA_CON_SCHEDULE:-staged_linear}"
      LAMBDA_CON_MID="${LAMBDA_CON_MID:-0.01}"
      ;;
  esac
}

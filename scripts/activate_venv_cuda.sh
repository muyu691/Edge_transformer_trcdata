#!/usr/bin/env bash

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

VENV_DIR="${PROJECT_ROOT}/venv_cuda"

PYTHON_HOME=""
if [[ -f "${VENV_DIR}/pyvenv.cfg" ]]; then
  PYTHON_HOME="$(
    awk -F= '
      /^home[[:space:]]*=/ {
        value=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        print value
        exit
      }
    ' "${VENV_DIR}/pyvenv.cfg"
  )"
fi

if command -v module >/dev/null 2>&1; then
  module purge

  if [[ -z "${PYTHON_MODULE:-}" && -n "${PYTHON_HOME}" ]]; then
    PYTHON_PREFIX="${PYTHON_HOME}"
    if [[ "$(basename "${PYTHON_PREFIX}")" == "bin" ]]; then
      PYTHON_PREFIX="$(dirname "${PYTHON_PREFIX}")"
    fi
    if [[ "${PYTHON_PREFIX}" == */Python/* ]]; then
      PYTHON_MODULE="Python/${PYTHON_PREFIX##*/Python/}"
    fi
  fi

  if [[ -n "${PYTHON_MODULE:-}" ]]; then
    if ! module load "${PYTHON_MODULE}"; then
      echo "Warning: failed to load PYTHON_MODULE=${PYTHON_MODULE}; continuing with LD_LIBRARY_PATH fallbacks." >&2
    fi
  else
    echo "Warning: PYTHON_MODULE was not set and could not be inferred from ${VENV_DIR}/pyvenv.cfg." >&2
  fi
else
  echo "Warning: module command not found; skipping module purge." >&2
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Missing virtual environment: ${VENV_DIR}/bin/activate" >&2
  return 1 2>/dev/null || exit 1
fi

prepend_ld_library_path() {
  local candidate="$1"
  if [[ ! -d "${candidate}" ]]; then
    return 0
  fi

  local resolved
  resolved="$(cd "${candidate}" && pwd -P)"
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${resolved}:"*) ;;
    *) export LD_LIBRARY_PATH="${resolved}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

# venv_cuda may point to a Python executable whose libpython lives in the
# original Python prefix. module purge removes that prefix from LD_LIBRARY_PATH,
# so add the likely lib directories before invoking python.
prepend_ld_library_path "${VENV_DIR}/lib"
prepend_ld_library_path "${VENV_DIR}/lib64"

if [[ -n "${PYTHON_HOME}" ]]; then
  prepend_ld_library_path "${PYTHON_HOME}/../lib"
  prepend_ld_library_path "${PYTHON_HOME}/../lib64"
  prepend_ld_library_path "${PYTHON_HOME}/lib"
  prepend_ld_library_path "${PYTHON_HOME}/lib64"
fi

PYTHON_REAL="$(readlink -f "${VENV_DIR}/bin/python" 2>/dev/null || true)"
if [[ -n "${PYTHON_REAL}" ]]; then
  PYTHON_REAL_BIN_DIR="$(dirname "${PYTHON_REAL}")"
  prepend_ld_library_path "${PYTHON_REAL_BIN_DIR}/../lib"
  prepend_ld_library_path "${PYTHON_REAL_BIN_DIR}/../lib64"
fi

for LIBPYTHON in \
  "${VENV_DIR}"/lib*/libpython*.so* \
  "${VENV_DIR}"/lib*/python*/config-*/libpython*.so*
do
  if [[ -e "${LIBPYTHON}" ]]; then
    prepend_ld_library_path "$(dirname "${LIBPYTHON}")"
  fi
done

# Fallback for Vera/EasyBuild Python modules if a dependency module was not
# reloaded correctly. These globs are no-ops outside that software tree.
for DEP_LIB_DIR in \
  /apps/Arch/software/bzip2/*/lib \
  /apps/Arch/software/bzip2/*/lib64 \
  /apps/Arch/software/zlib/*/lib \
  /apps/Arch/software/zlib/*/lib64 \
  /apps/Arch/software/XZ/*/lib \
  /apps/Arch/software/XZ/*/lib64 \
  /apps/Arch/software/libffi/*/lib \
  /apps/Arch/software/libffi/*/lib64 \
  /apps/Arch/software/OpenSSL/*/lib \
  /apps/Arch/software/OpenSSL/*/lib64 \
  /apps/Arch/software/SQLite/*/lib \
  /apps/Arch/software/SQLite/*/lib64 \
  /apps/Arch/software/ncurses/*/lib \
  /apps/Arch/software/ncurses/*/lib64 \
  /apps/Arch/software/Readline/*/lib \
  /apps/Arch/software/Readline/*/lib64
do
  prepend_ld_library_path "${DEP_LIB_DIR}"
done

source "${VENV_DIR}/bin/activate"

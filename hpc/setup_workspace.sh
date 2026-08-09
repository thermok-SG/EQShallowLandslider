#!/usr/bin/env bash
# Create the machine-local data workspace used by the HPC configurations.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/EQShallowLandslider_data" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$1"

if [[ "${DATA_ROOT}" != /* ]]; then
  echo "The data workspace path must be absolute: ${DATA_ROOT}" >&2
  exit 2
fi

mkdir -p \
  "${DATA_ROOT}/input_data/dem" \
  "${DATA_ROOT}/input_data/nepal" \
  "${DATA_ROOT}/input_data/japan" \
  "${DATA_ROOT}/runs" \
  "${DATA_ROOT}/analysis_output"

if [[ -L "${SCRIPT_DIR}/workspace" ]]; then
  current_target="$(readlink "${SCRIPT_DIR}/workspace")"
  if [[ "${current_target}" == "${DATA_ROOT}" ]]; then
    echo "Workspace already configured: ${SCRIPT_DIR}/workspace -> ${DATA_ROOT}"
    exit 0
  fi
  echo "Workspace link already points to ${current_target}; remove it explicitly to replace it." >&2
  exit 1
fi

if [[ -e "${SCRIPT_DIR}/workspace" ]]; then
  echo "Refusing to replace existing path: ${SCRIPT_DIR}/workspace" >&2
  exit 1
fi

ln -s "${DATA_ROOT}" "${SCRIPT_DIR}/workspace"
echo "Configured workspace: ${SCRIPT_DIR}/workspace -> ${DATA_ROOT}"

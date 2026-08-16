#!/bin/bash
#SBATCH --job-name=single-ls
#SBATCH --partition=cpu_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${EQSL_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" || ! -f "${PROJECT_DIR}/run_landslide_model_cli.py" ]]; then
  echo "Submit from the repository root or export EQSL_PROJECT_DIR=/absolute/repository/path." >&2
  exit 1
fi
PROJECT_DIR="$(cd -- "${PROJECT_DIR}" && pwd)"
WORKSPACE="${PROJECT_DIR}/hpc/workspace"
CONFIG="${CONFIG:-${PROJECT_DIR}/hpc/configs/nepal_config.yaml}"

module load conda
conda activate landlab_dev

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ ! -e "${WORKSPACE}" ]]; then
  echo "Missing hpc/workspace; run hpc/setup_workspace.sh first." >&2
  exit 1
fi
cd "${WORKSPACE}"

python -u "${PROJECT_DIR}/run_landslide_model_cli.py" \
  --config "${CONFIG}" \
  --validate-only

python -u "${PROJECT_DIR}/run_landslide_model_cli.py" \
  --config "${CONFIG}"

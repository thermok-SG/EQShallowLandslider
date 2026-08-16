#!/bin/bash
#SBATCH --job-name=ensem-ls
#SBATCH --partition=cpu_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${EQSL_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" || ! -f "${PROJECT_DIR}/run_landslide_ensemble.py" ]]; then
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

# This allocation runs one ensemble member at a time. Increase --jobs only if
# both --cpus-per-task and memory are increased to hold concurrent full models.
python -u "${PROJECT_DIR}/run_landslide_ensemble.py" \
  --config "${CONFIG}" \
  --dry-run

python -u "${PROJECT_DIR}/run_landslide_ensemble.py" \
  --config "${CONFIG}" \
  --jobs 1

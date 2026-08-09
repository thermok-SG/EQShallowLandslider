#!/bin/bash
#SBATCH --job-name=landslide-single
#SBATCH --partition=cpu_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-${PROJECT_DIR}/hpc/configs/nepal_config.yaml}"

module load conda
conda activate landlab_dev

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${PROJECT_DIR}"

if [[ ! -e "hpc/workspace" ]]; then
  echo "Missing hpc/workspace; run hpc/setup_workspace.sh first." >&2
  exit 1
fi

python -u run_landslide_model_cli.py \
  --config "${CONFIG}" \
  --validate-only

python -u run_landslide_model_cli.py \
  --config "${CONFIG}"

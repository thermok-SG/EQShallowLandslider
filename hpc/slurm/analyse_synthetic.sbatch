#!/bin/bash
#SBATCH --job-name=analy-synth
#SBATCH --partition=cpu_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${EQSL_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" || ! -f "${PROJECT_DIR}/analyse_landslide_outputs.py" ]]; then
  echo "Submit from the repository root or export EQSL_PROJECT_DIR=/absolute/repository/path." >&2
  exit 1
fi
PROJECT_DIR="$(cd -- "${PROJECT_DIR}" && pwd)"
WORKSPACE="${PROJECT_DIR}/hpc/workspace"
MODEL="${MODEL:-stability}"

if [[ "${MODEL}" != "stability" && "${MODEL}" != "runout" ]]; then
  echo "MODEL must be stability or runout, not: ${MODEL}" >&2
  exit 2
fi

RUNS_DIR="${WORKSPACE}/runs/synthetic_${MODEL}"
OUTPUT_DIR="${WORKSPACE}/analysis_output/synthetic_${MODEL}"

module load conda
conda activate landlab_dev

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ ! -d "${RUNS_DIR}" ]]; then
  echo "Run directory does not exist: ${RUNS_DIR}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
python -u "${PROJECT_DIR}/analyse_landslide_outputs.py" \
  --runs "${RUNS_DIR}" \
  --output "${OUTPUT_DIR}"

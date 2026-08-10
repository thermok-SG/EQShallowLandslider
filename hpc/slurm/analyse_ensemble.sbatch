#!/bin/bash
#SBATCH --job-name=landslide-analysis-ensemble
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
REGION="${REGION:-nepal}"
RUNS_DIR="${WORKSPACE}/runs/${REGION}/ensemble"
OUTPUT_DIR="${WORKSPACE}/analysis_output/${REGION}_ensemble"
OBSERVED_DIR="${WORKSPACE}/input_data/${REGION}"

module load conda
conda activate landlab_dev

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${PROJECT_DIR}"

if [[ ! -d "${RUNS_DIR}" ]]; then
  echo "Run directory does not exist: ${RUNS_DIR}" >&2
  exit 1
fi

COMPLETED_RUNS="$(find "${RUNS_DIR}" -type f -name manifest.json | wc -l | tr -d ' ')"
if [[ "${COMPLETED_RUNS}" -eq 0 ]]; then
  echo "No completed runs found beneath ${RUNS_DIR}" >&2
  exit 1
fi
echo "Analysing ${COMPLETED_RUNS} completed ${REGION} ensemble member(s)."

python -u "${PROJECT_DIR}/analyse_landslide_outputs.py" \
  --runs "${RUNS_DIR}" \
  --output "${OUTPUT_DIR}" \
  --observed-inventory "${OBSERVED_DIR}/measuredLandslides_all.csv" \
  --observed-zonal-stats "${OBSERVED_DIR}/measuredLandslides_all_ZonalStats.csv" \
  --min-observed-area 900

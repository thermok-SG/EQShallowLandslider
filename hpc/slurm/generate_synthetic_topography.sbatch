#!/bin/bash
#SBATCH --job-name=synthetic-topography
#SBATCH --partition=cpu_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

PROJECT_DIR="${EQSL_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_DIR}" || ! -f "${PROJECT_DIR}/generate_synthetic_topography.py" ]]; then
  echo "Submit from the repository root or export EQSL_PROJECT_DIR=/absolute/repository/path." >&2
  exit 1
fi
PROJECT_DIR="$(cd -- "${PROJECT_DIR}" && pwd)"
WORKSPACE="${PROJECT_DIR}/hpc/workspace"
OUTPUT_DIR="${OUTPUT_DIR:-input_data/dem/synthetic_ensemble}"

# Each seed is paired across the two regolith models. When this list changes,
# set the Slurm array upper bound to (2 * number_of_seeds) - 1.
read -r -a SEEDS <<< "${SYNTHETIC_SEEDS:-20260804 20260805 20260806}"
REGOLITH_MODELS=(space weathering_taylor)

TASK_ID="${SLURM_ARRAY_TASK_ID:?Run this file with sbatch as an array job}"
MODEL_COUNT="${#REGOLITH_MODELS[@]}"
TASK_COUNT="$(( ${#SEEDS[@]} * MODEL_COUNT ))"
if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
  echo "Array task ${TASK_ID} is outside the configured 0-$((TASK_COUNT - 1)) range" >&2
  exit 2
fi

SEED_INDEX="$((TASK_ID / MODEL_COUNT))"
MODEL_INDEX="$((TASK_ID % MODEL_COUNT))"
SEED="${SEEDS[SEED_INDEX]}"
REGOLITH_MODEL="${REGOLITH_MODELS[MODEL_INDEX]}"
NROWS_VALUE="${NROWS:-600}"
NCOLS_VALUE="${NCOLS:-800}"
SPACING_VALUE="${SPACING:-30}"
SPACING_LABEL="${SPACING_VALUE//./p}"
OUTPUT_BASENAME="synthetic_${REGOLITH_MODEL}_seed${SEED}_${NROWS_VALUE}x${NCOLS_VALUE}_${SPACING_LABEL}m.asc"

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
mkdir -p "${OUTPUT_DIR}"

echo "Generating model=${REGOLITH_MODEL} seed=${SEED} task=${TASK_ID}/${TASK_COUNT}"

python -u "${PROJECT_DIR}/generate_synthetic_topography.py" \
  --nrows "${NROWS_VALUE}" \
  --ncols "${NCOLS_VALUE}" \
  --spacing "${SPACING_VALUE}" \
  --iterations "${ITERATIONS:-1000}" \
  --timestep "${TIMESTEP:-250}" \
  --refinement-factor "${REFINEMENT_FACTOR:-2}" \
  --uplift-rate "${UPLIFT_RATE:-0.001}" \
  --rock-erodibility "${ROCK_ERODIBILITY:-2.5e-5}" \
  --sediment-erodibility "${SEDIMENT_ERODIBILITY:-2.5e-5}" \
  --initial-soil-depth "${INITIAL_SOIL_DEPTH:-0.5}" \
  --soil-production-maximum-rate "${SOIL_PRODUCTION_MAXIMUM_RATE:-1.0e-4}" \
  --soil-production-decay-depth "${SOIL_PRODUCTION_DECAY_DEPTH:-0.5}" \
  --soil-transport-velocity "${SOIL_TRANSPORT_VELOCITY:-0.02}" \
  --soil-transport-decay-depth "${SOIL_TRANSPORT_DECAY_DEPTH:-0.5}" \
  --critical-slope "${CRITICAL_SLOPE:-1.0}" \
  --seed "${SEED}" \
  --regolith-model "${REGOLITH_MODEL}" \
  --output "${OUTPUT_DIR}/${OUTPUT_BASENAME}"


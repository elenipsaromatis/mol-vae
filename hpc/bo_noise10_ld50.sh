#!/bin/bash
#PBS -N mol_bo_noise10_ld50
#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Mirrors bo/results/noise_10pct: 1000 iterations, 10% relative LD50
# noise, all 10 RANDOM_SEEDS, overlap pool, ucb + pareto + ei + random.
#
# Submit with:
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_noise10_ld50.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_noise10_ld50.sh}"

module purge
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"

CONDA_ENV="${CONDA_ENV:-mol-vae}"
conda activate "$CONDA_ENV"

PROJECT_ROOT="${PBS_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MLFLOW_ALLOW_FILE_STORE=true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python bo/run_bo.py \
  --checkpoint "$CHECKPOINT" \
  --out-dir bo/results_ld50/noise_10pct \
  --n-iterations 1000 \
  --noise-scale 0.1 \
  --selection ucb pareto ei \
  --run-random \
  --experiment-name bo-ld50-noise10-molkg

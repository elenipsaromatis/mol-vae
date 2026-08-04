#!/bin/bash
#PBS -N mol_bo_baseline_1000_ld50
#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Mirrors bo/results/baseline_1000: 1000 iterations, all 10 RANDOM_SEEDS,
# overlap candidate pool, ucb + pareto + ei + random.
# (Locally, the equivalent 1000-iter x 10-seed x 4-method noise sweep took
# ~9h on Apple Silicon MPS -- 20h walltime here is a generous buffer for an
# unknown CPU node; tighten once you've timed one run.)
#
# Submit with:
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_baseline_1000_ld50.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_baseline_1000_ld50.sh}"

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
  --out-dir bo/results_ld50/baseline_1000 \
  --n-iterations 1000 \
  --selection ucb pareto ei \
  --run-random \
  --experiment-name bo-ld50-1000-molkg

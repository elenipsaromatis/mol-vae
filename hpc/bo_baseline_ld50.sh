#!/bin/bash
#PBS -N mol_bo_baseline_ld50
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Mirrors bo/results/baseline_full: default 300 iterations, all 10
# RANDOM_SEEDS (11 22 33 44 55 66 77 88 99 110), overlap candidate pool,
# ucb + pareto + ei + random.
#
# Submit with (CHECKPOINT is required -- the new checkpoint from
# train_vae_ld50.sh):
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_baseline_ld50.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_baseline_ld50.sh}"

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
  --out-dir bo/results_ld50/baseline_full \
  --selection ucb pareto ei \
  --run-random \
  --experiment-name bo-ld50-baseline-molkg

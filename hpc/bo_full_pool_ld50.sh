#!/bin/bash
#PBS -N mol_bo_full_pool_ld50
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Mirrors bo/results/full_ld50_comparison: 300 iterations, seeds 11 22 33
# 44 only (matches what was actually run before -- not the full 10-seed
# RANDOM_SEEDS), candidate pool = full LD50_Zhu (not just the
# solubility x LD50 overlap), ucb + pareto + ei + random. Tests how well
# the overlap-trained latent space extrapolates.
#
# Submit with:
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_full_pool_ld50.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_full_pool_ld50.sh}"

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
  --out-dir bo/results_ld50/full_ld50_comparison \
  --pool full-ld50 \
  --seeds 11 22 33 44 \
  --selection ucb pareto ei \
  --run-random \
  --experiment-name bo-ld50-fullpool-molkg

#!/bin/bash
#PBS -N mol_bo_full_pool_pareto_expert
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Adds pareto_expert to bo/results_ld50/full_ld50_comparison: 300
# iterations, seeds 11 22 33 44 only (matches what was actually run,
# not the full 10-seed RANDOM_SEEDS), candidate pool = full LD50_Zhu.
# Writes into the SAME out-dir as hpc/bo_full_pool_ld50.sh, under
# seed_*/pareto_expert/ -- does NOT touch the existing ucb/pareto/ei/random
# results (see hpc/bo_baseline_ld50_pareto_expert.sh for why this is
# safe). Only pareto_expert runs here, so this is much faster than the
# 4-strategy original (12h) -- shorten walltime further once you've timed
# one run.
#
# --pool and --seeds MUST match the original run exactly -- they're part
# of what the shared initial_indices/pca depend on being reproduced from.
#
# Submit with the SAME checkpoint used for full_ld50_comparison:
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_full_pool_ld50_pareto_expert.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_full_pool_ld50_pareto_expert.sh}"

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
  --selection pareto_expert \
  --experiment-name bo-ld50-fullpool-molkg

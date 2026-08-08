#!/bin/bash
#PBS -N mol_bo_full_pool_pLD50_pareto_expert
#PBS -l walltime=18:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Adds pareto_expert to bo/results_pLD50/full_ld50_comparison
# (--objective minimize): 1000 iterations, all 10 RANDOM_SEEDS by
# default, candidate pool = full LD50_Zhu (not just the solubility x
# LD50 overlap). Writes into the SAME out-dir as hpc/bo_full_pool_pLD50.sh,
# under seed_*/pareto_expert/ -- does NOT touch the existing
# ucb/pareto/ei/random results there (see
# hpc/bo_baseline_1000_pLD50_pareto_expert.sh for why this is safe). Only
# pareto_expert runs here, so this is much faster than the 4-strategy
# original (~60h) -- shorten walltime further once you've timed one run.
#
# --pool full-ld50 MUST match the original run, and use the SAME
# checkpoint as the original full_ld50_comparison pLD50 run.
#
# Fresh run (all seeds):
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_full_pool_pLD50_pareto_expert.sh
#
# Resume/partial (SEEDS uses "+" as separator, e.g. "66+77+88" -- this
# cluster's qsub mangles -v values containing whitespace):
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS=66+77+88 hpc/bo_full_pool_pLD50_pareto_expert.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_full_pool_pLD50_pareto_expert.sh}"
SEEDS="${SEEDS:-}"
SEEDS="${SEEDS//+/ }"

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
  --out-dir bo/results_pLD50/full_ld50_comparison \
  --pool full-ld50 \
  --n-iterations 1000 \
  --objective minimize \
  ${SEEDS:+--seeds $SEEDS} \
  --selection pareto_expert \
  --experiment-name bo-pLD50-fullpool

#!/bin/bash
#PBS -N mol_bo_noise20_pLD50_pareto_expert
#PBS -l walltime=14:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Adds pareto_expert to bo/results_pLD50/noise_20pct (--objective
# minimize): 1000 iterations, 20% relative LD50 noise, all 10
# RANDOM_SEEDS by default, overlap pool. Writes into the SAME out-dir as
# hpc/bo_noise20_pLD50.sh, under seed_*/pareto_expert/ -- does NOT touch
# the existing ucb/pareto/ei/random results there (see
# hpc/bo_baseline_1000_pLD50_pareto_expert.sh for why this is safe). Only
# pareto_expert runs here, so this is much faster than the 4-strategy
# original (~48h) -- shorten walltime further once you've timed one run.
#
# --noise-scale MUST match the original run exactly (0.2), and use the
# SAME checkpoint as the original noise_20pct pLD50 run.
#
# Fresh run (all seeds):
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_noise20_pLD50_pareto_expert.sh
#
# Resume/partial (SEEDS uses "+" as separator, e.g. "66+77+88" -- this
# cluster's qsub mangles -v values containing whitespace):
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS=66+77+88 hpc/bo_noise20_pLD50_pareto_expert.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_noise20_pLD50_pareto_expert.sh}"
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
  --out-dir bo/results_pLD50/noise_20pct \
  --n-iterations 1000 \
  --noise-scale 0.2 \
  --objective minimize \
  ${SEEDS:+--seeds $SEEDS} \
  --selection pareto_expert \
  --experiment-name bo-pLD50-noise20

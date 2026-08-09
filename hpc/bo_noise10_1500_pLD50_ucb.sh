#!/bin/bash
#PBS -N mol_bo_noise10_1500_ucb
#PBS -l walltime=18:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Mirrors bo_noise10_pLD50.sh but --n-iterations 1500, 10% relative LD50 noise.
# Strategy: ucb (also carries --run-random for this dir's random baseline, all 10 seeds' worth accumulate across this strategy's batches)
#
# FRESH run (not a resume) into a NEW out-dir -- the existing
# bo/results_pLD50/noise_10pct/ (1000 iterations, already analyzed under
# analysis/TRUE_analysis/) is untouched. This intentionally does NOT try
# to continue/resume the 1000-iteration trace: splicing a resumed run
# onto a fresh one isn't guaranteed bit-identical to a single continuous
# run on a shared cluster (see thesis discussion -- different compute
# nodes can produce tiny floating-point differences that cascade through
# GP fits and flip close acquisition-function ties), so every seed here
# runs all 1500 iterations as one continuous process.
#
# Split into seed batches (not all 10 at once) because iteration cost is
# superlinear in the growing observed set -- 1500 iterations costs
# roughly 3x a 1000-iteration run, not 1.5x, so 10 seeds at once here
# would be impractically long for one job.
#
# SEEDS is REQUIRED and uses "+" as separator (e.g. "11+22+33") -- this
# cluster's qsub mangles -v values containing whitespace regardless of
# shell quoting. Converted back to space-separated below before being
# passed to run_bo.py's --seeds (nargs="+"). Submit once per batch:
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS=11+22+33 hpc/bo_noise10_1500_pLD50_ucb.sh
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS=44+55+66 hpc/bo_noise10_1500_pLD50_ucb.sh
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS=77+88+99 hpc/bo_noise10_1500_pLD50_ucb.sh
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS=110 hpc/bo_noise10_1500_pLD50_ucb.sh
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS=11+22+33 hpc/bo_noise10_1500_pLD50_ucb.sh}"
: "${SEEDS:?Set SEEDS via: qsub -v CHECKPOINT=...,SEEDS=11+22+33 hpc/bo_noise10_1500_pLD50_ucb.sh (batches: 11+22+33, 44+55+66, 77+88+99, 110)}"
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
  --out-dir bo/results_pLD50/noise_10pct_1500 \
  --n-iterations 1500 \
  --objective minimize \
  --seeds $SEEDS \
  --selection ucb \
  --run-random \
  --noise-scale 0.1 \
  --experiment-name bo-pLD50-1500-noise10-ucb

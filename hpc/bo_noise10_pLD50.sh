#!/bin/bash
#PBS -N mol_bo_noise10_pLD50
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# pLD50 (--objective minimize) counterpart of bo_noise10_ld50.sh. Mirrors
# bo/results_ld50/noise_10pct but minimizing pLD50: 1000 iterations, 10%
# relative LD50 noise, all 10 RANDOM_SEEDS by default, overlap pool,
# ucb + pareto + ei + random.
#
# 48h walltime is sized off the one measured rate available (~3.9h/seed
# from the maximize-objective noise_10pct run) x 10 seeds + margin. If
# this queue caps walltime below 48h, qsub will reject the submission --
# tell me the cap and I'll resize.
#
# Fresh run (all seeds):
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_noise10_pLD50.sh
#
# Resume after a walltime kill (or any other interruption) -- resubmit
# this SAME script, just restrict to the seeds still missing under
# bo/results_pLD50/noise_10pct/ (results merge into the same out-dir;
# already-finished seeds are left untouched):
#   qsub -l walltime=18:00:00 -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS=66+77+88 hpc/bo_noise10_pLD50.sh
#
# SEEDS uses "+" as the separator (e.g. "66+77+88"), not spaces -- this
# cluster's qsub mangles -v values containing whitespace regardless of
# shell quoting. Converted back to space-separated below before being
# passed to run_bo.py's --seeds (nargs="+").

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_noise10_pLD50.sh}"
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
  --out-dir bo/results_pLD50/noise_10pct \
  --n-iterations 1000 \
  --noise-scale 0.1 \
  --objective minimize \
  ${SEEDS:+--seeds $SEEDS} \
  --selection ucb pareto ei \
  --run-random \
  --experiment-name bo-pLD50-noise10

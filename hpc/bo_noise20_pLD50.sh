#!/bin/bash
#PBS -N mol_bo_noise20_pLD50
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# pLD50 (--objective minimize) counterpart of bo_noise20_ld50.sh. Mirrors
# bo/results_ld50/noise_20pct but minimizing pLD50: 1000 iterations, 20%
# relative LD50 noise, all 10 RANDOM_SEEDS by default, overlap pool,
# ucb + pareto + ei + random.
#
# 48h walltime is sized off the one measured rate available (~4.2h/seed
# from the maximize-objective noise_20pct run) x 10 seeds + margin. If
# this queue caps walltime below 48h, qsub will reject the submission --
# tell me the cap and I'll resize.
#
# Fresh run (all seeds):
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_noise20_pLD50.sh
#
# Resume after a walltime kill (or any other interruption) -- resubmit
# this SAME script, just restrict to the seeds still missing under
# bo/results_pLD50/noise_20pct/ (results merge into the same out-dir;
# already-finished seeds are left untouched):
#   qsub -l walltime=18:00:00 -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS="66 77 88" hpc/bo_noise20_pLD50.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_noise20_pLD50.sh}"
SEEDS="${SEEDS:-}"

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
  --selection ucb pareto ei \
  --run-random \
  --experiment-name bo-pLD50-noise20

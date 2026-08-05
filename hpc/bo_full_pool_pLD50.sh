#!/bin/bash
#PBS -N mol_bo_full_pool_pLD50
#PBS -l walltime=60:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# pLD50 (--objective minimize) counterpart of bo_full_pool_ld50.sh, but
# scaled up to match the other pLD50 sweeps: 1000 iterations, all 10
# RANDOM_SEEDS by default (the original maximize run only covered seeds
# 11 22 33 44 at 300 iterations -- this one covers the full sweep since
# it's running on the cluster). Candidate pool = full LD50_Zhu (not just
# the solubility x LD50 overlap), ucb + pareto + ei + random. Tests how
# well the overlap-trained latent space extrapolates when minimizing
# pLD50.
#
# 60h walltime is a padded guess, not a measurement: the full-ld50 pool
# is larger than the overlap pool, so per-iteration GP fit/predict cost
# is higher than bo_baseline_1000_pLD50.sh's ~4.2h/seed, and there's no
# prior full-pool x 1000-iteration run to measure from. If this queue
# caps walltime below 60h, qsub will reject the submission -- tell me
# the cap and I'll resize. Time the first seed and be ready to resume.
#
# Fresh run (all seeds):
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_full_pool_pLD50.sh
#
# Resume after a walltime kill (or any other interruption) -- resubmit
# this SAME script, just restrict to the seeds still missing under
# bo/results_pLD50/full_ld50_comparison/ (results merge into the same
# out-dir; already-finished seeds are left untouched):
#   qsub -l walltime=18:00:00 -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS="66 77 88" hpc/bo_full_pool_pLD50.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_full_pool_pLD50.sh}"
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
  --out-dir bo/results_pLD50/full_ld50_comparison \
  --pool full-ld50 \
  --n-iterations 1000 \
  --objective minimize \
  ${SEEDS:+--seeds $SEEDS} \
  --selection ucb pareto ei \
  --run-random \
  --experiment-name bo-pLD50-fullpool

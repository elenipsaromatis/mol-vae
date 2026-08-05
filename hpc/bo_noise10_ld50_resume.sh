#!/bin/bash
#PBS -N mol_bo_noise10_resume
#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Resume job for bo_noise10_ld50.sh. Covers only remaining seeds; writes
# into the SAME out-dir so results merge with what's already done.
# Measured rate on this queue: ~3.5s/iteration -> 1 seed (4 methods x 1000
# iters) takes roughly 3.9h. Size -l walltime= on the qsub command line to
# (num_seeds_in_this_job * ~4.2h) for a safety margin.
#
# Submit with:
#   qsub -l walltime=16:00:00 -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS="66 77 88" hpc/bo_noise10_ld50_resume.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via -v CHECKPOINT=...}"
: "${SEEDS:?Set SEEDS via -v SEEDS=\"66 77 88\" (space-separated, quoted)}"

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
  --out-dir bo/results_ld50/noise_10pct \
  --n-iterations 1000 \
  --noise-scale 0.1 \
  --seeds $SEEDS \
  --selection ucb pareto ei \
  --run-random \
  --experiment-name bo-ld50-noise10-molkg

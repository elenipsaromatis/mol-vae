#!/bin/bash
#PBS -N mol_bo_noise20_resume
#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Resume job for bo_noise20_ld50.sh. Covers only remaining seeds; writes
# into the SAME out-dir so results merge with what's already done.
# Measured rate on this queue: ~3.8s/iteration -> 1 seed (4 methods x 1000
# iters) takes roughly 4.3h. Size -l walltime= on the qsub command line to
# (num_seeds_in_this_job * ~4.5h) for a safety margin.
#
# Submit with:
#   qsub -l walltime=18:00:00 -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS="55 66 77" hpc/bo_noise20_ld50_resume.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via -v CHECKPOINT=...}"
: "${SEEDS:?Set SEEDS via -v SEEDS=\"55 66 77\" (space-separated, quoted)}"

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
  --out-dir bo/results_ld50/noise_20pct \
  --n-iterations 1000 \
  --noise-scale 0.2 \
  --seeds $SEEDS \
  --selection ucb pareto ei \
  --run-random \
  --experiment-name bo-ld50-noise20-molkg

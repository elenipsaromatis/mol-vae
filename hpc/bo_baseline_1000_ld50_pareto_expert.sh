#!/bin/bash
#PBS -N mol_bo_baseline_1000_pareto_expert
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Adds pareto_expert to bo/results_ld50/baseline_1000: 1000 iterations,
# all 10 RANDOM_SEEDS, overlap candidate pool. Writes into the SAME
# out-dir as hpc/bo_baseline_1000_ld50.sh, under seed_*/pareto_expert/ --
# does NOT touch the existing ucb/pareto/ei/random results (see
# hpc/bo_baseline_ld50_pareto_expert.sh for why this is safe). Only
# pareto_expert runs here, so this is much faster than the 4-strategy
# original (20h) -- shorten walltime further once you've timed one run.
#
# Submit with the SAME checkpoint used for baseline_1000:
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_baseline_1000_ld50_pareto_expert.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via: qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/bo_baseline_1000_ld50_pareto_expert.sh}"

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
  --out-dir bo/results_ld50/baseline_1000 \
  --n-iterations 1000 \
  --selection pareto_expert \
  --experiment-name bo-ld50-1000-molkg

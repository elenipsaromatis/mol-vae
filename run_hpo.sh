#!/bin/bash
#PBS -N mol_vae_hpo
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

set -euo pipefail

module purge
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"

CONDA_ENV="${CONDA_ENV:-mol-vae}"
conda activate "$CONDA_ENV"

PROJECT_ROOT="${PBS_O_WORKDIR:-$PWD}"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MLFLOW_ALLOW_FILE_STORE=true

# Prevent thread oversubscription across 4 parallel Optuna workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python hpo.py --n_trials 50 --n_epochs 50

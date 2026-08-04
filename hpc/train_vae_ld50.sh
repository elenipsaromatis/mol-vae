#!/bin/bash
#PBS -N mol_vae_train_ld50
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Runs train_vae.py -- NOT train.py. train_vae.py is what actually produced
# the vae_solubility_<hash>_best.pth checkpoints that bo/run_bo.py and the
# analysis/ scripts default to. Its loss only supervises reconstruction +
# solubility (train_vae.py:78); LD50 gets no gradient, so this run will not
# change the encoder or solubility-head weights at all versus before your
# data.py fix. It's still required because build_dataloaders_multitask
# recomputes ld50_mean/ld50_std fresh from the current data.py every call,
# and train_vae.py bakes those into the checkpoint's "standardise" dict --
# bo/problem.py depends on that dict matching the current data.py, or the
# raw LD50 values it reports get silently corrupted (mismatched
# standardize/destandardize stats). This run is about refreshing that
# metadata, not learning anything new -- should be quick.
#
# Submit with:
#   qsub hpc/train_vae_ld50.sh
#
# After it finishes, note the new checkpoint filename printed to stdout,
# e.g. checkpoints/vae_solubility_<hash>_best.pth -- every BO job below
# needs that path.

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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python train_vae.py

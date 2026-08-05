#!/bin/bash
#PBS -N mol_bo_baseline_resume
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe

# Resume job for bo_baseline_ld50.sh: covers only the seeds that weren't
# finished (walltime-killed) in the original run. Writes into the SAME
# out-dir, so results merge with the seeds already completed there --
# does NOT redo anything already done.
#
# Submit with (SEEDS = space-separated list of remaining seeds; override
# -l walltime= on the qsub command line if this job's seed count differs
# from what 4h was sized for):
#   qsub -v CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth,SEEDS="99 110" hpc/bo_baseline_ld50_resume.sh

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT via -v CHECKPOINT=...}"
: "${SEEDS:?Set SEEDS via -v SEEDS=\"99 110\" (space-separated, quoted)}"

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
  --out-dir bo/results_ld50/baseline_full \
  --seeds $SEEDS \
  --selection ucb pareto ei \
  --run-random \
  --experiment-name bo-ld50-baseline-molkg

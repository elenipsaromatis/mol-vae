#!/bin/bash
# Submits every seed-batch job for the fresh 1500-iteration pLD50 sweep.
# Run this ONCE from the repo root on the cluster after `git pull`.
#
# Usage: CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/submit_1500_sweep.sh
set -euo pipefail
: "${CHECKPOINT:?Set CHECKPOINT via: CHECKPOINT=checkpoints/vae_solubility_XXXXXXXX_best.pth hpc/submit_1500_sweep.sh}"

submitted=0
failed=0

echo "Submitting bo_baseline_1500_pLD50_ucb.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_baseline_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_ucb.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_ucb.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_baseline_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_ucb.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_ucb.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_baseline_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_ucb.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_ucb.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_baseline_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_ucb.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_pareto.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_baseline_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_pareto.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_pareto.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_baseline_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_pareto.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_pareto.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_baseline_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_pareto.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_pareto.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_baseline_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_pareto.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_ei.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_baseline_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_ei.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_ei.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_baseline_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_ei.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_ei.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_baseline_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_ei.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_ei.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_baseline_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_ei.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_pareto_expert.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_baseline_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_pareto_expert.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_pareto_expert.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_baseline_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_pareto_expert.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_pareto_expert.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_baseline_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_pareto_expert.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_baseline_1500_pLD50_pareto_expert.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_baseline_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_baseline_1500_pLD50_pareto_expert.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_ucb.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_noise10_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_ucb.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_ucb.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_noise10_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_ucb.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_ucb.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_noise10_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_ucb.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_ucb.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_noise10_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_ucb.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_pareto.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_noise10_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_pareto.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_pareto.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_noise10_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_pareto.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_pareto.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_noise10_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_pareto.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_pareto.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_noise10_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_pareto.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_ei.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_noise10_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_ei.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_ei.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_noise10_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_ei.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_ei.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_noise10_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_ei.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_ei.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_noise10_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_ei.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_pareto_expert.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_noise10_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_pareto_expert.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_pareto_expert.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_noise10_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_pareto_expert.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_pareto_expert.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_noise10_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_pareto_expert.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise10_1500_pLD50_pareto_expert.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_noise10_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise10_1500_pLD50_pareto_expert.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_ucb.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_noise20_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_ucb.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_ucb.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_noise20_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_ucb.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_ucb.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_noise20_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_ucb.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_ucb.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_noise20_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_ucb.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_pareto.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_noise20_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_pareto.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_pareto.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_noise20_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_pareto.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_pareto.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_noise20_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_pareto.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_pareto.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_noise20_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_pareto.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_ei.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_noise20_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_ei.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_ei.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_noise20_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_ei.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_ei.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_noise20_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_ei.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_ei.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_noise20_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_ei.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_pareto_expert.sh SEEDS=11+22+33"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22+33 hpc/bo_noise20_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_pareto_expert.sh SEEDS=11+22+33" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_pareto_expert.sh SEEDS=44+55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=44+55+66 hpc/bo_noise20_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_pareto_expert.sh SEEDS=44+55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_pareto_expert.sh SEEDS=77+88+99"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88+99 hpc/bo_noise20_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_pareto_expert.sh SEEDS=77+88+99" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_noise20_1500_pLD50_pareto_expert.sh SEEDS=110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=110 hpc/bo_noise20_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_noise20_1500_pLD50_pareto_expert.sh SEEDS=110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_ucb.sh SEEDS=11+22"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22 hpc/bo_fullpool_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_ucb.sh SEEDS=11+22" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_ucb.sh SEEDS=33+44"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=33+44 hpc/bo_fullpool_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_ucb.sh SEEDS=33+44" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_ucb.sh SEEDS=55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=55+66 hpc/bo_fullpool_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_ucb.sh SEEDS=55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_ucb.sh SEEDS=77+88"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88 hpc/bo_fullpool_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_ucb.sh SEEDS=77+88" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_ucb.sh SEEDS=99+110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=99+110 hpc/bo_fullpool_1500_pLD50_ucb.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_ucb.sh SEEDS=99+110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_pareto.sh SEEDS=11+22"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22 hpc/bo_fullpool_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_pareto.sh SEEDS=11+22" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_pareto.sh SEEDS=33+44"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=33+44 hpc/bo_fullpool_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_pareto.sh SEEDS=33+44" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_pareto.sh SEEDS=55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=55+66 hpc/bo_fullpool_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_pareto.sh SEEDS=55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_pareto.sh SEEDS=77+88"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88 hpc/bo_fullpool_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_pareto.sh SEEDS=77+88" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_pareto.sh SEEDS=99+110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=99+110 hpc/bo_fullpool_1500_pLD50_pareto.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_pareto.sh SEEDS=99+110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_ei.sh SEEDS=11+22"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22 hpc/bo_fullpool_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_ei.sh SEEDS=11+22" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_ei.sh SEEDS=33+44"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=33+44 hpc/bo_fullpool_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_ei.sh SEEDS=33+44" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_ei.sh SEEDS=55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=55+66 hpc/bo_fullpool_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_ei.sh SEEDS=55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_ei.sh SEEDS=77+88"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88 hpc/bo_fullpool_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_ei.sh SEEDS=77+88" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_ei.sh SEEDS=99+110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=99+110 hpc/bo_fullpool_1500_pLD50_ei.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_ei.sh SEEDS=99+110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_pareto_expert.sh SEEDS=11+22"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=11+22 hpc/bo_fullpool_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_pareto_expert.sh SEEDS=11+22" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_pareto_expert.sh SEEDS=33+44"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=33+44 hpc/bo_fullpool_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_pareto_expert.sh SEEDS=33+44" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_pareto_expert.sh SEEDS=55+66"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=55+66 hpc/bo_fullpool_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_pareto_expert.sh SEEDS=55+66" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_pareto_expert.sh SEEDS=77+88"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=77+88 hpc/bo_fullpool_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_pareto_expert.sh SEEDS=77+88" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Submitting bo_fullpool_1500_pLD50_pareto_expert.sh SEEDS=99+110"
if qsub -v CHECKPOINT="$CHECKPOINT",SEEDS=99+110 hpc/bo_fullpool_1500_pLD50_pareto_expert.sh; then
  submitted=$((submitted+1))
else
  echo "  FAILED: bo_fullpool_1500_pLD50_pareto_expert.sh SEEDS=99+110" >&2
  failed=$((failed+1))
fi
sleep 1

echo "Done: $submitted submitted, $failed failed."
if [ "$failed" -gt 0 ]; then
  echo "Some qsub calls failed (see FAILED lines above, e.g. a per-user queue-slot cap). Do NOT just re-run this whole script -- it has no memory of what already succeeded and would resubmit those as duplicates. Instead, copy the individual 'qsub -v CHECKPOINT=...,SEEDS=... hpc/<script>' command for each FAILED line above and submit just those by hand." >&2
  exit 1
fi

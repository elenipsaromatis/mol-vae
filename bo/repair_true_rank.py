"""Repair `selected_true_ld50_rank` / `selected_is_true_top_10` in existing
bo_trace.csv files written before the run_bo.py fix (see the "True LD50
rank/top-10 respect the objective direction" comment in run_bo.py).

Bug: the UCB/Pareto/EI loop hardcoded `ascending=False` when ranking
`problem.y_raw_true` (i.e. always "higher LD50 ranks better"), instead of
respecting `--objective`. For `--objective minimize` runs this scored "true
top 10" against the wrong tail of the distribution -- BO's actual molecule
selection was correct (mu/sigma/ucb/ei/best_ld50_after all correctly
minimized), only this one derived reporting column was wrong. The
random-search loop already computed this correctly, so its rows don't need
repair (re-repairing them is harmless -- same result).

This does NOT require re-running BO. `selected_recipe_id` (the candidate's
integer position in the pool) and the correct `selected_ld50_true_raw`
value are already in every row; we only need to reload the same candidate
pool once (no GP fitting) to get the ground-truth rank of every pool
member, then look up each row's corrected rank/top-10 flag by
selected_recipe_id.

Usage:
    python bo/repair_true_rank.py --results-dir bo/results_pLD50/baseline_1000 --checkpoint checkpoints/vae_solubility_e919f3e0_best.pth --objective minimize
    python bo/repair_true_rank.py --results-dir bo/results_pLD50/full_ld50_comparison --checkpoint checkpoints/vae_solubility_e919f3e0_best.pth --objective minimize --pool full-ld50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bo.problem import load_bo_problem, load_bo_problem_full_ld50


def repair_file(trace_path: Path, rank_series: pd.Series, backup: bool) -> tuple[int, int]:
    df = pd.read_csv(trace_path)
    if "selected_recipe_id" not in df.columns:
        print(f"  skip {trace_path} (no selected_recipe_id column)")
        return 0, 0

    old_rank = df["selected_true_ld50_rank"].copy()
    old_top10 = df["selected_is_true_top_10"].copy()

    df["selected_true_ld50_rank"] = df["selected_recipe_id"].map(rank_series)
    df["selected_is_true_top_10"] = df["selected_true_ld50_rank"] <= 10

    n_changed_rank = int((df["selected_true_ld50_rank"] != old_rank).sum())
    n_changed_top10 = int((df["selected_is_true_top_10"] != old_top10).sum())

    if n_changed_rank or n_changed_top10:
        if backup:
            trace_path.with_suffix(".csv.bak").write_bytes(trace_path.read_bytes())
        df.to_csv(trace_path, index=False)

    return n_changed_rank, n_changed_top10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--objective", choices=["maximize", "minimize"], required=True)
    parser.add_argument("--pool", choices=["overlap", "full-ld50"], default="overlap")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a .csv.bak copy before overwriting each fixed file.",
    )
    args = parser.parse_args()

    print(f"Loading candidate pool ({args.pool}) from {args.checkpoint} ...")
    if args.pool == "full-ld50":
        problem = load_bo_problem_full_ld50(
            args.checkpoint, batch_size=args.batch_size, device=args.device
        )
    else:
        problem = load_bo_problem(
            args.checkpoint, batch_size=args.batch_size, device=args.device
        )

    rank_ascending = args.objective == "minimize"
    rank_series = pd.Series(problem.y_raw_true).rank(method="min", ascending=rank_ascending)
    print(
        f"Pool size: {len(rank_series)} molecules. objective={args.objective} "
        f"(rank ascending={rank_ascending})"
    )

    trace_files = sorted(args.results_dir.glob("seed_*/*/bo_trace.csv"))
    if not trace_files:
        raise FileNotFoundError(f"No bo_trace.csv files found under {args.results_dir}")

    total_changed = 0
    for trace_path in trace_files:
        n_rank, n_top10 = repair_file(trace_path, rank_series, backup=not args.no_backup)
        status = f"{n_rank} rank values, {n_top10} top10 flags changed"
        print(f"{trace_path}: {status}")
        total_changed += n_rank

    print(f"\nDone. {total_changed} total rank values corrected across {len(trace_files)} files.")
    if total_changed and not args.no_backup:
        print("Originals backed up alongside each file as *.csv.bak.")


if __name__ == "__main__":
    main()

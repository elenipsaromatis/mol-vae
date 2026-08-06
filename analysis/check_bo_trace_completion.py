"""Sanity check: confirm every seed/method under a BO results directory has
a complete bo_trace.csv (exactly --n-iterations rows), not one left
truncated by a walltime kill.

run_bo.py writes bo_trace.csv incrementally, one row per iteration, and a
resumed seed is rerun from iteration 0 (not resumed mid-loop). A seed that
was interrupted mid-run and never resubmitted leaves behind a short file
that analysis/compare_bo_baselines.py will silently include, quietly
shrinking the effective seed count at the later iterations of a
convergence plot.

Usage:
    python analysis/check_bo_trace_completion.py --results-dir bo/results_ld50/baseline_1000 --n-iterations 1000
    python analysis/check_bo_trace_completion.py --results-dir bo/results_ld50/full_ld50_comparison --n-iterations 300 --seeds 11 22 33 44
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RANDOM_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
METHODS = ["ucb", "pareto", "ei", "random"]


def check(results_dir: Path, n_iterations: int, seeds: list[int], methods: list[str]) -> int:
    missing = []
    incomplete = []
    ok = []

    for seed in seeds:
        for method in methods:
            trace_path = results_dir / f"seed_{seed}" / method / "bo_trace.csv"
            if not trace_path.exists():
                missing.append((seed, method))
                continue
            n_rows = len(pd.read_csv(trace_path))
            if n_rows < n_iterations:
                incomplete.append((seed, method, n_rows))
            else:
                ok.append((seed, method, n_rows))

    print(f"\n=== {results_dir} (expecting {n_iterations} iterations) ===")
    print(f"complete:   {len(ok)}/{len(seeds) * len(methods)}")

    if incomplete:
        print(f"INCOMPLETE ({len(incomplete)}):")
        for seed, method, n_rows in incomplete:
            print(f"  seed_{seed}/{method}: {n_rows}/{n_iterations} rows")

    if missing:
        print(f"MISSING ({len(missing)}):")
        for seed, method in missing:
            print(f"  seed_{seed}/{method}: no bo_trace.csv")

    if not incomplete and not missing:
        print("All seed/method traces complete.")

    # Suggest a ready-to-paste resume seed list.
    bad_seeds = sorted({s for s, _ in missing} | {s for s, _, _ in incomplete})
    if bad_seeds:
        print(f"Seeds needing resume: {' '.join(str(s) for s in bad_seeds)}")

    return len(incomplete) + len(missing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--n-iterations", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_bad = check(args.results_dir, args.n_iterations, args.seeds, args.methods)
    raise SystemExit(1 if n_bad else 0)

# TRUE_analysis: minimizing pLD50

This folder mirrors `analysis/figures/` and `analysis/tables/`, but for the
BO sweeps run with `--objective minimize` instead of the default
`maximize`. The objective column BO operates on is pLD50
(`log(1/LD50)`, mol/kg — see the `switch back ld50 units to log(1/ld50)`
commit), so minimizing it is exactly "minimize mu of pLD50" while Pareto
selection (which already balances predicted mean against GP uncertainty)
covers the "maximize std" half — there's no separate flag for that.

Raw BO output lives in `bo/results_pLD50/` (parallel to `bo/results_ld50/`),
produced by the HPC scripts in `hpc/bo_*_pLD50*.sh`. Once a sweep's results
are pulled back from the cluster, populate the matching subfolder here by
re-running the existing analysis scripts against the new results dir:

| Subfolder | Source BO results | Analysis command |
|---|---|---|
| `pLD50_baseline_1000` | `bo/results_pLD50/baseline_1000` | `python analysis/compare_bo_baselines.py --results-dir bo/results_pLD50/baseline_1000 --figures-dir analysis/TRUE_analysis/figures/pLD50_baseline_1000 --tables-dir analysis/TRUE_analysis/tables/pLD50_baseline_1000` |
| `pLD50_noise_10` | `bo/results_pLD50/noise_10pct` | `python analysis/compare_bo_baselines.py --results-dir bo/results_pLD50/noise_10pct --figures-dir analysis/TRUE_analysis/figures/pLD50_noise_10 --tables-dir analysis/TRUE_analysis/tables/pLD50_noise_10` |
| `pLD50_noise_20` | `bo/results_pLD50/noise_20pct` | `python analysis/compare_bo_baselines.py --results-dir bo/results_pLD50/noise_20pct --figures-dir analysis/TRUE_analysis/figures/pLD50_noise_20 --tables-dir analysis/TRUE_analysis/tables/pLD50_noise_20` |
| `pLD50_noise_robustness` | the three dirs above | `python analysis/compare_noise_robustness.py --baseline-dir bo/results_pLD50/baseline_1000 --noise10-dir bo/results_pLD50/noise_10pct --noise20-dir bo/results_pLD50/noise_20pct --figures-dir analysis/TRUE_analysis/figures/pLD50_noise_robustness` (note: this script only writes figures, no tables dir) |
| `pLD50_full_pool_comparison` | `bo/results_pLD50/full_ld50_comparison` | `python analysis/compare_bo_baselines.py --results-dir bo/results_pLD50/full_ld50_comparison --figures-dir analysis/TRUE_analysis/figures/pLD50_full_pool_comparison --tables-dir analysis/TRUE_analysis/tables/pLD50_full_pool_comparison` |

Unlike the maximize-objective version, `full_ld50_comparison` here runs the
**full 1000-iteration x all-10-seed sweep** (not the original 300-iteration,
4-seed subset) since it's running on the cluster.

## `pareto_expert`: a fake human-in-the-loop expert

Alongside `ucb`/`pareto`/`ei`/`random`, `bo/run_bo.py` supports a fifth
selection strategy, `pareto_expert`. It builds the exact same non-dominated
(mu, sigma) front as `pareto`, but instead of picking the front point closest
to the utopia point, it picks whichever front point has the best true
(noiseless) pLD50 -- a stand-in for a human expert who could inspect the
candidates on the front and just knows which one is genuinely best, rather
than a geometric proxy for that judgement.

Because `bo/run_bo.py` recomputes `initial_indices`/PCA deterministically
from `checkpoint + pool + noise-scale + n-initial + seed`, and writes each
strategy to its own `seed_*/<selection>/` folder, `pareto_expert` can be run
**on its own**, after the fact, without redoing any of the `ucb`/`pareto`/
`ei`/`random` results already sitting in a results dir -- as long as every
other flag matches the original run exactly (checkpoint above all).

The follow-up HPC jobs live in `hpc/bo_*_pLD50_pareto_expert.sh` (one per
subfolder below, `--objective minimize` baked in, matching `--noise-scale`/
`--pool`/`--n-iterations` per dir):

| Subfolder | Pareto-Expert job script |
|---|---|
| `pLD50_baseline_1000` | `hpc/bo_baseline_1000_pLD50_pareto_expert.sh` |
| `pLD50_noise_10` | `hpc/bo_noise10_pLD50_pareto_expert.sh` |
| `pLD50_noise_20` | `hpc/bo_noise20_pLD50_pareto_expert.sh` |
| `pLD50_full_pool_comparison` | `hpc/bo_full_pool_pLD50_pareto_expert.sh` |

Submit with the same checkpoint as the original sweep, e.g.:

```
qsub -v CHECKPOINT=checkpoints/vae_solubility_e919f3e0_best.pth hpc/bo_baseline_1000_pLD50_pareto_expert.sh
```

Once `seed_*/pareto_expert/bo_trace.csv` is pulled back into the matching
`bo/results_pLD50/*` folder, `analysis/compare_bo_baselines.py` picks up
`pareto_expert` automatically (labelled "Pareto-Expert") and adds it to every
figure and table alongside UCB/Pareto/EI/Random -- no code changes needed.
`pLD50_noise_robustness` (via `analysis/compare_noise_robustness.py`) does
not include it, since that script only ever compares `pareto`/`ucb`/`ei`
across noise levels.

**Whether to actually include Pareto-Expert in each subfolder above is
still an open decision** (as of 2026-08-09), so the two versions are kept in
separate folders rather than picking one:

- `pLD50_<name>` -- the canonical folder from the table above, generated
  with `--exclude-methods pareto_expert` so it stays the original
  UCB/Pareto/EI/Random quartet even after `pareto_expert` results exist on
  disk.
- `pLD50_<name>_pareto_expert` -- the same results dir, generated *without*
  `--exclude-methods`, so it includes Pareto-Expert as a fifth method.

e.g. for `baseline_1000`:

```
python analysis/compare_bo_baselines.py --results-dir bo/results_pLD50/baseline_1000 --figures-dir analysis/TRUE_analysis/figures/pLD50_baseline_1000 --tables-dir analysis/TRUE_analysis/tables/pLD50_baseline_1000 --objective minimize --exclude-methods pareto_expert

python analysis/compare_bo_baselines.py --results-dir bo/results_pLD50/baseline_1000 --figures-dir analysis/TRUE_analysis/figures/pLD50_baseline_1000_pareto_expert --tables-dir analysis/TRUE_analysis/tables/pLD50_baseline_1000_pareto_expert --objective minimize
```

Once the decision is made, the `_pareto_expert` folder (if rejected) or the
`--exclude-methods` flag on the canonical command (if accepted) can go away.

## Not included: `ld50_extrapolation`

`analysis/check_ld50_extrapolation.py` fits a GP on the overlap and reports
how well it predicts held-out overlap molecules vs. the rest of LD50_Zhu.
It never runs BO acquisition and doesn't take an `--objective` flag, so its
output is identical regardless of minimize vs. maximize. See the existing
results in `analysis/figures/ld50_extrapolation/` and
`analysis/tables/ld50_extrapolation/` — no need to duplicate it here.

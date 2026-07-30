"""Relative-noise generator for the LD50 noise-robustness experiment.

LD50_noisy = LD50_true * (1 + epsilon), epsilon ~ N(0, scale^2).

`epsilon` is drawn once, as a single fixed-length vector over the whole
candidate pool, from a generator seeded at `seed_run + 1`. `seed_run` is
fixed at 100 for this experiment, so the noise realisation is identical
across every BO seed (11, 22, ..., 110) and selection strategy -- it is
decoupled from whichever seed happens to be driving a given BO run.
"""

import dataclasses

import numpy as np

from bo.problem import BOProblem


SEED_RUN = 100


def generate_relative_noise(n: int, scale: float, seed_run: int = SEED_RUN) -> np.ndarray:
    """Draw `n` iid N(0, scale^2) relative-noise values, reproducibly."""
    rng = np.random.default_rng(seed_run + 1)
    return rng.normal(loc=0.0, scale=scale, size=n)


def apply_relative_noise(
    problem: BOProblem, scale: float, seed_run: int = SEED_RUN
) -> BOProblem:
    """Return a copy of `problem` where BO observes noisy LD50 values.

    `problem.y_raw_true` is left untouched. `problem.y`/`problem.y_raw` --
    what the GP is fit on and what drives acquisition/best-observed
    tracking -- become the noisy values. A `scale` of 0.0 returns `problem`
    unchanged.
    """
    if scale == 0.0:
        return problem

    epsilon = generate_relative_noise(
        n=problem.y_raw_true.shape[0], scale=scale, seed_run=seed_run
    )

    y_raw_noisy = problem.y_raw_true * (1.0 + epsilon)
    y_noisy = (y_raw_noisy - problem.ld50_mean) / problem.ld50_std

    return dataclasses.replace(problem, y=y_noisy, y_raw=y_raw_noisy)

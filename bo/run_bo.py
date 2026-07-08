"""Minimal candidate-selection Bayesian optimisation over VAE latent embeddings.

Steps:
1. Load VAE latent embeddings and LD50 labels.
2. Randomly observe 20 molecules.
3. Fit a GP on observed LD50 values.
4. Predict mu and sigma for all molecules.
5. Compute UCB.
6. Plot all molecules on mu/sigma axes.
7. Keep non-dominated unobserved candidates.
8. Use ParetoSelector to pick the candidate closest to the utopia point.
9. Add that selected candidate to the observed set.
10. Repeat.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from bo.problem import load_bo_problem
from paretodo.selection.selector import ParetoSelector


CHECKPOINT = Path("checkpoints/vae_solubility_70a6568a_best.pth")

def fit_gp(X_observed, y_observed, seed):
    """Fit a Gaussian process on the currently observed molecules."""
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=seed,
        n_restarts_optimizer=3,
    )

    gp.fit(X_observed, y_observed)
    return gp


def non_dominated_indices(points):
    """Return local indices of non-dominated points.

    Both objectives are maximised.

    Each point is:
        [mu, sigma]

    A point is dominated if another point has:
        mu >= its mu
        sigma >= its sigma

    and is strictly better in at least one objective.
    """
    n_points = points.shape[0]
    keep = []

    for i in range(n_points):
        point = points[i]
        dominated = False

        for j in range(n_points):
            if i == j:
                continue

            other = points[j]

            better_or_equal = np.all(other >= point)
            strictly_better = np.any(other > point)

            if better_or_equal and strictly_better:
                dominated = True
                break

        if not dominated:
            keep.append(i)

    return np.array(keep, dtype=int)


def choose_next_candidate(candidate_indices, mu, sigma):
    """Choose the next molecule using ParetoSelector.

    candidate_indices are the global indices of molecules that have not yet
    been observed.
    """
    points = np.column_stack(
        [
            mu[candidate_indices],
            sigma[candidate_indices],
        ]
    )

    pareto_local_indices = non_dominated_indices(points)
    pareto_global_indices = candidate_indices[pareto_local_indices]

    pareto_front = pl.DataFrame(
        {
            "index": pareto_global_indices,
            "obj:mu": mu[pareto_global_indices],
            "obj:sigma": sigma[pareto_global_indices],
        }
    )

    bounds = pl.DataFrame(
        {
            "obj:mu": [
                pareto_front["obj:mu"].min(),
                pareto_front["obj:mu"].max(),
            ],
            "obj:sigma": [
                pareto_front["obj:sigma"].min(),
                pareto_front["obj:sigma"].max(),
            ],
        }
    )

    utopia_point = pl.DataFrame(
        {
            "obj:mu": [pareto_front["obj:mu"].max()],
            "obj:sigma": [pareto_front["obj:sigma"].max()],
        }
    )

    selector = ParetoSelector()

    selected_row = selector.select_closest_recipe_to_utopia(
        pareto_front=pareto_front,
        utopia_point=utopia_point,
        bounds=bounds,
    )

    selected_index = int(selected_row["index"][0])

    return selected_index, pareto_global_indices


def plot_mu_sigma(
    path,
    mu,
    sigma,
    ucb,
    observed_indices,
    pareto_indices,
    selected_index,
    iteration,
):
    """Plot all molecules on GP mu/sigma axes."""
    all_indices = np.arange(len(mu))
    unobserved_indices = np.setdiff1d(all_indices, observed_indices)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        mu[unobserved_indices],
        sigma[unobserved_indices],
        c=ucb[unobserved_indices],
        s=20,
        alpha=0.7,
        label="unobserved",
    )

    plt.scatter(
        mu[observed_indices],
        sigma[observed_indices],
        marker="x",
        s=50,
        label="observed",
    )

    plt.scatter(
        mu[pareto_indices],
        sigma[pareto_indices],
        facecolors="none",
        edgecolors="black",
        s=70,
        label="non-dominated",
    )

    plt.scatter(
        mu[selected_index],
        sigma[selected_index],
        marker="*",
        s=180,
        label="selected",
    )

    plt.xlabel("GP posterior mean, mu")
    plt.ylabel("GP posterior standard deviation, sigma")
    plt.title(f"BO iteration {iteration}")
    plt.colorbar(scatter, label="UCB = mu + kappa × sigma")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_bo(args):
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)

    problem = load_bo_problem(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        ld50_col=args.ld50_col,
    )

    X = problem.X
    y = problem.y

    if args.objective == "minimize":
        y_for_gp = -y
    else:
        y_for_gp = y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_candidates = X_scaled.shape[0]

    rng = np.random.default_rng(args.seed)

    observed_indices = rng.choice(
        n_candidates,
        size=args.n_initial,
        replace=False,
    ).tolist()

    history = []

    for iteration in range(args.n_iterations):
        observed_array = np.array(observed_indices, dtype=int)

        X_observed = X_scaled[observed_array]
        y_observed = y_for_gp[observed_array]

        gp = fit_gp(
            X_observed=X_observed,
            y_observed=y_observed,
            seed=args.seed + iteration,
        )

        mu, sigma = gp.predict(X_scaled, return_std=True)

        ucb = mu + args.kappa * sigma

        all_indices = np.arange(n_candidates)
        candidate_indices = np.setdiff1d(all_indices, observed_array)

        selected_index, pareto_indices = choose_next_candidate(
            candidate_indices=candidate_indices,
            mu=mu,
            sigma=sigma,
        )

        plot_mu_sigma(
            path=plots_dir / f"mu_sigma_iter_{iteration:03d}.png",
            mu=mu,
            sigma=sigma,
            ucb=ucb,
            observed_indices=observed_array,
            pareto_indices=pareto_indices,
            selected_index=selected_index,
            iteration=iteration,
        )

        prediction_table = pd.DataFrame(
            {
                "index": all_indices,
                "ld50_standardised_true": problem.y,
                "ld50_raw_true": problem.y_raw,
                "mu": mu,
                "sigma": sigma,
                "ucb": ucb,
                "observed": np.isin(all_indices, observed_array),
                "non_dominated": np.isin(all_indices, pareto_indices),
                "selected": all_indices == selected_index,
            }
        )

        prediction_table.to_csv(
            predictions_dir / f"predictions_iter_{iteration:03d}.csv",
            index=False,
        )

        history.append(
            {
                "iteration": iteration,
                "n_observed_before": len(observed_indices),
                "selected_index": int(selected_index),
                "selected_ld50_standardised": float(problem.y[selected_index]),
                "selected_ld50_raw": float(problem.y_raw[selected_index]),
                "selected_mu": float(mu[selected_index]),
                "selected_sigma": float(sigma[selected_index]),
                "selected_ucb": float(ucb[selected_index]),
                "n_non_dominated": int(len(pareto_indices)),
            }
        )

        print(
            f"iteration {iteration:03d} | "
            f"observed = {len(observed_indices)} | "
            f"selected = {selected_index} | "
            f"LD50 raw = {problem.y_raw[selected_index]:.4f} | "
            f"mu = {mu[selected_index]:.4f} | "
            f"sigma = {sigma[selected_index]:.4f} | "
            f"UCB = {ucb[selected_index]:.4f}"
        )

        observed_indices.append(int(selected_index))

        pd.DataFrame(history).to_csv(out_dir / "bo_trace.csv", index=False)
        np.save(out_dir / "observed_indices.npy", np.array(observed_indices))

    return pd.DataFrame(history)


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--out-dir", default="bo/results")

    parser.add_argument("--n-initial", type=int, default=100)
    parser.add_argument("--n-iterations", type=int, default=20)

    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ld50-col", type=int, default=1)

    parser.add_argument(
        "--objective",
        choices=["maximize", "minimize"],
        default="maximize",
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_bo(args)
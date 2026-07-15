"""Candidate selection Bayesian optimisation over VAE latent embeddings.

Steps:
1. Load VAE latent embeddings and LD50 labels.
2. Randomly observe 100 molecules.
3. Fit a GP on observed LD50 values.
4. Predict mu and sigma for all molecules.
5. Compute UCB.
6. Keep non-dominated unobserved candidates.
7. Select the next candidate (UCB argmax, or ParetoSelector utopia distance).
8. Add that selected candidate to the observed set.
9. Repeat.

Each iteration logs three CSVs into its own MLflow artifact folder: experiments, pareto front, selected recipes.
"""

from pathlib import Path
import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import polars as pl

import matplotlib
matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt

import mlflow

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from bo.problem import load_bo_problem
from bo.dashboard_artifacts import log_dashboard_general
from paretodo.selection.selector import ParetoSelector


CHECKPOINT = Path("checkpoints/vae_solubility_70a6568a_best.pth")


OBJ_MU_COL = "obj:max(\u03bc(LD50))"      # obj:max(mu(LD50))
OBJ_SIGMA_COL = "obj:max(\u03c3(LD50))"   # obj:max(sigma(LD50))


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


def compute_pareto_indices(candidate_indices, mu, sigma):
    """Global indices of the non-dominated unobserved candidates."""
    points = np.column_stack(
        [
            mu[candidate_indices],
            sigma[candidate_indices],
        ]
    )
    pareto_local_indices = non_dominated_indices(points)
    return candidate_indices[pareto_local_indices]


def select_via_ucb(candidate_indices, ucb):
    """Pick the unobserved candidate with the highest UCB."""
    best_local = int(np.argmax(ucb[candidate_indices]))
    return int(candidate_indices[best_local])


def select_via_pareto(candidate_indices, mu, sigma):
    """Pick the front point closest to the utopia point via ParetoSelector.

    Returns the selected global index and the non-dominated global indices.
    """
    pareto_global_indices = compute_pareto_indices(candidate_indices, mu, sigma)

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


def plot_pareto_front(
    iteration,
    candidate_indices,
    pareto_indices,
    selected_index,
    mu,
    sigma,
    save_path,
):
    """Scatter of the objective space for this iteration.

    x = mu(LD50), y = sigma(LD50), both maximised, so the utopia corner is
    top-right. Unobserved candidates are grey, the non-dominated front is
    highlighted, and the selected point is starred.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # unobserved candidates
    ax.scatter(
        mu[candidate_indices],
        sigma[candidate_indices],
        s=12,
        c="lightgrey",
        label="candidates",
        zorder=1,
    )

    # Non-dominated front
    order = np.argsort(mu[pareto_indices])
    front = pareto_indices[order]
    ax.plot(
        mu[front],
        sigma[front],
        color="tab:blue",
        linewidth=1.0,
        alpha=0.6,
        zorder=2,
    )
    ax.scatter(
        mu[front],
        sigma[front],
        s=40,
        c="tab:blue",
        edgecolors="black",
        linewidths=0.4,
        label="pareto front",
        zorder=3,
    )

    # Selected recipe
    ax.scatter(
        mu[selected_index],
        sigma[selected_index],
        s=220,
        marker="*",
        c="tab:red",
        edgecolors="black",
        linewidths=0.6,
        label="selected",
        zorder=4,
    )

    ax.set_xlabel("obj:max(mu(LD50))")
    ax.set_ylabel("obj:max(sigma(LD50))")
    ax.set_title(f"Pareto front, iteration {iteration:02d}")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def log_iteration_artifacts(
    iteration,
    all_indices,
    candidate_indices,
    observed_array,
    pareto_indices,
    selected_index,
    problem,
    mu,
    sigma,
    ucb,
    selection_method,
):
    """Write experiments/pareto front/selected recipes for this iteration
    and log them into a dedicated MLflow artifact folder."""
    smiles = problem.smiles

    experiments = pd.DataFrame(
        {
            "RecipeID": observed_array,
            "SMILES": smiles[observed_array],
            "LD50": problem.y_raw[observed_array],
        }
    )

    pareto_front = pd.DataFrame(
        {
            "RecipeID": pareto_indices,
            "SMILES": smiles[pareto_indices],
            OBJ_MU_COL: mu[pareto_indices],
            OBJ_SIGMA_COL: sigma[pareto_indices],
        }
    )

    selected = pd.DataFrame(
        [
            {
                "RecipeID": int(selected_index),
                "SMILES": str(smiles[selected_index]),
                "mu": float(mu[selected_index]),
                "sigma": float(sigma[selected_index]),
                "ucb": float(ucb[selected_index]),
                "ld50_raw": float(problem.y_raw[selected_index]),
                "ld50_standardised": float(problem.y[selected_index]),
                "selection_method": selection_method,
            }
        ]
    )

    folder = f"Iteration_{iteration:02d}"

    with tempfile.TemporaryDirectory() as tmp:
        exp_path = os.path.join(tmp, "experiments.csv")
        pf_path = os.path.join(tmp, "pareto_front.csv")
        pf_pred_path = os.path.join(tmp, "pareto_front_predictions.csv")
        sel_path = os.path.join(tmp, "selected_recipes.csv")
        plot_path = os.path.join(tmp, "pareto_front.png")

        experiments.to_csv(exp_path, index=False)
        pareto_front.to_csv(pf_path, index=False)
        # Same content under the name the paretodo dashboard reads.
        pareto_front.to_csv(pf_pred_path, index=False)
        selected.to_csv(sel_path, index=False)

        plot_pareto_front(
            iteration=iteration,
            candidate_indices=candidate_indices,
            pareto_indices=pareto_indices,
            selected_index=selected_index,
            mu=mu,
            sigma=sigma,
            save_path=plot_path,
        )

        mlflow.log_artifact(exp_path, artifact_path=folder)
        mlflow.log_artifact(pf_path, artifact_path=folder)
        mlflow.log_artifact(pf_pred_path, artifact_path=folder)
        mlflow.log_artifact(sel_path, artifact_path=folder)
        mlflow.log_artifact(plot_path, artifact_path=folder)


def plot_convergence(history, save_path):
    """Selected LD50 (raw) per iteration, with a running best line."""
    iters = [h["iteration"] for h in history]
    ld50 = np.array([h["selected_ld50_raw"] for h in history], dtype=float)
    running_best = np.maximum.accumulate(ld50)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        iters,
        ld50,
        marker="o",
        markersize=4,
        linewidth=1.0,
        color="tab:grey",
        label="selected LD50",
    )
    ax.plot(
        iters,
        running_best,
        linewidth=2.0,
        color="tab:red",
        label="running best",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("LD50 (raw)")
    ax.set_title("BO convergence")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_bo(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    all_indices = np.arange(n_candidates)

    rng = np.random.default_rng(args.seed)

    observed_indices = rng.choice(
        n_candidates,
        size=args.n_initial,
        replace=False,
    ).tolist()

    history = []

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "checkpoint": str(args.checkpoint),
                "n_initial": args.n_initial,
                "n_iterations": args.n_iterations,
                "kappa": args.kappa,
                "seed": args.seed,
                "objective": args.objective,
                "selection": args.selection,
                "n_candidates": n_candidates,
                "latent_dim": int(X.shape[1]),
                "mode": args.mode,
            }
        )

        # Always log the mode tag so the dashboard can react
        mlflow.set_tag("mode", args.mode)

        # Log the exact VAE checkpoint used
        mlflow.log_artifact(str(args.checkpoint), artifact_path="vae_checkpoint")

        log_dashboard_general(problem)

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

            candidate_indices = np.setdiff1d(all_indices, observed_array)

            if args.selection == "ucb":
                selected_index = select_via_ucb(candidate_indices, ucb)
                pareto_indices = compute_pareto_indices(
                    candidate_indices, mu, sigma
                )
            else:
                selected_index, pareto_indices = select_via_pareto(
                    candidate_indices, mu, sigma
                )

            log_iteration_artifacts(
                iteration=iteration,
                all_indices=all_indices,
                candidate_indices=candidate_indices,
                observed_array=observed_array,
                pareto_indices=pareto_indices,
                selected_index=selected_index,
                problem=problem,
                mu=mu,
                sigma=sigma,
                ucb=ucb,
                selection_method=args.selection,
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

            mlflow.log_metrics(
                {
                    "selected_ld50_raw": float(problem.y_raw[selected_index]),
                    "selected_mu": float(mu[selected_index]),
                    "selected_sigma": float(sigma[selected_index]),
                    "selected_ucb": float(ucb[selected_index]),
                    "n_non_dominated": int(len(pareto_indices)),
                },
                step=iteration,
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

        mlflow.log_artifact(str(out_dir / "bo_trace.csv"))

        # Run-level convergence plot across all iterations.
        with tempfile.TemporaryDirectory() as tmp:
            conv_path = os.path.join(tmp, "convergence.png")
            plot_convergence(history, conv_path)
            mlflow.log_artifact(conv_path)

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

    parser.add_argument(
        "--selection",
        choices=["pareto", "ucb"],
        default="pareto",
        help="pareto: ParetoSelector utopia distance. ucb: argmax UCB.",
    )

    parser.add_argument("--experiment-name", default="bo-ld50")
    parser.add_argument("--run-name", default=None)

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--default",
        dest="mode",
        action="store_const",
        const="default",
        help="Default mode: dashboard shows all sections.",
    )
    mode_group.add_argument(
        "--candidate-selection",
        dest="mode",
        action="store_const",
        const="candidate_selection",
        help="Candidate-selection mode: dashboard hides GP Predictions and "
        "Investigation Space Visualization.",
    )
    parser.set_defaults(mode="default")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_bo(args)
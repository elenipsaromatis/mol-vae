"""Candidate selection Bayesian optimisation over VAE latent embeddings.

Steps, per seed in RANDOM_SEEDS:
1. Load VAE latent embeddings and LD50 labels (once).
2. Fit a 3D PCA on the latent embeddings (once, shared by every seed/strategy).
3. Latin Hypercube sample 100 initial molecules in that PCA space.
4. For each selection strategy (UCB, Pareto), starting from that same initial set:
   a. Fit a GP on observed LD50 values.
   b. Predict mu and sigma for all molecules.
   c. Compute UCB.
   d. Keep non-dominated unobserved candidates.
   e. Select the next candidate (UCB argmax, or ParetoSelector utopia distance).
   f. Add that selected candidate to the observed set.
   g. Repeat.

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
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

import mlflow

from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc

from bo.problem import load_bo_problem
from bo.dashboard_artifacts import log_dashboard_general
from paretodo.selection.selector import ParetoSelector


CHECKPOINT = Path("checkpoints/vae_solubility_70a6568a_best.pth")

# Fixed seeds for the reproducible multi-seed BO sweep. Each seed draws its
# own Latin Hypercube initial sample from the same fitted 3D PCA space.
RANDOM_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]


OBJ_MU_COL = "obj:max(\u03bc(LD50))"      # obj:max(mu(LD50))
OBJ_SIGMA_COL = "obj:max(\u03c3(LD50))"   # obj:max(sigma(LD50))


def select_initial_indices_lhs_pca(pca_coords, n_initial, seed):
    """Latin Hypercube sample `n_initial` points in the fitted 3D PCA space,
    then map each sampled point to the nearest not-yet-selected real molecule.

    `pca_coords` must come from a single PCA fit on the fixed molecule pool
    (fit once by the caller, reused across every seed and strategy) so all
    seeds sample from the exact same 3D space. Returns unique global indices
    into that pool, deterministic for a given seed.
    """
    n_candidates = pca_coords.shape[0]
    if n_initial > n_candidates:
        raise ValueError(
            f"n_initial ({n_initial}) exceeds the molecule pool size "
            f"({n_candidates})"
        )

    sampler = qmc.LatinHypercube(d=pca_coords.shape[1], seed=seed)
    unit_samples = sampler.random(n=n_initial)

    lower = pca_coords.min(axis=0)
    upper = pca_coords.max(axis=0)
    lhs_points = qmc.scale(unit_samples, lower, upper)

    selected_indices = []
    selected_set = set()

    for point in lhs_points:
        distances = np.linalg.norm(pca_coords - point, axis=1)
        for candidate in np.argsort(distances):
            candidate = int(candidate)
            if candidate not in selected_set:
                selected_indices.append(candidate)
                selected_set.add(candidate)
                break

    return np.array(selected_indices, dtype=int)


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

    # Save predictions and statuses for every molecule
    all_predictions = pd.DataFrame(
        {
            "RecipeID": all_indices,
            "SMILES": smiles[all_indices],
            "LD50_raw": problem.y_raw[all_indices],
            "LD50_standardised": problem.y[all_indices],
            "mu": mu[all_indices],
            "sigma": sigma[all_indices],
            "ucb": ucb[all_indices],
            "observed": np.isin(all_indices, observed_array),
            "candidate": np.isin(all_indices, candidate_indices),
            "is_pareto": np.isin(all_indices, pareto_indices),
            "is_selected": all_indices == selected_index,
        }
    )

    # Rank candidates by acquisition score and true LD50
    all_predictions["ucb_rank"] = (
        all_predictions["ucb"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    all_predictions["true_ld50_rank"] = (
        all_predictions["LD50_raw"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    all_predictions["is_true_top_10"] = (
        all_predictions["true_ld50_rank"] <= 10
    )

    folder = f"Iteration_{iteration:02d}"

    with tempfile.TemporaryDirectory() as tmp:
        exp_path = os.path.join(tmp, "experiments.csv")
        pf_path = os.path.join(tmp, "pareto_front.csv")
        pf_pred_path = os.path.join(tmp, "pareto_front_predictions.csv")
        sel_path = os.path.join(tmp, "selected_recipes.csv")
        all_pred_path = os.path.join(tmp, "all_candidate_predictions.csv")
        plot_path = os.path.join(tmp, "pareto_front.png")

        experiments.to_csv(exp_path, index=False)
        pareto_front.to_csv(pf_path, index=False)

        # Same content under the name the paretodo dashboard reads.
        pareto_front.to_csv(pf_pred_path, index=False)

        selected.to_csv(sel_path, index=False)
        all_predictions.to_csv(all_pred_path, index=False)

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
        mlflow.log_artifact(all_pred_path, artifact_path=folder)
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


def run_bo_single(
    problem, X_scaled, initial_indices, seed, selection, out_dir, args, pca
):
    """Run one Bayesian optimisation loop for a single (seed, selection) pair.

    `initial_indices` is the exact array generated once for this seed by
    `select_initial_indices_lhs_pca`; the caller passes a `.copy()` per
    strategy so a paired UCB/Pareto comparison never shares mutable state.
    This function only reads `initial_indices` (via `.tolist()`, which
    copies) and only reads from `problem`/`X_scaled` -- neither the caller's
    array nor the shared problem data is mutated in place.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n_initial = args.n_initial
    candidate_pool = X_scaled

    assert len(initial_indices) == n_initial
    assert len(np.unique(initial_indices)) == n_initial
    assert initial_indices.min() >= 0
    assert initial_indices.max() < len(candidate_pool)

    y = problem.y

    if args.objective == "minimize":
        y_for_gp = -y
    else:
        y_for_gp = y

    n_candidates = X_scaled.shape[0]
    all_indices = np.arange(n_candidates)

    observed_indices = initial_indices.tolist()

    history = []

    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or f"seed{seed}_{selection}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "checkpoint": str(args.checkpoint),
                "n_initial": args.n_initial,
                "n_iterations": args.n_iterations,
                "kappa": args.kappa,
                "seed": seed,
                "objective": args.objective,
                "selection": selection,
                "n_candidates": n_candidates,
                "latent_dim": int(X_scaled.shape[1]),
                "mode": args.mode,
                "initial_sample_size": n_initial,
                "sampling_method": "latin_hypercube_pca_nearest_neighbor",
                "pca_n_components": int(pca.n_components_),
                "pca_explained_variance_ratio_pc1": float(
                    pca.explained_variance_ratio_[0]
                ),
                "pca_explained_variance_ratio_pc2": float(
                    pca.explained_variance_ratio_[1]
                ),
                "pca_explained_variance_ratio_pc3": float(
                    pca.explained_variance_ratio_[2]
                ),
                "pca_explained_variance_ratio_total": float(
                    pca.explained_variance_ratio_.sum()
                ),
            }
        )

        # Always log the mode/seed/selection tags so the dashboard can react
        mlflow.set_tag("mode", args.mode)
        mlflow.set_tag("seed", str(seed))
        mlflow.set_tag("selection", selection)

        # Log the exact VAE checkpoint used
        mlflow.log_artifact(str(args.checkpoint), artifact_path="vae_checkpoint")

        # Log the exact initial index array shared by this seed's UCB/Pareto pair
        mlflow.log_artifact(
            str(out_dir.parent / "initial_indices.npy"),
            artifact_path="initial_sample",
        )
        mlflow.log_artifact(
            str(out_dir.parent / "initial_sample.csv"),
            artifact_path="initial_sample",
        )

        log_dashboard_general(problem)

        for iteration in range(args.n_iterations):
            observed_array = np.array(observed_indices, dtype=int)

            X_observed = X_scaled[observed_array]
            y_observed = y_for_gp[observed_array]

            gp = fit_gp(
                X_observed=X_observed,
                y_observed=y_observed,
                seed=seed + iteration,
            )

            mu, sigma = gp.predict(X_scaled, return_std=True)

            ucb = mu + args.kappa * sigma

            candidate_indices = np.setdiff1d(all_indices, observed_array)

            if selection == "ucb":
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
                selection_method=selection,
            )

            history.append(
                {
                    "seed": seed,
                    "selection": selection,
                    "iteration": iteration,
                    "selected_recipe_id": int(selected_index),
                    "selected_smiles": str(problem.smiles[selected_index]),
                    "n_observed_before": len(observed_indices),
                    "selected_index": int(selected_index),
                    "selected_ld50_standardised": float(problem.y[selected_index]),
                    "selected_ld50_raw": float(problem.y_raw[selected_index]),
                    "selected_mu": float(mu[selected_index]),
                    "selected_sigma": float(sigma[selected_index]),
                    "selected_ucb": float(ucb[selected_index]),
                    "selected_ucb_rank": int(
                        pd.Series(ucb)
                        .rank(method="min", ascending=False)
                        .iloc[selected_index]
                    ),
                    "selected_true_ld50_rank": int(
                        pd.Series(problem.y_raw)
                        .rank(method="min", ascending=False)
                        .iloc[selected_index]
                    ),
                    "selected_is_true_top_10": bool(
                        pd.Series(problem.y_raw)
                        .rank(method="min", ascending=False)
                        .iloc[selected_index] <= 10
                    ),
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
                f"[seed {seed} | {selection}] iteration {iteration:03d} | "
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


def run_bo(args):
    """Reproducible multi-seed BO sweep.

    Loads the candidate pool and fits `StandardScaler` + 3D PCA exactly
    once. For every seed, one Latin Hypercube initial sample is generated in
    that fixed PCA space; the same index array (copied per strategy) is then
    handed to every requested selection strategy so UCB vs. Pareto is a
    paired, fair comparison. Each (seed, strategy) result is written to its
    own subfolder so nothing overwrites a sibling run.
    """
    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    problem = load_bo_problem(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        ld50_col=args.ld50_col,
    )

    if args.seed is not None and args.seeds is not None:
        raise ValueError(
            "Pass either --seed (legacy single-seed alias) or --seeds, not both."
        )

    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = RANDOM_SEEDS

    X = problem.X

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA once on the fixed molecule pool (latent embeddings only, no
    # LD50/target values involved). Every seed and every selection strategy
    # samples its initial set from this same 3D space; it is never refit.
    #
    # svd_solver="auto" resolves deterministically to "covariance_eigh" for
    # this pool shape (n_samples >> n_features), so no random_state is set
    # here -- there is no randomized-solver step to seed.
    pca = PCA(n_components=3)
    pca_coords = pca.fit_transform(X_scaled)
    print(f"PCA sampling-space shape: {pca_coords.shape}")
    print(
        "PCA explained variance ratio (pc1, pc2, pc3): "
        f"{pca.explained_variance_ratio_}"
    )

    strategies = args.selection

    results = {}

    for seed in seeds:
        initial_indices = select_initial_indices_lhs_pca(
            pca_coords, n_initial=args.n_initial, seed=seed
        )

        n_selected = len(initial_indices)
        all_unique = len(set(initial_indices.tolist())) == n_selected

        print(
            f"[seed {seed}] initial indices selected = {n_selected} | "
            f"all unique = {all_unique}"
        )

        assert n_selected == args.n_initial, (
            f"seed {seed}: expected {args.n_initial} initial indices, "
            f"got {n_selected}"
        )
        assert all_unique, f"seed {seed}: duplicate molecule in initial set"

        seed_dir = base_out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Saved once per seed (shared by every strategy) so nothing is
        # duplicated across the sibling seed_{seed}/{ucb,pareto} folders.
        np.save(seed_dir / "initial_indices.npy", initial_indices)

        initial_sample_frame = pd.DataFrame(
            {
                "dataset_index": initial_indices,
                # RecipeID elsewhere in this file is just the dataset index.
                "recipe_id": initial_indices,
                "smiles": problem.smiles[initial_indices],
                "pc1": pca_coords[initial_indices, 0],
                "pc2": pca_coords[initial_indices, 1],
                "pc3": pca_coords[initial_indices, 2],
            }
        )
        initial_sample_frame.to_csv(
            seed_dir / "initial_sample.csv", index=False
        )

        for selection in strategies:
            out_dir = seed_dir / selection
            results[(seed, selection)] = run_bo_single(
                problem=problem,
                X_scaled=X_scaled,
                initial_indices=initial_indices.copy(),
                seed=seed,
                selection=selection,
                out_dir=out_dir,
                args=args,
                pca=pca,
            )

    return results


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--out-dir", default="bo/results")

    parser.add_argument("--n-initial", type=int, default=100)
    parser.add_argument("--n-iterations", type=int, default=20)

    parser.add_argument("--kappa", type=float, default=1.96)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One or more seeds for the reproducible sweep. Each seed draws "
            "its own Latin Hypercube initial sample in PCA space. Defaults "
            f"to RANDOM_SEEDS = {RANDOM_SEEDS}. Pass a single value (e.g. "
            "'--seeds 11') to test one seed."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Legacy single-seed alias, equivalent to '--seeds SEED'. "
            "Mutually exclusive with --seeds; kept for backward "
            "compatibility with existing scripts/commands."
        ),
    )

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
        nargs="+",
        default=["ucb", "pareto"],
        help=(
            "pareto: ParetoSelector utopia distance. ucb: argmax UCB. "
            "Both run by default, from the same initial sample per seed, "
            "for a paired comparison. Pass one value to run only that "
            "strategy."
        ),
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
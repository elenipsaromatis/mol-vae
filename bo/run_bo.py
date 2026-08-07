"""Candidate selection Bayesian optimisation over VAE latent embeddings.

Steps, per seed in RANDOM_SEEDS:
1. Load VAE latent embeddings and LD50 labels (once).
2. Fit a 3D PCA on the latent embeddings (once, shared by every seed/strategy).
3. Latin Hypercube sample 100 initial molecules in that PCA space.
4. For each selection strategy (UCB, Pareto, EI), starting from that same initial set:
   a. Fit a GP on observed LD50 values.
   b. Predict mu and sigma for all molecules.
   c. Compute UCB and Expected Improvement.
   d. Keep non-dominated unobserved candidates.
   e. Select the next candidate (UCB argmax, EI argmax, or ParetoSelector utopia distance).
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
from scipy.stats import norm, qmc

from bo.problem import load_bo_problem, load_bo_problem_full_ld50
from bo.noise import apply_relative_noise
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


def compute_ei(mu, sigma, best_f, xi=0.01):
    """Expected Improvement for a maximised objective.

    `best_f` is the best (GP-scale) objective value observed so far. Points
    with sigma == 0 (e.g. already-observed molecules) get EI = 0 rather than
    a divide-by-zero NaN.
    """
    sigma_safe = np.where(sigma > 0, sigma, 1.0)
    improvement = mu - best_f - xi
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(sigma > 0, ei, 0.0)


def select_via_ei(candidate_indices, ei):
    """Pick the unobserved candidate with the highest Expected Improvement."""
    best_local = int(np.argmax(ei[candidate_indices]))
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
    ei,
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
                "ei": float(ei[selected_index]),
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
            "LD50_true_raw": problem.y_raw_true[all_indices],
            "mu": mu[all_indices],
            "sigma": sigma[all_indices],
            "ucb": ucb[all_indices],
            "ei": ei[all_indices],
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

    all_predictions["ei_rank"] = (
        all_predictions["ei"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Always derived from ground-truth LD50 (never the noisy observed
    # value), so this stays meaningful under --noise-scale.
    all_predictions["true_ld50_rank"] = (
        all_predictions["LD50_true_raw"]
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


def _assert_convergence_monotonic(
    initial_best_ld50, history, objective, seed, selection
):
    """Verify the best-observed-so-far sequence (x=0 initial, then
    best_ld50_after per iteration) never regresses for the given objective.
    """
    if not history:
        return

    sequence = np.array(
        [initial_best_ld50] + [h["best_ld50_after"] for h in history],
        dtype=float,
    )
    diffs = np.diff(sequence)

    if objective == "maximize":
        ok = np.all(diffs >= -1e-9)
    else:
        ok = np.all(diffs <= 1e-9)

    assert ok, (
        f"seed {seed} | {selection}: convergence sequence is not "
        f"monotonic for objective={objective}: {sequence.tolist()}"
    )


def plot_convergence(history, save_path):
    """Plot best observed raw LD50 including the initial sample."""
    if not history:
        return

    evaluations = np.array(
        [h["iteration"] + 1 for h in history],
        dtype=int,
    )

    best_observed = np.array(
        [h["best_ld50_after"] for h in history],
        dtype=float,
    )

    initial_best = float(
        history[0]["initial_best_ld50"]
    )

    x = np.concatenate(
        [
            np.array([0], dtype=int),
            evaluations,
        ]
    )

    y = np.concatenate(
        [
            np.array([initial_best], dtype=float),
            best_observed,
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.step(
        x,
        y,
        where="post",
        linewidth=2.0,
        label="Best observed LD50",
    )

    ax.scatter(
        [0],
        [initial_best],
        s=45,
        label="Best initial molecule",
        zorder=3,
    )

    ax.set_xlabel("Additional molecule evaluations")
    ax.set_ylabel("Best observed LD50")
    ax.set_title("Optimisation convergence")
    ax.xaxis.set_major_locator(
        plt.MaxNLocator(integer=True)
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
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

    if args.objective == "maximize":
        initial_best_ld50 = float(
            np.max(problem.y_raw[initial_indices])
        )
        initial_best_ld50_true = float(
            np.max(problem.y_raw_true[initial_indices])
        )
    else:
        initial_best_ld50 = float(
            np.min(problem.y_raw[initial_indices])
        )
        initial_best_ld50_true = float(
            np.min(problem.y_raw_true[initial_indices])
        )

    # Never request more acquisitions than there are unobserved molecules
    # left in the pool.
    n_available = n_candidates - len(initial_indices)
    n_run_iterations = min(args.n_iterations, n_available)
    if n_run_iterations < args.n_iterations:
        print(
            f"[seed {seed} | {selection}] requested n_iterations="
            f"{args.n_iterations} exceeds remaining unobserved molecules "
            f"({n_available}); capping to {n_run_iterations}."
        )

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
                "noise_scale": args.noise_scale,
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
                "n_iterations_actual": n_run_iterations,
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

        for iteration in range(n_run_iterations):
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
            best_f = float(np.max(y_observed))
            ei = compute_ei(mu, sigma, best_f)

            candidate_indices = np.setdiff1d(all_indices, observed_array)

            if selection == "ucb":
                selected_index = select_via_ucb(candidate_indices, ucb)
                pareto_indices = compute_pareto_indices(
                    candidate_indices, mu, sigma
                )
            elif selection == "ei":
                selected_index = select_via_ei(candidate_indices, ei)
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
                ei=ei,
                selection_method=selection,
            )

            selected_ld50_raw = float(
                problem.y_raw[selected_index]
            )
            selected_ld50_true_raw_value = float(
                problem.y_raw_true[selected_index]
            )

            if args.objective == "maximize":
                best_ld50_before = float(
                    np.max(problem.y_raw[observed_array])
                )
                best_ld50_after = float(
                    max(best_ld50_before, selected_ld50_raw)
                )
                assert best_ld50_after >= best_ld50_before, (
                    f"seed {seed} | {selection} | iteration {iteration}: "
                    f"best_ld50_after ({best_ld50_after}) < best_ld50_before "
                    f"({best_ld50_before}) for a maximisation objective"
                )
                best_ld50_true_before = float(
                    np.max(problem.y_raw_true[observed_array])
                )
                best_ld50_true_after = float(
                    max(best_ld50_true_before, selected_ld50_true_raw_value)
                )
            else:
                best_ld50_before = float(
                    np.min(problem.y_raw[observed_array])
                )
                best_ld50_after = float(
                    min(best_ld50_before, selected_ld50_raw)
                )
                assert best_ld50_after <= best_ld50_before, (
                    f"seed {seed} | {selection} | iteration {iteration}: "
                    f"best_ld50_after ({best_ld50_after}) > best_ld50_before "
                    f"({best_ld50_before}) for a minimisation objective"
                )
                best_ld50_true_before = float(
                    np.min(problem.y_raw_true[observed_array])
                )
                best_ld50_true_after = float(
                    min(best_ld50_true_before, selected_ld50_true_raw_value)
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
                    "selected_ld50_raw": selected_ld50_raw,
                    "selected_ld50_true_raw": selected_ld50_true_raw_value,
                    "initial_best_ld50": initial_best_ld50,
                    "initial_best_ld50_true": initial_best_ld50_true,
                    "best_ld50_before": best_ld50_before,
                    "best_ld50_after": best_ld50_after,
                    "best_ld50_true_before": best_ld50_true_before,
                    "best_ld50_true_after": best_ld50_true_after,
                    "improved_best": bool(
                        best_ld50_after != best_ld50_before
                    ),
                    "selected_mu": float(mu[selected_index]),
                    "selected_sigma": float(sigma[selected_index]),
                    "selected_ucb": float(ucb[selected_index]),
                    "selected_ucb_rank": int(
                        pd.Series(ucb)
                        .rank(method="min", ascending=False)
                        .iloc[selected_index]
                    ),
                    "selected_ei": float(ei[selected_index]),
                    "selected_ei_rank": int(
                        pd.Series(ei)
                        .rank(method="min", ascending=False)
                        .iloc[selected_index]
                    ),
                    # True LD50 rank/top-10 respect the objective direction: for
                    # "maximize" a larger LD50 ranks higher (ascending=False); for
                    # "minimize" a smaller LD50 ranks higher (ascending=True). Always
                    # derived from ground-truth LD50, never the noisy observed value.
                    # (Matches the random-search loop's rank_ascending below --
                    # this block used to hardcode ascending=False regardless of
                    # args.objective, silently scoring "true top 10" against the
                    # wrong tail for minimize runs.)
                    "selected_true_ld50_rank": int(
                        pd.Series(problem.y_raw_true)
                        .rank(method="min", ascending=(args.objective == "minimize"))
                        .iloc[selected_index]
                    ),
                    "selected_is_true_top_10": bool(
                        pd.Series(problem.y_raw_true)
                        .rank(method="min", ascending=(args.objective == "minimize"))
                        .iloc[selected_index] <= 10
                    ),
                    "n_non_dominated": int(len(pareto_indices)),
                }
            )

            mlflow.log_metrics(
                {
                    "selected_ld50_raw": selected_ld50_raw,
                    "best_ld50_after": best_ld50_after,
                    "selected_mu": float(mu[selected_index]),
                    "selected_sigma": float(sigma[selected_index]),
                    "selected_ucb": float(ucb[selected_index]),
                    "selected_ei": float(ei[selected_index]),
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

        _assert_convergence_monotonic(
            initial_best_ld50, history, args.objective, seed, selection
        )

        mlflow.log_artifact(str(out_dir / "bo_trace.csv"))

        # Run-level convergence plot across all iterations.
        with tempfile.TemporaryDirectory() as tmp:
            conv_path = os.path.join(tmp, "convergence.png")
            plot_convergence(history, conv_path)
            mlflow.log_artifact(conv_path)

    return pd.DataFrame(history)


def run_random_single(
    problem,
    initial_indices,
    seed,
    out_dir,
    args,
):
    """Run a random-search baseline for a single seed.

    Starts from the exact same `initial_indices` used by the UCB/Pareto BO
    strategies for this seed, then repeatedly picks a uniformly random
    unobserved candidate via a local `np.random.default_rng(seed)`. No GP,
    PCA, or Pareto front is involved -- this is a plain baseline against
    which the BO strategies are compared.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = "random"
    representation = "none"

    n_initial = args.n_initial
    n_candidates = problem.X.shape[0]
    all_indices = np.arange(n_candidates)

    observed_indices = initial_indices.tolist()

    if args.objective == "maximize":
        initial_best_ld50 = float(
            np.max(problem.y_raw[initial_indices])
        )
        initial_best_ld50_true = float(
            np.max(problem.y_raw_true[initial_indices])
        )
    else:
        initial_best_ld50 = float(
            np.min(problem.y_raw[initial_indices])
        )
        initial_best_ld50_true = float(
            np.min(problem.y_raw_true[initial_indices])
        )

    # Never request more acquisitions than there are unobserved molecules
    # left in the pool.
    n_available = n_candidates - len(initial_indices)
    n_run_iterations = min(args.n_iterations, n_available)
    if n_run_iterations < args.n_iterations:
        print(
            f"[seed {seed} | random] requested n_iterations="
            f"{args.n_iterations} exceeds remaining unobserved molecules "
            f"({n_available}); capping to {n_run_iterations}."
        )

    rng = np.random.default_rng(seed)

    history = []

    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or f"seed{seed}_random"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "checkpoint": str(args.checkpoint),
                "n_initial": args.n_initial,
                "n_iterations": args.n_iterations,
                "seed": seed,
                "objective": args.objective,
                "selection": selection,
                "representation": representation,
                "noise_scale": args.noise_scale,
                "n_candidates": n_candidates,
                "mode": args.mode,
                "initial_sample_size": n_initial,
                "sampling_method": "latin_hypercube_pca_nearest_neighbor",
                "n_iterations_actual": n_run_iterations,
            }
        )

        # Always log the mode/seed/selection/representation tags so the
        # dashboard can react.
        mlflow.set_tag("mode", args.mode)
        mlflow.set_tag("seed", str(seed))
        mlflow.set_tag("selection", selection)
        mlflow.set_tag("representation", representation)

        # Log the exact initial index array shared by this seed's
        # UCB/Pareto/random trio.
        mlflow.log_artifact(
            str(out_dir.parent / "initial_indices.npy"),
            artifact_path="initial_sample",
        )
        mlflow.log_artifact(
            str(out_dir.parent / "initial_sample.csv"),
            artifact_path="initial_sample",
        )

        for iteration in range(n_run_iterations):
            observed_array = np.array(observed_indices, dtype=int)

            candidate_indices = np.setdiff1d(all_indices, observed_array)
            selected_index = int(rng.choice(candidate_indices))

            selected_ld50_raw = float(problem.y_raw[selected_index])
            selected_ld50_standardised = float(problem.y[selected_index])
            selected_ld50_true_raw_value = float(
                problem.y_raw_true[selected_index]
            )

            if args.objective == "maximize":
                best_ld50_before = float(
                    np.max(problem.y_raw[observed_array])
                )
                best_ld50_after = float(
                    max(best_ld50_before, selected_ld50_raw)
                )
                assert best_ld50_after >= best_ld50_before, (
                    f"seed {seed} | random | iteration {iteration}: "
                    f"best_ld50_after ({best_ld50_after}) < best_ld50_before "
                    f"({best_ld50_before}) for a maximisation objective"
                )
                best_ld50_true_before = float(
                    np.max(problem.y_raw_true[observed_array])
                )
                best_ld50_true_after = float(
                    max(best_ld50_true_before, selected_ld50_true_raw_value)
                )
            else:
                best_ld50_before = float(
                    np.min(problem.y_raw[observed_array])
                )
                best_ld50_after = float(
                    min(best_ld50_before, selected_ld50_raw)
                )
                assert best_ld50_after <= best_ld50_before, (
                    f"seed {seed} | random | iteration {iteration}: "
                    f"best_ld50_after ({best_ld50_after}) > best_ld50_before "
                    f"({best_ld50_before}) for a minimisation objective"
                )
                best_ld50_true_before = float(
                    np.min(problem.y_raw_true[observed_array])
                )
                best_ld50_true_after = float(
                    min(best_ld50_true_before, selected_ld50_true_raw_value)
                )

            # True LD50 rank/top-10 respect the objective direction: for
            # "maximize" a larger LD50 ranks higher (ascending=False); for
            # "minimize" a smaller LD50 ranks higher (ascending=True). Always
            # derived from ground-truth LD50, never the noisy observed value.
            rank_ascending = args.objective == "minimize"
            selected_true_ld50_rank = int(
                pd.Series(problem.y_raw_true)
                .rank(method="min", ascending=rank_ascending)
                .iloc[selected_index]
            )
            selected_is_true_top_10 = bool(selected_true_ld50_rank <= 10)

            history.append(
                {
                    "seed": seed,
                    "selection": selection,
                    "representation": representation,
                    "iteration": iteration,
                    "selected_recipe_id": int(selected_index),
                    "selected_smiles": str(problem.smiles[selected_index]),
                    "n_observed_before": len(observed_indices),
                    "selected_index": int(selected_index),
                    "selected_ld50_standardised": selected_ld50_standardised,
                    "selected_ld50_raw": selected_ld50_raw,
                    "selected_ld50_true_raw": selected_ld50_true_raw_value,
                    "initial_best_ld50": initial_best_ld50,
                    "initial_best_ld50_true": initial_best_ld50_true,
                    "best_ld50_before": best_ld50_before,
                    "best_ld50_after": best_ld50_after,
                    "best_ld50_true_before": best_ld50_true_before,
                    "best_ld50_true_after": best_ld50_true_after,
                    "improved_best": bool(
                        best_ld50_after != best_ld50_before
                    ),
                    "selected_mu": np.nan,
                    "selected_sigma": np.nan,
                    "selected_ucb": np.nan,
                    "selected_ucb_rank": np.nan,
                    "selected_ei": np.nan,
                    "selected_ei_rank": np.nan,
                    "selected_true_ld50_rank": selected_true_ld50_rank,
                    "selected_is_true_top_10": selected_is_true_top_10,
                    "n_non_dominated": np.nan,
                }
            )

            mlflow.log_metrics(
                {
                    "selected_ld50_raw": selected_ld50_raw,
                    "best_ld50_before": best_ld50_before,
                    "best_ld50_after": best_ld50_after,
                },
                step=iteration,
            )

            print(
                f"[seed {seed} | random] iteration {iteration:03d} | "
                f"observed = {len(observed_indices)} | "
                f"selected = {selected_index} | "
                f"LD50 raw = {selected_ld50_raw:.4f} | "
                f"best before = {best_ld50_before:.4f} | "
                f"best after = {best_ld50_after:.4f}"
            )

            observed_indices.append(selected_index)

            pd.DataFrame(history).to_csv(out_dir / "bo_trace.csv", index=False)
            np.save(out_dir / "observed_indices.npy", np.array(observed_indices))

        _assert_convergence_monotonic(
            initial_best_ld50, history, args.objective, seed, selection
        )

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

    if args.pool == "full-ld50":
        problem = load_bo_problem_full_ld50(
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        problem = load_bo_problem(
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            ld50_col=args.ld50_col,
        )

    problem = apply_relative_noise(problem, scale=args.noise_scale)

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

        if "ucb" in strategies and "pareto" in strategies:
            ucb_initial_best = float(
                results[(seed, "ucb")]["initial_best_ld50"].iloc[0]
            )
            pareto_initial_best = float(
                results[(seed, "pareto")]["initial_best_ld50"].iloc[0]
            )
            assert np.isclose(ucb_initial_best, pareto_initial_best), (
                f"seed {seed}: UCB initial_best_ld50 ({ucb_initial_best}) != "
                f"Pareto initial_best_ld50 ({pareto_initial_best}) despite "
                "sharing the same initial_indices"
            )

        if args.run_random:
            random_out_dir = seed_dir / "random"

            results[(seed, "random")] = run_random_single(
                problem=problem,
                initial_indices=initial_indices.copy(),
                seed=seed,
                out_dir=random_out_dir,
                args=args,
            )

    return results


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--out-dir", default="bo/results")

    parser.add_argument("--n-initial", type=int, default=100)
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=300,
        help=(
            "Number of acquisition iterations to run (batch size 1). "
            "Automatically capped at (pool size - n-initial), i.e. the "
            "number of molecules not already in the initial sample."
        ),
    )

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
        "--pool",
        choices=["overlap", "full-ld50"],
        default="overlap",
        help=(
            "overlap: candidate pool is the solubility x LD50 overlap the "
            "VAE was trained on (default, unchanged behaviour). full-ld50: "
            "candidate pool is the full LD50_Zhu dataset, encoded with the "
            "same frozen checkpoint, to test how well the overlap-trained "
            "latent space extrapolates. --ld50-col is ignored in this mode."
        ),
    )

    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.0,
        help=(
            "Relative Gaussian noise scale applied to LD50 before BO ever "
            "sees it: LD50_noisy = LD50_true * (1 + epsilon), "
            "epsilon ~ N(0, noise_scale^2). One fixed noise vector is drawn "
            "per run (seeded at SEED_RUN + 1 = 101 from bo/noise.py), "
            "identical across every --seeds value. Default 0.0 is the "
            "noise-free baseline (no-op)."
        ),
    )

    parser.add_argument(
        "--objective",
        choices=["maximize", "minimize"],
        default="maximize",
    )

    parser.add_argument(
        "--selection",
        choices=["pareto", "ucb", "ei"],
        nargs="+",
        default=["ucb", "pareto"],
        help=(
            "pareto: ParetoSelector utopia distance. ucb: argmax UCB. "
            "ei: argmax Expected Improvement. ucb and pareto run by "
            "default, from the same initial sample per seed, for a paired "
            "comparison. Pass one or more values (e.g. '--selection ei') "
            "to run a different subset of strategies."
        ),
    )

    parser.add_argument("--experiment-name", default="bo-ld50")
    parser.add_argument("--run-name", default=None)

    parser.add_argument(
        "--run-random",
        action="store_true",
        help=(
            "Run a random-search baseline using the same initial "
            "molecule indices as the BO strategies."
        ),
    )

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
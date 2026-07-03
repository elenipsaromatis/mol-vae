"""Single-objective search over the VAE latent space: maximize LD50 (GP
surrogate fit on the solubility/LD50 overlap set), then decode the selected
latent points back into SMILES. Solubility is only used as a VAE training
signal (see train.py) and is not searched over here.

Usage: python bo/run_bo.py [--config bo/config.toml]
"""

import argparse
import sys
import tomllib
from pathlib import Path

BO_DIR = Path(__file__).resolve().parent
ROOT = BO_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BO_DIR))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: E402
from pymoo.optimize import minimize as pymoo_minimize  # noqa: E402
from pymoo.termination.default import DefaultMultiObjectiveTermination  # noqa: E402

from paretodo import (  # noqa: E402
    MlFlowLogger,
    OptimizationAxis,
    OptimizationProblem,
    ParetoSelector,
)

import init as bo_init  # noqa: E402
from problem import MolecularLatentProblem  # noqa: E402


def build_optimization_problem(config):
    return OptimizationProblem(
        problem=[
            OptimizationAxis(
                var_name="Y_ld50",
                minimize=config["minimize_ld50"],
                optimization_type="objective",
                lower_bound=None,
                upper_bound=None,
            ),
        ]
    )


def main(config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)["parameters"]

    print("Setting up: loading checkpoint, encoding overlap set, fitting LD50 GP...")
    setup_ns = bo_init.setup(config)
    print(f"LD50 GP train R^2: {setup_ns.ld50_gp_train_r2:.3f}")
    if setup_ns.ld50_gp_train_r2 < 0.3:
        print(
            "WARNING: LD50 GP fit is weak (train R^2 < 0.3) - the overlap "
            "dataset may be too small/high-dimensional for a useful surrogate."
        )

    optimization_problem = build_optimization_problem(config)

    ml_logger = MlFlowLogger(experiment_name=config["name"])
    for key, value in config.items():
        ml_logger.add_parameter(key, value)
    ml_logger.log_investigation_space(setup_ns.investigation_space)
    ml_logger.log_optimization_problem(optimization_problem)
    ml_logger.add_tag("Algorithm", "pfgs+bo")
    ml_logger.add_tag("Problem", "vae-solubility-ld50")
    ml_logger.log_metric("ld50_gp_train_r2", setup_ns.ld50_gp_train_r2)

    problem_instance = MolecularLatentProblem(
        investigation_space=setup_ns.investigation_space,
        optimization_problem=optimization_problem,
        ld50_gp=setup_ns.ld50_gp,
    )

    # paretodo's ParetoFrontGenerator assumes a multi-objective Pareto
    # *front*; with a single objective, pymoo's NSGA2 collapses its result
    # to one best point (res.X becomes 1D) instead of a population, which
    # breaks that assumption. Drive pymoo directly instead (same
    # algorithm/termination paretodo uses internally) and pull the full
    # final population via res.pop.
    print("Running NSGA2 search over the latent space (LD50 only)...")
    algorithm = NSGA2(
        pop_size=problem_instance.n_var * config["pop_size_per_dim"],
        seed=config["seed"],
    )
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8, ftol=1e-8, period=50, n_max_gen=config["n_max_gen"]
    )
    res = pymoo_minimize(
        problem_instance, algorithm, termination, seed=config["seed"], verbose=False
    )

    _predictions, pop_df = problem_instance.predict(res.pop.get("X"))

    # ParetoSelector's utopia point is hardcoded to 1.0 (maximize) / 0.0
    # (minimize) and assumes the objective column is already scaled to
    # [0,1]. Y_ld50 is a raw prediction (e.g. ~2-10), so it has to be
    # min-max normalized into the "obj:" column ParetoSelector reads -
    # otherwise it silently picks the *worst* point as "closest to utopia"
    # (verified empirically). The raw Y_ld50 column is kept for reporting.
    objective_label = optimization_problem.objective_labels[0]
    y_ld50 = pop_df["Y_ld50"].to_numpy()
    y_range = y_ld50.max() - y_ld50.min()
    y_norm = (y_ld50 - y_ld50.min()) / y_range if y_range > 0 else np.zeros_like(y_ld50)
    pop_df = pop_df.with_columns(
        pl.Series(objective_label, y_norm),
        pl.Series("RecipeID", np.arange(pop_df.height)),
    )
    ml_logger.log_dataframe(pop_df, datatype="pareto_front", iteration=1)

    selector = ParetoSelector()
    selected = selector.select_recipes(
        pareto_front=pop_df,
        investigation_space=setup_ns.investigation_space,
        optimization_problem=optimization_problem,
        n_samples=config["n_select"],
    )
    ml_logger.log_dataframe(selected, datatype="selected_recipes", iteration=1)

    z_pca = selected.select(setup_ns.investigation_space.labels).to_numpy()
    z_full = setup_ns.pca.inverse_transform(z_pca)
    smiles = bo_init.generate_smiles_from_latent(
        setup_ns.model, z_full, setup_ns.idx2char, setup_ns.device
    )

    result = pl.DataFrame(
        {
            "smiles": smiles,
            "Y_ld50": selected["Y_ld50"],
        }
    )
    print(result)
    result.write_csv(BO_DIR / "selected_recipes.csv")

    ml_logger.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(BO_DIR / "config.toml"),
        help="Path to the TOML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)

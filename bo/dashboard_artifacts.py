"""Build and log the paretodo dashboard schema for a BO run.

The Pareto dashboard reads two JSON files from each run's `general` artifact
folder, parsed by paretodo.data_models:

- investigation_space.json  -> InvestigationSpace.from_json
- optimization_problem.json -> OptimizationProblem.from_json

The objective column names the dashboard expects are produced by
OptimizationProblem.objective_labels, which uses Greek letters:

- optimization_type "objective"          -> obj:max(mu(LD50))    [Greek mu]
                                             obj:min(mu(LD50)) for a
                                             --objective minimize run
- optimization_type "standard_deviation" -> obj:max(sigma(LD50)) [Greek sigma]
                                             (always "max", regardless of
                                             --objective)

run_bo.py must write its pareto-front objective columns with those exact
names, so keep run_bo.py's _obj_mu_col()/OBJ_SIGMA_COL in sync with
objective_labels here.
"""

import json
import os
import tempfile

import mlflow


def build_investigation_space_dict(X):
    """One variable per VAE latent dimension, bounds taken from the pool.

    InvestigationSpaceVariable requires min < max strictly and float types.
    """
    lower = X.min(axis=0)
    upper = X.max(axis=0)

    variables = []
    for i in range(X.shape[1]):
        lo = float(lower[i])
        hi = float(upper[i])
        if hi <= lo:
            hi = lo + 1e-6
        variables.append(
            {
                "name": f"z{i}",
                "min": lo,
                "max": hi,
                "discretization": None,
                "expected_error": None,
            }
        )
    return {"variables": variables}


def build_optimization_problem_dict(objective: str = "maximize"):
    """Objective (mu) and predictive-std axes on LD50.

    minimize=False gives the max( prefix, minimize=True gives min(.
    optimization_type controls the inner symbol: objective -> mu,
    standard_deviation -> sigma.

    The mu axis's direction must match run_bo.py's --objective: for a
    "minimize" run, run_bo.py negates mu back to true (standardised) pLD50
    units before logging it to the dashboard artifacts (see
    log_iteration_artifacts's mu_display), so "lower is better" here is
    genuinely consistent with the numbers on that axis -- flipping this flag
    without that companion negation would make the label say the opposite of
    what the plotted values mean.

    The standard_deviation axis stays minimize=False regardless of
    objective: GP posterior std doesn't change sign under the internal
    y -> -y flip used for minimize runs, and it's always "more uncertainty"
    on that axis, not a signed quantity to invert.
    """
    return {
        "problem": [
            {
                "var_name": "LD50",
                "minimize": objective == "minimize",
                "optimization_type": "objective",
                "lower_bound": None,
                "upper_bound": None,
            },
            {
                "var_name": "LD50",
                "minimize": False,
                "optimization_type": "standard_deviation",
                "lower_bound": None,
                "upper_bound": None,
            },
        ]
    }


def log_dashboard_general(problem, objective: str = "maximize"):
    """Write the two schema JSONs and log them under the run's general/ folder.

    Call once inside an active MLflow run. `objective` must match the
    --objective this run was started with (see build_optimization_problem_dict).
    """
    inv = build_investigation_space_dict(problem.X)
    opt = build_optimization_problem_dict(objective)

    with tempfile.TemporaryDirectory() as tmp:
        inv_path = os.path.join(tmp, "investigation_space.json")
        opt_path = os.path.join(tmp, "optimization_problem.json")

        with open(inv_path, "w", encoding="utf-8") as f:
            json.dump(inv, f, indent=4)
        with open(opt_path, "w", encoding="utf-8") as f:
            json.dump(opt, f, indent=4)

        mlflow.log_artifact(inv_path, artifact_path="general")
        mlflow.log_artifact(opt_path, artifact_path="general")

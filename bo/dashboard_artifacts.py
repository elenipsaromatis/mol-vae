"""Build and log the paretodo dashboard schema for a BO run.

The Pareto dashboard reads two JSON files from each run's `general` artifact
folder, parsed by paretodo.data_models:

- investigation_space.json  -> InvestigationSpace.from_json
- optimization_problem.json -> OptimizationProblem.from_json

The objective column names the dashboard expects are produced by
OptimizationProblem.objective_labels, which uses Greek letters:

- optimization_type "objective"          -> obj:max(mu(LD50))   [Greek mu]
- optimization_type "standard_deviation" -> obj:max(sigma(LD50)) [Greek sigma]

run_bo.py must write its pareto-front objective columns with those exact names,
so keep OBJ_MU_COL / OBJ_SIGMA_COL there in sync with objective_labels here.
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


def build_optimization_problem_dict():
    """Two maximised objectives on LD50: predictive mean and predictive std.

    minimize=False gives the max( prefix. optimization_type controls the inner
    symbol: objective -> mu, standard_deviation -> sigma.
    """
    return {
        "problem": [
            {
                "var_name": "LD50",
                "minimize": False,
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


def log_dashboard_general(problem):
    """Write the two schema JSONs and log them under the run's general/ folder.

    Call once inside an active MLflow run.
    """
    inv = build_investigation_space_dict(problem.X)
    opt = build_optimization_problem_dict()

    with tempfile.TemporaryDirectory() as tmp:
        inv_path = os.path.join(tmp, "investigation_space.json")
        opt_path = os.path.join(tmp, "optimization_problem.json")

        with open(inv_path, "w", encoding="utf-8") as f:
            json.dump(inv, f, indent=4)
        with open(opt_path, "w", encoding="utf-8") as f:
            json.dump(opt, f, indent=4)

        mlflow.log_artifact(inv_path, artifact_path="general")
        mlflow.log_artifact(opt_path, artifact_path="general")

"""paretodo BaseProblem: single-objective LD50 search over the VAE latent space."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from paretodo import BaseProblem


class MolecularLatentProblem(BaseProblem):
    """Single-objective LD50 search over a PCA-compressed VAE latent space.

    Solubility is only used as a training signal for the VAE (see train.py)
    and is not searched over here. The search runs in the PCA subspace built
    in bo/init.py:fit_pca (needed to avoid a moocore stack-overflow above
    ~32 decision variables); the LD50 GP surrogate was fit directly in that
    same PCA space, so predict() can query it without any VAE involvement.
    """

    def __init__(
        self,
        investigation_space,
        optimization_problem,
        ld50_gp,
        samples_per_dimension=10,
    ):
        super().__init__(
            investigation_space=investigation_space,
            optimization_problem=optimization_problem,
            samples_per_dimension=samples_per_dimension,
        )
        self.ld50_gp = ld50_gp

    def _to_real_latent(self, x):
        lower, upper = self.investigation_space.bounds
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        return lower + np.asarray(x, dtype=np.float64) * (upper - lower)

    def predict(self, x, normalized=True):
        z = self._to_real_latent(x)
        ld50_mean_arr, ld50_sigma_arr = self.ld50_gp.predict(z, return_std=True)

        predictions = SimpleNamespace(
            x_data=z,
            y_labels=["Y_ld50"],
            y_data=ld50_mean_arr.reshape(-1, 1),
            y_errors=ld50_sigma_arr.reshape(-1, 1),
        )

        columns = dict(zip(self.investigation_space.labels, z.T.tolist()))
        columns["Y_ld50"] = ld50_mean_arr.tolist()
        columns["Y_ld50_std"] = ld50_sigma_arr.tolist()
        pred_df = pl.DataFrame(columns)

        return predictions, pred_df

    def create_objective_list(self, predictions, for_minimization=True):
        objective_list = []
        for axis in self.optimization_problem.problem:
            idx = predictions.y_labels.index(axis.var_name)
            objective_value = self._get_objective_value(
                axis,
                x_vals=predictions.x_data,
                mean=predictions.y_data[:, idx],
                sigma=predictions.y_errors[:, idx],
                for_minimization=for_minimization,
            )
            objective_list.append(objective_value)
        return objective_list

    def prediction_to_robust_objective_values(self, predictions, opt_axis):
        raise NotImplementedError(
            "robustness optimization_type is not used by MolecularLatentProblem."
        )

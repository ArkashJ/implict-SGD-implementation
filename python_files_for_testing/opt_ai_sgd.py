import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

from sgd import sgd


def tune_par(data, lr):
    """
    Tune parameter by using optimization, comparing errors over a subset of the data.

    Args:
      data: DATA object created through sample_data() (see functions.py)
      lr: The specified learning rate for AI-SGD.

    Returns:
      The parameter achieving the lowest error over the subset of data and the
      checked values.
    """

    def objective(par):
        # Convert range from (-inf, inf)^2 to [0, inf) x [-1, -1/2].
        par_adjusted = [
            np.exp(par[0]),
            interval_map(0, 1, -1, -1 / 2, logistic(par[1])),
        ]
        return eval_par(par_adjusted, data, lr)

    res = minimize(objective, x0=[1, -1], method="Nelder-Mead")
    res.x[0] = np.exp(res.x[0])
    res.x[1] = interval_map(0, 1, -1, -1 / 2, logistic(res.x[1]))
    return res


def eval_par(par, data, lr, idx_slice=slice(1000)):
    """
    Do a pass with AI-SGD using the fixed params to evaluate the error.

    Args:
      par: hyperparameters for the AI-SGD
      data: DATA object created through sample_data() (see functions.py)
      idx: Slice of indices to use as the subset of the data. Defaults to first 1000.
      lr: The specified learning rate for AI-SGD.

    Returns:
      The training error of AI-SGD using the fixed parameter values trained over
      the subset of the data.
    """

    # Subset data
    data_subset = {"X": data["X"][idx_slice, :], "Y": data["Y"][idx_slice]}

    # Run SGD
    theta_sgd = sgd(data_subset, sgd_method="AI-SGD", lr=lr, par=par)
    theta_sgd = theta_sgd[:, -1]

    # Use MSE of h(X*θ) from y
    cost = np.linalg.norm(
        data_subset["Y"] - data["model"]["h"](np.dot(data_subset["X"], theta_sgd)),
        ord=2,
    )

    # Logging the cost for each parameter set
    if len(par) == 1:
        print(f"Trying par={par[0]:.3f} yields cost {cost:.3f}")
    else:
        print(
            f"Trying par=({', '.join([f'{p:.3f}' for p in par])}) yields cost {cost:.3f}"
        )

    return cost


# Helper functions like logistic and interval_map need to be defined
def logistic(x):
    return 1 / (1 + np.exp(-x))


def interval_map(x, x_min, x_max, y_min, y_max):
    return y_min + (y_max - y_min) * ((x - x_min) / (x_max - x_min))

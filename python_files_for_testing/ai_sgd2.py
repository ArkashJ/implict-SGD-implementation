import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from ai_sgd_opt import *


def run(model, pars, n=1e4, p=1e1, plot_save=False, **kwargs):
    """
    Run AI-SGD for a set of parameters and any additionally selected methods,
    and plot error over training data size. The set of parameters affect only
    AI-SGD's learning rate.

    Args:
      model: the specified GLM
      pars: A npars x 2 array, where each row is a set of parameters to run
            AI-SGD on
      n: number of observations
      p: number of parameters
      add_methods: list of additional methods to benchmark. Options are
                   documented in sgd()
      plot_save: boolean specifying whether to save plot to disk or output it

    Returns:
      A plot object, plotting error over training data size for each
      optimization routine.
    """
    np.random.seed(42)
    X_list = generate_X_A(n, p)
    d = generate_data(
        X_list, glm_model=get_glm_model(model), theta=2 * np.exp(-np.arange(1, p + 1))
    )

    # Construct functions for learning rate
    def lr(n, par):
        D, alpha = par
        return D * n**alpha

    # Optimize!
    theta = []
    for i, par in enumerate(pars):
        print("Running AI-SGD..")
        theta_sgd = sgd(d, lr=lambda n: lr(n, par), **kwargs)
        theta.append((theta_sgd, f"AI-SGD ({par[0]}, {par[1]})"))

    def lr_implicit(n):
        alpha = 1 / 0.01
        return alpha / (alpha + n)

    theta_sgd = sgd(d, lr=lr_implicit, **kwargs)
    theta.append((theta_sgd, method))
    print("Running implicit SGD..")
    return plot_risk(
        d, [t[0] for t in theta]
    )  # This should be modified to pass correct format


# Example usage
model = "gaussian"
print("Running AI-SGD..")
pars = np.array([[0.1, 0.5], [0.2, 0.4]])
run(model, pars, n=10000, p=10)

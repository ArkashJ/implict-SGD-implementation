import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.stats as stats
from scipy.optimize import root

from functions import logistic, random_matrix


def sgd(data_v, lr=1, npass=1, lambda_v=0):
    n = data_v["X"].shape[0]
    p = data_v["X"].shape[1]
    n_iters = n * npass
    glm_model = get_glm_model(model="gaussian")
    # init param matrix for sgd with size p x (n)*npass+1
    theta = np.zeros((p, n_iters + 1))
    theta_new = None
    ai = None

    # run stochastic gradient descent
    for i in range(1, n_iters + 1):
        # calculate ai
        if i % n == 0:
            idx = n
        else:
            idx = i % n

        x_i = data_v["X"][idx - 1, :]
        y_i = data_v["Y"][idx - 1]
        theta_old = theta[:, i - 1]
        ai = lr(i)

        # update step
        xi_norm = np.sum(x_i**2)
        l_pred = np.dot(theta_old, x_i)
        # #   The scalar value yi - h(θ_i' xi + xi^2 ξ) + λ*||θ_i+ξ||_2

        def score(ksi):
            return (
                y_i
                - glm_model["h"](l_pred + xi_norm * ksi)
                + lambda_v * np.sqrt(np.sum((theta_old + ksi) ** 2))
            )

        # search interval
        ri = ai * score(0)
        B_i = [0, ri] if ri >= 0 else [ri, 0]

        def implict_func(u):
            return u - ai * score(u)

        ksi_new = None
        if B_i[1] != B_i[0]:
            ksi_new = root(implict_func, x0=B_i[0]).x[0]
        else:
            ksi_new = B_i[0]
        theta_new = theta_old + ksi_new * x_i
        theta[:, i] = theta_new

    # average over all estimates
    theta_avg = np.cumsum(theta, axis=1) / np.arange(1, theta.shape[1] + 1)
    return theta_avg


def get_glm_model(model="gaussian"):
    """Returns the link/link-deriv functions of the specified GLM model."""
    if model == "gaussian":
        return {"name": model, "h": lambda x: x, "hprime": lambda x: 1}
    elif model == "poisson":
        return {"name": model, "h": np.exp, "hprime": np.exp}
    elif model == "logistic":
        return {
            "name": model,
            "h": logistic,
            "hprime": lambda x: logistic(x) * (1 - logistic(x)),
        }
    else:
        raise ValueError(f"Model {model} is not supported...")


def generate_X_A(n, p, lambdas=None):
    """Generate observations from Normal(0, A)."""
    p = int(p)
    n = int(n)
    if lambdas is None:
        lambdas = np.linspace(0.01, 1, num=p)
    A = random_matrix(lambdas)  # Define this function to generate random matrix
    X = np.random.multivariate_normal(np.zeros(p), A, size=n)
    return {"X": X, "A": A}


def generate_X_corr(n, p, rho):
    """Generate normal observations with equally correlated covariates."""
    if not -1 < rho < 1:
        raise ValueError("Absolute value of rho must be less than 1")
    Z = np.random.normal(0, 1, n)
    if abs(rho) < 1:
        beta = np.sqrt(rho / (1 - rho))
        W = np.random.normal(0, 1, (n, p))
        Z_mat = np.tile(Z, (p, 1)).T
        X = beta * Z_mat + W
    else:  # rho == 1
        X = np.tile(Z, (p, 1)).T
    return {"X": X, "rho": rho}


def generate_data(X_list, theta=None, glm_model="gaussian", snr=1):
    """
    Generate the dataset.

    Args:
      X_list: dictionary with key 'X' for the design matrix and other elements for any stored data used to generate X
      theta: true parameters, numpy array of shape (n_features, 1)
      glm_model: GLM model name as a string ('gaussian', 'poisson', 'logistic')
      snr: signal-to-noise ratio (currently not in use)

    Returns:
      A dictionary with the following keys:
        'Y': outcomes (n_samples, 1)
        'X': covariates (n_samples, n_features)
        'theta': true params. (n_features, 1)
        'L': X * theta
        'model': GLM model name
        'obs_data': dictionary containing any data used to generate X
    """
    X = X_list["X"]
    if theta is None:
        theta = np.ones((X.shape[1], 1))

    lpred = X @ theta
    n = X.shape[0]

    if glm_model == "gaussian":
        epsilon = np.random.normal(0, 1, n)
        y = lpred + epsilon.reshape(-1, 1)
    elif glm_model == "poisson":
        y = np.random.poisson(lam=np.exp(lpred).flatten(), size=n)
    elif glm_model == "logistic":
        prob = 1 / (1 + np.exp(-lpred))
        y = np.random.binomial(1, prob.flatten(), size=n)
    else:
        raise ValueError(f"GLM model {glm_model} is not implemented.")

    # Removing 'X' from the X_list as it is not needed to be stored again
    obs_data = {key: value for key, value in X_list.items() if key != "X"}

    return {
        "Y": y,
        "X": X,
        "theta": theta,
        "L": lpred,
        "model": glm_model,
        "obs_data": obs_data,
    }


def plot_risk(data, est):
    """
    Plot estimated biases of the optimization routines performed.

    Args:
      data: DATA object created through generate_data (assumed to be a similar function in Python)
      est: A list of numpy arrays estimates, one for each optimization method run on data.

    Returns:
      A log-log scaled plot with a curve for each optimization routine, showing excess risk over training size.
    """
    list_bias = []
    for i, estimate in enumerate(est):
        values = np.apply_along_axis(
            lambda col: np.transpose(col - data["theta"])
            @ data["obs_data"]["A"]
            @ (col - data["theta"]),
            1,
            estimate,
        )
        t_values = (
            np.arange(1, len(values) + 1)
            if estimate.shape[1] == len(values)
            else np.array([int(col) for col in estimate.columns])
        )
        list_bias.append((t_values, values, f"Method {i+1}"))

    # Plotting
    plt.figure(figsize=(10, 6))
    for t_values, values, label in list_bias:
        plt.plot(t_values, values, label=label)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training size t")
    plt.ylabel("Excess risk")
    plt.title("Excess risk over training size")
    plt.legend()
    plt.show()


# set seed
np.random.seed(1)
nsamples = 100000
ncovs = 10
X_list = generate_X_A(n=nsamples, p=ncovs, lambdas=np.linspace(1, 1, ncovs))
lambda0 = np.min(scipy.linalg.eigh(X_list["A"], eigvals_only=True))
d = generate_data(X_list, theta=np.ones((ncovs, 1)), glm_model="gaussian")
subset_idx = np.linspace(10, nsamples, num=50, dtype=int)


def lr_implicit_avg(n):
    return (0 + lambda0 * n) ** (-0.8)


def test_ai_sgd():
    theta_sgd = sgd(data_v=d, lr=lr_implicit_avg, npass=1, lambda_v=0)
    theta_sgd = theta_sgd[:, subset_idx]
    return theta_sgd


def mse(theta_t):
    def calculate_mse(col):
        U = col - d["theta"]
        return np.log(np.sum(U.T @ X_list["A"] @ U))

    return np.apply_along_axis(calculate_mse, 0, theta_t)


def test_mse():
    theta_sgd = test_ai_sgd()
    mse_vals = mse(theta_sgd)
    plt.plot(mse_vals, color="red")
    plt.show()


#
# def main():
#     test_mse()
#
#
# main()

import time

import numpy as np
import scipy.stats as stats

# Generalized Linear Models


def logistic(x):
    """Return logit inverse."""
    return 1 / (1 + np.exp(-x))


def logit(x):
    """Return logit."""
    return np.log(x / (1 - x))


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


# Observation Matrix Generation


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


# Data Generation


def generate_data(X_list, theta=None, glm_model=None, snr=1):
    """Generate the dataset."""
    if glm_model is None:
        glm_model = get_glm_model("gaussian")
    if theta is None:
        theta = np.ones((X_list["X"].shape[1], 1))
    X = X_list["X"]
    n, p = X.shape
    lpred = X @ theta
    if glm_model["name"] == "gaussian":
        epsilon = np.random.normal(0, 1, n)
        y = lpred + epsilon
    elif glm_model["name"] == "poisson":
        y = np.random.poisson(lam=glm_model["h"](lpred))
    elif glm_model["name"] == "logistic":
        y = np.random.binomial(1, glm_model["h"](lpred), n)
    else:
        raise ValueError(f"GLM model {glm_model['name']} is not implemented..")
    X_list.pop("X", None)  # Remove X from the list
    return {
        "Y": y,
        "X": X,
        "theta": theta,
        "L": lpred,
        "model": glm_model,
        "obs_data": X_list,
    }


def print_data(data):
    """Pretty print of the object generated from generate_data."""
    nx, p = data["X"].shape
    ny = len(data["Y"])
    if nx != ny or p != len(data["theta"]):
        raise ValueError("Dimensions of X, Y, and theta do not match")
    lambdas = np.linalg.eigvals(np.cov(data["X"].T))
    print(lambdas)
    print(np.mean(data["Y"]))
    print(np.var(data["Y"]))
    print(1 + np.sum(np.cov(data["X"].T)))


def frac_sec():
    """
    Generate a seed number based on the current time.
    """
    now = time.time()
    return int(abs(now - int(now)) * 10**8)


def interval_map(a, b, c, d, x):
    """
    Scale values in [a, b] to [c, d].
    """
    return c + (d - c) / (b - a) * (x - a)


def random_orthogonal(p):
    """
    Get an orthogonal matrix.
    """
    B = np.random.rand(p, p)
    Q, _ = np.linalg.qr(B)
    return Q


def random_matrix(lambdas=None):
    """
    Generate a random matrix with the desired eigenvalues.
    """
    if lambdas is None:
        lambdas = np.linspace(0.01, 1, num=100)
    p = len(lambdas)
    Q = random_orthogonal(p)
    A = Q @ np.diag(lambdas) @ Q.T
    return A

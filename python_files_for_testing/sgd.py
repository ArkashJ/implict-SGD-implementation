import numpy as np
from scipy.optimize import root


def sgd(data, sgd_method, lr, npass=1, lambda_reg=0, **kwargs):
    """
    Find the optimal parameters using a stochastic gradient method for
    generalized linear models.

    Args:
      data: DATA object created through generate_data() (see functions.py)
      sgd_method: a string which is one of the following: "SGD", "ASGD", "LS-SGD", "ISGD", "AI-SGD", "LS-ISGD", "SVRG"
      lr: function which computes learning rate with input the iterate index
      npass: number of passes over data
      lambda_reg: L2 regularization parameter for cross validation. Defaults to performing no cross validation

    Returns:
      A p x n*npass+1 matrix where the jth column is the jth theta update.
    """

    # Validate input
    assert all(key in data for key in ["X", "Y", "model"])
    assert sgd_method in ["SGD", "ASGD", "LS-SGD", "ISGD", "AI-SGD", "LS-ISGD", "SVRG"]

    # Initialize constants
    n, p = data["X"].shape
    niters = n * npass
    glm_model = data["model"]
    m = None

    if sgd_method == "SVRG":
        assert npass % 2 == 0
        m = 2 * n
        niters = npass // 2  # do this many 2-passes over the data

    # Initialize parameter matrix for the stochastic gradient descent (p x n*npass+1)
    theta_sgd = np.zeros((p, niters + 1))

    if sgd_method == "SVRG":
        # Mark the true number of iterations for each sgd iterate in SVRG (0, m, 2*m, ...)
        col_names = np.arange(0, niters + 1) * m + 1

    for i in range(1, niters + 1):
        # Index
        idx = n if i % n == 0 else i % n  # sample index of data
        xi = data["X"][idx - 1, :]
        yi = data["Y"][idx - 1]
        theta_old = theta_sgd[:, i - 1]
        ai = lr(i, **kwargs)

        if sgd_method in ["SGD", "ASGD", "LS-SGD"]:
            theta_new = sgd_update(theta_old, xi, yi, ai, lambda_reg, glm_model)
        elif sgd_method in ["ISGD", "AI-SGD", "LS-ISGD"]:
            theta_new = isgd_update(theta_old, xi, yi, ai, lambda_reg, glm_model)
        elif sgd_method == "SVRG":
            theta_new = svrg_update(
                theta_old, data, lr, lambda_reg, glm_model, m, **kwargs
            )

        theta_sgd[:, i] = theta_new

    # Post-process parameters if the method requires it
    if sgd_method in ["ASGD", "AI-SGD"]:
        theta_sgd = average_post(theta_sgd)
    if sgd_method in ["LS-SGD", "LS-ISGD"]:
        theta_sgd = ls_post(theta_sgd, data)

    return theta_sgd


# Update Functions


def sgd_update(theta_old, xi, yi, ai, lambda_reg, glm_model):
    def score(theta):
        return (yi - glm_model["h"](np.dot(xi, theta))) * xi + lambda_reg * np.sqrt(
            np.sum(theta**2)
        )

    theta_new = theta_old + ai * score(theta_old)
    return theta_new


def isgd_update(theta_old, xi, yi, ai, lambda_reg, glm_model):
    xi_norm = np.sum(xi**2)
    lpred = np.dot(xi, theta_old)

    def get_score_coeff(ksi):
        # Ensure that the output is a scalar
        score = yi - glm_model["h"](lpred + xi_norm * ksi)
        regularization = lambda_reg * np.sqrt(np.sum((theta_old + ksi) ** 2))
        return score + regularization

    # Get the score for ksi = 0
    score_at_zero = get_score_coeff(0)

    # Check if score_at_zero is scalar
    if not np.isscalar(score_at_zero):
        raise ValueError("Score at zero should be a scalar.")

    ri = ai * score_at_zero
    Bi = [0, ri] if ri >= 0 else [ri, 0]

    def implicit_fn(u):
        return u - ai * get_score_coeff(u)

    ksi_new = root(implicit_fn, x0=Bi[0]).x[0] if Bi[1] != Bi[0] else Bi[0]
    theta_new = theta_old + ksi_new * xi
    return theta_new


def svrg_update(theta_old, data, lr, lambda_reg, glm_model, m, **kwargs):
    n, p = data["X"].shape

    def score(theta, xi, yi):
        return (yi - glm_model["h"](np.dot(xi, theta))) * xi + lambda_reg * np.sqrt(
            np.sum(theta**2)
        )

    mu = np.mean(
        [score(theta_old, data["X"][i], data["Y"][i]) for i in range(n)], axis=0
    )

    w = np.copy(theta_old)
    for mi in range(m):
        idx = np.random.choice(n)
        xi, yi = data["X"][idx], data["Y"][idx]
        ai = lr(mi, **kwargs)
        w += ai * (score(w, xi, yi) - score(theta_old, xi, yi) + mu)

    theta_new = w
    return theta_new


# Post-Processing Functions


def average_post(theta_sgd):
    return np.cumsum(theta_sgd, axis=1) / np.arange(1, theta_sgd.shape[1] + 1)


def ls_post(theta_sgd, data):
    n, p = data["X"].shape
    ncol_theta = theta_sgd.shape[1]
    y = np.zeros((p, ncol_theta))

    for i in range(1, ncol_theta):
        idx = n if i % n == 0 else i % n
        xi = data["X"][idx - 1]
        theta_old = theta_sgd[:, i - 1]
        y[:, i] = np.dot(data["obs_data"]["A"], xi - theta_old)

    beta_0 = np.zeros((p, ncol_theta))
    beta_1 = np.zeros((p, ncol_theta))

    for i in range(ncol_theta):
        x_i = theta_sgd[:, : i + 1]
        y_i = y[:, : i + 1]
        bar_x_i = np.mean(x_i, axis=1)
        bar_y_i = np.mean(y_i, axis=1)
        beta_1[:, i] = np.sum(y_i * (x_i - bar_x_i), axis=1) / np.sum(
            (x_i - bar_x_i) ** 2, axis=1
        )
        beta_0[:, i] = bar_y_i - beta_1[:, i] * bar_x_i

    return -beta_0 / beta_1

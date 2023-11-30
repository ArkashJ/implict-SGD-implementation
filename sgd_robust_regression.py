import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autograd import grad


def _robust_loss(psi, beta, nu, y, z):
    scaled_sq_errors = np.exp(-2 * psi) * (np.dot(z, beta) - y) ** 2
    if nu == np.inf:
        return scaled_sq_errors / 2 + psi
    return (nu + 1) / 2 * np.log(1 + scaled_sq_errors / nu) + psi


def make_sgd_robust_loss(y, z, nu):
    n = y.size
    sgd_loss = lambda param, inds: np.mean(
        _robust_loss(param[0], param[1:], nu, y[inds], z[inds])
    ) + np.sum(param**2) / (2 * n)
    grad_sgd_loss = grad(sgd_loss)
    return sgd_loss, grad_sgd_loss


def generate_data(n, d, seed):
    rng = np.random.default_rng(seed)
    # generate multivariate t covariates with 10 degrees
    # of freedom and non-diagonal covariance
    t_dof = 10
    locs = np.arange(d).reshape((d, 1))
    cov = (t_dof - 2) / t_dof * np.exp(-((locs - locs.T) ** 2) / 4)
    z = rng.multivariate_normal(np.zeros(d), cov, size=n)
    z *= np.sqrt(t_dof / rng.chisquare(t_dof, size=(n, 1)))
    # generate responses using regression coefficients beta = (1, 2, ..., d)
    # and t-distributed noise
    true_beta = np.arange(1, d + 1)
    y = z.dot(true_beta) + rng.standard_t(t_dof, size=n)
    # for simplicity, center responses
    y = y - np.mean(y)
    return true_beta, y, z


def run_sgd(
    grad_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, n
):
    k = (epochs * n) // batchsize
    d = init_param.size
    paramiters = np.zeros((k + 1, d))
    paramiters[0] = init_param
    for k in range(k):
        inds = np.random.choice(n, batchsize)
        stepsize = init_stepsize / (k + 1) ** stepsize_decayrate
        paramiters[k + 1] = paramiters[k] - stepsize * grad_loss(paramiters[k], inds)
    return paramiters


def plot_iterates_and_squared_errors(
    paramiters,
    true_beta,
    opt_param,
    skip_epochs,
    epochs,
    N,
    batchsize,
    fig_name,
    include_psi=True,
):
    D = true_beta.size
    param_names = [r"$\beta_{{{}}}$".format(i) for i in range(D)]
    if include_psi:
        param_names = [r"$\psi$"] + param_names
    else:
        paramiters = paramiters[:, 1:]
        opt_param = opt_param[1:]
    skip_epochs = 0
    skip_iters = int(skip_epochs * N // batchsize)
    xs = np.linspace(skip_epochs, epochs, paramiters.shape[0] - skip_iters)
    plt.plot(xs, paramiters[skip_iters:])
    plt.plot(np.array(D * [[xs[0], xs[-1]]]).T, np.array([true_beta, true_beta]), ":")
    plt.xlabel("epoch")
    plt.ylabel("parameter value")
    plt.savefig("{}.jpg".format(fig_name), bbox_inches="tight")
    # set name of figure
    plt.legend(
        param_names,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=4,
        frameon=False,
    )
    sns.despine()
    plt.show()
    plt.plot(xs, np.linalg.norm(paramiters - opt_param[np.newaxis, :], axis=1) ** 2)
    plt.xlabel("epoch")
    plt.ylabel(r"$\|x_k - x_{\star}\|_2^2$")
    plt.yscale("log")
    plt.savefig("{}_error.jpg".format(fig_name), bbox_inches="tight")
    # automatically close figures to avoid memory leak
    sns.despine()
    plt.show()

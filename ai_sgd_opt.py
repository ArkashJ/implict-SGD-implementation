import numpy as np


def sgd(data_v, lr=1, npass=1, lambda=0):
    n = data_v["X"].shape
    p = data_v["Y"].shape
    n_iters = n * npass
    glm_model = get_glm_model(model="gaussian")
    # init param matrix for sgd with size p x (n)*npass+1
    theta = np.zeros((p, n_iters+1))
    theta_new = None
    ai = None

    # run stochastic gradient descent
    for i in range(1, n_iters+1):
        # calculate ai
        if i%n == 0:
            idx = n 
        else:
            idx = i%n

        x_i, y_i = data_v["X"][idx-1, :], data_v["Y"][idx-1] 
        theta_old = theta[:, i-1]
        ai = lr(i)

        # update step
        xi_norm = np.sum(x_i**2)
        l_pred = np.dot(theta_old, x_i)
        # #   The scalar value yi - h(θ_i' xi + xi^2 ξ) + λ*||θ_i+ξ||_2
        
        def score(ksi):
            return y_i - glm_model["h"](l_pred+x_i*ksi) + lambda*np.sqrt(np.sum((theta_old+*ksi)**2))
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

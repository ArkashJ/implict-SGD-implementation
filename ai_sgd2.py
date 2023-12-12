import numpy as np
from scipy.optimize import root


def ai_sgd(data, initial_theta, lr, lambda_reg, glm_model, n_iters, batch_size=1):
    """
    Perform Averaged Implicit Stochastic Gradient Descent (AI-SGD).

    Args:
        data (dict): Dictionary containing 'X' and 'Y' data.
        initial_theta (numpy.ndarray): Initial parameters for the model.
        lr (function): Learning rate function.
        lambda_reg (float): Regularization parameter.
        glm_model (dict): Dictionary containing 'h' function for the model.
        n_iters (int): Number of iterations to run.
        batch_size (int): Size of each batch for stochastic updates.

    Returns:
        numpy.ndarray: Averaged parameters after AI-SGD.
    """

    def implicit_update(theta_old, xi, yi, ai):
        """Function to perform the implicit update."""

        def objective(ksi):
            adjusted_theta = theta_old + ksi
            score = yi - glm_model["h"](np.dot(xi, adjusted_theta))
            regularization = lambda_reg * np.sqrt(np.sum(adjusted_theta**2))
            return ksi - ai * (score + regularization)

        ksi_solution = root(objective, x0=np.zeros_like(theta_old)).x
        return theta_old + ksi_solution

    n, p = data["X"].shape
    theta = np.copy(initial_theta)
    avg_theta = np.copy(initial_theta)

    for i in range(1, n_iters + 1):
        idx = np.random.choice(n, batch_size, replace=False)
        xi, yi = data["X"][idx], data["Y"][idx]
        ai = lr(i)

        theta = implicit_update(theta, xi, yi, ai)
        avg_theta = (avg_theta * (i - 1) + theta) / i  # Averaging step

    return avg_theta


# Example usage
n, p = 100, 10  # Example dimensions for data
data = {"X": np.random.randn(n, p), "Y": np.random.randn(n)}
initial_theta = np.zeros(p)
lr = lambda n: 0.01 / (1 + 0.01 * n)  # Example learning rate function
lambda_reg = 0.1  # Regularization parameter
glm_model = {"h": lambda x: x}  # Example model (linear in this case)
n_iters = 1000

avg_theta = ai_sgd(data, initial_theta, lr, lambda_reg, glm_model, n_iters)

print(avg_theta)



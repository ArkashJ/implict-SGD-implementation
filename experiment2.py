import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

def generate_data(N, p):
    eigenvalues = 1 / np.arange(1, p + 1)
    Q, _ = np.linalg.qr(np.random.randn(p, p))
    H = Q @ np.diag(eigenvalues) @ Q.T
    X = np.random.multivariate_normal(np.zeros(p), H, size=N)
    theta_star = np.zeros(p)
    Y = X @ theta_star + np.random.normal(0, 1, N)
    return X, Y, H


def implicit_update(theta_old, xi, yi, lr):
    def objective(ksi):
        theta = theta_old + ksi
        return ksi + 2 * lr * ((yi - xi.T @ theta) * xi)

    ksi_solution = root(objective, x0=np.zeros_like(theta_old)).x
    return theta_old + ksi_solution


def ai_sgd(X, Y, lr, n_iters):
    n, p = X.shape
    theta = np.zeros(p)
    avg_theta = np.zeros(p)
    losses = []

    for i in range(n_iters):
        idx = np.random.randint(n)
        xi, yi = X[idx], Y[idx]
        theta = implicit_update(theta, xi, yi, lr)
        avg_theta = avg_theta * (i / (i + 1)) + theta / (i + 1)

        # Calculate and store the loss
        loss = np.mean((Y - X @ avg_theta) ** 2)
        losses.append(loss)

    return avg_theta, losses


# Experiment setup
N = 10**6
p = 20
X, Y, H = generate_data(N, p)

R_squared = np.trace(H)
gamma = 1 / R_squared

n_iters = 10000
avg_theta, losses = ai_sgd(X, Y, gamma, n_iters)

# Plotting
plt.figure()
plt.loglog(range(1, n_iters + 1), losses, label="AI-SGD Loss")
plt.xlabel("Iteration (log scale)")
plt.ylabel("Loss (log scale)")
plt.title("AI-SGD Convergence")
plt.legend()
plt.show()

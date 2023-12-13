import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from tqdm import tqdm

# Initialize parameters
N = 10**4
p = 20
theta_star = np.zeros(p)
H = np.diag(np.array([1 / k for k in range(1, p + 1)]))
X = np.random.multivariate_normal(np.zeros(p), H, N)
Y = np.array([np.random.normal(X[n].dot(theta_star), 1) for n in range(N)])


# Loss function
def loss_function(theta, x, y):
    return (y - x.dot(theta)) ** 2


# Gradient of the loss function
def gradient(theta, x, y):
    return -2 * x * (y - x.dot(theta))


# AI-SGD with dynamic learning rate
def ai_sgd(X, Y, gamma_start, gamma_end, N):
    theta = np.zeros(p)
    theta_avg = np.zeros(p)
    losses = []
    for n in range(N):
        gamma = gamma_start + (gamma_end - gamma_start) * (
            n / N
        )  # Dynamic learning rate
        grad = -2 * X[n] * (Y[n] - X[n].dot(theta))
        theta -= gamma * grad
        theta_avg = (theta_avg * n + theta) / (n + 1)
        loss = np.mean(
            [(Y[i] - X[i].dot(theta_avg)) ** 2 for i in range(n + 1)]
        )  # Excess risk
        losses.append(loss)
    return losses


# Run AI-SGD
gamma_start = 0.5 / np.trace(H)  # Below threshold
gamma_end = 2 / np.trace(H)  # Above threshold
losses = ai_sgd(X, Y, gamma_start, gamma_end, N)

# Plotting
plt.loglog(range(1, N + 1), losses)
plt.xlabel("Log N")
plt.ylabel("Log Excess Risk")
plt.title("AI-SGD with Dynamic Learning Rate")
plt.show()

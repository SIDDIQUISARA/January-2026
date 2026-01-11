import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# 1. Closed-form solution
theta_closed = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# 2. Gradient Descent
alpha = 0.1
n_iterations = 1000
theta_gd = np.random.randn(2, 1)

for _ in range(n_iterations):
    gradients = 2 / len(X_b) * X_b.T @ (X_b @ theta_gd - y)
    theta_gd -= alpha * gradients

# 3. Stochastic Gradient Descent
theta_sgd = np.random.randn(2, 1)
epochs = 50

for epoch in range(epochs):
    for i in range(len(X_b)):
        random_index = np.random.randint(len(X_b))
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta_sgd - yi)
        theta_sgd -= alpha * gradients

# Results
print("Closed-form theta:", theta_closed.ravel())
print("Gradient Descent theta:", theta_gd.ravel())
print("SGD theta:", theta_sgd.ravel())

# Plot
plt.scatter(X, y, color="blue")
X_plot = np.array([[0], [2]])
X_plot_b = np.c_[np.ones((2, 1)), X_plot]

plt.plot(X_plot, X_plot_b @ theta_closed, "r-", label="Closed-form")
plt.plot(X_plot, X_plot_b @ theta_gd, "g--", label="GD")
plt.plot(X_plot, X_plot_b @ theta_sgd, "k:", label="SGD")

plt.legend()
plt.title("Linear Regression Comparison")
plt.show()

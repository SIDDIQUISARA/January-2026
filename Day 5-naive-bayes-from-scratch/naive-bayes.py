import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes implemented from scratch
    using log-likelihood for numerical stability.
    """

    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon  # to avoid division by zero

    def fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        self.classes = np.unique(y)
        n_features = X.shape[1]

        self.priors = {}
        self.means = {}
        self.vars = {}

        for c in self.classes:
            X_c = X[y == c]

            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + self.epsilon

    def _log_gaussian_pdf(self, x, mean, var):
        """
        Compute log N(x | mean, var) element-wise
        """
        return (
            -0.5 * np.log(2 * np.pi * var)
            - ((x - mean) ** 2) / (2 * var)
        )

    def _log_posterior(self, x, c):
        """
        log P(y=c | x) ∝ log P(y=c) + sum log P(x_j | y=c)
        """
        log_prior = np.log(self.priors[c])
        log_likelihood = np.sum(
            self._log_gaussian_pdf(x, self.means[c], self.vars[c])
        )
        return log_prior + log_likelihood

    def predict(self, X):
        """
        X: (n_samples, n_features)
        """
        y_pred = []

        for x in X:
            posteriors = [
                self._log_posterior(x, c) for c in self.classes
            ]
            y_pred.append(self.classes[np.argmax(posteriors)])

        return np.array(y_pred)


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    # Toy dataset
    X = np.array([
        [1.0, 2.1],
        [1.2, 1.9],
        [3.0, 3.1],
        [3.2, 2.9]
    ])
    y = np.array([0, 0, 1, 1])

    model = GaussianNaiveBayes()
    model.fit(X, y)

    X_test = np.array([[1.1, 2.0], [3.1, 3.0]])
    preds = model.predict(X_test)

    print("Predictions:", preds)

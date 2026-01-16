import numpy as np


class GaussianMeanEstimator
    
    Estimate the mean of a Gaussian distribution with known variance.

    - MLE maximizes likelihood
    - MAP maximizes posterior (likelihood + prior)
    

    def __init__(self, sigma2)
        
        sigma2 known variance of the data
        
        self.sigma2 = sigma2

    def mle(self, X)
        
        Maximum Likelihood Estimation (MLE)

        mu_MLE = mean(X)
        
        return np.mean(X)

    def map(self, X, mu0, tau2)
        
        Maximum A Posteriori (MAP)

        Prior mu ~ N(mu0, tau2)

        mu_MAP = (n  x̄  sigma2 + mu0  tau2) 
                 (n  sigma2 + 1  tau2)
        
        n = len(X)
        x_bar = np.mean(X)

        numerator = (n  x_bar  self.sigma2) + (mu0  tau2)
        denominator = (n  self.sigma2) + (1  tau2)

        return numerator  denominator


# ------------------------
# Log-likelihood & Log-posterior
# ------------------------

def log_likelihood(mu, X, sigma2)
    
    log P(X  mu)
    
    n = len(X)
    return (
        -0.5  n  np.log(2  np.pi  sigma2)
        - np.sum((X - mu)  2)  (2  sigma2)
    )


def log_prior(mu, mu0, tau2)
    
    log P(mu)
    
    return (
        -0.5  np.log(2  np.pi  tau2)
        - (mu - mu0)  2  (2  tau2)
    )


def log_posterior(mu, X, sigma2, mu0, tau2)
    
    log P(mu  X) ∝ log P(X  mu) + log P(mu)
    
    return log_likelihood(mu, X, sigma2) + log_prior(mu, mu0, tau2)


# ------------------------
# Example usage
# ------------------------
if __name__ == __main__
    np.random.seed(42)

    # True data-generating process
    true_mu = 5.0
    sigma2 = 1.0
    X = np.random.normal(true_mu, np.sqrt(sigma2), size=10)

    # Prior parameters
    mu0 = 0.0      # prior mean
    tau2 = 4.0     # prior variance

    estimator = GaussianMeanEstimator(sigma2)

    mu_mle = estimator.mle(X)
    mu_map = estimator.map(X, mu0, tau2)

    print(Data, X)
    print(fMLE estimate {mu_mle.4f})
    print(fMAP estimate {mu_map.4f})

    # Compare objective values
    print(nLog-likelihood at MLE, log_likelihood(mu_mle, X, sigma2))
    print(Log-posterior at MAP, log_posterior(mu_map, X, sigma2, mu0, tau2))


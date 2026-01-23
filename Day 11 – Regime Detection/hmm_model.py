import numpy as np
from hmmlearn.hmm import GaussianHMM

def train_hmm(returns, n_states=2):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    model.fit(returns)
    hidden_states = model.predict(returns)
    return model, hidden_states

from data_loader import load_returns
from hmm_model import train_hmm
import numpy as np

def regime_statistics(returns, states):
    for i in range(len(set(states))):
        state_returns = returns[states == i]
        print(f"Regime {i}: Mean={state_returns.mean():.5f}, Vol={state_returns.std():.5f}")

def run():
    data = load_returns()
    returns = data["Returns"].values.reshape(-1, 1)

    model, states = train_hmm(returns)

    data["Regime"] = states
    regime_statistics(data["Returns"], data["Regime"])

    return data

if __name__ == "__main__":
    run()

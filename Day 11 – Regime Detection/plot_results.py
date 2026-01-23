import matplotlib.pyplot as plt

def plot_regimes(data):
    plt.figure(figsize=(12,6))
    for regime in data["Regime"].unique():
        subset = data[data["Regime"] == regime]
        plt.scatter(
            subset.index,
            subset["Adj Close"],
            label=f"Regime {regime}",
            s=10
        )
    plt.legend()
    plt.title("Market Regimes Detected by HMM")
    plt.show()

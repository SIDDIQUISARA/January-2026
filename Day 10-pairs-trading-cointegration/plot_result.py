import matplotlib.pyplot as plt

def plot_strategy(spread, signals):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(spread)
    axs[0].set_title("Spread")

    axs[1].plot(signals["z_score"])
    axs[1].axhline(1, color='r', linestyle='--')
    axs[1].axhline(-1, color='r', linestyle='--')
    axs[1].set_title("Z-Score")

    axs[2].plot(signals["cumulative_pnl"])
    axs[2].set_title("Cumulative PnL")

    plt.tight_layout()
    plt.show()

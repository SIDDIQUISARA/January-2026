import numpy as np
import matplotlib.pyplot as plt


# ------------------------
# Brier Score
# ------------------------

def brier_score(y_true, y_prob):
    """
    Brier Score for binary classification

    y_true: (n,) binary labels {0,1}
    y_prob: (n,) predicted probabilities for class 1
    """
    return np.mean((y_prob - y_true) ** 2)


# ------------------------
# Reliability Curve
# ------------------------

def reliability_curve(y_true, y_prob, n_bins=10):
    """
    Computes points for a reliability (calibration) curve.

    Returns:
        bin_confidence: average predicted probability per bin
        bin_accuracy: empirical accuracy per bin
        bin_counts: number of samples per bin
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    bin_confidence = []
    bin_accuracy = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            bin_confidence.append(np.mean(y_prob[mask]))
            bin_accuracy.append(np.mean(y_true[mask]))
            bin_counts.append(np.sum(mask))

    return (
        np.array(bin_confidence),
        np.array(bin_accuracy),
        np.array(bin_counts),
    )


def plot_reliability_curve(confidence, accuracy):
    """
    Plot reliability curve
    """
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(confidence, accuracy, "o-", label="Model")

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # Simulated classifier outputs
    n = 1000
    y_prob = np.random.rand(n)

    # Simulated true labels (slightly miscalibrated)
    y_true = (y_prob + 0.1 * np.random.randn(n) > 0.5).astype(int)

    # Brier score
    score = brier_score(y_true, y_prob)
    print(f"Brier Score: {score:.4f}")

    # Reliability curve
    conf, acc, counts = reliability_curve(y_true, y_prob, n_bins=10)

    print("\nBin confidence:", conf)
    print("Bin accuracy:", acc)
    print("Bin counts:", counts)

    plot_reliability_curve(conf, acc)

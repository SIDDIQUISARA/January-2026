import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from train_models import train_models

def evaluate(models, X_test, y_test):
    results = []

    for name, pipe in models.items():
        preds = pipe.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        coef = pipe.named_steps["model"].coef_
        non_zero = np.sum(coef != 0)

        results.append({
            "Model": name,
            "MSE": mse,
            "NonZeroCoefficients": non_zero
        })

        # Plot coefficients
        plt.figure(figsize=(10, 3))
        plt.bar(range(len(coef)), coef)
        plt.title(f"{name} Coefficient Magnitudes")
        plt.xlabel("Feature Index")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(f"results/{name}_coefficients.png")
        plt.close()

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = train_models(X_train, y_train)
    results = evaluate(models, X_test, y_test)

    results.to_csv("results/metrics_comparison.csv", index=False)
    print(results)


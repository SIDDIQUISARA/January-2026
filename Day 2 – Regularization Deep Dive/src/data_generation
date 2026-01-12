import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

def generate_data(
    n_samples=1000,
    n_features=100,
    n_informative=10,
    noise=20,
    random_state=42
):
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        coef=True,
        random_state=random_state
    )

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    df["target"] = y

    return df, coef


if __name__ == "__main__":
    df, true_coef = generate_data()
    df.to_csv("data/synthetic_data.csv", index=False)
    print("Data generated and saved.")


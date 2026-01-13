"""
Day 3 – Feature Selection Methods
Methods Covered:
1. Mutual Information
2. Chi-Square Test
3. Recursive Feature Elimination (RFE)

Dataset: Breast Cancer (sklearn)
"""

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import (
    mutual_info_classif,
    SelectKBest,
    chi2,
    RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset shape:", X.shape)

# --------------------------------------------------
# 1. Mutual Information
# --------------------------------------------------
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({
    "Feature": X.columns,
    "MI Score": mi_scores
}).sort_values(by="MI Score", ascending=False)

print("\nTop 5 Features (Mutual Information):")
print(mi_df.head())

# --------------------------------------------------
# 2. Chi-Square Test
# (Requires non-negative features)
# --------------------------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

chi2_selector = SelectKBest(score_func=chi2, k=5)
chi2_selector.fit(X_scaled, y)

chi2_features = X.columns[chi2_selector.get_support()]

print("\nTop 5 Features (Chi-Square):")
print(list(chi2_features))

# --------------------------------------------------
# 3. Recursive Feature Elimination (RFE)
# --------------------------------------------------
model = LogisticRegression(max_iter=5000)
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X, y)

rfe_features = X.columns[rfe.support_]

print("\nTop 5 Features (RFE):")
print(list(rfe_features))

# --------------------------------------------------
# Summary
# --------------------------------------------------
summary_df = pd.DataFrame({
    "Mutual Info Top 5": mi_df["Feature"].head(5).values,
    "Chi-Square Top 5": chi2_features,
    "RFE Top 5": rfe_features
})

print("\nFeature Selection Comparison:")
print(summary_df)

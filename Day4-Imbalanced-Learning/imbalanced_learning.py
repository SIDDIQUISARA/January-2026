"""
Day 4 – Imbalanced Learning

Techniques Covered:
1. Baseline Model
2. SMOTE (Oversampling)
3. Class Weighting
4. Threshold Tuning
5. ROC & Precision-Recall Curves

Author: Your Name
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE


# Create Imbalanced Dataset
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=5,
    n_redundant=2,
    weights=[0.9, 0.1],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training class distribution:", np.bincount(y_train))
print("Testing class distribution:", np.bincount(y_test))


# 1. Baseline Model
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)

y_pred_base = baseline_model.predict(X_test)
y_prob_base = baseline_model.predict_proba(X_test)[:, 1]

print("\n--- Baseline Model ---")
print(confusion_matrix(y_test, y_pred_base))
print(classification_report(y_test, y_pred_base))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_base))


# 2. SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:", np.bincount(y_train_smote))

smote_model = LogisticRegression(max_iter=1000)
smote_model.fit(X_train_smote, y_train_smote)

y_pred_smote = smote_model.predict(X_test)
y_prob_smote = smote_model.predict_proba(X_test)[:, 1]

print("\n--- SMOTE Model ---")
print(confusion_matrix(y_test, y_pred_smote))
print(classification_report(y_test, y_pred_smote))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_smote))


# 3. Class Weighting
weighted_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

weighted_model.fit(X_train, y_train)

y_pred_weighted = weighted_model.predict(X_test)
y_prob_weighted = weighted_model.predict_proba(X_test)[:, 1]

print("\n--- Class Weight Model ---")
print(confusion_matrix(y_test, y_pred_weighted))
print(classification_report(y_test, y_pred_weighted))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_weighted))


# 4. Threshold Tuning
threshold = 0.3
y_pred_custom = (y_prob_weighted >= threshold).astype(int)

print(f"\n--- Threshold Tuning (threshold={threshold}) ---")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))


# 5. ROC Curve
plt.figure(figsize=(8, 6))

for label, probs in [
    ("Baseline", y_prob_base),
    ("SMOTE", y_prob_smote),
    ("Class Weight", y_prob_weighted)
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=label)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show()


# 6. Precision-Recall Curve
plt.figure(figsize=(8, 6))

for label, probs in [
    ("Baseline", y_prob_base),
    ("SMOTE", y_prob_smote),
    ("Class Weight", y_prob_weighted)
]:
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.plot(recall, precision, label=label)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.show()

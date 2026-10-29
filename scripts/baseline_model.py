from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script trains and evaluates a simple baseline classifier.
#
# It is NOT the final model.
#
# The aim is to test whether a basic machine-learning pipeline
# can run on the prepared gene-expression data.
#
# Input:
# X = cleaned gene-expression probe features
# y = fast_growing target label
#
# Pipeline:
# StandardScaler -> SelectKBest -> LogisticRegression
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Find project folder
# ------------------------------------------------------------
# This script is inside the scripts/ folder.
# parent.parent moves up to the main project folder.

project_folder = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# 2. Load X and y
# ------------------------------------------------------------

X_path = project_folder / "outputs" / "X_expression_features.csv"
y_path = project_folder / "outputs" / "y_fast_growing_label.csv"

print("Loading X from:")
print(X_path)

print("\nLoading y from:")
print(y_path)

X = pd.read_csv(X_path)
y = pd.read_csv(y_path).squeeze()

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nTarget label counts:")
print(y.value_counts())

print("\nTarget label proportions:")
print(y.value_counts(normalize=True))


# ------------------------------------------------------------
# 3. Create baseline modelling pipeline
# ------------------------------------------------------------
# StandardScaler:
# Standardises expression features.
#
# SelectKBest:
# Selects the top 50 probes that differ most between
# fast-growing and remaining tumours.
#
# LogisticRegression:
# Simple baseline classifier.
#
# class_weight="balanced":
# Helps deal with the 60 vs 21 class imbalance.
# ------------------------------------------------------------

baseline_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_classif, k=50)),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])


# ------------------------------------------------------------
# 4. Set up stratified cross-validation
# ------------------------------------------------------------
# StratifiedKFold keeps the class proportions similar
# in each train/test split.
#
# We use 5 folds as an initial simple baseline.
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 5. Define evaluation metrics
# ------------------------------------------------------------
# Accuracy alone can be misleading because the classes are imbalanced.
#
# Balanced accuracy, precision, recall, F1, and ROC-AUC are more useful.
# ------------------------------------------------------------

scoring = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]


# ------------------------------------------------------------
# 6. Run cross-validation
# ------------------------------------------------------------

print("\nRunning 5-fold stratified cross-validation...")

cv_results = cross_validate(
    baseline_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=True
)


# ------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------

print("\nCross-validation results")
print("=" * 60)

for metric in scoring:
    train_scores = cv_results["train_" + metric]
    test_scores = cv_results["test_" + metric]

    print("\nMetric:", metric)
    print("Train scores:", train_scores)
    print("Test scores:", test_scores)
    print("Mean train score:", train_scores.mean())
    print("Mean test score:", test_scores.mean())

# ------------------------------------------------------------
# 9. Save mean results to a CSV file
# ------------------------------------------------------------

summary_rows = []

for metric in scoring:
    train_scores = cv_results["train_" + metric]
    test_scores = cv_results["test_" + metric]

    summary_rows.append({
        "metric": metric,
        "mean_train_score": train_scores.mean(),
        "mean_test_score": test_scores.mean(),
        "std_test_score": test_scores.std()
    })

results_summary = pd.DataFrame(summary_rows)

results_output_path = project_folder / "outputs" / "baseline_model_results.csv"
results_summary.to_csv(results_output_path, index=False)

print("\nSaved baseline model results:")
print(results_output_path)

print("\nBaseline model results summary:")
print(results_summary)

# ------------------------------------------------------------
# 8. Final explanation
# ------------------------------------------------------------

print("\nBaseline modelling complete.")
print("This is an initial baseline, not the final model.")
print("Pipeline used: StandardScaler -> SelectKBest(k=50) -> LogisticRegression")
print("Cross-validation used: 5-fold StratifiedKFold")
print("Feature selection was performed inside the pipeline to reduce data leakage risk.")
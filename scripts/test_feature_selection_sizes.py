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
# This script tests different numbers of selected expression probes.
#
# In the first baseline model, we used SelectKBest(k=50).
# Here, we test several k values to see whether the model's
# performance is stable or depends strongly on one choice of k.
#
# This is still early baseline modelling, not the final model.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Find project folder and load X and y
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

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


# ------------------------------------------------------------
# 2. Set up cross-validation
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 3. Define evaluation metrics
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
# 4. Try different numbers of selected probes
# ------------------------------------------------------------
# k means how many expression probes are kept after feature selection.
#
# Example:
# k=10 means keep only the top 10 probes.
# k=500 means keep the top 500 probes.
# ------------------------------------------------------------

k_values = [10, 25, 50, 100, 250, 500]

all_results = []

for k in k_values:
    print("\n" + "=" * 70)
    print("Testing SelectKBest with k =", k)
    print("=" * 70)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )

    for metric in scoring:
        train_scores = cv_results["train_" + metric]
        test_scores = cv_results["test_" + metric]

        result_row = {
            "k": k,
            "metric": metric,
            "mean_train_score": train_scores.mean(),
            "mean_test_score": test_scores.mean(),
            "std_test_score": test_scores.std()
        }

        all_results.append(result_row)

        print("\nMetric:", metric)
        print("Mean train score:", train_scores.mean())
        print("Mean test score:", test_scores.mean())
        print("Test score standard deviation:", test_scores.std())


# ------------------------------------------------------------
# 5. Save all results
# ------------------------------------------------------------

results_df = pd.DataFrame(all_results)

results_output_path = project_folder / "outputs" / "feature_selection_k_comparison.csv"

results_df.to_csv(results_output_path, index=False)

print("\n" + "=" * 70)
print("Saved full k comparison results:")
print(results_output_path)


# ------------------------------------------------------------
# 6. Create an easier summary table
# ------------------------------------------------------------
# This table puts the mean test scores side by side for each k.
# It is easier to read than the long results table.
# ------------------------------------------------------------

summary_table = results_df.pivot(
    index="k",
    columns="metric",
    values="mean_test_score"
)

summary_output_path = project_folder / "outputs" / "feature_selection_k_summary.csv"

summary_table.to_csv(summary_output_path)

print("\nSaved summary table:")
print(summary_output_path)

print("\nMean test score summary:")
print(summary_table)


# ------------------------------------------------------------
# 7. Final explanation
# ------------------------------------------------------------

print("\nFeature-selection comparison complete.")
print("This tested different values of k for SelectKBest.")
print("The aim is to see whether performance is stable across different numbers of selected probes.")
print("These results should be interpreted carefully because the dataset has only 81 samples.")
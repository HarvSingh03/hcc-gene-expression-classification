from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    permutation_test_score
)
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script performs a permutation-label sanity check on the
# strongest selected-probe model from the previous modelling work.
#
# Previous strongest selected-probe model:
#
#   StandardScaler
#   -> SelectKBest(k=10)
#   -> Random Forest
#
# Felipe suggested that if the LSA / TruncatedSVD results are poorer
# than the selected-probe results, then the previous selected-probe
# results may have been overfitted.
#
# This script checks whether the model performs meaningfully better
# with the real fast_growing labels than with randomly shuffled labels.
#
# If the real-label score is clearly above the shuffled-label scores,
# that supports the presence of real label-associated signal.
#
# If shuffled labels can achieve similar scores, that would suggest
# the previous result may be overly optimistic / overfitted.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Suppress common metric warnings
# ------------------------------------------------------------

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------------------------------------------
# 2. Set pandas display options
# ------------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)


# ------------------------------------------------------------
# 3. Find project folders
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

outputs_folder = project_folder / "outputs"
plots_folder = project_folder / "plots"

outputs_folder.mkdir(exist_ok=True)
plots_folder.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 4. Load X and y
# ------------------------------------------------------------

x_path = outputs_folder / "X_expression_features.csv"
y_path = outputs_folder / "y_fast_growing_label.csv"

print("Loading X expression features:")
print(x_path)

if not x_path.exists():
    raise FileNotFoundError(
        "Could not find X_expression_features.csv. "
        "Please run prepare_modelling_data.py first."
    )

print("\nLoading y fast-growing labels:")
print(y_path)

if not y_path.exists():
    raise FileNotFoundError(
        "Could not find y_fast_growing_label.csv. "
        "Please run prepare_modelling_data.py first."
    )

X = pd.read_csv(x_path)
y_df = pd.read_csv(y_path)

if "fast_growing" in y_df.columns:
    y = y_df["fast_growing"]
else:
    y = y_df.iloc[:, 0]

print("\nData loaded.")
print("X shape:")
print(X.shape)
print("y shape:")
print(y.shape)

print("\nTarget class counts:")
print(y.value_counts())

print("\nMissing values in X:")
print(X.isna().sum().sum())

print("\nMissing values in y:")
print(y.isna().sum())


# ------------------------------------------------------------
# 5. Define strongest selected-probe model
# ------------------------------------------------------------

# This is the strongest selected-probe baseline from the previous work:
#
#   StandardScaler -> SelectKBest(k=10) -> Random Forest
#
# Important:
# SelectKBest is inside the pipeline, so feature selection happens
# within each training fold during cross-validation.

best_model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(
        score_func=f_classif,
        k=10
    )),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nBest model pipeline:")
print("StandardScaler -> SelectKBest(k=10, f_classif) -> RandomForestClassifier")


# ------------------------------------------------------------
# 6. Re-evaluate real-label performance using repeated CV
# ------------------------------------------------------------

# This repeats the previous evaluation style:
#
#   5 folds x 20 repeats = 100 evaluations

repeated_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(
        precision_score,
        zero_division=0
    ),
    "recall": make_scorer(
        recall_score,
        zero_division=0
    ),
    "f1": make_scorer(
        f1_score,
        zero_division=0
    ),
    "roc_auc": "roc_auc"
}

print("\nRunning real-label repeated cross-validation...")
print("5 folds x 20 repeats = 100 evaluations")

real_cv_results = cross_validate(
    best_model_pipeline,
    X,
    y,
    cv=repeated_cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=1
)

real_results_df = pd.DataFrame({
    "fold_run": range(1, len(real_cv_results["test_accuracy"]) + 1),
    "fit_time": real_cv_results["fit_time"],
    "score_time": real_cv_results["score_time"],
    "train_accuracy": real_cv_results["train_accuracy"],
    "test_accuracy": real_cv_results["test_accuracy"],
    "train_balanced_accuracy": real_cv_results["train_balanced_accuracy"],
    "test_balanced_accuracy": real_cv_results["test_balanced_accuracy"],
    "train_precision": real_cv_results["train_precision"],
    "test_precision": real_cv_results["test_precision"],
    "train_recall": real_cv_results["train_recall"],
    "test_recall": real_cv_results["test_recall"],
    "train_f1": real_cv_results["train_f1"],
    "test_f1": real_cv_results["test_f1"],
    "train_roc_auc": real_cv_results["train_roc_auc"],
    "test_roc_auc": real_cv_results["test_roc_auc"]
})

real_results_output_path = outputs_folder / "permutation_test_real_label_repeated_cv_results.csv"
real_results_df.to_csv(real_results_output_path, index=False)

print("\nSaved real-label repeated CV results:")
print(real_results_output_path)


# ------------------------------------------------------------
# 7. Summarise real-label repeated CV performance
# ------------------------------------------------------------

real_summary_rows = []

for metric in [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]:
    test_col = f"test_{metric}"
    train_col = f"train_{metric}"

    real_summary_rows.append({
        "metric": metric,
        "mean_train_score": real_results_df[train_col].mean(),
        "std_train_score": real_results_df[train_col].std(),
        "mean_test_score": real_results_df[test_col].mean(),
        "std_test_score": real_results_df[test_col].std(),
        "test_score_mean_plus_minus_std": (
            f"{real_results_df[test_col].mean():.4f} ± {real_results_df[test_col].std():.4f}"
        )
    })

real_summary_df = pd.DataFrame(real_summary_rows)

real_summary_output_path = outputs_folder / "permutation_test_real_label_repeated_cv_summary.csv"
real_summary_df.to_csv(real_summary_output_path, index=False)

print("\nReal-label repeated CV summary:")
print(real_summary_df)

print("\nSaved real-label repeated CV summary:")
print(real_summary_output_path)


# ------------------------------------------------------------
# 8. Run permutation-label test
# ------------------------------------------------------------

# The permutation test is deliberately focused on balanced accuracy.
#
# Balanced accuracy is useful here because the dataset is imbalanced:
#
#   class 0 = 60 remaining tumours
#   class 1 = 21 fast-growing tumours
#
# The permutation test repeatedly shuffles y, then evaluates the same
# pipeline. This creates a null distribution showing how well the model
# can perform when the labels are random.

# Practical note:
# 100 permutations is a reasonable starting point.
# It gives a useful sanity check without making the script too slow.
#
# If needed later, this can be increased to 500 or 1000, but that may
# take much longer.

n_permutations = 100

permutation_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nRunning permutation-label test...")
print("Scoring metric: balanced accuracy")
print("Number of permutations:")
print(n_permutations)
print("CV used inside permutation test: 5-fold StratifiedKFold")

real_score, permutation_scores, p_value = permutation_test_score(
    estimator=best_model_pipeline,
    X=X,
    y=y,
    scoring="balanced_accuracy",
    cv=permutation_cv,
    n_permutations=n_permutations,
    n_jobs=1,
    random_state=42
)

print("\nPermutation test complete.")

print("\nReal-label balanced accuracy from permutation-test CV:")
print(real_score)

print("\nMean shuffled-label balanced accuracy:")
print(np.mean(permutation_scores))

print("\nStandard deviation of shuffled-label balanced accuracy:")
print(np.std(permutation_scores, ddof=1))

print("\nMinimum shuffled-label balanced accuracy:")
print(np.min(permutation_scores))

print("\nMaximum shuffled-label balanced accuracy:")
print(np.max(permutation_scores))

print("\nPermutation test p-value:")
print(p_value)


# ------------------------------------------------------------
# 9. Save permutation scores
# ------------------------------------------------------------

permutation_scores_df = pd.DataFrame({
    "permutation_number": range(1, len(permutation_scores) + 1),
    "shuffled_label_balanced_accuracy": permutation_scores
})

permutation_scores_output_path = outputs_folder / "permutation_test_shuffled_label_scores.csv"
permutation_scores_df.to_csv(permutation_scores_output_path, index=False)

print("\nSaved shuffled-label permutation scores:")
print(permutation_scores_output_path)


# ------------------------------------------------------------
# 10. Save permutation summary
# ------------------------------------------------------------

permutation_summary_df = pd.DataFrame([
    {
        "model": "StandardScaler_SelectKBest_k10_RandomForest",
        "metric": "balanced_accuracy",
        "n_permutations": n_permutations,
        "real_label_score": real_score,
        "mean_shuffled_label_score": np.mean(permutation_scores),
        "std_shuffled_label_score": np.std(permutation_scores, ddof=1),
        "min_shuffled_label_score": np.min(permutation_scores),
        "max_shuffled_label_score": np.max(permutation_scores),
        "p_value": p_value
    }
])

permutation_summary_output_path = outputs_folder / "permutation_test_best_model_summary.csv"
permutation_summary_df.to_csv(permutation_summary_output_path, index=False)

print("\nSaved permutation test summary:")
print(permutation_summary_output_path)

print("\nPermutation test summary:")
print(permutation_summary_df)


# ------------------------------------------------------------
# 11. Plot permutation-score distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    permutation_scores,
    bins=20,
    alpha=0.8,
    edgecolor="black"
)

plt.axvline(
    real_score,
    linestyle="--",
    linewidth=2,
    label=f"Real labels = {real_score:.3f}"
)

plt.xlabel("Balanced accuracy")
plt.ylabel("Frequency")
plt.title("Permutation test: real labels vs shuffled labels")
plt.legend()
plt.tight_layout()

permutation_plot_path = plots_folder / "permutation_test_best_model_balanced_accuracy.png"
plt.savefig(permutation_plot_path, dpi=300)
plt.close()

print("\nSaved permutation test plot:")
print(permutation_plot_path)


# ------------------------------------------------------------
# 12. Save text summary
# ------------------------------------------------------------

text_summary_path = outputs_folder / "permutation_test_best_model_summary.txt"

with open(text_summary_path, "w", encoding="utf-8") as f:
    f.write("Permutation Test for Best Selected-Probe Model\n")
    f.write("=============================================\n\n")

    f.write("Purpose\n")
    f.write("-------\n")
    f.write("This analysis tested whether the strongest selected-probe model performs meaningfully better with the real fast_growing labels than with shuffled labels.\n")
    f.write("This was done as a sanity check for possible overfitting, following supervisor feedback that weaker LSA results may indicate that previous selected-probe results were overfitted.\n\n")

    f.write("Model tested\n")
    f.write("------------\n")
    f.write("StandardScaler -> SelectKBest(k=10, f_classif) -> RandomForestClassifier\n\n")

    f.write("Dataset\n")
    f.write("-------\n")
    f.write(f"Samples: {X.shape[0]}\n")
    f.write(f"Expression probe features: {X.shape[1]}\n")
    f.write("Target: fast_growing\n\n")

    f.write("Class counts\n")
    f.write("------------\n")
    f.write(y.value_counts().to_string())
    f.write("\n\n")

    f.write("Real-label repeated CV setup\n")
    f.write("----------------------------\n")
    f.write("RepeatedStratifiedKFold with 5 folds and 20 repeats.\n")
    f.write("This gives 100 evaluations.\n\n")

    f.write("Real-label repeated CV summary\n")
    f.write("------------------------------\n")
    f.write(real_summary_df.to_string(index=False))
    f.write("\n\n")

    f.write("Permutation test setup\n")
    f.write("----------------------\n")
    f.write("Scoring metric: balanced accuracy\n")
    f.write(f"Number of permutations: {n_permutations}\n")
    f.write("CV inside permutation test: 5-fold StratifiedKFold\n\n")

    f.write("Permutation test results\n")
    f.write("------------------------\n")
    f.write(f"Real-label balanced accuracy: {real_score}\n")
    f.write(f"Mean shuffled-label balanced accuracy: {np.mean(permutation_scores)}\n")
    f.write(f"Standard deviation of shuffled-label balanced accuracy: {np.std(permutation_scores, ddof=1)}\n")
    f.write(f"Minimum shuffled-label balanced accuracy: {np.min(permutation_scores)}\n")
    f.write(f"Maximum shuffled-label balanced accuracy: {np.max(permutation_scores)}\n")
    f.write(f"Permutation p-value: {p_value}\n\n")

    f.write("Interpretation guide\n")
    f.write("--------------------\n")
    f.write("If the real-label score is clearly higher than the shuffled-label scores, this supports the presence of real label-associated signal.\n")
    f.write("If shuffled labels often achieve similar performance to the real labels, this suggests that the model may be overfitting or exploiting chance patterns in the high-dimensional data.\n")
    f.write("This test does not prove external generalisation, because an independent validation dataset is still needed for that.\n\n")

    f.write("Output files\n")
    f.write("------------\n")
    f.write(f"- {real_results_output_path}\n")
    f.write(f"- {real_summary_output_path}\n")
    f.write(f"- {permutation_scores_output_path}\n")
    f.write(f"- {permutation_summary_output_path}\n")
    f.write(f"- {text_summary_path}\n")
    f.write(f"- {permutation_plot_path}\n")

print("\nSaved text summary:")
print(text_summary_path)


# ------------------------------------------------------------
# 13. Final message
# ------------------------------------------------------------

print("\nPermutation test script complete.")

print("\nMain outputs:")
print(real_results_output_path)
print(real_summary_output_path)
print(permutation_scores_output_path)
print(permutation_summary_output_path)
print(text_summary_path)

print("\nMain plot:")
print(permutation_plot_path)

print("\nKey result to inspect:")
print("Real-label balanced accuracy vs shuffled-label balanced accuracy distribution.")
print("Also inspect the permutation p-value.")
from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.linear_model._logistic"
)


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script compares a small set of classification models for
# predicting fast-growing HCC tumours from gene-expression probes.
#
# The target is:
# fast_growing = 1: tumour doubling time <= 54 days
# fast_growing = 0: remaining tumour samples
#
# The dataset is high-dimensional:
# 81 tumour samples x 39,613 expression probes
#
# Because there are far more features than samples, feature
# selection is performed inside the cross-validation pipeline.
#
# This avoids data leakage.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Set pandas display options
# ------------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)


# ------------------------------------------------------------
# 2. Find project folders
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

outputs_folder = project_folder / "outputs"
plots_folder = project_folder / "plots"

outputs_folder.mkdir(exist_ok=True)
plots_folder.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 3. Load X and y
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

# y file may either have one column called fast_growing,
# or just one unnamed/first label column.
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
# 4. Define cross-validation strategy
# ------------------------------------------------------------
# Repeated stratified cross-validation is used because:
#
# - the dataset is small
# - the classes are imbalanced
# - a single train/test split could give unstable results
#
# For this first test run:
# 5 folds x 5 repeats = 25 test evaluations per model
#
# Later, once the script is confirmed to run correctly, this can
# be increased back to 20 repeats:
# 5 folds x 20 repeats = 100 evaluations per model
# ------------------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)

print("\nCross-validation strategy:")
print("RepeatedStratifiedKFold")
print("5 folds x 20 repeats = 100 evaluations per model")


# ------------------------------------------------------------
# 5. Define scoring metrics
# ------------------------------------------------------------
# Accuracy is included, but it is not the most important metric
# because the dataset is imbalanced.
#
# More important:
# - balanced accuracy
# - precision
# - recall
# - F1-score
# - ROC-AUC
#
# zero_division=0 prevents warnings when a model predicts no
# positive fast-growing samples. This happens with the dummy
# classifier and is expected.
# ------------------------------------------------------------

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc"
}


# ------------------------------------------------------------
# 6. Define feature selection
# ------------------------------------------------------------
# SelectKBest chooses the top k probes according to how strongly
# each probe differs between the two classes.
#
# k=25 is used because previous exploration showed this was a
# reasonable baseline size and produced stable candidate probes.
# ------------------------------------------------------------

k_features = 25


# ------------------------------------------------------------
# 7. Define models
# ------------------------------------------------------------

models = {}

# ------------------------------------------------------------
# Dummy Classifier
# ------------------------------------------------------------
# This is a trivial baseline.
# It does not learn useful expression patterns.
# It is included to check whether real models beat majority-class
# style prediction.
# ------------------------------------------------------------

models["Dummy_most_frequent"] = DummyClassifier(
    strategy="most_frequent"
)


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------
# Simple interpretable linear classifier.
# class_weight="balanced" helps account for class imbalance.
# ------------------------------------------------------------

models["Logistic_Regression"] = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    ))
])


# ------------------------------------------------------------
# Elastic Net Logistic Regression
# ------------------------------------------------------------
# Elastic Net combines L1 and L2 regularisation.
# This is useful when predictors may be correlated.
#
# solver="saga" supports elasticnet penalty.
# l1_ratio=0.5 means equal mix of L1 and L2.
# ------------------------------------------------------------

models["Elastic_Net_Logistic_Regression"] = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
    ("classifier", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        class_weight="balanced",
        max_iter=10000,
        random_state=42
    ))
])


# ------------------------------------------------------------
# Linear Support Vector Machine
# ------------------------------------------------------------
# A linear SVM is suitable for high-dimensional data.
#
# LinearSVC provides a decision_function, so ROC-AUC can still
# be calculated by scikit-learn.
# ------------------------------------------------------------

models["Linear_SVM"] = Pipeline([
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
    ("classifier", LinearSVC(
        class_weight="balanced",
        max_iter=10000,
        random_state=42
    ))
])


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------
# Non-linear comparison model.
# It can capture interactions between selected features.
#
# It is used cautiously because the sample size is small.
# Feature selection is applied before the forest.
# ------------------------------------------------------------

models["Random_Forest"] = Pipeline([
    ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42
    ))
])


# ------------------------------------------------------------
# Gaussian Naive Bayes
# ------------------------------------------------------------
# Simple probabilistic classifier for continuous features.
#
# Limitation:
# it assumes features are conditionally independent, which is not
# fully true for this dataset because correlation heatmaps showed
# correlated probes.
# ------------------------------------------------------------

models["Gaussian_Naive_Bayes"] = Pipeline([
    ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
    ("classifier", GaussianNB())
])


# ------------------------------------------------------------
# 8. Run model comparison
# ------------------------------------------------------------

all_results = []

print("\nStarting model comparison...")

for model_name, model in models.items():
    print("\n" + "=" * 70)
    print(f"Evaluating model: {model_name}")
    print("=" * 70)

    cv_results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1
    )

    for fold_index in range(len(cv_results["test_accuracy"])):
        result_row = {
            "model": model_name,
            "fold_run": fold_index + 1,
            "fit_time": cv_results["fit_time"][fold_index],
            "score_time": cv_results["score_time"][fold_index],
        }

        for metric in scoring.keys():
            result_row[f"train_{metric}"] = cv_results[f"train_{metric}"][fold_index]
            result_row[f"test_{metric}"] = cv_results[f"test_{metric}"][fold_index]

        all_results.append(result_row)

    print("Mean test scores:")
    for metric in scoring.keys():
        mean_score = cv_results[f"test_{metric}"].mean()
        std_score = cv_results[f"test_{metric}"].std()
        print(f"{metric}: mean={mean_score:.4f}, std={std_score:.4f}")


results_df = pd.DataFrame(all_results)


# ------------------------------------------------------------
# 9. Save full model comparison results
# ------------------------------------------------------------

results_output_path = outputs_folder / "model_comparison_results.csv"
results_df.to_csv(results_output_path, index=False)

print("\nSaved full model comparison results:")
print(results_output_path)


# ------------------------------------------------------------
# 10. Create summary table
# ------------------------------------------------------------

summary_rows = []

for model_name in results_df["model"].unique():
    model_results = results_df[results_df["model"] == model_name]

    summary_row = {
        "model": model_name,
        "n_evaluations": model_results.shape[0]
    }

    for metric in scoring.keys():
        summary_row[f"mean_test_{metric}"] = model_results[f"test_{metric}"].mean()
        summary_row[f"std_test_{metric}"] = model_results[f"test_{metric}"].std()
        summary_row[f"mean_train_{metric}"] = model_results[f"train_{metric}"].mean()

    summary_rows.append(summary_row)

summary_df = pd.DataFrame(summary_rows)

# Sort by balanced accuracy first because this is important for imbalanced classes
summary_df = summary_df.sort_values(
    by="mean_test_balanced_accuracy",
    ascending=False
)

summary_output_path = outputs_folder / "model_comparison_summary.csv"
summary_df.to_csv(summary_output_path, index=False)

print("\nModel comparison summary:")
print(summary_df)

print("\nSaved model comparison summary:")
print(summary_output_path)


# ------------------------------------------------------------
# 11. Save text summary
# ------------------------------------------------------------

text_summary_path = outputs_folder / "model_comparison_summary.txt"

with open(text_summary_path, "w", encoding="utf-8") as f:
    f.write("Model Comparison Summary\n")
    f.write("========================\n\n")

    f.write("Task:\n")
    f.write("Binary classification of fast-growing vs remaining HCC tumour samples.\n\n")

    f.write("Dataset:\n")
    f.write(f"Samples: {X.shape[0]}\n")
    f.write(f"Expression probe features: {X.shape[1]}\n")
    f.write("Target: fast_growing\n\n")

    f.write("Class counts:\n")
    f.write(y.value_counts().to_string())
    f.write("\n\n")

    f.write("Cross-validation:\n")
    f.write("RepeatedStratifiedKFold with 5 folds and 20 repeats.\n")
    f.write("This gives 100 evaluations per model.\n\n")

    f.write("Feature selection:\n")
    f.write("SelectKBest(f_classif, k=25) was used inside each real model pipeline.\n")
    f.write("This means feature selection was fitted only on training folds during cross-validation.\n\n")

    f.write("Models compared:\n")
    for model_name in models.keys():
        f.write(f"- {model_name}\n")

    f.write("\nSummary table sorted by mean test balanced accuracy:\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")

    f.write("Interpretation notes:\n")
    f.write("- Accuracy should not be interpreted alone because the classes are imbalanced.\n")
    f.write("- Balanced accuracy, recall, F1-score and ROC-AUC are more informative.\n")
    f.write("- High train scores compared with test scores may indicate overfitting.\n")
    f.write("- Random Forest is included as a non-linear comparison but should be interpreted cautiously because of the small sample size.\n")
    f.write("- Gaussian Naive Bayes is included as a simple probabilistic comparison, but its independence assumption is a limitation because expression probes can be correlated.\n")
    f.write("- These results are initial experimental results and may be refined after interpretation.\n")

print("\nSaved text summary:")
print(text_summary_path)


# ------------------------------------------------------------
# 12. Plot helper function
# ------------------------------------------------------------

def save_metric_plot(summary_dataframe, metric_column, ylabel, title, output_filename):
    """
    Create and save a horizontal bar plot for a model-comparison metric.
    """

    plot_df = summary_dataframe.sort_values(
        by=metric_column,
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_df["model"],
        plot_df[metric_column]
    )

    plt.xlabel(ylabel)
    plt.ylabel("Model")
    plt.title(title)

    plt.tight_layout()

    output_path = plots_folder / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


# ------------------------------------------------------------
# 13. Save model comparison plots
# ------------------------------------------------------------

save_metric_plot(
    summary_df,
    "mean_test_balanced_accuracy",
    "Mean test balanced accuracy",
    "Model Comparison: Balanced Accuracy",
    "model_comparison_balanced_accuracy.png"
)

save_metric_plot(
    summary_df,
    "mean_test_f1",
    "Mean test F1-score",
    "Model Comparison: F1-score",
    "model_comparison_f1_score.png"
)

save_metric_plot(
    summary_df,
    "mean_test_roc_auc",
    "Mean test ROC-AUC",
    "Model Comparison: ROC-AUC",
    "model_comparison_roc_auc.png"
)

save_metric_plot(
    summary_df,
    "mean_test_recall",
    "Mean test recall",
    "Model Comparison: Recall",
    "model_comparison_recall.png"
)


# ------------------------------------------------------------
# 14. Final message
# ------------------------------------------------------------

print("\nModel comparison complete.")
print("Main outputs:")
print(results_output_path)
print(summary_output_path)
print(text_summary_path)
print("\nMain plots saved in:")
print(plots_folder)
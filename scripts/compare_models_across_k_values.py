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


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script checks whether model performance is stable across
# different numbers of selected gene-expression probes.
#
# Previous model comparison used SelectKBest(k=25).
# This script tests several k values to see whether results depend
# strongly on the number of selected probes.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Hide repeated sklearn future warnings
# ------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn.linear_model._logistic"
)


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
# 5. Define cross-validation strategy
# ------------------------------------------------------------
# 5 folds x 20 repeats = 100 evaluations for each model/k pair.
# ------------------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)

print("\nCross-validation strategy:")
print("RepeatedStratifiedKFold")
print("5 folds x 20 repeats = 100 evaluations per model/k pair")


# ------------------------------------------------------------
# 6. Define scoring metrics
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
# 7. Define k values to test
# ------------------------------------------------------------
# These values test whether model performance changes when using
# smaller or larger selected feature sets.
# ------------------------------------------------------------

k_values = [10, 25, 50, 100]

print("\nFeature-selection k values to test:")
print(k_values)


# ------------------------------------------------------------
# 8. Function to create models for each k
# ------------------------------------------------------------

def create_models_for_k(k_features):
    """
    Create model pipelines for a given SelectKBest k value.
    """

    models = {}

    # Dummy classifier does not use feature selection.
    # It is included once for each k only so that the summary table
    # has the same structure for all k values.
    models["Dummy_most_frequent"] = DummyClassifier(
        strategy="most_frequent"
    )

    models["Logistic_Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ))
    ])

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

    models["Linear_SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
        ("classifier", LinearSVC(
            class_weight="balanced",
            max_iter=10000,
            random_state=42
        ))
    ])

    models["Random_Forest"] = Pipeline([
        ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42
        ))
    ])

    models["Gaussian_Naive_Bayes"] = Pipeline([
        ("feature_selection", SelectKBest(score_func=f_classif, k=k_features)),
        ("classifier", GaussianNB())
    ])

    return models


# ------------------------------------------------------------
# 9. Run model comparison across k values
# ------------------------------------------------------------

all_results = []

print("\nStarting model comparison across k values...")

for k_features in k_values:
    print("\n" + "#" * 80)
    print(f"Testing k = {k_features}")
    print("#" * 80)

    models = create_models_for_k(k_features)

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Evaluating model: {model_name}, k = {k_features}")
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
                "k": k_features,
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
# 10. Save full results
# ------------------------------------------------------------

results_output_path = outputs_folder / "model_comparison_across_k_results.csv"
results_df.to_csv(results_output_path, index=False)

print("\nSaved full across-k model comparison results:")
print(results_output_path)


# ------------------------------------------------------------
# 11. Create summary table
# ------------------------------------------------------------

summary_rows = []

for k_features in sorted(results_df["k"].unique()):
    k_results = results_df[results_df["k"] == k_features]

    for model_name in k_results["model"].unique():
        model_results = k_results[k_results["model"] == model_name]

        summary_row = {
            "k": k_features,
            "model": model_name,
            "n_evaluations": model_results.shape[0]
        }

        for metric in scoring.keys():
            summary_row[f"mean_test_{metric}"] = model_results[f"test_{metric}"].mean()
            summary_row[f"std_test_{metric}"] = model_results[f"test_{metric}"].std()
            summary_row[f"mean_train_{metric}"] = model_results[f"train_{metric}"].mean()

        summary_rows.append(summary_row)

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    by=["mean_test_balanced_accuracy", "mean_test_f1"],
    ascending=False
)

summary_output_path = outputs_folder / "model_comparison_across_k_summary.csv"
summary_df.to_csv(summary_output_path, index=False)

print("\nModel comparison across k summary:")
print(summary_df)

print("\nSaved across-k summary:")
print(summary_output_path)


# ------------------------------------------------------------
# 12. Identify best result for key metrics
# ------------------------------------------------------------

best_balanced_accuracy = summary_df.loc[
    summary_df["mean_test_balanced_accuracy"].idxmax()
]

best_f1 = summary_df.loc[
    summary_df["mean_test_f1"].idxmax()
]

best_recall = summary_df.loc[
    summary_df["mean_test_recall"].idxmax()
]

best_roc_auc = summary_df.loc[
    summary_df["mean_test_roc_auc"].idxmax()
]

print("\nBest model/k combinations:")
print("\nBest balanced accuracy:")
print(best_balanced_accuracy[[
    "k", "model", "mean_test_balanced_accuracy", "mean_test_f1", "mean_test_recall", "mean_test_roc_auc"
]])

print("\nBest F1-score:")
print(best_f1[[
    "k", "model", "mean_test_balanced_accuracy", "mean_test_f1", "mean_test_recall", "mean_test_roc_auc"
]])

print("\nBest recall:")
print(best_recall[[
    "k", "model", "mean_test_balanced_accuracy", "mean_test_f1", "mean_test_recall", "mean_test_roc_auc"
]])

print("\nBest ROC-AUC:")
print(best_roc_auc[[
    "k", "model", "mean_test_balanced_accuracy", "mean_test_f1", "mean_test_recall", "mean_test_roc_auc"
]])


# ------------------------------------------------------------
# 13. Save text summary
# ------------------------------------------------------------

text_summary_path = outputs_folder / "model_comparison_across_k_summary.txt"

with open(text_summary_path, "w", encoding="utf-8") as f:
    f.write("Model Comparison Across k Values\n")
    f.write("================================\n\n")

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
    f.write("This gives 100 evaluations per model/k pair.\n\n")

    f.write("Feature-selection k values tested:\n")
    for k_features in k_values:
        f.write(f"- k = {k_features}\n")

    f.write("\nModels compared:\n")
    for model_name in create_models_for_k(k_values[0]).keys():
        f.write(f"- {model_name}\n")

    f.write("\nSummary table sorted by mean test balanced accuracy and F1-score:\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")

    f.write("Best model/k combinations:\n\n")

    f.write("Best balanced accuracy:\n")
    f.write(best_balanced_accuracy.to_string())
    f.write("\n\n")

    f.write("Best F1-score:\n")
    f.write(best_f1.to_string())
    f.write("\n\n")

    f.write("Best recall:\n")
    f.write(best_recall.to_string())
    f.write("\n\n")

    f.write("Best ROC-AUC:\n")
    f.write(best_roc_auc.to_string())
    f.write("\n\n")

    f.write("Interpretation notes:\n")
    f.write("- This experiment checks whether model performance depends strongly on the number of selected probes.\n")
    f.write("- If a model performs well across several k values, that suggests its performance is less dependent on one specific feature-selection size.\n")
    f.write("- Accuracy should not be interpreted alone because the classes are imbalanced.\n")
    f.write("- Balanced accuracy, recall, F1-score and ROC-AUC are more informative.\n")
    f.write("- High training scores compared with test scores may suggest possible overfitting.\n")
    f.write("- Results should be interpreted cautiously because the dataset is small and high-dimensional.\n")

print("\nSaved text summary:")
print(text_summary_path)


# ------------------------------------------------------------
# 14. Plot helper function
# ------------------------------------------------------------

def save_metric_line_plot(summary_dataframe, metric_column, ylabel, title, output_filename):
    """
    Create and save a line plot showing how each model performs
    across different k values.
    """

    plt.figure(figsize=(10, 6))

    for model_name in summary_dataframe["model"].unique():
        model_df = summary_dataframe[summary_dataframe["model"] == model_name]
        model_df = model_df.sort_values(by="k")

        plt.plot(
            model_df["k"],
            model_df[metric_column],
            marker="o",
            label=model_name
        )

    plt.xlabel("Number of selected probes, k")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    output_path = plots_folder / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


# ------------------------------------------------------------
# 15. Save plots
# ------------------------------------------------------------

save_metric_line_plot(
    summary_df,
    "mean_test_balanced_accuracy",
    "Mean test balanced accuracy",
    "Model Performance Across k Values: Balanced Accuracy",
    "model_comparison_across_k_balanced_accuracy.png"
)

save_metric_line_plot(
    summary_df,
    "mean_test_f1",
    "Mean test F1-score",
    "Model Performance Across k Values: F1-score",
    "model_comparison_across_k_f1_score.png"
)

save_metric_line_plot(
    summary_df,
    "mean_test_recall",
    "Mean test recall",
    "Model Performance Across k Values: Recall",
    "model_comparison_across_k_recall.png"
)

save_metric_line_plot(
    summary_df,
    "mean_test_roc_auc",
    "Mean test ROC-AUC",
    "Model Performance Across k Values: ROC-AUC",
    "model_comparison_across_k_roc_auc.png"
)


# ------------------------------------------------------------
# 16. Final message
# ------------------------------------------------------------

print("\nModel comparison across k values complete.")
print("Main outputs:")
print(results_output_path)
print(summary_output_path)
print(text_summary_path)
print("\nMain plots saved in:")
print(plots_folder)
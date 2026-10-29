from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script implements an LSA-style classification approach
# using TruncatedSVD.
#
# In text analysis:
#   documents x terms -> TruncatedSVD -> latent semantic components
#
# In this project:
#   tumour samples x expression probes -> TruncatedSVD -> latent expression components
#
# The aim is to test whether latent expression components can classify
# fast-growing HCC tumours, and whether this co-expression-inspired
# representation performs similarly or better than the previous
# selected-probe baseline models.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Suppress common warnings
# ------------------------------------------------------------

warnings.filterwarnings("ignore", category=ConvergenceWarning)
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
# 5. Define LSA component sizes
# ------------------------------------------------------------

# These values are deliberately kept small because the dataset only
# contains 81 tumour samples.
#
# TruncatedSVD components must be fewer than the number of samples/features.
# These values are safe for this dataset.

component_sizes = [2, 5, 10, 20, 40]

print("\nLSA / TruncatedSVD component sizes to test:")
print(component_sizes)


# ------------------------------------------------------------
# 6. Define classifiers
# ------------------------------------------------------------

# We start with a small set of classifiers after LSA.
#
# Logistic Regression:
#   simple linear classifier on latent components
#
# Random Forest:
#   non-linear classifier on latent components
#
# Gaussian Naive Bayes:
#   probabilistic comparison model

classifiers = {
    "LSA_Logistic_Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),
    "LSA_Random_Forest": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ),
    "LSA_Gaussian_Naive_Bayes": GaussianNB()
}

print("\nClassifiers to test after LSA components:")
for model_name in classifiers:
    print("-", model_name)


# ------------------------------------------------------------
# 7. Define repeated stratified cross-validation
# ------------------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)

print("\nCross-validation setup:")
print("5 folds x 20 repeats = 100 evaluations per model/component setting")


# ------------------------------------------------------------
# 8. Define scoring metrics
# ------------------------------------------------------------

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

print("\nScoring metrics:")
for metric_name in scoring:
    print("-", metric_name)


# ------------------------------------------------------------
# 9. Run LSA classification experiments
# ------------------------------------------------------------

all_fold_results = []
all_summary_results = []
explained_variance_results = []

for n_components in component_sizes:
    print("\n============================================================")
    print(f"Testing LSA / TruncatedSVD with n_components = {n_components}")
    print("============================================================")

    for model_name, classifier in classifiers.items():
        print(f"\nRunning model: {model_name}")
        print(f"Components: {n_components}")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lsa", TruncatedSVD(
                n_components=n_components,
                random_state=42
            )),
            ("classifier", classifier)
        ])

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=1
        )

        n_runs = len(cv_results["test_accuracy"])

        fold_result_df = pd.DataFrame({
            "model": model_name,
            "n_components": n_components,
            "fold_run": range(1, n_runs + 1),
            "fit_time": cv_results["fit_time"],
            "score_time": cv_results["score_time"],
            "train_accuracy": cv_results["train_accuracy"],
            "test_accuracy": cv_results["test_accuracy"],
            "train_balanced_accuracy": cv_results["train_balanced_accuracy"],
            "test_balanced_accuracy": cv_results["test_balanced_accuracy"],
            "train_precision": cv_results["train_precision"],
            "test_precision": cv_results["test_precision"],
            "train_recall": cv_results["train_recall"],
            "test_recall": cv_results["test_recall"],
            "train_f1": cv_results["train_f1"],
            "test_f1": cv_results["test_f1"],
            "train_roc_auc": cv_results["train_roc_auc"],
            "test_roc_auc": cv_results["test_roc_auc"]
        })

        all_fold_results.append(fold_result_df)

        summary_row = {
            "model": model_name,
            "n_components": n_components,
            "n_evaluations": n_runs
        }

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]:
            train_col = f"train_{metric}"
            test_col = f"test_{metric}"

            summary_row[f"mean_train_{metric}"] = fold_result_df[train_col].mean()
            summary_row[f"std_train_{metric}"] = fold_result_df[train_col].std()
            summary_row[f"mean_test_{metric}"] = fold_result_df[test_col].mean()
            summary_row[f"std_test_{metric}"] = fold_result_df[test_col].std()
            summary_row[f"test_{metric}_mean_plus_minus_std"] = (
                f"{fold_result_df[test_col].mean():.4f} ± {fold_result_df[test_col].std():.4f}"
            )

        all_summary_results.append(summary_row)

        print("\nCompleted:")
        print(model_name)
        print("n_components:", n_components)
        print("Mean test balanced accuracy:", round(summary_row["mean_test_balanced_accuracy"], 4))
        print("Mean test recall:", round(summary_row["mean_test_recall"], 4))
        print("Mean test F1:", round(summary_row["mean_test_f1"], 4))
        print("Mean test ROC-AUC:", round(summary_row["mean_test_roc_auc"], 4))

    # Fit SVD once on full scaled dataset for exploratory explained variance.
    # This is not used for model scoring. It is only used to understand how
    # much variance the components capture overall.

    scaler_for_exploration = StandardScaler()
    X_scaled = scaler_for_exploration.fit_transform(X)

    svd_for_exploration = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    svd_for_exploration.fit(X_scaled)

    explained_variance_ratio = svd_for_exploration.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    for component_index, variance_value in enumerate(explained_variance_ratio, start=1):
        explained_variance_results.append({
            "n_components_setting": n_components,
            "component": component_index,
            "explained_variance_ratio": variance_value,
            "cumulative_explained_variance_ratio": cumulative_explained_variance[component_index - 1]
        })


# ------------------------------------------------------------
# 10. Combine and save fold-level results
# ------------------------------------------------------------

all_fold_results_df = pd.concat(
    all_fold_results,
    ignore_index=True
)

fold_results_output_path = outputs_folder / "lsa_component_classification_fold_results.csv"
all_fold_results_df.to_csv(fold_results_output_path, index=False)

print("\nSaved LSA fold-level results:")
print(fold_results_output_path)


# ------------------------------------------------------------
# 11. Save summary results
# ------------------------------------------------------------

summary_results_df = pd.DataFrame(all_summary_results)

summary_results_df = summary_results_df.sort_values(
    by="mean_test_balanced_accuracy",
    ascending=False
)

summary_output_path = outputs_folder / "lsa_component_classification_summary.csv"
summary_results_df.to_csv(summary_output_path, index=False)

print("\nSaved LSA summary results:")
print(summary_output_path)

print("\nLSA summary results sorted by balanced accuracy:")
print(summary_results_df[
    [
        "model",
        "n_components",
        "n_evaluations",
        "test_balanced_accuracy_mean_plus_minus_std",
        "test_recall_mean_plus_minus_std",
        "test_precision_mean_plus_minus_std",
        "test_f1_mean_plus_minus_std",
        "test_roc_auc_mean_plus_minus_std"
    ]
])


# ------------------------------------------------------------
# 12. Save explained variance results
# ------------------------------------------------------------

explained_variance_df = pd.DataFrame(explained_variance_results)

explained_variance_output_path = outputs_folder / "lsa_explained_variance_by_component.csv"
explained_variance_df.to_csv(explained_variance_output_path, index=False)

print("\nSaved LSA explained variance results:")
print(explained_variance_output_path)


# ------------------------------------------------------------
# 13. Identify best LSA settings by metric
# ------------------------------------------------------------

best_balanced_accuracy = summary_results_df.sort_values(
    by="mean_test_balanced_accuracy",
    ascending=False
).iloc[0]

best_f1 = summary_results_df.sort_values(
    by="mean_test_f1",
    ascending=False
).iloc[0]

best_recall = summary_results_df.sort_values(
    by="mean_test_recall",
    ascending=False
).iloc[0]

best_roc_auc = summary_results_df.sort_values(
    by="mean_test_roc_auc",
    ascending=False
).iloc[0]

print("\nBest LSA setting by balanced accuracy:")
print(best_balanced_accuracy[[
    "model",
    "n_components",
    "test_balanced_accuracy_mean_plus_minus_std",
    "test_recall_mean_plus_minus_std",
    "test_f1_mean_plus_minus_std",
    "test_roc_auc_mean_plus_minus_std"
]])

print("\nBest LSA setting by F1-score:")
print(best_f1[[
    "model",
    "n_components",
    "test_balanced_accuracy_mean_plus_minus_std",
    "test_recall_mean_plus_minus_std",
    "test_f1_mean_plus_minus_std",
    "test_roc_auc_mean_plus_minus_std"
]])

print("\nBest LSA setting by recall:")
print(best_recall[[
    "model",
    "n_components",
    "test_balanced_accuracy_mean_plus_minus_std",
    "test_recall_mean_plus_minus_std",
    "test_f1_mean_plus_minus_std",
    "test_roc_auc_mean_plus_minus_std"
]])

print("\nBest LSA setting by ROC-AUC:")
print(best_roc_auc[[
    "model",
    "n_components",
    "test_balanced_accuracy_mean_plus_minus_std",
    "test_recall_mean_plus_minus_std",
    "test_f1_mean_plus_minus_std",
    "test_roc_auc_mean_plus_minus_std"
]])


# ------------------------------------------------------------
# 14. Create plots
# ------------------------------------------------------------

metrics_to_plot = [
    ("mean_test_balanced_accuracy", "Balanced accuracy"),
    ("mean_test_recall", "Recall"),
    ("mean_test_f1", "F1-score"),
    ("mean_test_roc_auc", "ROC-AUC")
]

for metric_column, metric_label in metrics_to_plot:
    plt.figure(figsize=(10, 6))

    for model_name in summary_results_df["model"].unique():
        model_df = summary_results_df[
            summary_results_df["model"] == model_name
        ].sort_values(by="n_components")

        plt.plot(
            model_df["n_components"],
            model_df[metric_column],
            marker="o",
            label=model_name
        )

    plt.xlabel("Number of LSA / TruncatedSVD components")
    plt.ylabel(metric_label)
    plt.title(f"LSA classification performance: {metric_label}")
    plt.legend()
    plt.tight_layout()

    safe_metric_name = metric_label.lower().replace("-", "_").replace(" ", "_")
    plot_path = plots_folder / f"lsa_component_classification_{safe_metric_name}.png"

    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("\nSaved plot:")
    print(plot_path)


# Explained variance plot for largest component setting
largest_component_setting = max(component_sizes)

largest_explained_df = explained_variance_df[
    explained_variance_df["n_components_setting"] == largest_component_setting
].copy()

plt.figure(figsize=(10, 6))
plt.plot(
    largest_explained_df["component"],
    largest_explained_df["cumulative_explained_variance_ratio"],
    marker="o"
)
plt.xlabel("Number of LSA / TruncatedSVD components")
plt.ylabel("Cumulative explained variance ratio")
plt.title("Cumulative explained variance from LSA / TruncatedSVD components")
plt.tight_layout()

explained_variance_plot_path = plots_folder / "lsa_cumulative_explained_variance.png"
plt.savefig(explained_variance_plot_path, dpi=300)
plt.close()

print("\nSaved explained variance plot:")
print(explained_variance_plot_path)


# ------------------------------------------------------------
# 15. Save text summary
# ------------------------------------------------------------

text_summary_path = outputs_folder / "lsa_component_classification_summary.txt"

with open(text_summary_path, "w", encoding="utf-8") as f:
    f.write("LSA / TruncatedSVD Component Classification Summary\n")
    f.write("=================================================\n\n")

    f.write("Purpose\n")
    f.write("-------\n")
    f.write("This analysis tested an LSA-style latent-component representation of the gene-expression matrix.\n")
    f.write("Tumour samples were treated analogously to documents, and expression probes were treated analogously to terms.\n")
    f.write("TruncatedSVD was used to transform the expression matrix into latent expression components.\n")
    f.write("These components were then used for classification of fast-growing versus remaining HCC tumours.\n\n")

    f.write("Dataset\n")
    f.write("-------\n")
    f.write(f"Samples: {X.shape[0]}\n")
    f.write(f"Expression probe features: {X.shape[1]}\n")
    f.write("Target: fast_growing\n\n")

    f.write("Class counts\n")
    f.write("------------\n")
    f.write(y.value_counts().to_string())
    f.write("\n\n")

    f.write("Replication setup\n")
    f.write("-----------------\n")
    f.write("Repeated stratified cross-validation was used.\n")
    f.write("The setup was 5 folds x 20 repeats = 100 evaluations per model/component setting.\n\n")

    f.write("Component sizes tested\n")
    f.write("----------------------\n")
    f.write(str(component_sizes))
    f.write("\n\n")

    f.write("Classifiers tested\n")
    f.write("------------------\n")
    for model_name in classifiers:
        f.write(f"- {model_name}\n")
    f.write("\n")

    f.write("Summary results sorted by balanced accuracy\n")
    f.write("-------------------------------------------\n")
    f.write(summary_results_df[
        [
            "model",
            "n_components",
            "n_evaluations",
            "test_balanced_accuracy_mean_plus_minus_std",
            "test_recall_mean_plus_minus_std",
            "test_precision_mean_plus_minus_std",
            "test_f1_mean_plus_minus_std",
            "test_roc_auc_mean_plus_minus_std"
        ]
    ].to_string(index=False))
    f.write("\n\n")

    f.write("Best LSA setting by balanced accuracy\n")
    f.write("-------------------------------------\n")
    f.write(best_balanced_accuracy.to_string())
    f.write("\n\n")

    f.write("Best LSA setting by F1-score\n")
    f.write("----------------------------\n")
    f.write(best_f1.to_string())
    f.write("\n\n")

    f.write("Best LSA setting by recall\n")
    f.write("--------------------------\n")
    f.write(best_recall.to_string())
    f.write("\n\n")

    f.write("Best LSA setting by ROC-AUC\n")
    f.write("---------------------------\n")
    f.write(best_roc_auc.to_string())
    f.write("\n\n")

    f.write("Output files\n")
    f.write("------------\n")
    f.write(f"- {fold_results_output_path}\n")
    f.write(f"- {summary_output_path}\n")
    f.write(f"- {explained_variance_output_path}\n")
    f.write(f"- {text_summary_path}\n")

print("\nSaved text summary:")
print(text_summary_path)


# ------------------------------------------------------------
# 16. Final message
# ------------------------------------------------------------

print("\nLSA / TruncatedSVD classification complete.")

print("\nMain outputs:")
print(fold_results_output_path)
print(summary_output_path)
print(explained_variance_output_path)
print(text_summary_path)

print("\nMain plots:")
for metric_column, metric_label in metrics_to_plot:
    safe_metric_name = metric_label.lower().replace("-", "_").replace(" ", "_")
    print(plots_folder / f"lsa_component_classification_{safe_metric_name}.png")
print(explained_variance_plot_path)
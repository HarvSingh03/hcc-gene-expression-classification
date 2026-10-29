from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script evaluates the strongest model/k combinations from
# the across-k model comparison using confusion-matrix style results.
#
# It focuses on:
#
# 1. Random Forest with k=10
#    - best balanced accuracy, F1-score and ROC-AUC
#
# 2. Gaussian Naive Bayes with k=25
#    - best recall for fast-growing tumours
#
# 3. Elastic Net Logistic Regression with k=10
#    - strong regularised linear comparison
#
# The aim is to understand what types of errors the models make:
#
# - How many fast-growing tumours are correctly detected?
# - How many fast-growing tumours are missed?
# - How many remaining tumours are correctly classified?
# - How many remaining tumours are incorrectly labelled fast-growing?
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
# Same repeated stratified cross-validation as the model comparison:
#
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
# 6. Define strongest candidate models
# ------------------------------------------------------------

candidate_models = {
    "Random_Forest_k10": Pipeline([
        ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Gaussian_Naive_Bayes_k25": Pipeline([
        ("feature_selection", SelectKBest(score_func=f_classif, k=25)),
        ("classifier", GaussianNB())
    ]),

    "Elastic_Net_Logistic_Regression_k10": Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
        ("classifier", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            class_weight="balanced",
            max_iter=10000,
            random_state=42
        ))
    ])
}


# ------------------------------------------------------------
# 7. Helper function for confusion-matrix metrics
# ------------------------------------------------------------

def calculate_confusion_metrics(y_true, y_pred):
    """
    Calculate confusion-matrix values and derived metrics.

    Class labels:
    0 = remaining tumour
    1 = fast-growing tumour

    Confusion matrix layout:
                    Predicted 0    Predicted 1
    Actual 0            TN             FP
    Actual 1            FN             TP
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall = sensitivity

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    balanced_accuracy = (sensitivity + specificity) / 2

    false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate
    }


# ------------------------------------------------------------
# 8. Evaluate each model across CV folds
# ------------------------------------------------------------

all_fold_results = []

print("\nStarting confusion-matrix analysis...")

for model_name, model in candidate_models.items():
    print("\n" + "=" * 80)
    print(f"Evaluating model: {model_name}")
    print("=" * 80)

    fold_number = 0

    for train_index, test_index in cv.split(X, y):
        fold_number += 1

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = calculate_confusion_metrics(y_test, y_pred)

        row = {
            "model": model_name,
            "fold_run": fold_number,
            "test_samples": len(y_test),
            "actual_remaining_count": int((y_test == 0).sum()),
            "actual_fast_growing_count": int((y_test == 1).sum())
        }

        row.update(metrics)

        all_fold_results.append(row)

    print(f"Completed {fold_number} evaluations.")


fold_results_df = pd.DataFrame(all_fold_results)


# ------------------------------------------------------------
# 9. Save full fold-level confusion results
# ------------------------------------------------------------

fold_results_path = outputs_folder / "best_model_confusion_matrix_fold_results.csv"
fold_results_df.to_csv(fold_results_path, index=False)

print("\nSaved fold-level confusion-matrix results:")
print(fold_results_path)


# ------------------------------------------------------------
# 10. Create summary table
# ------------------------------------------------------------

summary_rows = []

for model_name in fold_results_df["model"].unique():
    model_results = fold_results_df[fold_results_df["model"] == model_name]

    summary_row = {
        "model": model_name,
        "n_evaluations": model_results.shape[0],

        "mean_tn": model_results["tn"].mean(),
        "mean_fp": model_results["fp"].mean(),
        "mean_fn": model_results["fn"].mean(),
        "mean_tp": model_results["tp"].mean(),

        "mean_accuracy": model_results["accuracy"].mean(),
        "std_accuracy": model_results["accuracy"].std(),

        "mean_balanced_accuracy": model_results["balanced_accuracy"].mean(),
        "std_balanced_accuracy": model_results["balanced_accuracy"].std(),

        "mean_sensitivity_recall": model_results["sensitivity_recall"].mean(),
        "std_sensitivity_recall": model_results["sensitivity_recall"].std(),

        "mean_specificity": model_results["specificity"].mean(),
        "std_specificity": model_results["specificity"].std(),

        "mean_precision": model_results["precision"].mean(),
        "std_precision": model_results["precision"].std(),

        "mean_f1": model_results["f1"].mean(),
        "std_f1": model_results["f1"].std(),

        "mean_false_negative_rate": model_results["false_negative_rate"].mean(),
        "mean_false_positive_rate": model_results["false_positive_rate"].mean()
    }

    summary_rows.append(summary_row)

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    by=["mean_balanced_accuracy", "mean_f1"],
    ascending=False
)

summary_path = outputs_folder / "best_model_confusion_matrix_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nConfusion-matrix summary:")
print(summary_df)

print("\nSaved confusion-matrix summary:")
print(summary_path)


# ------------------------------------------------------------
# 11. Save text summary
# ------------------------------------------------------------

text_summary_path = outputs_folder / "best_model_confusion_matrix_summary.txt"

with open(text_summary_path, "w", encoding="utf-8") as f:
    f.write("Best Model Confusion-Matrix Summary\n")
    f.write("===================================\n\n")

    f.write("Task:\n")
    f.write("Binary classification of fast-growing vs remaining HCC tumour samples.\n\n")

    f.write("Class labels:\n")
    f.write("0 = remaining tumour\n")
    f.write("1 = fast-growing tumour\n\n")

    f.write("Dataset:\n")
    f.write(f"Samples: {X.shape[0]}\n")
    f.write(f"Expression probe features: {X.shape[1]}\n\n")

    f.write("Class counts:\n")
    f.write(y.value_counts().to_string())
    f.write("\n\n")

    f.write("Cross-validation:\n")
    f.write("RepeatedStratifiedKFold with 5 folds and 20 repeats.\n")
    f.write("This gives 100 evaluations per model.\n\n")

    f.write("Models evaluated:\n")
    for model_name in candidate_models.keys():
        f.write(f"- {model_name}\n")

    f.write("\nConfusion-matrix definitions:\n")
    f.write("- TP: fast-growing tumour correctly predicted as fast-growing\n")
    f.write("- FN: fast-growing tumour incorrectly predicted as remaining\n")
    f.write("- TN: remaining tumour correctly predicted as remaining\n")
    f.write("- FP: remaining tumour incorrectly predicted as fast-growing\n")
    f.write("- Sensitivity/recall: ability to detect fast-growing tumours\n")
    f.write("- Specificity: ability to correctly identify remaining tumours\n\n")

    f.write("Summary table sorted by mean balanced accuracy and F1-score:\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")

    f.write("Interpretation notes:\n")
    f.write("- False negatives are important because they represent fast-growing tumours that were missed.\n")
    f.write("- False positives are also important because they represent remaining tumours incorrectly labelled as fast-growing.\n")
    f.write("- Sensitivity/recall measures how well the model detects fast-growing tumours.\n")
    f.write("- Specificity measures how well the model correctly identifies remaining tumours.\n")
    f.write("- Balanced accuracy averages sensitivity and specificity, making it useful for imbalanced classification.\n")
    f.write("- These results are still cross-validation results and should be interpreted cautiously without an independent external validation set.\n")

print("\nSaved text summary:")
print(text_summary_path)


# ------------------------------------------------------------
# 12. Plot helper function
# ------------------------------------------------------------

def save_bar_plot(summary_dataframe, metric_column, ylabel, title, output_filename):
    """
    Save a horizontal bar plot for a selected summary metric.
    """

    plot_df = summary_dataframe.sort_values(
        by=metric_column,
        ascending=True
    )

    plt.figure(figsize=(10, 5))

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
# 13. Save metric plots
# ------------------------------------------------------------

save_bar_plot(
    summary_df,
    "mean_balanced_accuracy",
    "Mean balanced accuracy",
    "Best Candidate Models: Balanced Accuracy",
    "best_model_confusion_balanced_accuracy.png"
)

save_bar_plot(
    summary_df,
    "mean_sensitivity_recall",
    "Mean sensitivity / recall",
    "Best Candidate Models: Sensitivity / Recall",
    "best_model_confusion_sensitivity_recall.png"
)

save_bar_plot(
    summary_df,
    "mean_specificity",
    "Mean specificity",
    "Best Candidate Models: Specificity",
    "best_model_confusion_specificity.png"
)

save_bar_plot(
    summary_df,
    "mean_f1",
    "Mean F1-score",
    "Best Candidate Models: F1-score",
    "best_model_confusion_f1_score.png"
)


# ------------------------------------------------------------
# 14. Final message
# ------------------------------------------------------------

print("\nConfusion-matrix analysis complete.")
print("Main outputs:")
print(fold_results_path)
print(summary_path)
print(text_summary_path)

print("\nMain plots saved in:")
print(plots_folder)
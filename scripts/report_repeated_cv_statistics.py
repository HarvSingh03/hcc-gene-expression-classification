from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script reports repeated cross-validation statistics in a
# standard way: mean ± standard deviation.
#
# This directly addresses the supervisor feedback about making
# replication clear and reporting statistics properly.
#
# It uses the repeated cross-validation results already saved from
# previous model-comparison scripts.
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
outputs_folder.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 3. Define input files
# ------------------------------------------------------------

input_files = {
    "model_comparison_k25": outputs_folder / "model_comparison_results.csv",
    "model_comparison_across_k": outputs_folder / "model_comparison_across_k_results.csv",
    "best_model_confusion_matrices": outputs_folder / "best_model_confusion_matrix_fold_results.csv"
}


# ------------------------------------------------------------
# 4. Helper function: load file safely
# ------------------------------------------------------------

def load_results_file(label, path):
    print("\nChecking file:")
    print(path)

    if not path.exists():
        print(f"WARNING: File not found for {label}. Skipping this file.")
        return None

    df = pd.read_csv(path)

    print(f"Loaded {label}")
    print("Shape:")
    print(df.shape)

    print("Columns:")
    print(df.columns.tolist())

    return df


# ------------------------------------------------------------
# 5. Helper function: find useful grouping columns
# ------------------------------------------------------------

def find_group_columns(df):
    possible_group_columns = [
        "model",
        "model_name",
        "model_setting",
        "classifier",
        "k",
        "k_features",
        "feature_selection_k"
    ]

    group_columns = [
        col for col in possible_group_columns
        if col in df.columns
    ]

    return group_columns


# ------------------------------------------------------------
# 6. Helper function: find metric columns
# ------------------------------------------------------------

def find_metric_columns(df):
    possible_metric_columns = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "f1_score",
        "roc_auc",
        "sensitivity_recall",
        "specificity",
        "false_negative_rate",
        "false_positive_rate",
        "tn",
        "fp",
        "fn",
        "tp"
    ]

    metric_columns = [
        col for col in possible_metric_columns
        if col in df.columns
    ]

    return metric_columns


# ------------------------------------------------------------
# 7. Helper function: calculate mean ± standard deviation
# ------------------------------------------------------------

def summarise_repeated_cv_results(df, result_label):
    group_columns = find_group_columns(df)
    metric_columns = find_metric_columns(df)

    print("\n--------------------------------------------")
    print(f"Summarising: {result_label}")
    print("--------------------------------------------")

    print("\nGrouping columns:")
    print(group_columns)

    print("\nMetric columns:")
    print(metric_columns)

    if len(group_columns) == 0:
        raise ValueError(
            f"No grouping columns found for {result_label}. "
            "Expected columns such as model, model_name, model_setting, k, or k_features."
        )

    if len(metric_columns) == 0:
        raise ValueError(
            f"No metric columns found for {result_label}. "
            "Expected columns such as balanced_accuracy, recall, f1_score, or roc_auc."
        )

    summary_rows = []

    grouped = df.groupby(group_columns)

    for group_values, group_df in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = {
            "result_source": result_label,
            "n_evaluations": len(group_df)
        }

        for col_name, col_value in zip(group_columns, group_values):
            row[col_name] = col_value

        for metric in metric_columns:
            row[f"{metric}_mean"] = group_df[metric].mean()
            row[f"{metric}_std"] = group_df[metric].std()

            row[f"{metric}_mean_plus_minus_std"] = (
                f"{group_df[metric].mean():.4f} ± {group_df[metric].std():.4f}"
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    return summary_df


# ------------------------------------------------------------
# 8. Load and summarise each available result file
# ------------------------------------------------------------

all_summary_dfs = []

for label, path in input_files.items():
    df = load_results_file(label, path)

    if df is not None:
        summary_df = summarise_repeated_cv_results(df, label)
        all_summary_dfs.append(summary_df)

        individual_output_path = outputs_folder / f"{label}_mean_std_summary.csv"
        summary_df.to_csv(individual_output_path, index=False)

        print("\nSaved individual mean ± std summary:")
        print(individual_output_path)

        print("\nSummary preview:")
        print(summary_df.head())


if len(all_summary_dfs) == 0:
    raise FileNotFoundError(
        "No repeated cross-validation result files were found. "
        "Please run the model-comparison scripts first."
    )


# ------------------------------------------------------------
# 9. Combine all summaries into one table
# ------------------------------------------------------------

combined_summary_df = pd.concat(
    all_summary_dfs,
    ignore_index=True,
    sort=False
)

combined_output_path = outputs_folder / "repeated_cv_statistics_mean_std_summary.csv"
combined_summary_df.to_csv(combined_output_path, index=False)

print("\nCombined repeated CV statistics saved:")
print(combined_output_path)


# ------------------------------------------------------------
# 10. Create a cleaner reporting table for the most important metrics
# ------------------------------------------------------------

important_metric_display_columns = [
    "result_source",
    "model",
    "model_name",
    "model_setting",
    "k",
    "k_features",
    "feature_selection_k",
    "n_evaluations",
    "test_balanced_accuracy_mean_plus_minus_std",
    "balanced_accuracy_mean_plus_minus_std",
    "test_recall_mean_plus_minus_std",
    "recall_mean_plus_minus_std",
    "sensitivity_recall_mean_plus_minus_std",
    "test_precision_mean_plus_minus_std",
    "precision_mean_plus_minus_std",
    "test_f1_mean_plus_minus_std",
    "f1_mean_plus_minus_std",
    "f1_score_mean_plus_minus_std",
    "test_roc_auc_mean_plus_minus_std",
    "roc_auc_mean_plus_minus_std",
    "specificity_mean_plus_minus_std",
    "false_negative_rate_mean_plus_minus_std",
    "false_positive_rate_mean_plus_minus_std"
]

existing_display_columns = [
    col for col in important_metric_display_columns
    if col in combined_summary_df.columns
]

reporting_table_df = combined_summary_df[existing_display_columns].copy()

reporting_output_path = outputs_folder / "repeated_cv_statistics_reporting_table.csv"
reporting_table_df.to_csv(reporting_output_path, index=False)

print("\nCleaner reporting table saved:")
print(reporting_output_path)

print("\nCleaner reporting table preview:")
print(reporting_table_df)


# ------------------------------------------------------------
# 11. Save text summary
# ------------------------------------------------------------

text_summary_path = outputs_folder / "repeated_cv_statistics_summary.txt"

with open(text_summary_path, "w", encoding="utf-8") as f:
    f.write("Repeated Cross-Validation Statistics Summary\n")
    f.write("===========================================\n\n")

    f.write("Purpose\n")
    f.write("-------\n")
    f.write("This file reports model-evaluation statistics using mean ± standard deviation.\n")
    f.write("This addresses the need to make replication clear and report results in a standard way.\n\n")

    f.write("Replication setup\n")
    f.write("-----------------\n")
    f.write("The model-comparison experiments used repeated stratified cross-validation.\n")
    f.write("The main setup was:\n")
    f.write("- 5 folds\n")
    f.write("- 20 repeats\n")
    f.write("- 100 evaluations per model configuration\n\n")

    f.write("Input result files checked\n")
    f.write("--------------------------\n")
    for label, path in input_files.items():
        exists_text = "FOUND" if path.exists() else "NOT FOUND"
        f.write(f"- {label}: {path} [{exists_text}]\n")

    f.write("\nCombined reporting table\n")
    f.write("------------------------\n")
    f.write(reporting_table_df.to_string(index=False))
    f.write("\n\n")

    f.write("Output files\n")
    f.write("------------\n")
    f.write(f"- {combined_output_path}\n")
    f.write(f"- {reporting_output_path}\n")
    f.write(f"- {text_summary_path}\n")

print("\nText summary saved:")
print(text_summary_path)


# ------------------------------------------------------------
# 12. Final message
# ------------------------------------------------------------

print("\nRepeated CV statistics reporting complete.")
print("\nMain outputs:")
print(combined_output_path)
print(reporting_output_path)
print(text_summary_path)
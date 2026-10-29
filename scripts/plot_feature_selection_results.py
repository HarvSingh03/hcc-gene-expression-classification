from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script plots the baseline model performance across
# different SelectKBest k values.
#
# k = number of selected expression probes.
#
# The aim is to see whether model performance improves,
# decreases, or stays stable when more probes are included.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Find project folder
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# 2. Load the feature-selection summary results
# ------------------------------------------------------------

summary_path = project_folder / "outputs" / "feature_selection_k_summary.csv"

print("Loading feature-selection summary from:")
print(summary_path)

summary_df = pd.read_csv(summary_path)

print("\nLoaded summary table:")
print(summary_df)


# ------------------------------------------------------------
# 3. Plot selected performance metrics across k values
# ------------------------------------------------------------
# We focus on:
# - balanced_accuracy: useful because the classes are imbalanced
# - f1: balances precision and recall
# - roc_auc: shows class separation ability
#
# Accuracy is not the main focus because class 0 is larger than class 1.
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    summary_df["k"],
    summary_df["balanced_accuracy"],
    marker="o",
    label="Balanced accuracy"
)

plt.plot(
    summary_df["k"],
    summary_df["f1"],
    marker="o",
    label="F1 score"
)

plt.plot(
    summary_df["k"],
    summary_df["roc_auc"],
    marker="o",
    label="ROC-AUC"
)

plt.xlabel("Number of selected expression probes (k)")
plt.ylabel("Mean cross-validation test score")
plt.title("Baseline Model Performance Across Feature-Selection Sizes")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_path = project_folder / "plots" / "feature_selection_k_performance.png"

plt.savefig(output_path)
plt.close()

print("\nSaved plot:")
print(output_path)


# ------------------------------------------------------------
# 4. Identify the best k values for selected metrics
# ------------------------------------------------------------

best_balanced_accuracy_row = summary_df.loc[
    summary_df["balanced_accuracy"].idxmax()
]

best_f1_row = summary_df.loc[
    summary_df["f1"].idxmax()
]

best_roc_auc_row = summary_df.loc[
    summary_df["roc_auc"].idxmax()
]

print("\nBest k for balanced accuracy:")
print(best_balanced_accuracy_row[["k", "balanced_accuracy"]])

print("\nBest k for F1 score:")
print(best_f1_row[["k", "f1"]])

print("\nBest k for ROC-AUC:")
print(best_roc_auc_row[["k", "roc_auc"]])


# ------------------------------------------------------------
# 5. Final interpretation helper
# ------------------------------------------------------------

print("\nInterpretation helper:")
print("If smaller k values perform better, this suggests that using fewer selected probes may reduce noise.")
print("If larger k values do not improve performance, adding more probes may not help the baseline model.")
print("These results are still preliminary because the dataset has only 81 tumour samples.")
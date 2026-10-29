from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script checks the stability of SelectKBest feature selection.
#
# Previously, the baseline model suggested that k=25 selected probes
# gave a strong early result.
#
# However, we need to check whether the selected probes are stable.
#
# In simple terms:
# - Do the same probes keep being selected across different folds?
# - Or does SelectKBest choose different probes depending on the split?
#
# This does NOT train a classifier.
# It only checks feature-selection stability.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Find project folder and create output folders
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

outputs_folder = project_folder / "outputs"
plots_folder = project_folder / "plots"

outputs_folder.mkdir(exist_ok=True)
plots_folder.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. Load X and y
# ------------------------------------------------------------
# X = cleaned gene-expression probe values
# y = fast_growing label
# ------------------------------------------------------------

X_path = outputs_folder / "X_expression_features.csv"
y_path = outputs_folder / "y_fast_growing_label.csv"

print("Loading X from:")
print(X_path)

X = pd.read_csv(X_path)

print("\nLoading y from:")
print(y_path)

y = pd.read_csv(y_path).squeeze()

print("\nLoaded data:")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget label counts:")
print(y.value_counts())


# ------------------------------------------------------------
# 3. Basic checks
# ------------------------------------------------------------

missing_X = X.isna().sum().sum()
missing_y = y.isna().sum()

print("\nMissing value check:")
print("Missing values in X:", missing_X)
print("Missing values in y:", missing_y)

if missing_X != 0 or missing_y != 0:
    raise ValueError("Missing values found. Please fix missing values before running this script.")


# ------------------------------------------------------------
# 4. Set up repeated stratified cross-validation
# ------------------------------------------------------------
# We use repeated cross-validation because one 5-fold split may be
# too dependent on the exact random split.
#
# n_splits=5 means each repeat has 5 folds.
# n_repeats=20 means the 5-fold CV is repeated 20 times.
#
# Total number of feature-selection runs:
# 5 folds × 20 repeats = 100 runs
#
# In each run, SelectKBest is fitted only on the training fold.
# ------------------------------------------------------------

n_splits = 5
n_repeats = 20
random_state = 42

cv = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=random_state
)

k = 25

total_runs = n_splits * n_repeats

print("\nFeature-selection stability setup:")
print("SelectKBest k:", k)
print("Number of CV splits:", n_splits)
print("Number of repeats:", n_repeats)
print("Total feature-selection runs:", total_runs)


# ------------------------------------------------------------
# 5. Track selected probes
# ------------------------------------------------------------
# We create a counter table with one row per probe.
#
# Each time a probe is selected in a training fold, we add 1.
#
# At the end, each probe will have a selection_count.
# For example:
# - 100 means selected in every run
# - 50 means selected in half the runs
# - 1 means selected only once
# ------------------------------------------------------------

selection_counts = pd.Series(0, index=X.columns, dtype=int)

fold_records = []

run_number = 0

for train_index, test_index in cv.split(X, y):
    run_number += 1

    print(f"Running feature selection fold {run_number} of {total_runs}...")

    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    selected_mask = selector.get_support()
    selected_probe_ids = X.columns[selected_mask]

    # Add 1 to the count for each selected probe
    selection_counts.loc[selected_probe_ids] += 1

    # Save fold-level selected features
    for probe_id in selected_probe_ids:
        fold_records.append({
            "run_number": run_number,
            "probe_id": probe_id
        })


# ------------------------------------------------------------
# 6. Create stability table
# ------------------------------------------------------------

stability_df = pd.DataFrame({
    "probe_id": selection_counts.index,
    "selection_count": selection_counts.values
})

stability_df["selection_frequency"] = (
    stability_df["selection_count"] / total_runs
)

stability_df = stability_df.sort_values(
    by=["selection_count", "probe_id"],
    ascending=[False, True]
)

print("\nTop selected probes by stability:")
print(stability_df.head(30))


# ------------------------------------------------------------
# 7. Save outputs
# ------------------------------------------------------------

stability_output_path = outputs_folder / "feature_selection_stability_k25.csv"

stability_df.to_csv(stability_output_path, index=False)

print("\nSaved feature-selection stability table:")
print(stability_output_path)


fold_records_df = pd.DataFrame(fold_records)

fold_records_output_path = outputs_folder / "feature_selection_stability_fold_records_k25.csv"

fold_records_df.to_csv(fold_records_output_path, index=False)

print("\nSaved fold-level selected feature records:")
print(fold_records_output_path)


# ------------------------------------------------------------
# 8. Summarise stability
# ------------------------------------------------------------
# This tells us how many probes were selected very often.
# ------------------------------------------------------------

selected_at_least_once = stability_df[stability_df["selection_count"] > 0]
selected_every_time = stability_df[stability_df["selection_count"] == total_runs]
selected_at_least_half = stability_df[stability_df["selection_frequency"] >= 0.5]
selected_at_least_80_percent = stability_df[stability_df["selection_frequency"] >= 0.8]

print("\nFeature-selection stability summary:")
print("Total probes:", X.shape[1])
print("Probes selected at least once:", selected_at_least_once.shape[0])
print("Probes selected in at least 50% of runs:", selected_at_least_half.shape[0])
print("Probes selected in at least 80% of runs:", selected_at_least_80_percent.shape[0])
print("Probes selected in every run:", selected_every_time.shape[0])


# ------------------------------------------------------------
# 9. Save summary text file
# ------------------------------------------------------------

summary_text_path = outputs_folder / "feature_selection_stability_summary_k25.txt"

with open(summary_text_path, "w", encoding="utf-8") as f:
    f.write("Feature Selection Stability Summary\n")
    f.write("===================================\n\n")

    f.write("Purpose:\n")
    f.write("This analysis checks whether SelectKBest(k=25) selects similar probes across repeated cross-validation folds.\n\n")

    f.write("Data:\n")
    f.write(f"X shape: {X.shape}\n")
    f.write(f"y shape: {y.shape}\n\n")

    f.write("Target label counts:\n")
    f.write(y.value_counts().to_string())
    f.write("\n\n")

    f.write("Cross-validation setup:\n")
    f.write(f"n_splits: {n_splits}\n")
    f.write(f"n_repeats: {n_repeats}\n")
    f.write(f"total feature-selection runs: {total_runs}\n")
    f.write(f"SelectKBest k: {k}\n\n")

    f.write("Stability summary:\n")
    f.write(f"Total probes: {X.shape[1]}\n")
    f.write(f"Probes selected at least once: {selected_at_least_once.shape[0]}\n")
    f.write(f"Probes selected in at least 50% of runs: {selected_at_least_half.shape[0]}\n")
    f.write(f"Probes selected in at least 80% of runs: {selected_at_least_80_percent.shape[0]}\n")
    f.write(f"Probes selected in every run: {selected_every_time.shape[0]}\n\n")

    f.write("Top 30 probes by selection frequency:\n")
    f.write(stability_df.head(30).to_string(index=False))
    f.write("\n\n")

    f.write("Interpretation guide:\n")
    f.write("- Probes selected very often may be more stable candidate features.\n")
    f.write("- Probes selected only rarely may be sensitive to the train/test split.\n")
    f.write("- Stability does not prove biological relevance or biomarker validity.\n")
    f.write("- Stable probes should still be mapped to gene names and interpreted carefully.\n")

print("\nSaved stability summary text file:")
print(summary_text_path)


# ------------------------------------------------------------
# 10. Plot top 30 most stable probes
# ------------------------------------------------------------

top_n = 30

top_stable_probes = stability_df.head(top_n).copy()

# Sort ascending for horizontal bar plot, so highest appears at top visually
top_stable_probes_plot = top_stable_probes.sort_values(
    by="selection_frequency",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_stable_probes_plot["probe_id"],
    top_stable_probes_plot["selection_frequency"]
)

plt.xlabel("Selection frequency across repeated CV runs")
plt.ylabel("Probe ID")
plt.title("Top 30 Most Stable Selected Probes: SelectKBest(k=25)")
plt.xlim(0, 1)

plt.tight_layout()

stability_plot_path = plots_folder / "feature_selection_stability_top30_k25.png"

plt.savefig(stability_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved stability plot:")
print(stability_plot_path)


# ------------------------------------------------------------
# 11. Plot distribution of selection counts
# ------------------------------------------------------------
# This shows whether most selected probes are selected rarely
# or repeatedly.
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    selected_at_least_once["selection_count"],
    bins=30
)

plt.xlabel("Number of times selected")
plt.ylabel("Number of probes")
plt.title("Distribution of Selection Counts for Probes Selected At Least Once")

plt.tight_layout()

selection_count_distribution_path = plots_folder / "feature_selection_stability_count_distribution_k25.png"

plt.savefig(selection_count_distribution_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved selection count distribution plot:")
print(selection_count_distribution_path)


# ------------------------------------------------------------
# 12. Final message
# ------------------------------------------------------------

print("\nFeature-selection stability analysis complete.")
print("This script did not train a classifier.")
print("It checked how often each probe was selected by SelectKBest(k=25)")
print("across repeated stratified cross-validation folds.")
print("Use this to judge whether the selected features are stable or split-dependent.")
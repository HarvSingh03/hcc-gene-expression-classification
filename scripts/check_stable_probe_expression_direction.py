from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)
# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script checks the expression direction of the stable
# selected probes.
#
# We already know these probes were selected frequently by
# SelectKBest(k=25) across repeated cross-validation folds.
#
# Now we want to answer:
#
# Are these probes higher or lower in fast-growing tumours
# compared with the remaining tumours?
#
# This is not a classifier.
# This is an interpretation step.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Find project folders
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

outputs_folder = project_folder / "outputs"
plots_folder = project_folder / "plots"

outputs_folder.mkdir(exist_ok=True)
plots_folder.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. Load cleaned labelled tumour dataset
# ------------------------------------------------------------
# This file contains:
# - metadata columns
# - cleaned expression probe columns
# - fast_growing label
# ------------------------------------------------------------

labelled_data_path = outputs_folder / "tumour_labelled_no_missing_probes.csv"

print("Loading cleaned labelled tumour dataset:")
print(labelled_data_path)

if not labelled_data_path.exists():
    raise FileNotFoundError(
        "Could not find tumour_labelled_no_missing_probes.csv. "
        "Please check the outputs folder."
    )

tumour_df = pd.read_csv(labelled_data_path)

print("\nTumour dataset loaded.")
print("Shape:")
print(tumour_df.shape)

if "fast_growing" not in tumour_df.columns:
    raise ValueError("Column 'fast_growing' not found in tumour dataset.")

print("\nTarget label counts:")
print(tumour_df["fast_growing"].value_counts())


# ------------------------------------------------------------
# 3. Load stable annotated probes
# ------------------------------------------------------------
# This file was created by:
#
# scripts/map_stable_probes_to_genes.py
#
# It contains the stable probe IDs and gene annotations.
# ------------------------------------------------------------

stable_probes_path = outputs_folder / "stable_selected_probes_gene_annotations_threshold_80.csv"

print("\nLoading stable annotated probes:")
print(stable_probes_path)

if not stable_probes_path.exists():
    raise FileNotFoundError(
        "Could not find stable_selected_probes_gene_annotations_threshold_80.csv. "
        "Please run map_stable_probes_to_genes.py first."
    )

stable_probes_df = pd.read_csv(stable_probes_path)

print("\nStable probes loaded.")
print("Shape:")
print(stable_probes_df.shape)

print("\nStable probes:")
print(stable_probes_df[["probe_id", "selection_count", "selection_frequency", "GENE_SYMBOL", "GENE_NAME"]])


# ------------------------------------------------------------
# 4. Check that stable probes exist in the tumour dataset
# ------------------------------------------------------------

stable_probe_ids = stable_probes_df["probe_id"].tolist()

missing_probes = [
    probe_id for probe_id in stable_probe_ids
    if probe_id not in tumour_df.columns
]

if missing_probes:
    raise ValueError(
        "Some stable probes were not found in the tumour dataset: "
        + str(missing_probes)
    )

print("\nAll stable probes found in tumour dataset.")


# ------------------------------------------------------------
# 5. Compare expression between classes
# ------------------------------------------------------------
# Class 1 = fast-growing tumours
# Class 0 = remaining tumours
#
# For each probe, calculate:
# - mean expression in class 1
# - mean expression in class 0
# - difference
# - median expression in each class
# - Mann-Whitney U test p-value
#
# Mann-Whitney U is used because the sample size is small and
# we do not want to assume normal distributions.
# ------------------------------------------------------------

results = []

for _, probe_row in stable_probes_df.iterrows():
    probe_id = probe_row["probe_id"]
    gene_symbol = probe_row.get("GENE_SYMBOL", "")
    gene_name = probe_row.get("GENE_NAME", "")
    selection_count = probe_row["selection_count"]
    selection_frequency = probe_row["selection_frequency"]

    class_0_values = tumour_df.loc[
        tumour_df["fast_growing"] == 0,
        probe_id
    ]

    class_1_values = tumour_df.loc[
        tumour_df["fast_growing"] == 1,
        probe_id
    ]

    mean_class_0 = class_0_values.mean()
    mean_class_1 = class_1_values.mean()

    median_class_0 = class_0_values.median()
    median_class_1 = class_1_values.median()

    mean_difference = mean_class_1 - mean_class_0
    median_difference = median_class_1 - median_class_0

    if mean_difference > 0:
        direction = "higher in fast-growing tumours"
    elif mean_difference < 0:
        direction = "lower in fast-growing tumours"
    else:
        direction = "no mean difference"

    # Mann-Whitney U test
    # This compares the two groups without assuming normal distribution.
    test_result = mannwhitneyu(
        class_1_values,
        class_0_values,
        alternative="two-sided"
    )

    p_value = test_result.pvalue

    results.append({
        "probe_id": probe_id,
        "GENE_SYMBOL": gene_symbol,
        "GENE_NAME": gene_name,
        "selection_count": selection_count,
        "selection_frequency": selection_frequency,
        "n_fast_growing": class_1_values.shape[0],
        "n_remaining": class_0_values.shape[0],
        "mean_expression_fast_growing": mean_class_1,
        "mean_expression_remaining": mean_class_0,
        "mean_difference_fast_minus_remaining": mean_difference,
        "median_expression_fast_growing": median_class_1,
        "median_expression_remaining": median_class_0,
        "median_difference_fast_minus_remaining": median_difference,
        "direction_based_on_mean": direction,
        "mannwhitneyu_p_value": p_value
    })


direction_df = pd.DataFrame(results)

# Sort by absolute mean difference so the biggest differences appear first
direction_df["absolute_mean_difference"] = direction_df[
    "mean_difference_fast_minus_remaining"
].abs()

direction_df = direction_df.sort_values(
    by="absolute_mean_difference",
    ascending=False
)

print("\nStable probe expression direction summary:")
print(direction_df[
    [
        "probe_id",
        "GENE_SYMBOL",
        "selection_frequency",
        "mean_expression_fast_growing",
        "mean_expression_remaining",
        "mean_difference_fast_minus_remaining",
        "direction_based_on_mean",
        "mannwhitneyu_p_value"
    ]
])


# ------------------------------------------------------------
# 6. Save expression direction summary
# ------------------------------------------------------------

direction_output_path = outputs_folder / "stable_probe_expression_direction_summary.csv"

direction_df.to_csv(direction_output_path, index=False)

print("\nSaved expression direction summary:")
print(direction_output_path)


# ------------------------------------------------------------
# 7. Save a text summary
# ------------------------------------------------------------

summary_text_path = outputs_folder / "stable_probe_expression_direction_summary.txt"

with open(summary_text_path, "w", encoding="utf-8") as f:
    f.write("Stable Probe Expression Direction Summary\n")
    f.write("========================================\n\n")

    f.write("Purpose:\n")
    f.write(
        "This analysis compares expression values of stable selected probes "
        "between fast-growing tumours and remaining tumours.\n\n"
    )

    f.write("Input files:\n")
    f.write(str(labelled_data_path) + "\n")
    f.write(str(stable_probes_path) + "\n\n")

    f.write("Class definition:\n")
    f.write("fast_growing = 1: tumour doubling time <= 54 days\n")
    f.write("fast_growing = 0: remaining tumour samples\n\n")

    f.write("Class counts:\n")
    f.write(tumour_df["fast_growing"].value_counts().to_string())
    f.write("\n\n")

    f.write("Expression direction results:\n")
    f.write(direction_df[
        [
            "probe_id",
            "GENE_SYMBOL",
            "GENE_NAME",
            "selection_frequency",
            "mean_expression_fast_growing",
            "mean_expression_remaining",
            "mean_difference_fast_minus_remaining",
            "direction_based_on_mean",
            "mannwhitneyu_p_value"
        ]
    ].to_string(index=False))
    f.write("\n\n")

    f.write("Interpretation guide:\n")
    f.write("- Positive mean difference means expression is higher in fast-growing tumours.\n")
    f.write("- Negative mean difference means expression is lower in fast-growing tumours.\n")
    f.write("- P-values are exploratory and should not be overinterpreted because the dataset is small.\n")
    f.write("- These probes remain candidate expression features, not confirmed biomarkers.\n")

print("\nSaved text summary:")
print(summary_text_path)


# ------------------------------------------------------------
# 8. Create boxplots for each stable probe
# ------------------------------------------------------------
# One plot with six small boxplots, one for each stable probe.
# ------------------------------------------------------------

num_probes = len(stable_probe_ids)

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(15, 8)
)

axes = axes.flatten()

for index, probe_id in enumerate(stable_probe_ids):
    ax = axes[index]

    gene_symbol = stable_probes_df.loc[
        stable_probes_df["probe_id"] == probe_id,
        "GENE_SYMBOL"
    ].iloc[0]

    class_0_values = tumour_df.loc[
        tumour_df["fast_growing"] == 0,
        probe_id
    ]

    class_1_values = tumour_df.loc[
        tumour_df["fast_growing"] == 1,
        probe_id
    ]

    ax.boxplot(
        [class_0_values, class_1_values],
        tick_labels=["0: remaining", "1: fast-growing"]
    )

    title_gene = gene_symbol if pd.notna(gene_symbol) else "unclear"
    ax.set_title(f"{probe_id}\n{title_gene}")
    ax.set_ylabel("Expression value")

# If there are unused subplot spaces, hide them
for j in range(num_probes, len(axes)):
    axes[j].axis("off")

plt.suptitle("Stable Probe Expression by Fast-Growing Label")
plt.tight_layout()

boxplot_path = plots_folder / "stable_probe_expression_boxplots.png"

plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved stable probe expression boxplots:")
print(boxplot_path)


# ------------------------------------------------------------
# 9. Create a simple direction bar plot
# ------------------------------------------------------------
# This shows the mean difference:
#
# mean fast-growing expression - mean remaining expression
#
# Positive = higher in fast-growing tumours
# Negative = lower in fast-growing tumours
# ------------------------------------------------------------

plot_df = direction_df.copy()

plot_df["display_label"] = plot_df.apply(
    lambda row: f"{row['probe_id']} ({row['GENE_SYMBOL'] if pd.notna(row['GENE_SYMBOL']) else 'unclear'})",
    axis=1
)

plot_df = plot_df.sort_values(
    by="mean_difference_fast_minus_remaining",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_df["display_label"],
    plot_df["mean_difference_fast_minus_remaining"]
)

plt.axvline(0)

plt.xlabel("Mean expression difference: fast-growing minus remaining")
plt.ylabel("Stable probe")
plt.title("Expression Direction of Stable Selected Probes")

plt.tight_layout()

direction_plot_path = plots_folder / "stable_probe_expression_direction_barplot.png"

plt.savefig(direction_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved expression direction bar plot:")
print(direction_plot_path)


# ------------------------------------------------------------
# 10. Final message
# ------------------------------------------------------------

print("\nStable probe expression direction analysis complete.")
print("Use the CSV and plots to interpret whether each stable candidate")
print("probe is higher or lower in fast-growing tumours.")
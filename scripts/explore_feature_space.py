from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script explores the statistical properties of the cleaned
# gene-expression feature matrix X.
#
# It does NOT train a model.
#
# The aim is to understand the feature space before making
# decisions about:
# - feature selection
# - cross-validation strategy
# - model choice
# - possible dimensionality reduction
# - handling low-variance features
# - whether samples appear unusual/outlying
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
#
# X should only contain expression probe columns.
# y should contain the target label.
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
# 3. Confirm missing values
# ------------------------------------------------------------
# This checks whether any expression values or labels are missing.
# Since we already removed probes with missing values, this should
# hopefully print 0 for both.
# ------------------------------------------------------------

missing_X = X.isna().sum().sum()
missing_y = y.isna().sum()

print("\nMissing value check:")
print("Missing values in X:", missing_X)
print("Missing values in y:", missing_y)


# ------------------------------------------------------------
# 4. Overall distribution of expression values
# ------------------------------------------------------------
# This summarises all expression values together.
#
# It answers:
# - What is the general range of expression values?
# - Are there extreme minimum or maximum values?
# - What is the average expression level?
# ------------------------------------------------------------

all_expression_values = X.to_numpy().ravel()

overall_expression_summary = pd.Series(all_expression_values).describe()

print("\nOverall expression value summary:")
print(overall_expression_summary)

overall_expression_summary.to_csv(
    outputs_folder / "feature_space_overall_expression_summary.csv"
)


# Plot overall expression value distribution

plt.figure(figsize=(8, 5))
plt.hist(all_expression_values, bins=100)
plt.xlabel("Expression value")
plt.ylabel("Frequency")
plt.title("Overall Distribution of Expression Values")
plt.tight_layout()

overall_hist_path = plots_folder / "feature_space_overall_expression_distribution.png"
plt.savefig(overall_hist_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved overall expression distribution plot:")
print(overall_hist_path)


# ------------------------------------------------------------
# 5. Probe-level variance
# ------------------------------------------------------------
# Each column in X is one expression probe.
#
# The variance tells us how much that probe changes across the
# 81 tumour samples.
#
# Very low-variance probes barely change across samples, so they
# are unlikely to help classification.
# ------------------------------------------------------------

probe_variances = X.var(axis=0)

probe_variance_summary = probe_variances.describe()

print("\nProbe variance summary:")
print(probe_variance_summary)

probe_variance_df = pd.DataFrame({
    "probe_id": probe_variances.index,
    "variance": probe_variances.values
})

probe_variance_df = probe_variance_df.sort_values(
    by="variance",
    ascending=False
)

probe_variance_output_path = outputs_folder / "feature_space_probe_variances.csv"
probe_variance_df.to_csv(probe_variance_output_path, index=False)

print("\nSaved probe variances:")
print(probe_variance_output_path)


# Plot probe variance distribution

plt.figure(figsize=(8, 5))
plt.hist(probe_variances, bins=100)
plt.xlabel("Probe variance across tumour samples")
plt.ylabel("Number of probes")
plt.title("Distribution of Probe Variances")
plt.tight_layout()

variance_hist_path = plots_folder / "feature_space_probe_variance_distribution.png"
plt.savefig(variance_hist_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved probe variance distribution plot:")
print(variance_hist_path)


# ------------------------------------------------------------
# 6. Identify very low-variance probes
# ------------------------------------------------------------
# There is no single universal threshold, so we start with a
# simple descriptive threshold.
#
# Here we flag probes in the lowest 1% of variance values.
# This does not remove them yet. It only identifies them.
# ------------------------------------------------------------

low_variance_threshold = probe_variances.quantile(0.01)

low_variance_probes = probe_variance_df[
    probe_variance_df["variance"] <= low_variance_threshold
].copy()

low_variance_output_path = outputs_folder / "feature_space_low_variance_probes_bottom_1_percent.csv"
low_variance_probes.to_csv(low_variance_output_path, index=False)

print("\nLow-variance probe check:")
print("Low-variance threshold, bottom 1%:", low_variance_threshold)
print("Number of probes in bottom 1% variance:", low_variance_probes.shape[0])

print("\nSaved low-variance probes:")
print(low_variance_output_path)


# ------------------------------------------------------------
# 7. Sample-level mean and standard deviation
# ------------------------------------------------------------
# Each row in X is one tumour sample.
#
# For each sample, we calculate:
# - mean expression across all probes
# - standard deviation across all probes
#
# This can help identify unusual samples that have much higher,
# lower, or more variable expression profiles than the others.
# ------------------------------------------------------------

sample_summary = pd.DataFrame({
    "sample_index": X.index,
    "fast_growing": y.values,
    "sample_mean_expression": X.mean(axis=1),
    "sample_std_expression": X.std(axis=1)
})

print("\nSample-level expression summary:")
print(sample_summary[["sample_mean_expression", "sample_std_expression"]].describe())

sample_summary_output_path = outputs_folder / "feature_space_sample_expression_summary.csv"
sample_summary.to_csv(sample_summary_output_path, index=False)

print("\nSaved sample-level expression summary:")
print(sample_summary_output_path)


# Plot sample mean expression

plt.figure(figsize=(8, 5))
plt.scatter(
    sample_summary.index,
    sample_summary["sample_mean_expression"],
    c=sample_summary["fast_growing"]
)
plt.xlabel("Sample index")
plt.ylabel("Mean expression across probes")
plt.title("Sample-Level Mean Expression")
plt.tight_layout()

sample_mean_plot_path = plots_folder / "feature_space_sample_mean_expression.png"
plt.savefig(sample_mean_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved sample mean expression plot:")
print(sample_mean_plot_path)


# Plot sample standard deviation

plt.figure(figsize=(8, 5))
plt.scatter(
    sample_summary.index,
    sample_summary["sample_std_expression"],
    c=sample_summary["fast_growing"]
)
plt.xlabel("Sample index")
plt.ylabel("Standard deviation across probes")
plt.title("Sample-Level Expression Standard Deviation")
plt.tight_layout()

sample_std_plot_path = plots_folder / "feature_space_sample_std_expression.png"
plt.savefig(sample_std_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved sample standard deviation plot:")
print(sample_std_plot_path)


# ------------------------------------------------------------
# 8. PCA scatterplot
# ------------------------------------------------------------
# PCA reduces the high-dimensional expression matrix into a few
# new summary axes called principal components.
#
# PC1 captures the largest direction of variation in the data.
# PC2 captures the next largest direction of variation.
#
# This helps us see whether fast-growing and remaining tumours
# separate naturally in the expression feature space.
# ------------------------------------------------------------

print("\nRunning PCA...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "fast_growing": y.values
})

explained_variance_ratio = pca.explained_variance_ratio_

print("\nPCA explained variance ratio:")
print("PC1:", explained_variance_ratio[0])
print("PC2:", explained_variance_ratio[1])
print("PC1 + PC2:", explained_variance_ratio.sum())

pca_output_path = outputs_folder / "feature_space_pca_coordinates.csv"
pca_df.to_csv(pca_output_path, index=False)

print("\nSaved PCA coordinates:")
print(pca_output_path)


# PCA scatterplot

plt.figure(figsize=(7, 6))

for label_value in sorted(pca_df["fast_growing"].unique()):
    subset = pca_df[pca_df["fast_growing"] == label_value]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"fast_growing = {label_value}",
        alpha=0.8
    )

plt.xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.2f}% variance)")
plt.ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.2f}% variance)")
plt.title("PCA of Cleaned Gene-Expression Feature Space")
plt.legend()
plt.tight_layout()

pca_plot_path = plots_folder / "feature_space_pca_pc1_pc2.png"
plt.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved PCA scatterplot:")
print(pca_plot_path)


# ------------------------------------------------------------
# 9. Correlation heatmap for top 50 highest-variance probes
# ------------------------------------------------------------
# A full 39,613 x 39,613 correlation matrix would be too large.
#
# Instead, we take the 50 highest-variance probes and calculate
# correlations between them.
#
# This gives a manageable first look at whether some probes move
# together, which relates to Felipe's point about feature-space
# structure and co-expression.
# ------------------------------------------------------------

top_50_variance_probe_ids = probe_variance_df.head(50)["probe_id"].tolist()

top_50_X = X[top_50_variance_probe_ids]

correlation_matrix = top_50_X.corr()

correlation_output_path = outputs_folder / "feature_space_top50_variance_probe_correlations.csv"
correlation_matrix.to_csv(correlation_output_path)

print("\nSaved top 50 variance probe correlation matrix:")
print(correlation_output_path)


# Plot correlation heatmap using matplotlib only

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.title("Correlation Heatmap: Top 50 Highest-Variance Probes")
plt.xlabel("Probe index")
plt.ylabel("Probe index")
plt.tight_layout()

correlation_heatmap_path = plots_folder / "feature_space_top50_variance_probe_correlation_heatmap.png"
plt.savefig(correlation_heatmap_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved top 50 variance probe correlation heatmap:")
print(correlation_heatmap_path)


# ------------------------------------------------------------
# 10. Optional: correlation heatmap for selected k=25 probes
# ------------------------------------------------------------
# If selected_features_k25.csv exists, we also create a heatmap
# for those selected features.
#
# This helps compare:
# - high-variance probes
# - previously selected baseline probes
# ------------------------------------------------------------

selected_features_path = outputs_folder / "selected_features_k25.csv"

if selected_features_path.exists():
    print("\nselected_features_k25.csv found.")
    print("Creating correlation heatmap for selected k=25 probes...")

    selected_features = pd.read_csv(selected_features_path)
    selected_probe_ids = selected_features["probe_id"].tolist()

    selected_X = X[selected_probe_ids]

    selected_correlation_matrix = selected_X.corr()

    selected_corr_output_path = outputs_folder / "feature_space_selected_k25_probe_correlations.csv"
    selected_correlation_matrix.to_csv(selected_corr_output_path)

    print("\nSaved selected k=25 probe correlation matrix:")
    print(selected_corr_output_path)

    plt.figure(figsize=(8, 7))
    plt.imshow(selected_correlation_matrix, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title("Correlation Heatmap: Selected k=25 Probes")
    plt.xlabel("Selected probe index")
    plt.ylabel("Selected probe index")
    plt.tight_layout()

    selected_corr_heatmap_path = plots_folder / "feature_space_selected_k25_probe_correlation_heatmap.png"
    plt.savefig(selected_corr_heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved selected k=25 probe correlation heatmap:")
    print(selected_corr_heatmap_path)

else:
    print("\nselected_features_k25.csv not found.")
    print("Skipping selected k=25 probe correlation heatmap.")


# ------------------------------------------------------------
# 11. Save a short text summary
# ------------------------------------------------------------
# This creates a simple text file you can use for GitLab evidence
# or project notes.
# ------------------------------------------------------------

summary_text_path = outputs_folder / "feature_space_exploration_summary.txt"

with open(summary_text_path, "w", encoding="utf-8") as f:
    f.write("Feature Space Exploration Summary\n")
    f.write("=================================\n\n")

    f.write(f"X shape: {X.shape}\n")
    f.write(f"y shape: {y.shape}\n\n")

    f.write("Target label counts:\n")
    f.write(y.value_counts().to_string())
    f.write("\n\n")

    f.write(f"Missing values in X: {missing_X}\n")
    f.write(f"Missing values in y: {missing_y}\n\n")

    f.write("Overall expression value summary:\n")
    f.write(overall_expression_summary.to_string())
    f.write("\n\n")

    f.write("Probe variance summary:\n")
    f.write(probe_variance_summary.to_string())
    f.write("\n\n")

    f.write(f"Low-variance threshold, bottom 1%: {low_variance_threshold}\n")
    f.write(f"Number of probes in bottom 1% variance: {low_variance_probes.shape[0]}\n\n")

    f.write("Sample-level mean/std expression summary:\n")
    f.write(sample_summary[["sample_mean_expression", "sample_std_expression"]].describe().to_string())
    f.write("\n\n")

    f.write("PCA explained variance ratio:\n")
    f.write(f"PC1: {explained_variance_ratio[0]}\n")
    f.write(f"PC2: {explained_variance_ratio[1]}\n")
    f.write(f"PC1 + PC2: {explained_variance_ratio.sum()}\n\n")

    f.write("Generated plots:\n")
    f.write(str(overall_hist_path) + "\n")
    f.write(str(variance_hist_path) + "\n")
    f.write(str(sample_mean_plot_path) + "\n")
    f.write(str(sample_std_plot_path) + "\n")
    f.write(str(pca_plot_path) + "\n")
    f.write(str(correlation_heatmap_path) + "\n")

    if selected_features_path.exists():
        f.write(str(selected_corr_heatmap_path) + "\n")

print("\nSaved feature-space summary text file:")
print(summary_text_path)


# ------------------------------------------------------------
# 12. Final message
# ------------------------------------------------------------

print("\nFeature space exploration complete.")
print("This script did not train a model.")
print("It explored expression distributions, probe variance, sample-level behaviour, PCA structure, and probe correlations.")
print("Use these outputs to understand the feature space before deciding the next modelling strategy.")
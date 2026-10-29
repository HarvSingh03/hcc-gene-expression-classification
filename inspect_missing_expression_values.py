import pandas as pd

df = pd.read_csv("outputs/tumour_metadata_expression_labelled.csv")

metadata_columns = [
    "title",
    "sample_id",
    "sample_type",
    "source_name",
    "tissue type",
    "tissue",
    "gender",
    "doubling time (days)",
    "survival time(months)",
    "amplification and labeling kit",
    "scanner update",
    "feature extraction software update",
    "fast_growing"
]

probe_columns = [col for col in df.columns if col not in metadata_columns]

expression_df = df[probe_columns]

print("Expression table shape:")
print(expression_df.shape)

# ------------------------------------------------------------
# Missing values per sample
# ------------------------------------------------------------

missing_per_sample = expression_df.isna().sum(axis=1)

sample_missing_summary = pd.DataFrame({
    "sample_id": df["sample_id"],
    "title": df["title"],
    "fast_growing": df["fast_growing"],
    "missing_expression_values": missing_per_sample
}).sort_values("missing_expression_values", ascending=False)

print("\nMissing expression values per sample:")
print(sample_missing_summary.head(20))

print("\nSummary of missing values per sample:")
print(missing_per_sample.describe())

# ------------------------------------------------------------
# Missing values per probe
# ------------------------------------------------------------

missing_per_probe = expression_df.isna().sum(axis=0)

probe_missing_summary = missing_per_probe.sort_values(ascending=False)

print("\nTop 20 probes with most missing values:")
print(probe_missing_summary.head(20))

print("\nSummary of missing values per probe:")
print(missing_per_probe.describe())

# ------------------------------------------------------------
# Count probes with any missing values
# ------------------------------------------------------------

num_probes_with_missing = (missing_per_probe > 0).sum()
num_probes_without_missing = (missing_per_probe == 0).sum()

print("\nNumber of probes with at least one missing value:")
print(num_probes_with_missing)

print("\nNumber of probes with no missing values:")
print(num_probes_without_missing)

print("\nPercentage of probes with at least one missing value:")
print((num_probes_with_missing / len(probe_columns)) * 100)

# ------------------------------------------------------------
# Save missingness summaries
# ------------------------------------------------------------

sample_missing_summary.to_csv(
    "outputs/missing_values_per_sample.csv",
    index=False
)

probe_missing_summary.to_csv(
    "outputs/missing_values_per_probe.csv",
    header=["missing_count"]
)

print("\nSaved files:")
print("outputs/missing_values_per_sample.csv")
print("outputs/missing_values_per_probe.csv")
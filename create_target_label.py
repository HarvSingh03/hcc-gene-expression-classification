import pandas as pd

# Load the combined tumour metadata + expression table
combined_df = pd.read_csv("outputs/tumour_metadata_plus_expression.csv")

print("Original combined table shape:")
print(combined_df.shape)

# Define threshold using the 25th percentile / fastest quartile
fast_growth_threshold = 54

# Create binary target label
combined_df["fast_growing"] = (
    combined_df["doubling time (days)"] <= fast_growth_threshold
).astype(int)

print("\nFast-growing threshold:")
print(f"Doubling time <= {fast_growth_threshold} days")

print("\nClass counts:")
print(combined_df["fast_growing"].value_counts())

print("\nClass proportions:")
print(combined_df["fast_growing"].value_counts(normalize=True))

print("\nCheck first 15 samples with labels:")
print(
    combined_df[[
        "sample_id",
        "title",
        "doubling time (days)",
        "survival time(months)",
        "fast_growing"
    ]]
    .sort_values("doubling time (days)")
    .head(15)
)

print("\nCheck slowest 10 samples with labels:")
print(
    combined_df[[
        "sample_id",
        "title",
        "doubling time (days)",
        "survival time(months)",
        "fast_growing"
    ]]
    .sort_values("doubling time (days)", ascending=False)
    .head(10)
)

# Save labelled dataset
combined_df.to_csv("outputs/tumour_metadata_expression_labelled.csv", index=False)

print("\nSaved file:")
print("outputs/tumour_metadata_expression_labelled.csv")
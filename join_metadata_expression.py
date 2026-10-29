import pandas as pd

# Load cleaned tumour metadata
tumour_metadata_df = pd.read_csv("outputs/tumour_metadata_clean.csv")

# Load transposed tumour expression table
tumour_expression_df = pd.read_csv("outputs/tumour_expression_sample_by_probe.csv")

# Remove any unwanted column index name if present
tumour_expression_df.columns.name = None

print("Tumour metadata shape:")
print(tumour_metadata_df.shape)

print("\nTumour expression shape:")
print(tumour_expression_df.shape)

print("\nFirst 5 metadata sample IDs:")
print(tumour_metadata_df["sample_id"].head())

print("\nFirst 5 expression sample IDs:")
print(tumour_expression_df["sample_id"].head())

# Join metadata and expression data using sample_id
combined_df = pd.merge(
    tumour_metadata_df,
    tumour_expression_df,
    on="sample_id",
    how="inner"
)

print("\nCombined table shape:")
print(combined_df.shape)

print("\nFirst 5 rows, first 10 columns:")
print(combined_df.iloc[:5, :10])

print("\nMissing values in key clinical columns:")
print(combined_df[[
    "sample_id",
    "title",
    "gender",
    "doubling time (days)",
    "survival time(months)"
]].isna().sum())

print("\nNumber of duplicated sample IDs:")
print(combined_df["sample_id"].duplicated().sum())

# Save combined table
combined_df.to_csv("outputs/tumour_metadata_plus_expression.csv", index=False)

print("\nSaved file:")
print("outputs/tumour_metadata_plus_expression.csv")
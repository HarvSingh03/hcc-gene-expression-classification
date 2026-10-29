import pandas as pd

# Load the tumour-only expression table we saved earlier
tumour_expression_df = pd.read_csv("outputs/tumour_expression_probe_by_sample.csv")

print("Original tumour expression shape:")
print(tumour_expression_df.shape)

print("\nOriginal first 5 rows:")
print(tumour_expression_df.head())

# Set probe IDs as the index
expression_indexed = tumour_expression_df.set_index("ID_REF")

# Transpose so rows become samples and columns become probes
tumour_expression_transposed = expression_indexed.T

# Move sample IDs from the index into a normal column
tumour_expression_transposed = tumour_expression_transposed.reset_index()

# Rename the sample ID column
tumour_expression_transposed = tumour_expression_transposed.rename(
    columns={"index": "sample_id"}
)

print("\nTransposed tumour expression shape:")
print(tumour_expression_transposed.shape)

print("\nFirst 5 rows of transposed expression table:")
print(tumour_expression_transposed.head())

print("\nFirst 10 columns:")
print(tumour_expression_transposed.columns[:10])

# Save the transposed version
tumour_expression_transposed.to_csv(
    "outputs/tumour_expression_sample_by_probe.csv",
    index=False
)

print("\nSaved file:")
print("outputs/tumour_expression_sample_by_probe.csv")
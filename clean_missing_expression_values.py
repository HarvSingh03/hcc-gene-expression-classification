import pandas as pd

# Load labelled tumour dataset
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

metadata_df = df[metadata_columns].copy()
expression_df = df[probe_columns].copy()

print("Original labelled dataset shape:")
print(df.shape)

print("\nOriginal expression shape:")
print(expression_df.shape)

print("\nOriginal total missing expression values:")
print(expression_df.isna().sum().sum())

# Identify probes with no missing values
complete_probe_columns = expression_df.columns[expression_df.isna().sum(axis=0) == 0]

print("\nNumber of probes before removing missing-value probes:")
print(len(probe_columns))

print("\nNumber of complete probes kept:")
print(len(complete_probe_columns))

print("\nNumber of probes removed:")
print(len(probe_columns) - len(complete_probe_columns))

# Keep only complete probes
expression_complete_df = expression_df[complete_probe_columns].copy()

print("\nTotal missing expression values after cleaning:")
print(expression_complete_df.isna().sum().sum())

# Recombine metadata and cleaned expression values
cleaned_df = pd.concat([metadata_df, expression_complete_df], axis=1)

print("\nCleaned labelled dataset shape:")
print(cleaned_df.shape)

print("\nTarget label counts after cleaning:")
print(cleaned_df["fast_growing"].value_counts())

print("\nFirst 5 rows, first 15 columns:")
print(cleaned_df.iloc[:5, :15])

# Save cleaned complete-case probe dataset
cleaned_df.to_csv(
    "outputs/tumour_labelled_no_missing_probes.csv",
    index=False
)

print("\nSaved file:")
print("outputs/tumour_labelled_no_missing_probes.csv")
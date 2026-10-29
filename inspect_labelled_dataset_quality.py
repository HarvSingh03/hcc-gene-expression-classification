import pandas as pd

# Load the labelled tumour dataset
df = pd.read_csv("outputs/tumour_metadata_expression_labelled.csv")

print("Labelled dataset shape:")
print(df.shape)

print("\nFirst 5 rows, first 12 columns:")
print(df.iloc[:5, :12])

print("\nColumn names for first 20 columns:")
print(df.columns[:20].tolist())

# ------------------------------------------------------------
# 1. Check target label
# ------------------------------------------------------------

print("\nTarget label counts:")
print(df["fast_growing"].value_counts())

print("\nTarget label proportions:")
print(df["fast_growing"].value_counts(normalize=True))

# ------------------------------------------------------------
# 2. Separate metadata columns from expression/probe columns
# ------------------------------------------------------------

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

print("\nNumber of metadata columns:")
print(len(metadata_columns))

print("\nNumber of probe/expression columns:")
print(len(probe_columns))

print("\nFirst 10 probe columns:")
print(probe_columns[:10])

# ------------------------------------------------------------
# 3. Check missing values
# ------------------------------------------------------------

print("\nMissing values in metadata columns:")
print(df[metadata_columns].isna().sum())

missing_probe_values = df[probe_columns].isna().sum().sum()

print("\nTotal missing values in expression/probe columns:")
print(missing_probe_values)

# ------------------------------------------------------------
# 4. Check whether expression columns are numeric
# ------------------------------------------------------------

non_numeric_probe_columns = []

for col in probe_columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        non_numeric_probe_columns.append(col)

print("\nNumber of non-numeric probe columns:")
print(len(non_numeric_probe_columns))

print("\nFirst 10 non-numeric probe columns, if any:")
print(non_numeric_probe_columns[:10])

# ------------------------------------------------------------
# 5. Basic summary of expression values
# ------------------------------------------------------------

expression_values = df[probe_columns]

print("\nExpression value summary across all probes/samples:")
print(expression_values.stack().describe())

print("\nMinimum expression value:")
print(expression_values.min().min())

print("\nMaximum expression value:")
print(expression_values.max().max())

# ------------------------------------------------------------
# 6. Check duplicate sample IDs and duplicate probe columns
# ------------------------------------------------------------

print("\nNumber of duplicated sample IDs:")
print(df["sample_id"].duplicated().sum())

print("\nNumber of duplicated probe column names:")
print(pd.Index(probe_columns).duplicated().sum())

# ------------------------------------------------------------
# 7. Check probes with zero variance
# ------------------------------------------------------------

probe_variances = expression_values.var(axis=0)

zero_variance_probes = probe_variances[probe_variances == 0]

print("\nNumber of zero-variance probes:")
print(len(zero_variance_probes))

print("\nFirst 10 zero-variance probes, if any:")
print(zero_variance_probes.head(10))
import gzip
import pandas as pd

file_path = "data/GSE54236_series_matrix.txt.gz"


def clean_value(value):
    """Remove surrounding quotes and whitespace."""
    return value.strip().strip('"')


sample_metadata_lines = []

with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("!Sample_"):
            sample_metadata_lines.append(line.strip())


# These fields usually occur once and line up clearly across all samples
basic_fields = {
    "!Sample_title": "title",
    "!Sample_geo_accession": "sample_id",
    "!Sample_type": "sample_type",
    "!Sample_source_name_ch1": "source_name",
}

# First, collect the basic fields
basic_metadata = {}

# Store all the Sample_characteristics rows separately
characteristics_rows = []

for line in sample_metadata_lines:
    parts = line.split("\t")
    field_name = parts[0]
    values = [clean_value(v) for v in parts[1:]]

    if field_name in basic_fields:
        basic_metadata[basic_fields[field_name]] = values

    elif field_name == "!Sample_characteristics_ch1":
        characteristics_rows.append(values)


# Create a list of dictionaries, one dictionary per sample
num_samples = len(basic_metadata["sample_id"])
sample_records = []

for i in range(num_samples):
    record = {
        "title": basic_metadata["title"][i],
        "sample_id": basic_metadata["sample_id"][i],
        "sample_type": basic_metadata["sample_type"][i],
        "source_name": basic_metadata["source_name"][i],
    }

    # Now parse each characteristics row for this particular sample
    for row in characteristics_rows:
        if i < len(row):
            value = row[i]

            if ":" in value:
                key, val = value.split(":", 1)
                key = key.strip()
                val = val.strip()
                record[key] = val

    sample_records.append(record)


metadata_df = pd.DataFrame(sample_records)

print("Metadata table shape:")
print(metadata_df.shape)

print("\nFirst 10 rows:")
print(metadata_df.head(10))

print("\nRows around position 78 to 85:")
print(metadata_df.iloc[78:86])

print("\nColumn names:")
print(metadata_df.columns.tolist())

print("\nMissing values per column:")
print(metadata_df.isna().sum())

print("\nData types before conversion:")
print(metadata_df.dtypes)


# Convert numeric columns safely
numeric_columns = ["doubling time (days)", "survival time(months)"]

for col in numeric_columns:
    if col in metadata_df.columns:
        metadata_df[col] = pd.to_numeric(metadata_df[col], errors="coerce")

print("\nData types after conversion:")
print(metadata_df.dtypes)

print("\nSummary statistics for numeric clinical columns:")
print(metadata_df[numeric_columns].describe())

print("\nNumber of samples with doubling time:")
print(metadata_df["doubling time (days)"].notna().sum())

print("\nNumber of samples without doubling time:")
print(metadata_df["doubling time (days)"].isna().sum())

print("\nSource name counts:")
print(metadata_df["source_name"].value_counts())

if "tissue type" in metadata_df.columns:
    print("\nTissue type counts:")
    print(metadata_df["tissue type"].value_counts(dropna=False))

# Create a tumour-only metadata table
tumour_metadata_df = metadata_df[
    metadata_df["source_name"] == "Biopsy of tumor tissue"
].copy()

print("\nTumour metadata shape:")
print(tumour_metadata_df.shape)

print("\nFirst 10 tumour samples:")
print(tumour_metadata_df[[
    "title",
    "sample_id",
    "gender",
    "doubling time (days)",
    "survival time(months)"
]].head(10))

print("\nMissing values in tumour metadata:")
print(tumour_metadata_df[[
    "title",
    "sample_id",
    "gender",
    "doubling time (days)",
    "survival time(months)"
]].isna().sum())

print("\nDoubling time summary for tumour samples:")
print(tumour_metadata_df["doubling time (days)"].describe())

print("\nSurvival time summary for tumour samples:")
print(tumour_metadata_df["survival time(months)"].describe())

# ------------------------------------------------------------
# Check that tumour sample IDs match expression table columns
# ------------------------------------------------------------

# Load the expression table again
expression_df = pd.read_csv(
    file_path,
    sep="\t",
    compression="gzip",
    skiprows=90,
    comment="!"
)

print("\nExpression table shape:")
print(expression_df.shape)

# Expression sample IDs are all columns except ID_REF
expression_sample_ids = list(expression_df.columns[1:])

# Tumour sample IDs come from the cleaned tumour metadata table
tumour_sample_ids = tumour_metadata_df["sample_id"].tolist()

print("\nNumber of sample IDs in expression table:")
print(len(expression_sample_ids))

print("\nNumber of tumour sample IDs in metadata:")
print(len(tumour_sample_ids))

# Check whether all tumour sample IDs are present in the expression table
missing_tumour_ids = set(tumour_sample_ids) - set(expression_sample_ids)

print("\nTumour sample IDs missing from expression table:")
print(missing_tumour_ids)

# Create tumour-only expression table
tumour_expression_df = expression_df[["ID_REF"] + tumour_sample_ids].copy()

print("\nTumour-only expression table shape:")
print(tumour_expression_df.shape)

print("\nFirst 5 rows of tumour-only expression table:")
print(tumour_expression_df.head())

print("\nFirst 10 columns of tumour-only expression table:")
print(tumour_expression_df.columns[:10])

# ------------------------------------------------------------
# Save cleaned/intermediate files
# ------------------------------------------------------------

metadata_df.to_csv("outputs/all_sample_metadata_clean.csv", index=False)

tumour_metadata_df.to_csv("outputs/tumour_metadata_clean.csv", index=False)

tumour_expression_df.to_csv("outputs/tumour_expression_probe_by_sample.csv", index=False)

print("\nSaved files:")
print("outputs/all_sample_metadata_clean.csv")
print("outputs/tumour_metadata_clean.csv")
print("outputs/tumour_expression_probe_by_sample.csv")
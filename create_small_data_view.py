import pandas as pd

# ------------------------------------------------------------
# Load cleaned tumour dataset
# ------------------------------------------------------------

df = pd.read_csv("outputs/tumour_labelled_no_missing_probes.csv")

print("Full cleaned tumour dataset shape:")
print(df.shape)

# ------------------------------------------------------------
# Define metadata / clinical columns to show
# ------------------------------------------------------------

clinical_columns = [
    "sample_id",
    "title",
    "source_name",
    "gender",
    "doubling time (days)",
    "survival time(months)",
    "fast_growing"
]

# ------------------------------------------------------------
# Identify expression probe columns
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

print("\nNumber of expression probe columns:")
print(len(probe_columns))

# ------------------------------------------------------------
# Choose a small number of example expression probes
# ------------------------------------------------------------
# These are just example features, not selected biomarkers.
# The purpose is only to show what the expression data looks like.

example_probe_columns = probe_columns[:8]

print("\nExample probe columns used:")
print(example_probe_columns)

# ------------------------------------------------------------
# Create a small balanced sample view
# ------------------------------------------------------------
# Select 5 fast-growing tumours and 5 other tumours.
# This makes the table easier to understand visually.

fast_growing_samples = df[df["fast_growing"] == 1].head(5)
other_samples = df[df["fast_growing"] == 0].head(5)

small_view = pd.concat(
    [fast_growing_samples, other_samples],
    axis=0
)

# Keep only clinical columns + a few expression probes
small_view = small_view[clinical_columns + example_probe_columns]

# Sort so fast-growing samples appear first
small_view = small_view.sort_values(
    by=["fast_growing", "doubling time (days)"],
    ascending=[False, True]
)

# ------------------------------------------------------------
# Rename columns to make the table easier to read
# ------------------------------------------------------------

small_view = small_view.rename(columns={
    "sample_id": "Sample ID",
    "title": "Sample title",
    "source_name": "Sample source",
    "gender": "Gender",
    "doubling time (days)": "Doubling time (days)",
    "survival time(months)": "Survival time (months)",
    "fast_growing": "Fast-growing label"
})

# ------------------------------------------------------------
# Save the small data view
# ------------------------------------------------------------

small_view.to_csv("outputs/small_data_view_for_supervisor.csv", index=False)

print("\nSmall data view shape:")
print(small_view.shape)

print("\nSmall data view:")
print(small_view)

print("\nSaved file:")
print("outputs/small_data_view_for_supervisor.csv")
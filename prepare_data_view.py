import pandas as pd



tumour_df = pd.read_csv("outputs/tumour_labelled_no_missing_probes.csv")
all_metadata_df = pd.read_csv("outputs/all_sample_metadata_clean.csv")

print("DATA VIEW FOR GSE54236")
print("=" * 50)



print("\n1. Basic dataset overview")
print("-" * 50)

print("Dataset: GSE54236")
print("Source: NCBI Gene Expression Omnibus (GEO)")
print("File used: GEO Series Matrix file")
print("Raw archive used: No")

print("\nAll metadata shape:")
print(all_metadata_df.shape)

print("\nCleaned tumour labelled dataset shape:")
print(tumour_df.shape)

print("\nSample type counts:")
print(all_metadata_df["source_name"].value_counts())



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

probe_columns = [col for col in tumour_df.columns if col not in metadata_columns]

print("\nNumber of metadata/label columns:")
print(len(metadata_columns))

print("\nNumber of cleaned expression probe columns:")
print(len(probe_columns))

print("\nFirst 10 expression probe columns:")
print(probe_columns[:10])


print("\n2. Target label summary")
print("-" * 50)

print("\nFast-growing label counts:")
print(tumour_df["fast_growing"].value_counts())

print("\nFast-growing label proportions:")
print(tumour_df["fast_growing"].value_counts(normalize=True))



print("\n3. Clinical descriptive statistics")
print("-" * 50)

clinical_columns = [
    "doubling time (days)",
    "survival time(months)"
]

print("\nSummary statistics for doubling time and survival time:")
print(tumour_df[clinical_columns].describe())

print("\nGender counts:")
print(tumour_df["gender"].value_counts())



print("\n4. Missing values after cleaning")
print("-" * 50)

print("\nMissing values in metadata/label columns:")
print(tumour_df[metadata_columns].isna().sum())

print("\nTotal missing values in cleaned expression probes:")
print(tumour_df[probe_columns].isna().sum().sum())



print("\n5. Data view summary")
print("-" * 50)

print("Rows in final tumour dataset: tumour samples/patients")
print("Expression columns: cleaned gene-expression probe measurements")
print("Target label: fast_growing")
print("fast_growing = 1 means doubling time <= 54 days")
print("fast_growing = 0 means doubling time > 54 days")
print("Raw expression probes: 41000")
print("Cleaned expression probes retained: 39613")
print("Probes removed due to missing values: 1387")
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script prepares the cleaned tumour dataset for future
# machine-learning modelling.
#
# It does NOT train a model yet.
#
# It separates:
# X = gene-expression input features
# y = fast_growing target label
#
# Important:
# Doubling time is NOT included in X because it was used to
# create the fast_growing label.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Load the cleaned labelled tumour dataset
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

file_path = project_folder / "outputs" / "tumour_labelled_no_missing_probes.csv"

print("Looking for file here:")
print(file_path)

df = pd.read_csv(file_path)

print("\nLoaded cleaned tumour dataset:")
print(df.shape)


# ------------------------------------------------------------
# 2. Define metadata and clinical columns
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


# ------------------------------------------------------------
# 3. Identify expression probe columns
# ------------------------------------------------------------

probe_columns = []

for column in df.columns:
    if column not in metadata_columns:
        probe_columns.append(column)

print("\nNumber of metadata/clinical/label columns:")
print(len(metadata_columns))

print("\nNumber of expression probe columns:")
print(len(probe_columns))

print("\nFirst 10 expression probe columns:")
print(probe_columns[:10])


# ------------------------------------------------------------
# 4. Create X and y
# ------------------------------------------------------------

X = df[probe_columns].copy()
y = df["fast_growing"].copy()

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)


# ------------------------------------------------------------
# 5. Check target label counts
# ------------------------------------------------------------

print("\nTarget label counts:")
print(y.value_counts())

print("\nTarget label proportions:")
print(y.value_counts(normalize=True))


# ------------------------------------------------------------
# 6. Confirm that clinical/metadata columns are not in X
# ------------------------------------------------------------

columns_that_should_not_be_in_X = [
    "sample_id",
    "title",
    "gender",
    "source_name",
    "tissue type",
    "doubling time (days)",
    "survival time(months)",
    "fast_growing"
]

print("\nChecking that metadata/clinical columns are excluded from X:")

for column in columns_that_should_not_be_in_X:
    if column in X.columns:
        print(column, "is incorrectly included in X")
    else:
        print(column, "is correctly excluded from X")


# ------------------------------------------------------------
# 7. Check that X contains only numeric values
# ------------------------------------------------------------

non_numeric_columns = []

for column in X.columns:
    if not pd.api.types.is_numeric_dtype(X[column]):
        non_numeric_columns.append(column)

print("\nNumber of non-numeric columns in X:")
print(len(non_numeric_columns))

if len(non_numeric_columns) > 0:
    print("\nFirst 10 non-numeric columns:")
    print(non_numeric_columns[:10])
else:
    print("All X columns are numeric.")


# ------------------------------------------------------------
# 8. Check missing values in X and y
# ------------------------------------------------------------

print("\nMissing values in X:")
print(X.isna().sum().sum())

print("\nMissing values in y:")
print(y.isna().sum())


# ------------------------------------------------------------
# 9. Save X and y for later modelling
# ------------------------------------------------------------

X_output_path = project_folder / "outputs" / "X_expression_features.csv"
y_output_path = project_folder / "outputs" / "y_fast_growing_label.csv"

X.to_csv(X_output_path, index=False)
y.to_csv(y_output_path, index=False)

print("\nSaved modelling files:")
print(X_output_path)
print(y_output_path)


# ------------------------------------------------------------
# 10. Final summary
# ------------------------------------------------------------

print("\nFinal modelling preparation summary:")
print("Rows/samples in X:", X.shape[0])
print("Expression features in X:", X.shape[1])
print("Labels in y:", y.shape[0])
print("Target label: fast_growing")
print("Class 1 means: fast-growing tumour")
print("Class 0 means: remaining tumour")
print("Doubling time excluded from X: yes")
print("Survival time excluded from X: yes")
print("Metadata excluded from X: yes")
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script inspects the GPL6480 annotation file.
#
# The aim is to understand:
# - what the file looks like
# - where the annotation table starts
# - what columns are available
# - which column contains probe IDs
# - which columns might contain gene symbols/names
#
# This script does NOT map probes yet.
# It only inspects the annotation file.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Find project folder and annotation file
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "data"
outputs_folder = project_folder / "outputs"

outputs_folder.mkdir(exist_ok=True)

annotation_path = data_folder / "GPL6480-9577.txt"

print("Looking for annotation file:")
print(annotation_path)

if not annotation_path.exists():
    raise FileNotFoundError(
        "GPL annotation file not found. Please check it is inside the data folder."
    )


# ------------------------------------------------------------
# 2. Print the first lines of the file
# ------------------------------------------------------------
# GEO platform files often contain metadata/header lines before
# the actual table starts.
#
# We print the first 80 lines so we can see the structure.
# ------------------------------------------------------------

print("\nFirst 80 lines of annotation file:")
print("----------------------------------")

with open(annotation_path, "r", encoding="utf-8", errors="replace") as file:
    for line_number in range(1, 81):
        line = file.readline()

        if not line:
            break

        print(f"{line_number}: {line.rstrip()}")


# ------------------------------------------------------------
# 3. Find where the annotation table starts
# ------------------------------------------------------------
# In GEO platform files, the table usually starts after a line:
#
# !platform_table_begin
#
# The table usually ends at:
#
# !platform_table_end
# ------------------------------------------------------------

table_start_line = None
table_end_line = None

with open(annotation_path, "r", encoding="utf-8", errors="replace") as file:
    for line_number, line in enumerate(file):
        line = line.strip()

        if line == "!platform_table_begin":
            table_start_line = line_number

        if line == "!platform_table_end":
            table_end_line = line_number
            break

print("\nPlatform table location:")
print("table_start_line:", table_start_line)
print("table_end_line:", table_end_line)

if table_start_line is None:
    print("No !platform_table_begin marker found.")
    print("This is okay for this file because the table starts after comment lines beginning with #.")

if table_end_line is None:
    print("Warning: could not find !platform_table_end, but we can still try to load the table.")


# ------------------------------------------------------------
# 4. Load the annotation table
# ------------------------------------------------------------
# This downloaded GPL file does not contain !platform_table_begin.
# Instead, it has comment lines starting with #, then the actual
# table header starts with:
#
# ID    SPOT_ID    CONTROL_TYPE ...
#
# So we tell pandas to ignore comment lines beginning with #.
# ------------------------------------------------------------

annotation_df = pd.read_csv(
    annotation_path,
    sep="\t",
    comment="#",
    low_memory=False
)

print("\nAnnotation table loaded.")
print("Shape:")
print(annotation_df.shape)

print("\nColumn names:")
for column in annotation_df.columns:
    print(column)

print("\nFirst 5 rows:")
print(annotation_df.head())


# ------------------------------------------------------------
# 5. Save a small preview
# ------------------------------------------------------------

preview_output_path = outputs_folder / "gpl6480_annotation_preview.csv"

annotation_df.head(20).to_csv(preview_output_path, index=False)

print("\nSaved annotation preview:")
print(preview_output_path)


# ------------------------------------------------------------
# 6. Check likely probe ID and gene annotation columns
# ------------------------------------------------------------
# We search column names for useful words.
# ------------------------------------------------------------

keywords = [
    "id",
    "probe",
    "gene",
    "symbol",
    "title",
    "name",
    "entrez",
    "refseq",
    "unigene",
    "description"
]

print("\nPotentially useful columns:")
for column in annotation_df.columns:
    column_lower = column.lower()

    if any(keyword in column_lower for keyword in keywords):
        print(column)


# ------------------------------------------------------------
# 7. Check whether known stable probe IDs appear
# ------------------------------------------------------------
# These are the most stable probe IDs from the stability analysis.
# We check whether they appear in the likely ID column.
# ------------------------------------------------------------

known_probe_ids = [
    "A_23_P60079",
    "A_23_P372516",
    "A_32_P208136",
    "A_24_P932435",
    "A_23_P144843",
    "A_23_P49145"
]

print("\nChecking known stable probe IDs:")

if "ID" in annotation_df.columns:
    annotation_ids = set(annotation_df["ID"].astype(str))

    for probe_id in known_probe_ids:
        print(probe_id, "found in ID column:", probe_id in annotation_ids)
else:
    print("No column called 'ID' found. Need to inspect column names manually.")


# ------------------------------------------------------------
# 8. Final message
# ------------------------------------------------------------

print("\nGPL annotation inspection complete.")
print("Next step will be to use the correct ID/gene columns to map stable probes.")
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)
# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script maps stable selected probe IDs to gene annotation.
#
# We previously checked SelectKBest(k=25) stability across
# repeated cross-validation folds.
#
# Now we use the GPL6480 annotation file to map probe IDs like:
#
# A_23_P60079
#
# to readable gene information, such as:
# - gene symbol
# - gene name
# - Entrez Gene ID
# - RefSeq accession
# - description
#
# Important:
# These are still preliminary candidate expression features,
# not confirmed biomarkers.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Find project folders
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "data"
outputs_folder = project_folder / "outputs"

outputs_folder.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. Load feature-selection stability results
# ------------------------------------------------------------
# This file was created by:
#
# scripts/check_selected_feature_stability.py
#
# It contains each probe and how often it was selected across
# 100 repeated cross-validation feature-selection runs.
# ------------------------------------------------------------

stability_path = outputs_folder / "feature_selection_stability_k25.csv"

print("Loading feature-selection stability file:")
print(stability_path)

if not stability_path.exists():
    raise FileNotFoundError(
        "Could not find feature_selection_stability_k25.csv. "
        "Please run check_selected_feature_stability.py first."
    )

stability_df = pd.read_csv(stability_path)

print("\nStability table loaded.")
print("Shape:")
print(stability_df.shape)

print("\nFirst 5 rows:")
print(stability_df.head())


# ------------------------------------------------------------
# 3. Select the most stable probes
# ------------------------------------------------------------
# We will start by focusing on probes selected in at least 80%
# of repeated cross-validation feature-selection runs.
#
# In your result, this should give 6 probes.
# ------------------------------------------------------------

stability_threshold = 0.80

stable_probes_df = stability_df[
    stability_df["selection_frequency"] >= stability_threshold
].copy()

print("\nStable probes selected with threshold:")
print("selection_frequency >=", stability_threshold)

print("\nNumber of stable probes:")
print(stable_probes_df.shape[0])

print("\nStable probes:")
print(stable_probes_df)


# ------------------------------------------------------------
# 4. Also keep the top 30 most stable probes
# ------------------------------------------------------------
# This is useful because sometimes the strict 80% threshold gives
# only a small number of probes.
#
# The top 30 table gives a broader candidate list for inspection.
# ------------------------------------------------------------

top_30_stable_df = stability_df.head(30).copy()

print("\nTop 30 most frequently selected probes:")
print(top_30_stable_df)


# ------------------------------------------------------------
# 5. Load the GPL6480 annotation file
# ------------------------------------------------------------
# The GPL file has comment lines beginning with #.
#
# The real table starts after those comment lines.
#
# Using comment="#" tells pandas to ignore those comment lines.
# ------------------------------------------------------------

annotation_path = data_folder / "GPL6480-9577.txt"

print("\nLoading GPL6480 annotation file:")
print(annotation_path)

if not annotation_path.exists():
    raise FileNotFoundError(
        "Could not find GPL6480-9577.txt. "
        "Please check it is inside the data folder."
    )

annotation_df = pd.read_csv(
    annotation_path,
    sep="\t",
    comment="#",
    low_memory=False
)

print("\nAnnotation table loaded.")
print("Shape:")
print(annotation_df.shape)

print("\nAnnotation columns:")
for column in annotation_df.columns:
    print(column)


# ------------------------------------------------------------
# 6. Choose useful annotation columns
# ------------------------------------------------------------
# These columns make the probe IDs easier to understand.
# ------------------------------------------------------------

annotation_columns_to_keep = [
    "ID",
    "GENE_SYMBOL",
    "GENE_NAME",
    "GENE",
    "REFSEQ",
    "GB_ACC",
    "UNIGENE_ID",
    "ENSEMBL_ID",
    "CHROMOSOMAL_LOCATION",
    "CYTOBAND",
    "DESCRIPTION",
    "GO_ID"
]

# Keep only columns that actually exist in the file
annotation_columns_to_keep = [
    column for column in annotation_columns_to_keep
    if column in annotation_df.columns
]

annotation_subset_df = annotation_df[annotation_columns_to_keep].copy()

print("\nAnnotation columns kept:")
for column in annotation_subset_df.columns:
    print(column)


# ------------------------------------------------------------
# 7. Map stable probes to annotation
# ------------------------------------------------------------
# stability_df has a column called probe_id.
# annotation_df has a column called ID.
#
# We join:
#
# probe_id  ->  ID
# ------------------------------------------------------------

stable_annotated_df = stable_probes_df.merge(
    annotation_subset_df,
    left_on="probe_id",
    right_on="ID",
    how="left"
)

top_30_annotated_df = top_30_stable_df.merge(
    annotation_subset_df,
    left_on="probe_id",
    right_on="ID",
    how="left"
)


# ------------------------------------------------------------
# 8. Check for unmapped probes
# ------------------------------------------------------------
# If GENE_SYMBOL is missing, the probe may not have a gene symbol
# annotation in this GPL file.
# ------------------------------------------------------------

stable_unmapped = stable_annotated_df[
    stable_annotated_df["ID"].isna()
]

top_30_unmapped = top_30_annotated_df[
    top_30_annotated_df["ID"].isna()
]

print("\nMapping check:")
print("Stable probes unmapped:", stable_unmapped.shape[0])
print("Top 30 probes unmapped:", top_30_unmapped.shape[0])


# ------------------------------------------------------------
# 9. Print annotated stable probes
# ------------------------------------------------------------

print("\nAnnotated stable probes:")
columns_to_print = [
    "probe_id",
    "selection_count",
    "selection_frequency",
    "GENE_SYMBOL",
    "GENE_NAME",
    "GENE",
    "REFSEQ",
    "DESCRIPTION"
]

columns_to_print = [
    column for column in columns_to_print
    if column in stable_annotated_df.columns
]

print(stable_annotated_df[columns_to_print])


# ------------------------------------------------------------
# 10. Save mapped outputs
# ------------------------------------------------------------

stable_output_path = outputs_folder / "stable_selected_probes_gene_annotations_threshold_80.csv"
top_30_output_path = outputs_folder / "top30_selected_probes_gene_annotations.csv"

stable_annotated_df.to_csv(stable_output_path, index=False)
top_30_annotated_df.to_csv(top_30_output_path, index=False)

print("\nSaved stable annotated probes:")
print(stable_output_path)

print("\nSaved top 30 annotated probes:")
print(top_30_output_path)


# ------------------------------------------------------------
# 11. Save a simple text summary
# ------------------------------------------------------------

summary_text_path = outputs_folder / "stable_probe_gene_annotation_summary.txt"

with open(summary_text_path, "w", encoding="utf-8") as f:
    f.write("Stable Probe Gene Annotation Summary\n")
    f.write("====================================\n\n")

    f.write("Purpose:\n")
    f.write(
        "This analysis maps stable selected probe IDs from the SelectKBest(k=25) "
        "feature-selection stability analysis to GPL6480 gene annotation.\n\n"
    )

    f.write("Input files:\n")
    f.write(str(stability_path) + "\n")
    f.write(str(annotation_path) + "\n\n")

    f.write("Stability threshold:\n")
    f.write(f"selection_frequency >= {stability_threshold}\n\n")

    f.write("Number of stable probes mapped:\n")
    f.write(str(stable_annotated_df.shape[0]) + "\n\n")

    f.write("Mapping check:\n")
    f.write(f"Stable probes unmapped: {stable_unmapped.shape[0]}\n")
    f.write(f"Top 30 probes unmapped: {top_30_unmapped.shape[0]}\n\n")

    f.write("Stable annotated probes:\n")
    f.write(stable_annotated_df[columns_to_print].to_string(index=False))
    f.write("\n\n")

    f.write("Interpretation guide:\n")
    f.write("- These are stable candidate expression features from the current workflow.\n")
    f.write("- They are not confirmed biomarkers.\n")
    f.write("- Gene annotation helps make the probe IDs biologically interpretable.\n")
    f.write("- Further literature review and biological validation are needed.\n")

print("\nSaved annotation summary text file:")
print(summary_text_path)


# ------------------------------------------------------------
# 12. Final message
# ------------------------------------------------------------

print("\nStable probe annotation mapping complete.")
print("The stable selected probe IDs have now been mapped to gene annotation where available.")
print("Treat these as stable candidate expression features, not confirmed biomarkers.")
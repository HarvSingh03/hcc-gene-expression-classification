from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script inspects the probes selected by SelectKBest for the
# strongest model settings found during model comparison.
#
# Strongest settings:
#
# 1. Random Forest with k=10
#    - best balanced accuracy, F1-score and ROC-AUC
#
# 2. Gaussian Naive Bayes with k=25
#    - best recall for fast-growing tumours
#
# 3. Elastic Net Logistic Regression with k=10
#    - strong regularised linear comparison
#
# Important:
# The classifiers themselves are not directly selecting probes here.
# The selected probes come from SelectKBest.
#
# This script fits SelectKBest on the full cleaned dataset only for
# exploratory interpretation.
#
# Model performance was already evaluated properly with feature
# selection inside cross-validation.
# ------------------------------------------------------------


from sklearn.feature_selection import SelectKBest, f_classif


# ------------------------------------------------------------
# 1. Set pandas display options
# ------------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)


# ------------------------------------------------------------
# 2. Find project folders
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "data"
outputs_folder = project_folder / "outputs"

outputs_folder.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 3. Load X and y
# ------------------------------------------------------------

x_path = outputs_folder / "X_expression_features.csv"
y_path = outputs_folder / "y_fast_growing_label.csv"

print("Loading X expression features:")
print(x_path)

if not x_path.exists():
    raise FileNotFoundError(
        "Could not find X_expression_features.csv. "
        "Please run prepare_modelling_data.py first."
    )

print("\nLoading y fast-growing labels:")
print(y_path)

if not y_path.exists():
    raise FileNotFoundError(
        "Could not find y_fast_growing_label.csv. "
        "Please run prepare_modelling_data.py first."
    )

X = pd.read_csv(x_path)
y_df = pd.read_csv(y_path)

if "fast_growing" in y_df.columns:
    y = y_df["fast_growing"]
else:
    y = y_df.iloc[:, 0]

print("\nData loaded.")
print("X shape:")
print(X.shape)
print("y shape:")
print(y.shape)

print("\nTarget class counts:")
print(y.value_counts())

print("\nMissing values in X:")
print(X.isna().sum().sum())

print("\nMissing values in y:")
print(y.isna().sum())


# ------------------------------------------------------------
# 4. Function to get selected probes for a given k
# ------------------------------------------------------------

def get_selected_probes_for_k(X, y, k_features):
    """
    Fit SelectKBest on the full cleaned dataset and return selected probes.

    This is for exploratory interpretation only.
    """

    selector = SelectKBest(
        score_func=f_classif,
        k=k_features
    )

    selector.fit(X, y)

    selected_mask = selector.get_support()
    selected_probe_ids = X.columns[selected_mask]

    selected_scores = selector.scores_[selected_mask]
    selected_p_values = selector.pvalues_[selected_mask]

    selected_df = pd.DataFrame({
        "probe_id": selected_probe_ids,
        "f_score": selected_scores,
        "p_value": selected_p_values
    })

    selected_df = selected_df.sort_values(
        by="f_score",
        ascending=False
    )

    selected_df["rank"] = range(1, len(selected_df) + 1)

    selected_df = selected_df[
        ["rank", "probe_id", "f_score", "p_value"]
    ]

    return selected_df


# ------------------------------------------------------------
# 5. Select probes for k=10 and k=25
# ------------------------------------------------------------

print("\nSelecting probes using SelectKBest(k=10)...")
selected_k10_df = get_selected_probes_for_k(X, y, k_features=10)

print("\nSelected probes for k=10:")
print(selected_k10_df)

k10_output_path = outputs_folder / "best_model_selected_probes_k10.csv"
selected_k10_df.to_csv(k10_output_path, index=False)

print("\nSaved k=10 selected probes:")
print(k10_output_path)


print("\nSelecting probes using SelectKBest(k=25)...")
selected_k25_df = get_selected_probes_for_k(X, y, k_features=25)

print("\nSelected probes for k=25:")
print(selected_k25_df)

k25_output_path = outputs_folder / "best_model_selected_probes_k25.csv"
selected_k25_df.to_csv(k25_output_path, index=False)

print("\nSaved k=25 selected probes:")
print(k25_output_path)


# ------------------------------------------------------------
# 6. Load GPL6480 annotation file
# ------------------------------------------------------------

annotation_path = data_folder / "GPL6480-9577.txt"

print("\nLoading GPL6480 annotation file:")
print(annotation_path)

if not annotation_path.exists():
    raise FileNotFoundError(
        "Could not find GPL6480-9577.txt in the data folder. "
        "Please download the GPL6480 annotation file and place it in data/."
    )

annotation_df = pd.read_csv(
    annotation_path,
    sep="\t",
    comment="#",
    low_memory=False
)

print("\nAnnotation table loaded.")
print("Annotation shape:")
print(annotation_df.shape)

print("\nAnnotation columns:")
print(annotation_df.columns.tolist())


# ------------------------------------------------------------
# 7. Prepare annotation columns
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

existing_annotation_columns = [
    col for col in annotation_columns_to_keep
    if col in annotation_df.columns
]

annotation_subset_df = annotation_df[existing_annotation_columns].copy()

annotation_subset_df = annotation_subset_df.rename(
    columns={"ID": "probe_id"}
)


# ------------------------------------------------------------
# 8. Add model-setting labels
# ------------------------------------------------------------

selected_k10_labelled_df = selected_k10_df.copy()
selected_k10_labelled_df["feature_selection_setting"] = "k10"
selected_k10_labelled_df["model_settings_using_this_k"] = "Random_Forest_k10; Elastic_Net_Logistic_Regression_k10"

selected_k25_labelled_df = selected_k25_df.copy()
selected_k25_labelled_df["feature_selection_setting"] = "k25"
selected_k25_labelled_df["model_settings_using_this_k"] = "Gaussian_Naive_Bayes_k25"


combined_selected_df = pd.concat(
    [selected_k10_labelled_df, selected_k25_labelled_df],
    ignore_index=True
)


# ------------------------------------------------------------
# 9. Map selected probes to gene annotations
# ------------------------------------------------------------

combined_annotated_df = combined_selected_df.merge(
    annotation_subset_df,
    on="probe_id",
    how="left"
)

combined_annotated_output_path = outputs_folder / "best_model_selected_probes_with_annotations.csv"
combined_annotated_df.to_csv(combined_annotated_output_path, index=False)

print("\nSaved selected probes with annotations:")
print(combined_annotated_output_path)


# ------------------------------------------------------------
# 10. Check unmapped probes
# ------------------------------------------------------------

if "GENE_SYMBOL" in combined_annotated_df.columns:
    unmapped_df = combined_annotated_df[
        combined_annotated_df["GENE_SYMBOL"].isna()
    ].copy()
else:
    unmapped_df = pd.DataFrame()

print("\nNumber of selected probe rows:")
print(combined_annotated_df.shape[0])

print("\nNumber of selected probe rows without gene symbol:")
print(unmapped_df.shape[0])

unmapped_output_path = outputs_folder / "best_model_selected_probes_unmapped.csv"
unmapped_df.to_csv(unmapped_output_path, index=False)

print("\nSaved unmapped selected probes:")
print(unmapped_output_path)


# ------------------------------------------------------------
# 11. Check overlap between k=10 and k=25 selected probes
# ------------------------------------------------------------

k10_probe_set = set(selected_k10_df["probe_id"])
k25_probe_set = set(selected_k25_df["probe_id"])

overlap_probes = sorted(list(k10_probe_set.intersection(k25_probe_set)))

overlap_df = pd.DataFrame({
    "probe_id": overlap_probes
})

overlap_df = overlap_df.merge(
    annotation_subset_df,
    on="probe_id",
    how="left"
)

overlap_output_path = outputs_folder / "best_model_selected_probe_overlap_k10_k25.csv"
overlap_df.to_csv(overlap_output_path, index=False)

print("\nNumber of overlapping probes between k=10 and k=25:")
print(len(overlap_probes))

print("\nSaved k=10/k=25 overlap probes:")
print(overlap_output_path)


# ------------------------------------------------------------
# 12. Check overlap with previous stable selected probes
# ------------------------------------------------------------

stable_features_path = outputs_folder / "feature_selection_stability_k25.csv"

print("\nChecking overlap with previous stability analysis:")
print(stable_features_path)

if stable_features_path.exists():
    stable_df = pd.read_csv(stable_features_path)

    print("\nStable feature table loaded.")
    print("Stable feature table shape:")
    print(stable_df.shape)

    # Expected columns from earlier stability script:
    # probe_id, selection_count, selection_frequency

    if "probe_id" not in stable_df.columns:
        raise ValueError(
            "feature_selection_stability_k25.csv does not contain a probe_id column."
        )

    stable_focus_df = stable_df.copy()

    if "selection_frequency" in stable_focus_df.columns:
        stable_focus_df = stable_focus_df[
            stable_focus_df["selection_frequency"] >= 0.80
        ].copy()

    stable_probe_set = set(stable_focus_df["probe_id"])

    combined_selected_probe_set = set(combined_selected_df["probe_id"])

    overlap_with_stable = sorted(
        list(combined_selected_probe_set.intersection(stable_probe_set))
    )

    overlap_stable_df = pd.DataFrame({
        "probe_id": overlap_with_stable
    })

    overlap_stable_df = overlap_stable_df.merge(
        combined_annotated_df.drop_duplicates(subset=["probe_id"]),
        on="probe_id",
        how="left"
    )

    stable_overlap_output_path = outputs_folder / "best_model_selected_probe_overlap_with_stable_features.csv"
    overlap_stable_df.to_csv(stable_overlap_output_path, index=False)

    print("\nNumber of selected probes overlapping with stable >=80% probes:")
    print(len(overlap_with_stable))

    print("\nOverlapping stable probes:")
    print(overlap_stable_df)

    print("\nSaved overlap with stable features:")
    print(stable_overlap_output_path)

else:
    print("\nPrevious stability file not found. Skipping stable feature overlap check.")


# ------------------------------------------------------------
# 13. Create simple gene-level summary
# ------------------------------------------------------------

gene_summary_df = combined_annotated_df.copy()

if "GENE_SYMBOL" in gene_summary_df.columns:
    gene_summary_df["GENE_SYMBOL"] = gene_summary_df["GENE_SYMBOL"].fillna("Unmapped/unclear")

    gene_counts_df = (
        gene_summary_df
        .groupby("GENE_SYMBOL")
        .agg(
            probe_count=("probe_id", "nunique"),
            feature_selection_settings=("feature_selection_setting", lambda x: "; ".join(sorted(set(x)))),
            model_settings=("model_settings_using_this_k", lambda x: "; ".join(sorted(set(x))))
        )
        .reset_index()
        .sort_values(by="probe_count", ascending=False)
    )

    gene_summary_output_path = outputs_folder / "best_model_selected_gene_summary.csv"
    gene_counts_df.to_csv(gene_summary_output_path, index=False)

    print("\nGene-level summary:")
    print(gene_counts_df)

    print("\nSaved gene-level summary:")
    print(gene_summary_output_path)

else:
    print("\nGENE_SYMBOL column not available. Skipping gene-level summary.")


# ------------------------------------------------------------
# 14. Save text summary
# ------------------------------------------------------------

text_summary_path = outputs_folder / "best_model_selected_probe_summary.txt"

with open(text_summary_path, "w", encoding="utf-8") as f:
    f.write("Best Model Selected Probe Summary\n")
    f.write("=================================\n\n")

    f.write("Purpose:\n")
    f.write("Inspect probes selected by SelectKBest for the strongest model settings.\n\n")

    f.write("Important note:\n")
    f.write("The selected probes in this file come from fitting SelectKBest on the full cleaned dataset for exploratory interpretation.\n")
    f.write("Model performance was previously evaluated using feature selection inside cross-validation to avoid data leakage.\n\n")

    f.write("Model settings inspected:\n")
    f.write("- Random Forest with k=10\n")
    f.write("- Gaussian Naive Bayes with k=25\n")
    f.write("- Elastic Net Logistic Regression with k=10\n\n")

    f.write("Dataset:\n")
    f.write(f"Samples: {X.shape[0]}\n")
    f.write(f"Expression probe features: {X.shape[1]}\n\n")

    f.write("Class counts:\n")
    f.write(y.value_counts().to_string())
    f.write("\n\n")

    f.write("Selected probes for k=10:\n")
    f.write(selected_k10_df.to_string(index=False))
    f.write("\n\n")

    f.write("Selected probes for k=25:\n")
    f.write(selected_k25_df.to_string(index=False))
    f.write("\n\n")

    f.write("Number of overlapping probes between k=10 and k=25:\n")
    f.write(str(len(overlap_probes)))
    f.write("\n\n")

    if stable_features_path.exists():
        f.write("Number of selected probes overlapping with previous stable >=80% probes:\n")
        f.write(str(len(overlap_with_stable)))
        f.write("\n\n")

        f.write("Overlapping stable probes:\n")
        f.write(overlap_stable_df.to_string(index=False))
        f.write("\n\n")

    if "GENE_SYMBOL" in combined_annotated_df.columns:
        f.write("Gene-level summary:\n")
        f.write(gene_counts_df.to_string(index=False))
        f.write("\n\n")

    f.write("Output files:\n")
    f.write(f"- {k10_output_path}\n")
    f.write(f"- {k25_output_path}\n")
    f.write(f"- {combined_annotated_output_path}\n")
    f.write(f"- {overlap_output_path}\n")
    f.write(f"- {unmapped_output_path}\n")

print("\nSaved text summary:")
print(text_summary_path)


# ------------------------------------------------------------
# 15. Final message
# ------------------------------------------------------------

print("\nSelected probe inspection complete.")
print("Main outputs:")
print(k10_output_path)
print(k25_output_path)
print(combined_annotated_output_path)
print(overlap_output_path)
print(unmapped_output_path)
print(text_summary_path)
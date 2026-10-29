from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script inspects the LSA / TruncatedSVD components from the
# strongest LSA classification setting.
#
# From the LSA classification experiment, the best setting was:
#
#   TruncatedSVD with 40 components
#   followed by Logistic Regression
#
# This script fits TruncatedSVD with 40 components on the full scaled
# expression dataset for exploratory interpretation.
#
# It then identifies which probes contribute most strongly to each
# component and maps those probes to gene annotations using GPL6480.
#
# Important:
# This is an exploratory interpretation step.
# It is not used to calculate model performance.
# ------------------------------------------------------------


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
plots_folder = project_folder / "plots"

outputs_folder.mkdir(exist_ok=True)
plots_folder.mkdir(exist_ok=True)


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
# 4. Load GPL6480 annotation file
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
# 5. Prepare annotation table
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
# 6. Fit TruncatedSVD with best LSA setting
# ------------------------------------------------------------

# From the previous LSA classification experiment, the strongest LSA
# setting was Logistic Regression with 40 TruncatedSVD components.
#
# Here, we fit 40 components on the full scaled dataset for exploratory
# interpretation of the components.

n_components = 40

print("\nScaling X...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFitting TruncatedSVD...")
print("n_components:")
print(n_components)

svd = TruncatedSVD(
    n_components=n_components,
    random_state=42
)

X_lsa = svd.fit_transform(X_scaled)

print("\nLSA transformed matrix shape:")
print(X_lsa.shape)

explained_variance_ratio = svd.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

explained_variance_df = pd.DataFrame({
    "component": range(1, n_components + 1),
    "explained_variance_ratio": explained_variance_ratio,
    "cumulative_explained_variance_ratio": cumulative_explained_variance
})

explained_variance_output_path = outputs_folder / "lsa_40_component_explained_variance.csv"
explained_variance_df.to_csv(explained_variance_output_path, index=False)

print("\nSaved explained variance table:")
print(explained_variance_output_path)

print("\nExplained variance summary:")
print(explained_variance_df.head(10))
print("\nCumulative explained variance at 40 components:")
print(cumulative_explained_variance[-1])


# ------------------------------------------------------------
# 7. Save sample-level LSA component scores
# ------------------------------------------------------------

lsa_scores_df = pd.DataFrame(
    X_lsa,
    columns=[f"LSA_component_{i}" for i in range(1, n_components + 1)]
)

lsa_scores_df.insert(0, "fast_growing", y.values)

lsa_scores_output_path = outputs_folder / "lsa_40_component_sample_scores.csv"
lsa_scores_df.to_csv(lsa_scores_output_path, index=False)

print("\nSaved sample-level LSA component scores:")
print(lsa_scores_output_path)


# ------------------------------------------------------------
# 8. Identify top contributing probes per component
# ------------------------------------------------------------

# svd.components_ has shape:
#   n_components x n_features
#
# Each value is a loading/weight showing how strongly a probe contributes
# to a component.
#
# Positive and negative loadings both matter, so we rank probes using
# absolute loading size.

top_n_probes_per_component = 25

all_top_probe_rows = []

for component_index in range(n_components):
    component_number = component_index + 1

    loadings = svd.components_[component_index]

    component_loadings_df = pd.DataFrame({
        "probe_id": X.columns,
        "loading": loadings
    })

    component_loadings_df["absolute_loading"] = component_loadings_df["loading"].abs()

    component_loadings_df = component_loadings_df.sort_values(
        by="absolute_loading",
        ascending=False
    )

    top_component_df = component_loadings_df.head(top_n_probes_per_component).copy()

    top_component_df.insert(0, "component", component_number)
    top_component_df.insert(
        1,
        "explained_variance_ratio",
        explained_variance_ratio[component_index]
    )
    top_component_df.insert(
        2,
        "cumulative_explained_variance_ratio",
        cumulative_explained_variance[component_index]
    )

    top_component_df["rank_within_component"] = range(1, len(top_component_df) + 1)

    all_top_probe_rows.append(top_component_df)

top_probes_df = pd.concat(
    all_top_probe_rows,
    ignore_index=True
)

top_probes_output_path = outputs_folder / "lsa_40_top_probe_loadings_by_component.csv"
top_probes_df.to_csv(top_probes_output_path, index=False)

print("\nSaved top probe loadings by component:")
print(top_probes_output_path)


# ------------------------------------------------------------
# 9. Map top LSA probes to gene annotations
# ------------------------------------------------------------

top_probes_annotated_df = top_probes_df.merge(
    annotation_subset_df,
    on="probe_id",
    how="left"
)

top_probes_annotated_output_path = outputs_folder / "lsa_40_top_probe_loadings_by_component_annotated.csv"
top_probes_annotated_df.to_csv(top_probes_annotated_output_path, index=False)

print("\nSaved annotated top probe loadings:")
print(top_probes_annotated_output_path)


# ------------------------------------------------------------
# 10. Check whether previous candidate genes appear in LSA components
# ------------------------------------------------------------

candidate_genes = [
    "ANGPT2",
    "ESM1",
    "MMS22L",
    "ZG16",
    "UROC1",
    "GSTA2",
    "NDC80"
]

print("\nChecking candidate genes in top LSA probe loadings:")
print(candidate_genes)

if "GENE_SYMBOL" in top_probes_annotated_df.columns:
    candidate_lsa_hits_df = top_probes_annotated_df[
        top_probes_annotated_df["GENE_SYMBOL"].isin(candidate_genes)
    ].copy()
else:
    candidate_lsa_hits_df = pd.DataFrame()

candidate_hits_output_path = outputs_folder / "lsa_40_candidate_gene_hits_in_top_loadings.csv"
candidate_lsa_hits_df.to_csv(candidate_hits_output_path, index=False)

print("\nCandidate gene hits in top LSA loadings:")
if candidate_lsa_hits_df.empty:
    print("No candidate genes found among top LSA loadings.")
else:
    print(candidate_lsa_hits_df[
        [
            "component",
            "rank_within_component",
            "probe_id",
            "GENE_SYMBOL",
            "GENE_NAME",
            "loading",
            "absolute_loading"
        ]
    ])

print("\nSaved candidate gene hits:")
print(candidate_hits_output_path)


# ------------------------------------------------------------
# 11. Gene-level summary of top LSA loadings
# ------------------------------------------------------------

gene_summary_df = top_probes_annotated_df.copy()

if "GENE_SYMBOL" in gene_summary_df.columns:
    gene_summary_df["GENE_SYMBOL"] = gene_summary_df["GENE_SYMBOL"].fillna("Unmapped/unclear")

    lsa_gene_summary_df = (
        gene_summary_df
        .groupby("GENE_SYMBOL")
        .agg(
            probe_count=("probe_id", "nunique"),
            component_count=("component", "nunique"),
            components=("component", lambda x: ", ".join(str(v) for v in sorted(set(x)))),
            max_absolute_loading=("absolute_loading", "max")
        )
        .reset_index()
        .sort_values(
            by=["component_count", "max_absolute_loading"],
            ascending=[False, False]
        )
    )

    lsa_gene_summary_output_path = outputs_folder / "lsa_40_top_loading_gene_summary.csv"
    lsa_gene_summary_df.to_csv(lsa_gene_summary_output_path, index=False)

    print("\nGene-level summary of top LSA loadings:")
    print(lsa_gene_summary_df.head(30))

    print("\nSaved gene-level summary:")
    print(lsa_gene_summary_output_path)

else:
    lsa_gene_summary_df = pd.DataFrame()
    print("\nGENE_SYMBOL column not available. Skipping gene-level summary.")


# ------------------------------------------------------------
# 12. Compare LSA top-loading probes with previous stable selected probes
# ------------------------------------------------------------

stable_features_path = outputs_folder / "feature_selection_stability_k25.csv"

print("\nChecking overlap with previous stable selected probes:")
print(stable_features_path)

if stable_features_path.exists():
    stable_df = pd.read_csv(stable_features_path)

    print("\nStable feature table loaded.")
    print("Stable table shape:")
    print(stable_df.shape)

    if "probe_id" not in stable_df.columns:
        raise ValueError(
            "feature_selection_stability_k25.csv does not contain a probe_id column."
        )

    if "selection_frequency" in stable_df.columns:
        stable_80_df = stable_df[
            stable_df["selection_frequency"] >= 0.80
        ].copy()
    else:
        stable_80_df = stable_df.copy()

    stable_80_probe_set = set(stable_80_df["probe_id"])
    lsa_top_probe_set = set(top_probes_annotated_df["probe_id"])

    stable_lsa_overlap_probes = sorted(
        list(stable_80_probe_set.intersection(lsa_top_probe_set))
    )

    stable_lsa_overlap_df = pd.DataFrame({
        "probe_id": stable_lsa_overlap_probes
    })

    stable_lsa_overlap_df = stable_lsa_overlap_df.merge(
        top_probes_annotated_df,
        on="probe_id",
        how="left"
    )

    stable_lsa_overlap_output_path = outputs_folder / "lsa_40_overlap_with_stable_selected_probes.csv"
    stable_lsa_overlap_df.to_csv(stable_lsa_overlap_output_path, index=False)

    print("\nNumber of top LSA probes overlapping with stable >=80% selected probes:")
    print(len(stable_lsa_overlap_probes))

    print("\nStable selected probes appearing in top LSA loadings:")
    if stable_lsa_overlap_df.empty:
        print("No overlap found.")
    else:
        print(stable_lsa_overlap_df[
            [
                "component",
                "rank_within_component",
                "probe_id",
                "GENE_SYMBOL",
                "GENE_NAME",
                "loading",
                "absolute_loading"
            ]
        ])

    print("\nSaved stable overlap file:")
    print(stable_lsa_overlap_output_path)

else:
    stable_lsa_overlap_df = pd.DataFrame()
    print("\nPrevious stable feature file not found. Skipping stable overlap check.")


# ------------------------------------------------------------
# 13. Plot cumulative explained variance
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(
    explained_variance_df["component"],
    explained_variance_df["cumulative_explained_variance_ratio"],
    marker="o"
)
plt.xlabel("LSA / TruncatedSVD component")
plt.ylabel("Cumulative explained variance ratio")
plt.title("Cumulative explained variance for 40 LSA components")
plt.tight_layout()

explained_variance_plot_path = plots_folder / "lsa_40_cumulative_explained_variance.png"
plt.savefig(explained_variance_plot_path, dpi=300)
plt.close()

print("\nSaved cumulative explained variance plot:")
print(explained_variance_plot_path)


# ------------------------------------------------------------
# 14. Plot top 15 genes by component count
# ------------------------------------------------------------

if not lsa_gene_summary_df.empty:
    top_gene_plot_df = lsa_gene_summary_df.head(15).copy()

    plt.figure(figsize=(12, 6))
    plt.bar(
        top_gene_plot_df["GENE_SYMBOL"],
        top_gene_plot_df["component_count"]
    )
    plt.xlabel("Gene symbol")
    plt.ylabel("Number of components where gene appears in top loadings")
    plt.title("Top LSA-loading genes by component count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    gene_count_plot_path = plots_folder / "lsa_40_top_genes_by_component_count.png"
    plt.savefig(gene_count_plot_path, dpi=300)
    plt.close()

    print("\nSaved top genes by component-count plot:")
    print(gene_count_plot_path)


# ------------------------------------------------------------
# 15. Save text summary
# ------------------------------------------------------------

text_summary_path = outputs_folder / "lsa_40_component_interpretation_summary.txt"

with open(text_summary_path, "w", encoding="utf-8") as f:
    f.write("LSA 40-Component Interpretation Summary\n")
    f.write("=======================================\n\n")

    f.write("Purpose\n")
    f.write("-------\n")
    f.write("This analysis inspected the TruncatedSVD components from the strongest LSA classification setting.\n")
    f.write("The strongest LSA classification setting was TruncatedSVD with 40 components followed by Logistic Regression.\n")
    f.write("The aim was to identify which probes and genes contributed most strongly to the latent expression components.\n\n")

    f.write("Important methodological note\n")
    f.write("-----------------------------\n")
    f.write("TruncatedSVD was fitted on the full scaled dataset for exploratory interpretation only.\n")
    f.write("This step was not used to calculate classification performance.\n")
    f.write("Classification performance was evaluated separately using repeated stratified cross-validation.\n\n")

    f.write("Dataset\n")
    f.write("-------\n")
    f.write(f"Samples: {X.shape[0]}\n")
    f.write(f"Expression probe features: {X.shape[1]}\n")
    f.write("Target: fast_growing\n\n")

    f.write("Class counts\n")
    f.write("------------\n")
    f.write(y.value_counts().to_string())
    f.write("\n\n")

    f.write("Explained variance\n")
    f.write("------------------\n")
    f.write(explained_variance_df.to_string(index=False))
    f.write("\n\n")

    f.write("Cumulative explained variance at 40 components\n")
    f.write("----------------------------------------------\n")
    f.write(str(cumulative_explained_variance[-1]))
    f.write("\n\n")

    f.write("Candidate genes checked\n")
    f.write("-----------------------\n")
    for gene in candidate_genes:
        f.write(f"- {gene}\n")
    f.write("\n")

    f.write("Candidate gene hits in top LSA loadings\n")
    f.write("---------------------------------------\n")
    if candidate_lsa_hits_df.empty:
        f.write("No candidate genes found among top LSA loadings.\n")
    else:
        f.write(candidate_lsa_hits_df[
            [
                "component",
                "rank_within_component",
                "probe_id",
                "GENE_SYMBOL",
                "GENE_NAME",
                "loading",
                "absolute_loading"
            ]
        ].to_string(index=False))
        f.write("\n")
    f.write("\n")

    f.write("Overlap with stable selected probes\n")
    f.write("-----------------------------------\n")
    if stable_features_path.exists():
        f.write(f"Number of top LSA probes overlapping with stable >=80% selected probes: {len(stable_lsa_overlap_probes)}\n\n")

        if stable_lsa_overlap_df.empty:
            f.write("No overlap found.\n")
        else:
            f.write(stable_lsa_overlap_df[
                [
                    "component",
                    "rank_within_component",
                    "probe_id",
                    "GENE_SYMBOL",
                    "GENE_NAME",
                    "loading",
                    "absolute_loading"
                ]
            ].to_string(index=False))
            f.write("\n")
    else:
        f.write("Stable feature file not found.\n")
    f.write("\n")

    if not lsa_gene_summary_df.empty:
        f.write("Top gene-level summary\n")
        f.write("----------------------\n")
        f.write(lsa_gene_summary_df.head(30).to_string(index=False))
        f.write("\n\n")

    f.write("Output files\n")
    f.write("------------\n")
    f.write(f"- {explained_variance_output_path}\n")
    f.write(f"- {lsa_scores_output_path}\n")
    f.write(f"- {top_probes_output_path}\n")
    f.write(f"- {top_probes_annotated_output_path}\n")
    f.write(f"- {candidate_hits_output_path}\n")
    if not lsa_gene_summary_df.empty:
        f.write(f"- {lsa_gene_summary_output_path}\n")
    if stable_features_path.exists():
        f.write(f"- {stable_lsa_overlap_output_path}\n")
    f.write(f"- {text_summary_path}\n")

print("\nSaved text summary:")
print(text_summary_path)


# ------------------------------------------------------------
# 16. Final message
# ------------------------------------------------------------

print("\nLSA component inspection complete.")

print("\nMain outputs:")
print(explained_variance_output_path)
print(lsa_scores_output_path)
print(top_probes_output_path)
print(top_probes_annotated_output_path)
print(candidate_hits_output_path)

if not lsa_gene_summary_df.empty:
    print(lsa_gene_summary_output_path)

if stable_features_path.exists():
    print(stable_lsa_overlap_output_path)

print(text_summary_path)

print("\nMain plots:")
print(explained_variance_plot_path)

if not lsa_gene_summary_df.empty:
    print(gene_count_plot_path)
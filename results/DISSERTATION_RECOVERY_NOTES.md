# Recovered figures and reported results from the MSc dissertation

**Source:** the submitted MSc dissertation PDF, *Exploratory Classification of Fast-Growing Hepatocellular Carcinoma Tumours Using Gene-Expression Data* (September 2026).

## Provenance and limitations

- The `plots/extracted_from_dissertation/` PNGs are **the original image objects embedded in the submitted dissertation PDF**, extracted without recomputing or redrawing the charts. They are not necessarily the original high-resolution files saved by the Python scripts. Figure 5.1 spans two pages and is provided as two images.
- The `results/reported_in_dissertation/` CSVs are **manual transcriptions of the published tables** in Section 7 of the submitted PDF. They are not the original CSV outputs of the experiments and are not newly computed results.
- Some report tables contain rounded means/standard deviations; raw per-fold scores cannot be reconstructed from these summary statistics.
- `Table_7_1` has `N/A` where the dissertation reports N/A for k=100 precision and recall.
- `Table_7_8` retains the report's stated difference for A_23_P372516 (+0.559), although subtraction of the displayed rounded means gives +0.560. It is not silently changed.
- The experiment used repeated stratified five-fold cross-validation (20 repeats) for the main comparisons. The preliminary k comparison used a single stratified five-fold CV. The project was exploratory and had no external validation or nested CV.
- Do not label these assets as independently rerun results. If using them in GitHub, cite the dissertation and label as **reported results** and **figures extracted from the dissertation**.

## Figure source pages (PDF page numbers)

| File | Page | Figure |
| --- | --- | --- |
| `Figure_4_1_tumour_doubling_time_boxplot.png` | 16 | 4 |
| `Figure_5_1a_sample_level_expression_mean.png` | 19 | 5 |
| `Figure_5_1b_sample_level_expression_std.png` | 20 | 5 |
| `Figure_5_2_PCA_gene_expression.png` | 21 | 5 |
| `Figure_5_3_top_50_variance_probe_correlation.png` | 22 | 5 |
| `Figure_5_4_selected_25_probe_correlation.png` | 23 | 5 |
| `Figure_7_1_balanced_accuracy_across_k.png` | 31 | 7 |
| `Figure_7_2_ROC_AUC_across_k.png` | 31 | 7 |
| `Figure_7_3_TruncatedSVD_balanced_accuracy.png` | 33 | 7 |
| `Figure_C_1_doubling_time_vs_survival.png` | 49 | C |
| `Figure_C_2_recall_across_k.png` | 50 | C |
| `Figure_C_3_F1_across_k.png` | 51 | C |
| `Figure_C_4_TruncatedSVD_recall.png` | 51 | C |
| `Figure_C_5_TruncatedSVD_F1.png` | 52 | C |
| `Figure_C_6_TruncatedSVD_top_probes.png` | 52 | C |

## CSV source tables (Section 7)

| File | Table | PDF page |
| --- | --- | --- |
| `Table_7_1_preliminary_k_comparison.csv` | 7.1 | 29 |
| `Table_7_2_classifier_comparison_k25.csv` | 7.2 | 29 |
| `Table_7_3_strongest_model_settings_mean_std.csv` | 7.3 | 30 |
| `Table_7_4_strongest_models_metrics.csv` | 7.4 | 32 |
| `Table_7_5_truncated_svd_model_comparison.csv` | 7.5 | 33 |
| `Table_7_6_permutation_test.csv` | 7.6 | 34 |
| `Table_7_7_feature_selection_stability_summary.csv` | 7.7 | 34 |
| `Table_7_8_stable_probes_gene_annotations.csv` | 7.8 | 35 |

## Suggested repository integration

Copy the `plots/` and `results/` directories into the project root. Keep the existing `outputs/` directory as generated local outputs, and use the recovered files as documented historical findings rather than experiment outputs. Verify each figure and CSV against the submitted PDF before public release.

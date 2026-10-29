
# Exploratory Classification of Fast-Growing Hepatocellular Carcinoma Tumours Using Gene-Expression Data

An MSc Computer Science project investigating whether gene-expression patterns can be used to distinguish fast-growing hepatocellular carcinoma (HCC) tumours from other HCC tumours using supervised machine learning.

The project explores high-dimensional biological data, feature selection, classification algorithms and model evaluation, with an emphasis on identifying potentially informative gene-expression probes.

## Project Overview

Hepatocellular carcinoma (HCC) is a common form of primary liver cancer. Tumours can differ in their growth behaviour, making it valuable to investigate biological patterns associated with more aggressive disease.

This project frames tumour growth behaviour as a binary classification problem:

- **Class 1 — Fast-growing tumours:** tumour doubling time of 54 days or less.
- **Class 0 — Remaining tumours:** tumour doubling time greater than 54 days.

The 54-day threshold corresponds to the lowest quartile of tumour doubling times in the dataset. It is a dataset-derived definition rather than an established clinical diagnostic threshold.

The main objective was to evaluate whether gene-expression measurements could distinguish these groups and to explore which probes might be associated with faster tumour growth.

## Dataset

**Source:** [NCBI Gene Expression Omnibus — GSE54236](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54236)

The study uses the GSE54236 Series Matrix dataset.

| Dataset characteristic | Value |
|---|---:|
| Total samples | 161 |
| Tumour samples | 81 |
| Non-tumour/cirrhotic samples | 80 |
| Original expression probes | 41,000 |
| Probes retained after missing-value removal | 39,613 |
| Fast-growing tumour samples | 21 |
| Remaining tumour samples | 60 |

Only the 81 tumour samples were used for the classification task.

The dataset contains substantially more features than samples, making overfitting and feature selection important considerations.

## Methodology

### 1. Data preparation

The preprocessing workflow:

1. Extracts sample metadata and gene-expression measurements from the GEO Series Matrix file.
2. Identifies the 81 tumour samples.
3. Transposes the expression data so that rows represent samples and columns represent expression probes.
4. Joins tumour metadata with expression measurements.
5. Creates the binary target label using tumour doubling time.
6. Removes expression probes containing missing values.
7. Separates expression features (`X`) from the target label (`y`).

Clinical metadata, including doubling time and survival time, is excluded from the predictive features.

### 2. Exploratory data analysis

The project examines:

- Tumour doubling-time distributions.
- Gene-expression distributions and variability.
- Sample-level expression statistics.
- Principal component analysis (PCA).
- Correlations between selected expression probes.

These analyses help describe the dataset and investigate the structure of its high-dimensional feature space.

### 3. Machine learning

The classification methods investigated include:

- Logistic Regression.
- Elastic Net Logistic Regression.
- Linear Support Vector Machine.
- Gaussian Naive Bayes.
- Random Forest.

Supervised feature selection was explored using `SelectKBest` with different numbers of selected probes.

An alternative approach using `TruncatedSVD` for dimensionality reduction was also investigated.

### 4. Model evaluation

The main model comparisons used repeated stratified five-fold cross-validation with 20 repeats.

Scaling and supervised feature selection were incorporated into modelling pipelines so that these operations could be fitted within the training folds.

Evaluation considered:

- Balanced accuracy.
- ROC AUC.
- Recall for fast-growing tumours.
- Specificity.
- Accuracy and F1-score.

Balanced accuracy was particularly relevant because the two classes were imbalanced.

## Key Findings

The dissertation reported the following results among the stronger model configurations:

| Model | Selected probes | Mean balanced accuracy | Mean ROC AUC |
|---|---:|---:|---:|
| Random Forest | 10 | 0.8299 | 0.9245 |
| Gaussian Naive Bayes | 25 | 0.8293 | 0.9117 |
| Elastic Net Logistic Regression | 10 | 0.8203 | 0.9080 |

These are cross-validation results reported in the original dissertation, not results from a new independent validation study.

Feature-selection stability analysis also highlighted probes annotated to genes including **ANGPT2, ESM1, MMS22L and ZG16** as candidates for further investigation.

These findings are exploratory and do not establish clinically validated biomarkers.

## Limitations

Important limitations include:

- The relatively small number of tumour samples (81).
- Class imbalance, with only 21 fast-growing tumours.
- A very high number of gene-expression features relative to the sample count.
- A target definition based on the dataset's tumour doubling-time distribution.
- Potential optimism from comparing and selecting models using cross-validation results.
- No independent external validation cohort.
- No fully nested cross-validation evaluation of the complete model-selection procedure.

The results should therefore be interpreted as exploratory evidence rather than clinically deployable predictions.

## Repository Structure

```text
hcc-gene-expression-classification/
├── data/                  # Local GEO dataset files (not tracked)
├── outputs/               # Generated intermediate files (not tracked)
├── plots/
│   └── extracted_from_dissertation/
├── results/
│   ├── reported_in_dissertation/
│   └── DISSERTATION_RECOVERY_NOTES.md
├── scripts/               # Modelling and analysis scripts
├── *.py                   # Dataset preparation and inspection scripts
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### Requirements

The recovered project has been tested using Python 3.14.4 in Ubuntu under Windows Subsystem for Linux (WSL).

Install the required Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Download the dataset

Download the GSE54236 Series Matrix file from:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54236

Place the file in the `data/` directory with the filename:

```text
data/GSE54236_series_matrix.txt.gz
```

The GEO platform annotation file is also used by some of the later probe-annotation scripts.

### Run preprocessing

From the project root, execute:

```bash
mkdir -p data outputs

python build_metadata_table.py
python transpose_expression.py
python join_metadata_expression.py
python create_target_label.py
python inspect_missing_expression_values.py
python clean_missing_expression_values.py
python scripts/prepare_modelling_data.py
```

The final modelling inputs are saved to:

```text
outputs/X_expression_features.csv
outputs/y_fast_growing_label.csv
```

### Run a baseline model

```bash
python scripts/baseline_model.py
```

Additional scripts in `scripts/` perform model comparisons, dimensionality reduction, feature-stability analysis and related evaluations.

Some experiments involve repeated cross-validation and may take substantial time to complete.

## Results and Figure Provenance

The repository includes figures and summary tables recovered from the submitted MSc dissertation.

- `plots/extracted_from_dissertation/` contains 15 image objects extracted from the dissertation PDF.
- `results/reported_in_dissertation/` contains eight CSV files manually transcribed from dissertation tables.
- `results/DISSERTATION_RECOVERY_NOTES.md` documents the recovery process and its limitations.

These files preserve the **reported dissertation findings**. They are not the original fold-level experiment outputs and should not be interpreted as newly regenerated results.

The preprocessing workflow and a baseline classifier have also been executed successfully in the recovered Python environment.

## Project Context

Originally developed as an MSc Computer Science conversion project at the University of Birmingham.

The repository is intended to demonstrate practical experience with Python, data preprocessing, exploratory data analysis, supervised machine learning, model evaluation and reproducible research workflows.

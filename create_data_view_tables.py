import pandas as pd

# ------------------------------------------------------------
# Create variable type table manually
# ------------------------------------------------------------

variable_table = pd.DataFrame([
    {
        "Variable / data field": "sample_id / GSM ID",
        "Example value": "GSM1310570",
        "Meaning": "Unique GEO sample identifier",
        "Variable type": "Identifier / nominal",
        "Role in project": "Matches metadata rows to expression columns"
    },
    {
        "Variable / data field": "title",
        "Example value": "Tumor T_A_001",
        "Meaning": "Human-readable sample name",
        "Variable type": "Identifier / nominal",
        "Role in project": "Helps interpret sample origin"
    },
    {
        "Variable / data field": "source_name",
        "Example value": "Biopsy of tumor tissue",
        "Meaning": "Describes sample source",
        "Variable type": "Categorical / nominal",
        "Role in project": "Used to separate tumour from non-tumour samples"
    },
    {
        "Variable / data field": "tissue type",
        "Example value": "Biopsy of tumor tissue",
        "Meaning": "Tissue category from metadata",
        "Variable type": "Categorical / nominal",
        "Role in project": "Used to confirm tumour vs cirrhotic non-malignant samples"
    },
    {
        "Variable / data field": "tumour vs non-tumour status",
        "Example value": "tumour",
        "Meaning": "Whether sample is HCC tumour or cirrhotic non-malignant tissue",
        "Variable type": "Binary categorical",
        "Role in project": "Used for filtering; main classification dataset keeps tumour samples only"
    },
    {
        "Variable / data field": "gender",
        "Example value": "male",
        "Meaning": "Patient sex/gender metadata",
        "Variable type": "Binary categorical",
        "Role in project": "Descriptive clinical/sample variable"
    },
    {
        "Variable / data field": "doubling time (days)",
        "Example value": "54",
        "Meaning": "Estimated time for tumour to double in size",
        "Variable type": "Ratio numerical / continuous",
        "Role in project": "Used to define fast-growing tumours"
    },
    {
        "Variable / data field": "survival time(months)",
        "Example value": "25",
        "Meaning": "Survival time in months",
        "Variable type": "Ratio numerical / continuous",
        "Role in project": "Prognosis-related clinical variable"
    },
    {
        "Variable / data field": "fast_growing",
        "Example value": "1",
        "Meaning": "1 if doubling time <= 54 days, otherwise 0",
        "Variable type": "Binary categorical target",
        "Role in project": "Proposed supervised classification target"
    },
    {
        "Variable / data field": "gene-expression probe value",
        "Example value": "8.232",
        "Meaning": "Processed expression measurement for a probe in a sample",
        "Variable type": "Continuous numerical",
        "Role in project": "Machine learning input feature"
    },
    {
        "Variable / data field": "probe ID / expression feature name",
        "Example value": "A_23_P100001",
        "Meaning": "Agilent microarray probe identifier",
        "Variable type": "Identifier / nominal",
        "Role in project": "Column name for expression feature"
    },
])

variable_table.to_csv("outputs/data_view_variable_types.csv", index=False)

print("Saved:")
print("outputs/data_view_variable_types.csv")
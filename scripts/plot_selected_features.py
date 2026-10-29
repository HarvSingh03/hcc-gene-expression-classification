from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script creates a bar chart of the top 25 selected
# expression probes from SelectKBest(k=25).
#
# The probes are ranked by F-score.
#
# Important:
# These are preliminary selected expression features,
# not confirmed biomarkers.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Find project folder
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# 2. Load selected features
# ------------------------------------------------------------

selected_features_path = project_folder / "outputs" / "selected_features_k25.csv"

print("Loading selected features from:")
print(selected_features_path)

selected_features = pd.read_csv(selected_features_path)

print("\nSelected features table:")
print(selected_features)


# ------------------------------------------------------------
# 3. Sort features by F-score
# ------------------------------------------------------------

selected_features = selected_features.sort_values(
    by="f_score",
    ascending=True
)

print("\nSelected features sorted for plotting:")
print(selected_features)


# ------------------------------------------------------------
# 4. Create horizontal bar chart
# ------------------------------------------------------------
# A horizontal bar chart is easier to read because probe IDs
# are long labels.
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.barh(
    selected_features["probe_id"],
    selected_features["f_score"]
)

plt.xlabel("F-score")
plt.ylabel("Selected expression probe")
plt.title("Top 25 Selected Expression Probes by F-score")

plt.tight_layout()

output_path = project_folder / "plots" / "selected_features_k25_f_scores.png"

plt.savefig(output_path)
plt.close()

print("\nSaved plot:")
print(output_path)


# ------------------------------------------------------------
# 5. Final explanation
# ------------------------------------------------------------

print("\nSelected feature plot complete.")
print("Higher F-score means the probe showed stronger separation between")
print("fast-growing and remaining tumour samples in this simple feature-selection test.")
print("These selected probes are preliminary model features, not confirmed biomarkers.")
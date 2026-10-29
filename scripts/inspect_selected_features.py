from pathlib import Path

import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif


# ------------------------------------------------------------
# Goal of this script
# ------------------------------------------------------------
# This script identifies which expression probes are selected
# by SelectKBest when using k=25.
#
# This is based on the current best baseline setting from the
# feature-selection comparison.
#
# Important:
# These selected probes are NOT confirmed biomarkers yet.
# They are only the top-ranked probes according to a simple
# statistical test comparing fast-growing vs remaining tumours.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Find project folder and load X and y
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

X_path = project_folder / "outputs" / "X_expression_features.csv"
y_path = project_folder / "outputs" / "y_fast_growing_label.csv"

print("Loading X from:")
print(X_path)

print("\nLoading y from:")
print(y_path)

X = pd.read_csv(X_path)
y = pd.read_csv(y_path).squeeze()

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nTarget label counts:")
print(y.value_counts())


# ------------------------------------------------------------
# 2. Apply SelectKBest with k=25
# ------------------------------------------------------------
# SelectKBest ranks each probe based on how different its
# expression values are between:
#
# class 1 = fast-growing tumours
# class 0 = remaining tumours
#
# f_classif returns:
# - an F-score for each probe
# - a p-value for each probe
# ------------------------------------------------------------

k = 25

selector = SelectKBest(score_func=f_classif, k=k)

selector.fit(X, y)


# ------------------------------------------------------------
# 3. Get selected probe names and scores
# ------------------------------------------------------------

selected_mask = selector.get_support()

selected_probe_names = X.columns[selected_mask]

all_scores = selector.scores_
all_p_values = selector.pvalues_

selected_features = pd.DataFrame({
    "probe_id": selected_probe_names,
    "f_score": all_scores[selected_mask],
    "p_value": all_p_values[selected_mask]
})

selected_features = selected_features.sort_values(
    by="f_score",
    ascending=False
)

print("\nSelected features using SelectKBest(k=25):")
print(selected_features)


# ------------------------------------------------------------
# 4. Save selected features
# ------------------------------------------------------------

output_path = project_folder / "outputs" / "selected_features_k25.csv"

selected_features.to_csv(output_path, index=False)

print("\nSaved selected features:")
print(output_path)


# ------------------------------------------------------------
# 5. Also save all feature scores
# ------------------------------------------------------------
# This gives a full ranking of all 39,613 probes.
# It may be useful later, but the selected_features_k25 file
# is the main file to look at now.
# ------------------------------------------------------------

all_feature_scores = pd.DataFrame({
    "probe_id": X.columns,
    "f_score": all_scores,
    "p_value": all_p_values
})

all_feature_scores = all_feature_scores.sort_values(
    by="f_score",
    ascending=False
)

all_scores_output_path = project_folder / "outputs" / "all_feature_scores.csv"

all_feature_scores.to_csv(all_scores_output_path, index=False)

print("\nSaved all feature scores:")
print(all_scores_output_path)


# ------------------------------------------------------------
# 6. Final explanation
# ------------------------------------------------------------

print("\nSelected feature inspection complete.")
print("These probes are selected using a simple statistical comparison between classes.")
print("They should be treated as preliminary model features, not confirmed biomarkers.")
print("Further validation and biological annotation would be needed before making biomarker claims.")
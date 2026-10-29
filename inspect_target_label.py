import pandas as pd

combined_df = pd.read_csv("outputs/tumour_metadata_plus_expression.csv")

print("Combined table shape:")
print(combined_df.shape)

print("\nDoubling time summary:")
print(combined_df["doubling time (days)"].describe())

print("\nSurvival time summary:")
print(combined_df["survival time(months)"].describe())

print("\nDoubling time sorted from fastest to slowest:")
print(
    combined_df[[
        "sample_id",
        "title",
        "doubling time (days)",
        "survival time(months)"
    ]]
    .sort_values("doubling time (days)")
    .head(25)
)

print("\nDoubling time quartiles:")
print(combined_df["doubling time (days)"].quantile([0, 0.25, 0.5, 0.75, 1]))

print("\nNumber of samples with doubling time <= 54 days:")
print((combined_df["doubling time (days)"] <= 54).sum())

print("\nNumber of samples with doubling time > 54 days:")
print((combined_df["doubling time (days)"] > 54).sum())
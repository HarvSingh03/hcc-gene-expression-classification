import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# ------------------------------------------------------------
# Load cleaned tumour dataset
# ------------------------------------------------------------

df = pd.read_csv("outputs/tumour_labelled_no_missing_probes.csv")

print("Loaded cleaned tumour dataset:")
print(df.shape)

# ------------------------------------------------------------
# 1. Histogram of tumour doubling time
# ------------------------------------------------------------

plt.figure()
plt.hist(df["doubling time (days)"], bins=15)
plt.xlabel("Doubling time (days)")
plt.ylabel("Number of tumour samples")
plt.title("Distribution of Tumour Doubling Time")
plt.tight_layout()
plt.savefig("plots/doubling_time_histogram.png")
plt.close()

print("Saved: plots/doubling_time_histogram.png")

# ------------------------------------------------------------
# 2. Boxplot of tumour doubling time
# ------------------------------------------------------------

plt.figure()
plt.boxplot(df["doubling time (days)"])
plt.ylabel("Doubling time (days)")
plt.title("Boxplot of Tumour Doubling Time")
plt.tight_layout()
plt.savefig("plots/doubling_time_boxplot.png")
plt.close()

print("Saved: plots/doubling_time_boxplot.png")

# ------------------------------------------------------------
# 3. Histogram of survival time
# ------------------------------------------------------------

plt.figure()
plt.hist(df["survival time(months)"], bins=15)
plt.xlabel("Survival time (months)")
plt.ylabel("Number of tumour samples")
plt.title("Distribution of Survival Time")
plt.tight_layout()
plt.savefig("plots/survival_time_histogram.png")
plt.close()

print("Saved: plots/survival_time_histogram.png")

# ------------------------------------------------------------
# 4. Boxplot of survival time
# ------------------------------------------------------------

plt.figure()
plt.boxplot(df["survival time(months)"])
plt.ylabel("Survival time (months)")
plt.title("Boxplot of Survival Time")
plt.tight_layout()
plt.savefig("plots/survival_time_boxplot.png")
plt.close()

print("Saved: plots/survival_time_boxplot.png")

# ------------------------------------------------------------
# 5. Bar chart of fast-growing class counts
# ------------------------------------------------------------

class_counts = df["fast_growing"].value_counts().sort_index()

plt.figure()
plt.bar(["Other tumours (0)", "Fast-growing (1)"], class_counts)
plt.xlabel("Class")
plt.ylabel("Number of tumour samples")
plt.title("Fast-growing Label Class Counts")
plt.tight_layout()
plt.savefig("plots/fast_growing_class_counts.png")
plt.close()

print("Saved: plots/fast_growing_class_counts.png")

# ------------------------------------------------------------
# 6. Scatterplot: doubling time vs survival time
# ------------------------------------------------------------

plt.figure()
plt.scatter(
    df["doubling time (days)"],
    df["survival time(months)"]
)
plt.xlabel("Doubling time (days)")
plt.ylabel("Survival time (months)")
plt.title("Doubling Time vs Survival Time")
plt.tight_layout()
plt.savefig("plots/doubling_vs_survival_scatter.png")
plt.close()

print("Saved: plots/doubling_vs_survival_scatter.png")

# ------------------------------------------------------------
# 7. Simple clinical scatterplot matrix / SPLOM
# ------------------------------------------------------------

clinical_view = df[[
    "doubling time (days)",
    "survival time(months)",
    "fast_growing"
]]

scatter_matrix(clinical_view, figsize=(8, 8), diagonal="hist")
plt.suptitle("Scatterplot Matrix of Clinical Variables", y=1.02)
plt.tight_layout()
plt.savefig("plots/clinical_splom.png")
plt.close()

print("Saved: plots/clinical_splom.png")

print("\nAll plots created successfully.")
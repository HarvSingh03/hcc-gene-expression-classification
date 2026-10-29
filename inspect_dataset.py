import gzip

file_path = "data/GSE54236_series_matrix.txt.gz"

sample_lines = []

with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("!Sample_"):
            sample_lines.append(line.strip())

print("Number of sample metadata lines:", len(sample_lines))

print("\nFirst 20 sample metadata lines:\n")
for line in sample_lines[:20]:
    print(line[:1000])
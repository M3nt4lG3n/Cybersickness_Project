import pandas as pd
from tkinter import Tk, filedialog
from pathlib import Path

# ----------------------------
# Select CSV files
# ----------------------------
root = Tk()
root.withdraw()

csv_files = filedialog.askopenfilenames(
    title="Select CSV files",
    filetypes=[("CSV Files", "*.csv")]
)

if not csv_files:
    print("No files selected.")
    exit()

dfs = []

min_time = None
max_time = None

# ----------------------------
# Load each CSV
# ----------------------------
for file in csv_files:

    df = pd.read_csv(file)

    if "UnixTime_ms" not in df.columns:
        print(f"Skipping {Path(file).name} (missing UnixTime_ms)")
        continue

    df["UnixTime_ms"] = pd.to_numeric(df["UnixTime_ms"])

    suffix = Path(file).stem

    rename = {
        c: f"{c}_{suffix}"
        for c in df.columns
        if c != "UnixTime_ms"
    }

    df.rename(columns=rename, inplace=True)

    df.sort_values("UnixTime_ms", inplace=True)

    dfs.append(df)

    if min_time is None:
        min_time = df["UnixTime_ms"].min()
        max_time = df["UnixTime_ms"].max()
    else:
        min_time = min(min_time, df["UnixTime_ms"].min())
        max_time = max(max_time, df["UnixTime_ms"].max())

# ----------------------------
# Create complete millisecond timeline
# ----------------------------
master = pd.DataFrame({
    "UnixTime_ms": range(int(min_time), int(max_time) + 1)
})

# ----------------------------
# Merge each dataset
# ----------------------------
merged = master

for df in dfs:
    merged = merged.merge(df, how="left", on="UnixTime_ms")

# ----------------------------
# Save
# ----------------------------
output = Path(csv_files[0]).parent / "merged_every_ms.csv"
merged.to_csv(output, index=False)

print(f"Saved to:\n{output}")
print(f"Rows: {len(merged):,}")
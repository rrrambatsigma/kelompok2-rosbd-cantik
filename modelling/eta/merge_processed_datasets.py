# FIX
import pandas as pd
import os
import glob

# =========================
# CONFIG
# =========================
INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_ROUTE_SAMPLES = 3000
# bisa ubah:
# 1000 / 3000 / 5000

# =========================
# LOAD ALL PARQUET
# =========================
files = glob.glob(
    os.path.join(INPUT_DIR, "processed_eta_training_*.parquet")
)

print("Files found:")
for f in files:
    print("-", f)

if len(files) == 0:
    print("No parquet files found.")
    exit()

# =========================
# MERGE
# =========================
all_df = []

for file in files:
    print(f"\nLoading {file}")
    df = pd.read_parquet(file)
    print("Shape:", df.shape)
    all_df.append(df)

print("\nMerging datasets...")
merged_df = pd.concat(
    all_df,
    ignore_index=True
)

print("Merged shape:")
print(merged_df.shape)

# =========================
# ADD ROUTE
# =========================
merged_df["route"] = (
    merged_df["departure_airport"]
    + "_"
    + merged_df["arrival_airport"]
)

print("\nUnique routes:")
print(merged_df["route"].nunique())

# =========================
# ROUTE COUNTS
# =========================
route_counts = (
    merged_df["route"]
    .value_counts()
    .reset_index()
)

route_counts.columns = [
    "route",
    "samples"
]

print("\nTop 20 routes:")
print(route_counts.head(20))

# save stats route
route_counts.to_csv(
    os.path.join(OUTPUT_DIR, "route_statistics.csv"),
    index=False
)

# =========================
# FILTER ROUTES
# =========================
valid_routes = route_counts[
    route_counts["samples"] >= MIN_ROUTE_SAMPLES
]["route"]

filtered_df = merged_df[
    merged_df["route"].isin(valid_routes)
]

print("\nAfter route filtering:")
print(filtered_df.shape)

print("Remaining routes:")
print(filtered_df["route"].nunique())

# =========================
# FEATURE ENGINEERING
# =========================
filtered_df["elapsed_time"] = (
    filtered_df["track_time"]
    - filtered_df["firstseen"]
)

flight_duration = (
    filtered_df["lastseen"]
    - filtered_df["firstseen"]
)

filtered_df["progress_ratio"] = (
    filtered_df["elapsed_time"]
    / flight_duration
)

# =========================
# SAVE FINAL DATASET
# =========================
final_file = os.path.join(
    OUTPUT_DIR,
    "eta_training_merged.parquet"
)

filtered_df.to_parquet(
    final_file,
    index=False
)

print("\nSaved final dataset:")
print(final_file)

print("\nFinal columns:")
print(filtered_df.columns)
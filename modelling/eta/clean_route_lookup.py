import pandas as pd
import os

# =========================
# CONFIG
# =========================
INPUT_FILE = "data/final/route_avg_duration.csv"
OUTPUT_FILE = "data/final/route_avg_duration_clean.csv"

MIN_SAMPLES = 100
MIN_DURATION_SEC = 600      # 10 menit
MAX_DURATION_SEC = 86400    # 24 jam

# =========================
# LOAD DATA
# =========================
print("Loading route lookup...")
df = pd.read_csv(INPUT_FILE)

print("Original shape:")
print(df.shape)

# backup count
original_count = len(df)

# =========================
# FILTER 1
# remove self-route
# =========================
print("\nRemoving self-routes...")

df["departure_airport"] = df["route"].apply(
    lambda x: x.split("_")[0]
)

df["arrival_airport"] = df["route"].apply(
    lambda x: x.split("_")[1]
)

before = len(df)

df = df[
    df["departure_airport"] != df["arrival_airport"]
]

removed = before - len(df)

print(f"Removed self-routes: {removed}")

# =========================
# FILTER 2
# minimum samples
# =========================
print("\nFiltering low-sample routes...")

before = len(df)

df = df[
    df["samples"] >= MIN_SAMPLES
]

removed = before - len(df)

print(f"Removed low-sample routes: {removed}")

# =========================
# FILTER 3
# duration sanity check
# =========================
print("\nFiltering weird durations...")

before = len(df)

df = df[
    (df["avg_duration_sec"] >= MIN_DURATION_SEC) &
    (df["avg_duration_sec"] <= MAX_DURATION_SEC)
]

removed = before - len(df)

print(f"Removed weird durations: {removed}")

# =========================
# SORT
# =========================
df = df.sort_values(
    by="samples",
    ascending=False
)

# optional: drop helper cols
df = df.drop(
    columns=[
        "departure_airport",
        "arrival_airport"
    ]
)

# =========================
# SAVE
# =========================
os.makedirs("data/final", exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================
# OUTPUT
# =========================
print("\n===== SUMMARY =====")
print(f"Original routes : {original_count}")
print(f"Remaining routes: {len(df)}")
print(f"Removed routes  : {original_count - len(df)}")

print("\nTop 20 cleaned routes:")
print(df.head(20))

print("\nSaved:")
print(OUTPUT_FILE)
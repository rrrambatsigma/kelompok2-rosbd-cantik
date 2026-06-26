import pandas as pd
import os

# =========================
# CONFIG
# =========================
INPUT_FILE = "data/final/eta_training_merged.parquet"
OUTPUT_FILE = "data/final/route_avg_duration.csv"

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_parquet(INPUT_FILE)

print("Shape:")
print(df.shape)

# =========================
# CHECK REQUIRED COLUMNS
# =========================
required_columns = [
    "route",
    "firstseen",
    "lastseen"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# =========================
# COMPUTE FLIGHT DURATION
# =========================
print("\nComputing flight duration...")

df["flight_duration"] = (
    df["lastseen"] - df["firstseen"]
)

# remove weird duration
df = df[
    (df["flight_duration"] > 0) &
    (df["flight_duration"] < 86400)   # < 24 jam
]

print("After duration filtering:")
print(df.shape)

# =========================
# ROUTE AGGREGATION
# =========================
print("\nCalculating route statistics...")

route_stats = (
    df.groupby("route")
      .agg(
          avg_duration_sec=("flight_duration", "mean"),
          median_duration_sec=("flight_duration", "median"),
          min_duration_sec=("flight_duration", "min"),
          max_duration_sec=("flight_duration", "max"),
          samples=("route", "count")
      )
      .reset_index()
)

# rounding
route_stats["avg_duration_sec"] = (
    route_stats["avg_duration_sec"].round(2)
)

route_stats["median_duration_sec"] = (
    route_stats["median_duration_sec"].round(2)
)

# convert to minutes
route_stats["avg_duration_min"] = (
    route_stats["avg_duration_sec"] / 60
).round(2)

# =========================
# SORT
# =========================
route_stats = route_stats.sort_values(
    by="samples",
    ascending=False
)

# =========================
# SAVE
# =========================
os.makedirs("data/final", exist_ok=True)

route_stats.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================
# OUTPUT
# =========================
print("\nRoute stats shape:")
print(route_stats.shape)

print("\nTop 20 routes:")
print(route_stats.head(20))

print("\nSaved:")
print(OUTPUT_FILE)
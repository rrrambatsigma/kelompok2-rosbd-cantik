import pandas as pd
import os

# =========================
# CONFIG
# =========================
DATASET_FILE = "data/final/eta_training_merged.parquet"
VALID_ROUTES_FILE = "data/final/route_avg_duration_clean.csv"
OUTPUT_FILE = "data/final/callsign_route_confidence.csv"

# threshold confidence
MIN_CONFIDENCE = 0.90

# =========================
# LOAD DATA
# =========================
print("Loading training dataset...")
df = pd.read_parquet(DATASET_FILE)

print("Dataset shape:")
print(df.shape)

print("\nLoading valid routes...")
valid_routes = pd.read_csv(VALID_ROUTES_FILE)

valid_route_set = set(valid_routes["route"])

print("Valid routes:")
print(len(valid_route_set))

# =========================
# FILTER VALID ROUTES ONLY
# =========================
print("\nFiltering valid routes...")

before = len(df)

df = df[df["route"].isin(valid_route_set)]

after = len(df)

print(f"Before: {before}")
print(f"After : {after}")

# =========================
# DROP BAD CALLSIGN
# =========================
df["callsign"] = df["callsign"].astype(str).str.strip()
df = df[df["callsign"] != ""]
df = df[df["callsign"] != "nan"]

print("\nAfter callsign cleaning:")
print(df.shape)

# =========================
# COUNT CALLSIGN x ROUTE
# =========================
print("\nCounting callsign-route frequency...")

route_counts = (
    df.groupby(["callsign", "route"])
      .size()
      .reset_index(name="frequency")
)

print("Grouped rows:")
print(route_counts.shape)

# =========================
# TOTAL SAMPLE PER CALLSIGN
# =========================
callsign_total = (
    route_counts.groupby("callsign")["frequency"]
    .sum()
    .reset_index(name="total_samples")
)

# =========================
# GET TOP ROUTE PER CALLSIGN
# =========================
best_route = (
    route_counts.sort_values(
        by="frequency",
        ascending=False
    )
    .drop_duplicates(subset=["callsign"])
)

# merge total
best_route = best_route.merge(
    callsign_total,
    on="callsign"
)

# confidence
best_route["confidence"] = (
    best_route["frequency"] /
    best_route["total_samples"]
).round(4)

# =========================
# SORT
# =========================
best_route = best_route.sort_values(
    by=["confidence", "frequency"],
    ascending=[False, False]
)

# =========================
# SAVE
# =========================
os.makedirs("data/final", exist_ok=True)

best_route.to_csv(
    OUTPUT_FILE,
    index=False
)

# =========================
# STATS
# =========================
high_conf = best_route[
    best_route["confidence"] >= MIN_CONFIDENCE
]

print("\n===== SUMMARY =====")
print("Total callsign:", len(best_route))
print(f"Callsign confidence >= {MIN_CONFIDENCE}: {len(high_conf)}")

ratio = len(high_conf) / len(best_route) * 100
print(f"Reliable ratio: {ratio:.2f}%")

print("\nTop 20:")
print(best_route.head(20))

print("\nLow confidence examples:")
print(
    best_route[best_route["confidence"] < MIN_CONFIDENCE]
    .head(20)
)

print("\nSaved:")
print(OUTPUT_FILE)
import pandas as pd
import os

# =========================
# CONFIG
# =========================
INPUT_FILE = "data/final/eta_training_merged.parquet"
OUTPUT_FILE = "data/final/prefix_route_confidence.csv"

MIN_PREFIX_LEN = 3

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_parquet(INPUT_FILE)

print("Shape:")
print(df.shape)

# =========================
# CLEAN CALLSIGN
# =========================
df["callsign"] = df["callsign"].astype(str).str.strip()

df = df[
    (df["callsign"] != "") &
    (df["callsign"] != "nan")
]

print("\nAfter callsign cleaning:")
print(df.shape)

# =========================
# EXTRACT PREFIX
# =========================
print("\nExtracting prefix...")

df["prefix"] = df["callsign"].str[:3].str.upper()

# filter prefix valid alphabetic
df = df[
    df["prefix"].str.match(r"^[A-Z]{3}$", na=False)
]

print("After prefix filtering:")
print(df.shape)

# =========================
# COUNT PREFIX-ROUTE
# =========================
print("\nCounting prefix-route frequency...")

prefix_route = (
    df.groupby(["prefix", "route"])
      .size()
      .reset_index(name="frequency")
)

print("Grouped shape:")
print(prefix_route.shape)

# =========================
# TOTAL PER PREFIX
# =========================
prefix_total = (
    prefix_route.groupby("prefix")["frequency"]
    .sum()
    .reset_index(name="total_samples")
)

# =========================
# BEST ROUTE PER PREFIX
# =========================
best_route = (
    prefix_route
    .sort_values(by="frequency", ascending=False)
    .drop_duplicates(subset=["prefix"])
)

best_route = best_route.merge(
    prefix_total,
    on="prefix"
)

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
print("\n===== SUMMARY =====")
print("Total prefixes:", len(best_route))

high_conf = best_route[
    best_route["confidence"] >= 0.7
]

print("Prefix confidence >= 0.7:", len(high_conf))
print(
    f"Reliable ratio: {len(high_conf)/len(best_route)*100:.2f}%"
)

print("\nTop 30 prefixes:")
print(best_route.head(30))

print("\nLow confidence prefixes:")
print(
    best_route[
        best_route["confidence"] < 0.7
    ].head(30)
)

print("\nSaved:")
print(OUTPUT_FILE)
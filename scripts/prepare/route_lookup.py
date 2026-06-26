import pandas as pd

INPUT_FILE = "data/final/eta_training_merged.parquet"

print("Loading dataset...")
df = pd.read_parquet(INPUT_FILE)

lookup = (
    df.groupby(["callsign", "route"])
    .size()
    .reset_index(name="count")
)

# ambil route paling sering per callsign
lookup = (
    lookup.sort_values("count", ascending=False)
    .drop_duplicates(subset=["callsign"])
)

lookup = lookup[["callsign", "route"]]

print("Total mappings:", len(lookup))
print(lookup.head())

lookup.to_csv(
    "data/final/callsign_route_lookup.csv",
    index=False
)

print("Saved callsign_route_lookup.csv")
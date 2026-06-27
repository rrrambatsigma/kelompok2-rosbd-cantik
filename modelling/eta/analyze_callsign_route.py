import pandas as pd

INPUT_FILE = "data/final/eta_training_merged.parquet"

print("Loading dataset...")
df = pd.read_parquet(INPUT_FILE)

print("Shape:", df.shape)

# satu callsign muncul di berapa route
callsign_route_counts = (
    df.groupby("callsign")["route"]
    .nunique()
    .reset_index(name="num_routes")
)

print("\n===== STATISTICS =====")
print("Total unique callsign:", len(callsign_route_counts))

print("\nCallsign with exactly 1 route:")
print((callsign_route_counts["num_routes"] == 1).sum())

print("\nCallsign with >1 route:")
print((callsign_route_counts["num_routes"] > 1).sum())

print("\nTop 20 callsign with most routes:")
print(
    callsign_route_counts
    .sort_values("num_routes", ascending=False)
    .head(20)
)
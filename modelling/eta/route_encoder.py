import pandas as pd
import joblib

lookup = pd.read_csv("data/final/callsign_route_lookup.csv")
encoder = joblib.load("models/route_encoder.pkl")

known_routes = set(encoder.classes_)
lookup_routes = set(lookup["route"])

missing = lookup_routes - known_routes

print("Total route lookup:", len(lookup_routes))
print("Known by encoder:", len(known_routes))
print("Missing:", len(missing))

if len(missing) > 0:
    print(list(missing)[:20])
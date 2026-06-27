# build_airport_lookup.py

import pandas as pd

print("Loading callsign route lookup...")
route_df = pd.read_csv(
    "data/final/callsign_route_lookup.csv"
)

print("Loading airports...")
airport_df = pd.read_csv(
    "data/raw/airports.csv"
)

used_airports = set()

for route in route_df["route"]:

    try:
        origin, dest = route.split("_")

        used_airports.add(origin)
        used_airports.add(dest)

    except:
        pass

print("Unique airports:", len(used_airports))

airport_lookup = airport_df[
    airport_df["ident"].isin(used_airports)
].copy()

airport_lookup = airport_lookup[
    [
        "ident",
        "latitude_deg",
        "longitude_deg",
        "name",
        "iso_country"
    ]
]

airport_lookup.columns = [
    "icao",
    "lat",
    "lon",
    "airport_name",
    "country"
]

print("Rows:", len(airport_lookup))

airport_lookup.to_csv(
    "data/final/airport_lookup.csv",
    index=False
)

print("Saved airport lookup")
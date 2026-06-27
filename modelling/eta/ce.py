import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

airport_df = pd.read_csv("data/final/airport_lookup.csv")
route_df = pd.read_csv("data/final/callsign_route_lookup.csv")

airport_lookup = airport_df.set_index("icao").to_dict("index")

# contoh input
callsign = "EZY52NZ"
current_lat = 52.2
current_lon = 4.8
groundspeed = 750

row = route_df[route_df["callsign"] == callsign]

if row.empty:
    print("Callsign tidak ditemukan")
    exit()

route = row.iloc[0]["route"]

origin, destination = route.split("_")

if destination not in airport_lookup:
    print("Airport tujuan tidak ditemukan")
    exit()

dest_lat = airport_lookup[destination]["lat"]
dest_lon = airport_lookup[destination]["lon"]

remaining_distance = haversine(
    current_lat,
    current_lon,
    dest_lat,
    dest_lon
)

eta_hours = remaining_distance / groundspeed

print("Callsign :", callsign)
print("Route :", route)
print("Destination :", destination)
print(f"Remaining Distance : {remaining_distance:.2f} km")
print(f"ETA : {eta_hours:.2f} jam")
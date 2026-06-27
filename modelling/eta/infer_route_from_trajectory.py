import pandas as pd
import math

# ==================================================
# CONFIG
# ==================================================

CURRENT_LAT = -6.30
CURRENT_LON = 107.50
CURRENT_HEADING = 95

TOP_N = 10

# ==================================================
# HELPERS
# ==================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2)

    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1)
        * math.cos(lat2)
        * math.cos(dlon)
    )

    bearing = math.degrees(math.atan2(x, y))

    return (bearing + 360) % 360


def heading_difference(h1, h2):

    diff = abs(h1 - h2)

    return min(diff, 360 - diff)


# ==================================================
# LOAD DATA
# ==================================================

print("Loading airport lookup...")
airports = pd.read_csv(
    "data/final/airport_lookup.csv"
)

print("Loading route lookup...")
routes = pd.read_csv(
    "data/final/callsign_route_lookup.csv"
)

# ==================================================
# BUILD AIRPORT DICT
# ==================================================

airport_dict = {}

for _, row in airports.iterrows():

    airport_dict[row["icao"]] = {
        "lat": row["lat"],
        "lon": row["lon"],
        "name": row["airport_name"]
    }

# ==================================================
# SCORE ROUTES
# ==================================================

results = []

for _, row in routes.iterrows():

    route = row["route"]

    try:
        origin, destination = route.split("-")

    except:
        continue

    if destination not in airport_dict:
        continue

    dest_lat = airport_dict[destination]["lat"]
    dest_lon = airport_dict[destination]["lon"]

    distance = haversine(
        CURRENT_LAT,
        CURRENT_LON,
        dest_lat,
        dest_lon
    )

    bearing = calculate_bearing(
        CURRENT_LAT,
        CURRENT_LON,
        dest_lat,
        dest_lon
    )

    heading_diff = heading_difference(
        CURRENT_HEADING,
        bearing
    )

    # score arah
    heading_score = max(
        0,
        1 - (heading_diff / 180)
    )

    # score jarak
    distance_score = 1 / (1 + distance / 1000)

    final_score = (
        0.8 * heading_score
        + 0.2 * distance_score
    )

    results.append({
        "route": route,
        "destination": destination,
        "distance_km": round(distance, 1),
        "bearing": round(bearing, 1),
        "heading_diff": round(heading_diff, 1),
        "score": round(final_score, 4)
    })

# ==================================================
# OUTPUT
# ==================================================

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "score",
    ascending=False
)

print("\nTOP ROUTE CANDIDATES\n")

print(
    result_df.head(TOP_N)
    .to_string(index=False)
)
import pandas as pd
import math

INPUT_FILE = "data/historical/flights_data5_europe.parquet"
# ganti sesuai nama parquet raw kamu

print("Loading parquet...")
df = pd.read_parquet(INPUT_FILE)

print("Shape:", df.shape)

# ambil 1 flight
row = df.iloc[0]

track = row["track"]

print("\nFlight:")
print("ICAO24:", row["icao24"])
print("Callsign:", row["callsign"])
print("Jumlah titik track:", len(track))

# minimal 2 titik
if len(track) < 2:
    print("Track terlalu pendek")
    exit()

p1 = track[0]
p2 = track[1]

print("\nPoint 1:", p1)
print("Point 2:", p2)

# format track:
# [timestamp, lat, lon, altitude, heading, onground]

t1, lat1, lon1, alt1, heading1, _ = p1
t2, lat2, lon2, alt2, heading2, _ = p2

dt = abs(t2 - t1)

print("\nDelta time:", dt)

if dt == 0:
    print("Delta time = 0")
    exit()

# =========================
# HAVERSINE DISTANCE
# =========================
R = 6371  # km

lat1_rad = math.radians(lat1)
lon1_rad = math.radians(lon1)
lat2_rad = math.radians(lat2)
lon2_rad = math.radians(lon2)

dlat = lat2_rad - lat1_rad
dlon = lon2_rad - lon1_rad

a = (
    math.sin(dlat / 2) ** 2
    + math.cos(lat1_rad)
    * math.cos(lat2_rad)
    * math.sin(dlon / 2) ** 2
)

c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
distance_km = R * c

velocity_kmh = distance_km / (dt / 3600)

# =========================
# VERTICAL RATE
# =========================
vertical_rate = (alt2 - alt1) / dt

print("\n===== DERIVED FEATURES =====")
print(f"Distance       : {distance_km:.4f} km")
print(f"Velocity       : {velocity_kmh:.2f} km/h")
print(f"Vertical rate  : {vertical_rate:.2f} m/s")
print(f"Heading        : {heading2}")
print(f"Altitude       : {alt2}")
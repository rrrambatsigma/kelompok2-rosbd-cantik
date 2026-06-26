import pandas as pd
import os
import glob

# =========================
# CONFIG
# =========================

INPUT_DIR = "data/historical"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# cari semua parquet historical
parquet_files = glob.glob(
    os.path.join(INPUT_DIR, "*.parquet")
)

print("Found files:")
for f in parquet_files:
    print("-", f)


# =========================
# PROCESS FUNCTION
# =========================

def process_file(input_file):

    print("\n========================")
    print("Loading:", input_file)
    print("========================")

    df = pd.read_parquet(input_file)

    print("Raw shape:", df.shape)

    processed_rows = []
    total_flights = len(df)

    for idx, row in df.iterrows():

        if idx % 100 == 0:
            print(f"Processing flight {idx}/{total_flights}")

        track = row["track"]

        if track is None:
            continue

        icao24 = row["icao24"]
        callsign = row["callsign"]
        dep = row["estdepartureairport"]
        arr = row["estarrivalairport"]
        firstseen = row["firstseen"]
        lastseen = row["lastseen"]

        for point in track:

            if len(point) < 6:
                continue

            track_time = point[0]
            lat = point[1]
            lon = point[2]
            alt = point[3]
            heading = point[4]
            onground = point[5]

            remaining_time = lastseen - track_time

            if remaining_time < 0:
                continue

            processed_rows.append({
                "icao24": icao24,
                "callsign": callsign,
                "departure_airport": dep,
                "arrival_airport": arr,
                "firstseen": firstseen,
                "lastseen": lastseen,
                "track_time": track_time,
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "heading": heading,
                "onground": onground,
                "remaining_time": remaining_time
            })

    processed_df = pd.DataFrame(processed_rows)

    print("\nBefore cleaning:", processed_df.shape)

    processed_df = processed_df.dropna()
    processed_df = processed_df[
        processed_df["onground"] == 0
    ]
    processed_df = processed_df[
        processed_df["remaining_time"] > 0
    ]

    print("After cleaning:", processed_df.shape)

    # nama output
    base_name = os.path.basename(input_file)
    output_name = "processed_" + base_name
    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    processed_df.to_parquet(
        output_path,
        index=False
    )

    print("Saved:", output_path)


# =========================
# LOOP ALL FILES
# =========================

for file in parquet_files:
    process_file(file)

print("\nALL FILES DONE.")
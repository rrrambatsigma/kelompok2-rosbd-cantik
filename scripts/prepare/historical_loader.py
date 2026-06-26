# FIX

from trino.dbapi import connect
from trino.auth import OAuth2Authentication
import pandas as pd
import os

# =========================
# CONFIG
# =========================
DATA_DIR = "data/historical"
os.makedirs(DATA_DIR, exist_ok=True)

days = [
    1724544000,
    1724457600,
    1724371200,
    1724284800,
    1724198400,
    1724112000,
    1724025600
]

# =========================
# CONNECT TRINO
# =========================
print("Connecting to OpenSky Trino...")

conn = connect(
    host="trino.opensky-network.org",
    port=443,
    user="dheameidiyana@student.uns.ac.id",
    catalog="minio",
    schema="osky",
    http_scheme="https",
    auth=OAuth2Authentication()
)

cursor = conn.cursor()

# =========================
# LOOP PER HARI
# =========================
for day in days:

    print("\n========================")
    print(f"Processing day: {day}")
    print("========================")

    query = f"""
    WITH valid_routes AS (
        SELECT
            estdepartureairport,
            estarrivalairport
        FROM flights_data5
        WHERE day = {day}
          AND track IS NOT NULL
          AND estdepartureairport IS NOT NULL
          AND estarrivalairport IS NOT NULL
          AND any_match(
                track,
                x -> x.latitude BETWEEN 35 AND 72
                  AND x.longitude BETWEEN -9 AND 66
          )
        GROUP BY
            estdepartureairport,
            estarrivalairport
        HAVING COUNT(*) >= 10
    )

    SELECT
        f.icao24,
        f.firstseen,
        f.lastseen,
        f.estdepartureairport,
        f.estarrivalairport,
        f.callsign,
        f.track,
        f.day
    FROM flights_data5 f
    JOIN valid_routes vr
        ON f.estdepartureairport = vr.estdepartureairport
       AND f.estarrivalairport = vr.estarrivalairport
    WHERE f.day = {day}
      AND f.track IS NOT NULL
      AND f.estdepartureairport IS NOT NULL
      AND f.estarrivalairport IS NOT NULL
      AND any_match(
            f.track,
            x -> x.latitude BETWEEN 35 AND 72
              AND x.longitude BETWEEN -9 AND 66
      )
    """

    print("Running query...")
    cursor.execute(query)

    print("Fetching rows...")
    rows = cursor.fetchall()

    if len(rows) == 0:
        print("No data found.")
        continue

    columns = [col[0] for col in cursor.description]

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    print("\nData shape:")
    print(df.shape)

    unique_routes = (
        df[
            ["estdepartureairport", "estarrivalairport"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print("Unique routes:", unique_routes)

    filename = f"{DATA_DIR}/eta_training_{day}.parquet"

    df.to_parquet(
        filename,
        index=False
    )

    print("Saved:")
    print(filename)

print("\nDONE.")
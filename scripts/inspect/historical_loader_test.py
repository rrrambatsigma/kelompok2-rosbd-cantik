from trino.dbapi import connect
from trino.auth import OAuth2Authentication
import pandas as pd
import os
import time

# =========================
# CONFIG
# =========================

TEST_DAY = 1724544000
LIMIT_ROWS = 100

os.makedirs("data/historical", exist_ok=True)

# =========================
# CONNECT
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
# QUERY
# =========================

query = f"""
SELECT
    icao24,
    firstseen,
    lastseen,
    estdepartureairport,
    estarrivalairport,
    callsign,
    track,
    day
FROM flights_data5
WHERE day = {TEST_DAY}
  AND track IS NOT NULL
  AND any_match(
        track,
        x -> x.latitude BETWEEN 35 AND 72
          AND x.longitude BETWEEN -9 AND 66
  )
LIMIT {LIMIT_ROWS}
"""

print("Submitting query...")
start = time.time()

cursor.execute(query)

print("Query submitted.")
print("Fetching rows...")

rows = cursor.fetchall()

end = time.time()

print(f"Fetch selesai dalam {end - start:.2f} detik")
print(f"Rows fetched: {len(rows)}")

# =========================
# DATAFRAME
# =========================

columns = [col[0] for col in cursor.description]

df = pd.DataFrame(rows, columns=columns)

print("\nSample data:")
print(df.head())

# =========================
# SAVE
# =========================

output = "data/historical/test_100_rows.parquet"

df.to_parquet(output, index=False)

print("\nSaved to:")
print(output)

print("Shape:")
print(df.shape)
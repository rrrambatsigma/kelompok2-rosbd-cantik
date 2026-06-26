from trino.dbapi import connect
from trino.auth import OAuth2Authentication

# ==========================
# KONEKSI TRINO
# ==========================
conn = connect(
    host="trino.opensky-network.org",
    port=443,
    user="dheameidiyana@student.uns.ac.id",  # ganti dengan emailmu
    catalog="minio",
    schema="osky",
    http_scheme="https",
    auth=OAuth2Authentication()
)

cursor = conn.cursor()

# ==========================
# AMBIL 1 CONTOH DATA
# ==========================
query = """
SELECT
    estdepartureairport,
    estarrivalairport,
    track
FROM flights_data5
WHERE day = 1724544000
AND track IS NOT NULL
LIMIT 1
"""

cursor.execute(query)

row = cursor.fetchone()

print("Departure Airport:")
print(row[0])

print("\nArrival Airport:")
print(row[1])

print("\nTrack:")
print(row[2])
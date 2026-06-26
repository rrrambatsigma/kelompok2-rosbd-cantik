from trino.dbapi import connect
from trino.auth import OAuth2Authentication

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

print("Running query...")

cursor.execute("""
SELECT
    estdepartureairport,
    estarrivalairport,
    COUNT(*) AS total_flights
FROM flights_data5
WHERE day IN (
    1724544000,
    1724457600,
    1724371200,
    1724284800,
    1724198400,
    1724112000,
    1724025600
)
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
ORDER BY total_flights DESC
""")

rows = cursor.fetchall()

print("\nJumlah unique routes:", len(rows))

print("\nTop 20 routes:")
for row in rows[:20]:
    print(row)

# cek distribusi threshold
route_counts = [row[2] for row in rows]

print("\n===== ROUTE THRESHOLD =====")
print(">= 5 flights :", sum(c >= 5 for c in route_counts))
print(">= 10 flights:", sum(c >= 10 for c in route_counts))
print(">= 20 flights:", sum(c >= 20 for c in route_counts))
print(">= 50 flights:", sum(c >= 50 for c in route_counts))
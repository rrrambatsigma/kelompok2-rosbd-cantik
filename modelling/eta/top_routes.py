from trino.dbapi import connect
from trino.auth import OAuth2Authentication
import pandas as pd

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
# AMBIL 7 HARI TERAKHIR
# =========================
cursor.execute("""
SELECT DISTINCT day
FROM flights_data5
ORDER BY day DESC
LIMIT 7
""")

days = [row[0] for row in cursor.fetchall()]

print("Days:")
print(days)

day_list = ",".join(str(d) for d in days)

# =========================
# TOP ROUTES
# =========================
query = f"""
SELECT
    estdepartureairport,
    estarrivalairport,
    COUNT(*) AS total_flights
FROM flights_data5
WHERE day IN ({day_list})
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
LIMIT 20
"""

print("Running query...")

cursor.execute(query)
rows = cursor.fetchall()

columns = [
    col[0]
    for col in cursor.description
]

df = pd.DataFrame(rows, columns=columns)

print("\nTOP 20 ROUTES:")
print(df)

df.to_csv("top_routes.csv", index=False)

print("\nSaved to top_routes.csv")
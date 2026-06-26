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

print("\n7 hari terakhir:")
print(days)

# =========================
# HITUNG JUMLAH FLIGHT PER HARI
# =========================
print("\nJumlah flight per hari:\n")

for day in days:
    query = f"""
    SELECT COUNT(*)
    FROM flights_data5
    WHERE day = {day}
      AND track IS NOT NULL
      AND any_match(
            track,
            x -> x.latitude BETWEEN 35 AND 72
              AND x.longitude BETWEEN -9 AND 66
      )
    """

    cursor.execute(query)
    total = cursor.fetchone()[0]

    print(f"Day {day} -> {total} flights")
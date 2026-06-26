from trino.dbapi import connect
from trino.auth import OAuth2Authentication

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

cursor.execute("""
SELECT COUNT(*)
FROM flights_data5
WHERE day = 1724544000
AND track IS NOT NULL
AND any_match(
    track,
    x -> x.latitude BETWEEN 35 AND 72
      AND x.longitude BETWEEN -25 AND 45
)
""")

print(cursor.fetchone())
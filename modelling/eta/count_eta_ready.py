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

days = [
    1724544000,
    1724457600,
    1724371200,
    1724284800,
    1724198400,
    1724112000,
    1724025600
]

for day in days:
    query = f"""
    SELECT COUNT(*)
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
    """

    cursor.execute(query)
    total = cursor.fetchone()[0]
    print(day, total)
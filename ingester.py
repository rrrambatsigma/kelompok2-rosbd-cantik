import os, time, json, requests
from datetime import datetime, timedelta
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = "flights"
INTERVAL = 30

# Bounding box Eropa
BOUNDING_BOX = {
    "lamin": 34.5,
    "lomin": -10.0,
    "lamax": 71.0,
    "lomax": 40.0
}

class TokenManager:
    def __init__(self, clientId, clientSecret):
        self.clientId = clientId
        self.clientSecret = clientSecret
        self.token = None
        self.expires_at = None
        self.token_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"

    def get_token(self):
        if self.token and self.expires_at and datetime.now() < self.expires_at:
            return self.token
        return self._refresh()

    def _refresh(self):
        print("Refreshing token...")
        r = requests.post(self.token_url, data={
            "grant_type": "client_credentials",
            "clientId": self.clientId,
            "clientSecret": self.clientSecret,
        })
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        expires_in = data.get("expires_in", 1800)
        self.expires_at = datetime.now() + timedelta(seconds=expires_in)
        print("Token refreshed successfully")
        return self.token

    def headers(self):
        return {"Authorization": f"Bearer {self.get_token()}"}

# Load credentials
try:
    with open("credentials.json", "r") as f:
        creds = json.load(f)
    clientId = creds["clientId"]
    clientSecret = creds["clientSecret"]
    print("Credentials loaded.")
except Exception as e:
    print(f"Error loading credentials.json: {e}")
    exit(1)

token_manager = TokenManager(clientId, clientSecret)

# === RETRY MECHANISM FOR KAFKA ===
print(f"Waiting for Kafka at {KAFKA_BOOTSTRAP}...")
producer = None
while producer is None:
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        print("Kafka connected!")
    except NoBrokersAvailable:
        print("Kafka not ready yet, retrying in 3 seconds...")
        time.sleep(3)
    except Exception as e:
        print(f"Unexpected error connecting to Kafka: {e}")
        time.sleep(3)

print("Starting OpenSky ingester...")
while True:
    try:
        resp = requests.get(
            "https://opensky-network.org/api/states/all",
            params=BOUNDING_BOX,
            headers=token_manager.headers(),
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        states = data.get("states", [])
        count = 0
        for state in states:
            flight = {
                "icao24": state[0],
                "callsign": state[1].strip() if state[1] else None,
                "origin_country": state[2],
                "time_position": state[3],
                "last_contact": state[4],
                "longitude": state[5],
                "latitude": state[6],
                "baro_altitude": state[7],
                "on_ground": state[8],
                "velocity": state[9],
                "true_track": state[10],
                "vertical_rate": state[11],
                "geo_altitude": state[13] if len(state) > 13 else None,
                "timestamp": time.time()
            }
            producer.send(TOPIC, flight)
            count += 1
        print(f"Sent {count} flights at {datetime.now().isoformat()}")
        time.sleep(INTERVAL)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
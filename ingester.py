import os
import time
import json
import requests
from datetime import datetime, timedelta
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from elasticsearch import Elasticsearch 
import warnings
warnings.filterwarnings("ignore")  

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC           = "flights"
INTERVAL        = 5

BOUNDING_BOX = {
    "lamin": 34.5,   
    "lomin": -10.0,  
    "lamax": 71.0,   
    "lomax": 40.0    
}

TOKEN_URL            = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
TOKEN_REFRESH_MARGIN = 30
CREDENTIALS_FILE     = os.getenv("CREDENTIALS_FILE", "credentials.json")

# Elasticsearch
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch:9200")
es = Elasticsearch(f"http://{ES_HOST}")


class TokenManager:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.token         = None
        self.expires_at    = None

    def get_token(self) -> str:
        if self.token and self.expires_at and datetime.now() < self.expires_at:
            return self.token
        return self._refresh()

    def _refresh(self) -> str:
        print("[TOKEN] Refreshing...")
        r = requests.post(
            TOKEN_URL,
            data={
                "grant_type":    "client_credentials",
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        expires_in = data.get("expires_in", 1800)
        self.expires_at = datetime.now() + timedelta(seconds=expires_in - TOKEN_REFRESH_MARGIN)
        print("[TOKEN] OK")
        return self.token

    def auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}


def load_credentials(path: str):
    with open(path, "r") as f:
        creds = json.load(f)
    if "clientId" in creds:
        return creds["clientId"], creds["clientSecret"]
    elif "client_id" in creds:
        return creds["client_id"], creds["client_secret"]
    else:
        raise KeyError("Missing clientId/client_id in credentials.json")


def connect_kafka(bootstrap: str) -> KafkaProducer:
    print(f"[KAFKA] Connecting to {bootstrap}...")
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print("[KAFKA] Connected")
            return producer
        except NoBrokersAvailable:
            print("[KAFKA] Broker not ready, retry 5s")
            time.sleep(5)
        except Exception as e:
            print(f"[KAFKA] Error: {e}, retry 5s")
            time.sleep(5)


def preprocess_and_save(flight_dict: dict) -> None:
    """Preprocessing dan simpan ke Elasticsearch."""
    if not flight_dict.get("callsign"):
        return
    if flight_dict.get("longitude") is None or flight_dict.get("latitude") is None:
        return

    flight_dict["region"] = "Europe"

    if flight_dict.get("velocity") is not None:
        flight_dict["velocity_kmh"] = flight_dict["velocity"] * 3.6
    else:
        flight_dict["velocity_kmh"] = None

    # Filter altitude tidak wajar (misal > 20km)
    if flight_dict.get("geo_altitude") and flight_dict["geo_altitude"] > 20000:
        flight_dict["geo_altitude"] = None


    try:
        es.index(index="flights", document=flight_dict)
    except Exception as e:
        print(f"[ES] Error saving: {e}")


def parse_state(state: list) -> dict:
    """Konversi list state OpenSky → dict flight."""
    return {
        "icao24":         state[0],
        "callsign":       state[1].strip() if state[1] else None,
        "origin_country": state[2],
        "time_position":  state[3],
        "last_contact":   state[4],
        "longitude":      state[5],
        "latitude":       state[6],
        "baro_altitude":  state[7],
        "on_ground":      state[8],
        "velocity":       state[9],
        "true_track":     state[10],
        "vertical_rate":  state[11],
        "geo_altitude":   state[13] if len(state) > 13 else None,
        "squawk":         state[14] if len(state) > 14 else None,
        "timestamp":      time.time(),
        "ingested_at":    datetime.utcnow().isoformat() + "Z",
    }


def fetch_flights(token_manager: TokenManager) -> list:
    resp = requests.get(
        "https://opensky-network.org/api/states/all",
        params=BOUNDING_BOX,
        headers=token_manager.auth_headers(),
        timeout=15,
    )
    if resp.status_code == 429:
        print("[FETCH] Rate limited, wait 60s")
        time.sleep(60)
        return []
    if resp.status_code == 401:
        print("[FETCH] Token invalid, force refresh")
        token_manager.token = None
        return []
    resp.raise_for_status()
    data = resp.json()
    return data.get("states") or []


def main():
    client_id, client_secret = load_credentials(CREDENTIALS_FILE)
    token_manager = TokenManager(client_id, client_secret)

    try:
        token_manager.get_token()
        print("[INIT] Token OK")
    except Exception as e:
        print(f"[INIT] Token failed: {e}")
        exit(1)

    producer = connect_kafka(KAFKA_BOOTSTRAP)

    print(f"[MAIN] Ingester Europe started, interval {INTERVAL}s")
    consecutive_errors = 0

    while True:
        try:
            states = fetch_flights(token_manager)
            if states:
                count = 0
                for state in states:
                    flight = parse_state(state)
                    # Kirim ke Kafka
                    producer.send(TOPIC, flight)
                    # Preprocess & simpan ke Elasticsearch
                    preprocess_and_save(flight.copy())
                    count += 1
                producer.flush()
                print(f"[MAIN] Sent {count} flights, saved to ES")
                consecutive_errors = 0
            else:
                print("[MAIN] No data")
            time.sleep(INTERVAL)
        except Exception as e:
            consecutive_errors += 1
            print(f"[MAIN] Error: {e}")
            time.sleep(min(5 * consecutive_errors, 60))


if __name__ == "__main__":
    main()
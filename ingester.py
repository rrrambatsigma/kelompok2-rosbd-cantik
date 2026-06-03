import os
import time
import json
import requests
from datetime import datetime, timedelta
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ── Config ──────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC           = "flights"
INTERVAL        = 30  # detik antar fetch

BOUNDING_BOX = {
    "lamin": -11.0,   # Latitude min  (ujung selatan Indonesia)
    "lomin": 94.0,    # Longitude min (ujung barat Indonesia)
    "lamax":  6.0,    # Latitude max  (ujung utara Indonesia)
    "lomax": 141.0,   # Longitude max (ujung timur Indonesia)
}

TOKEN_URL            = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
TOKEN_REFRESH_MARGIN = 30  # refresh token 30 detik sebelum expired
CREDENTIALS_FILE     = os.getenv("CREDENTIALS_FILE", "credentials.json")
# ────────────────────────────────────────────────────────


class TokenManager:
    """Handle OAuth2 client_credentials token untuk OpenSky API."""

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
        print("[TOKEN] Refreshing token...")
        try:
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
        except requests.exceptions.HTTPError as e:
            print(f"[TOKEN] HTTP error saat refresh token: {r.status_code} - {r.text}")
            raise
        except Exception as e:
            print(f"[TOKEN] Gagal refresh token: {e}")
            raise

        data            = r.json()
        self.token      = data["access_token"]
        expires_in      = data.get("expires_in", 1800)
        self.expires_at = datetime.now() + timedelta(seconds=expires_in - TOKEN_REFRESH_MARGIN)
        print(f"[TOKEN] Token refreshed, berlaku {expires_in}s")
        return self.token

    def auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}


def load_credentials(path: str) -> tuple[str, str]:
    """Baca credentials.json dan return (client_id, client_secret)."""
    print(f"[CREDS] Membaca credentials dari: {path}")
    try:
        with open(path, "r") as f:
            creds = json.load(f)

        # Debug: tampilkan key yang tersedia
        print(f"[CREDS] Keys ditemukan: {list(creds.keys())}")

        client_id     = creds["clientId"]
        client_secret = creds["clientSecret"]
        print(f"[CREDS] Loaded client_id: {client_id[:20]}...")
        return client_id, client_secret

    except FileNotFoundError:
        print(f"[CREDS] ❌ File tidak ditemukan: {path}")
        raise
    except KeyError as e:
        print(f"[CREDS] ❌ Key tidak ada di credentials.json: {e}")
        print(f"[CREDS] Pastikan format file:")
        print('         { "clientId": "...", "clientSecret": "..." }')
        raise
    except json.JSONDecodeError as e:
        print(f"[CREDS] ❌ Format JSON tidak valid: {e}")
        raise


def connect_kafka(bootstrap: str) -> KafkaProducer:
    """Koneksi ke Kafka dengan retry loop."""
    print(f"[KAFKA] Menghubungkan ke {bootstrap}...")
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=5,
                acks="all",
            )
            print("[KAFKA] ✅ Terhubung!")
            return producer
        except NoBrokersAvailable:
            print("[KAFKA] Broker belum siap, retry 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"[KAFKA] Error: {e}, retry 5s...")
            time.sleep(5)


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
    """Fetch data penerbangan dari OpenSky API."""
    resp = requests.get(
        "https://opensky-network.org/api/states/all",
        params=BOUNDING_BOX,
        headers=token_manager.auth_headers(),
        timeout=15,
    )

    if resp.status_code == 429:
        print("[FETCH] ⚠️  Rate limited (429), tunggu 60s...")
        time.sleep(60)
        return []

    if resp.status_code == 401:
        print("[FETCH] ⚠️  Token tidak valid (401), force refresh...")
        token_manager.token = None  # force refresh next call
        return []

    resp.raise_for_status()
    data = resp.json()
    return data.get("states") or []


def main():
    # 1. Load credentials
    client_id, client_secret = load_credentials(CREDENTIALS_FILE)

    # 2. Setup token manager
    token_manager = TokenManager(client_id, client_secret)

    # 3. Test token sekali di awal
    print("[INIT] Test ambil token...")
    try:
        token_manager.get_token()
        print("[INIT] ✅ Token OK")
    except Exception as e:
        print(f"[INIT] ❌ Gagal ambil token: {e}")
        exit(1)

    # 4. Connect Kafka
    producer = connect_kafka(KAFKA_BOOTSTRAP)

    # 5. Main loop
    print(f"[MAIN] 🚀 Ingester berjalan, fetch setiap {INTERVAL}s...")
    consecutive_errors = 0

    while True:
        try:
            states = fetch_flights(token_manager)

            if states:
                count = 0
                for state in states:
                    flight = parse_state(state)
                    producer.send(TOPIC, flight)
                    count += 1

                producer.flush()
                print(f"[MAIN] ✅ Terkirim {count} penerbangan — {datetime.now().strftime('%H:%M:%S')}")
                consecutive_errors = 0
            else:
                print(f"[MAIN] ℹ️  Tidak ada data — {datetime.now().strftime('%H:%M:%S')}")

            time.sleep(INTERVAL)

        except requests.exceptions.HTTPError as e:
            consecutive_errors += 1
            print(f"[MAIN] ❌ HTTP Error: {e}")
            time.sleep(min(10 * consecutive_errors, 60))  # backoff max 60s

        except Exception as e:
            consecutive_errors += 1
            print(f"[MAIN] ❌ Error: {e}")
            time.sleep(min(5 * consecutive_errors, 30))   # backoff max 30s


if __name__ == "__main__":
    main()
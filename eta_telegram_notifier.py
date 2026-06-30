import os
import time
import json
from datetime import datetime
from elasticsearch import Elasticsearch

from utils.eta_telegram import send_eta_prediction, send_landing_alert

ES_HOST = os.getenv("ES_HOST", "http://127.0.0.1:9200")
PRED_INDEX = "flight_predictions"
POLL_INTERVAL = 15
CURSOR_FILE = "data/telegram_cursor.txt"

LANDING_DIST_KM = 50
LANDING_ALT_FT = 5000
LANDING_SPEED_KMH = 500
LANDING_COOLDOWN_SEC = 600

es = Elasticsearch(ES_HOST)


def load_cursor():
    try:
        with open(CURSOR_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "1970-01-01T00:00:00Z"


def save_cursor(timestamp):
    os.makedirs(os.path.dirname(CURSOR_FILE), exist_ok=True)
    with open(CURSOR_FILE, "w") as f:
        f.write(timestamp)


def get_new_predictions(cursor):
    query = {
        "query": {
            "range": {
                "predicted_at": {"gt": cursor}
            }
        },
        "sort": [{"predicted_at": {"order": "asc"}}],
        "size": 50
    }
    try:
        res = es.search(index=PRED_INDEX, body=query)
        return [hit["_source"] for hit in res["hits"]["hits"]]
    except Exception as e:
        print(f"[NOTIFIER] ES query error: {e}")
        return []


def check_landing_imminent(pred: dict) -> bool:
    cp = pred.get("current_position", {})
    dist = pred.get("distance_km_to_dest", 999)
    alt = cp.get("altitude", 9999) if isinstance(cp, dict) else 9999
    spd = cp.get("speed_kmh", 999) if isinstance(cp, dict) else 999
    return dist < LANDING_DIST_KM and alt < LANDING_ALT_FT and spd < LANDING_SPEED_KMH


def main():
    print("[NOTIFIER] Starting ETA Telegram Notifier...")
    try:
        es.info()
        print(f"[NOTIFIER] Connected to ES at {ES_HOST}")
    except Exception as e:
        print(f"[NOTIFIER] ES connection failed: {e}")
        print("[NOTIFIER] Make sure Elasticsearch is running at", ES_HOST)
        exit(1)

    landing_cooldown = {}

    while True:
        try:
            cursor = load_cursor()
            predictions = get_new_predictions(cursor)
            now = time.time()

            for pred in predictions:
                send_eta_prediction(pred)

                icao24 = pred.get("icao24")
                if not icao24:
                    continue

                if icao24 in landing_cooldown and now - landing_cooldown[icao24] < LANDING_COOLDOWN_SEC:
                    continue

                if check_landing_imminent(pred):
                    send_landing_alert(pred)
                    landing_cooldown[icao24] = now
                    print(f"[NOTIFIER] Landing alert sent for {icao24}")

            if predictions:
                latest_ts = predictions[-1].get("predicted_at", cursor)
                save_cursor(latest_ts)
                print(f"[NOTIFIER] Processed {len(predictions)} new predictions, cursor: {latest_ts}")

        except Exception as e:
            print(f"[NOTIFIER] Error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

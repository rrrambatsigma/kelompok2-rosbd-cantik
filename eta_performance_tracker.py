import os
import time
import json
from datetime import datetime
from elasticsearch import Elasticsearch, NotFoundError

ES_HOST = os.getenv("ES_HOST", "http://127.0.0.1:9200")
PRED_INDEX = "flight_predictions"
PRED_HISTORY_INDEX = "flight_predictions_history"
FLIGHT_INDEX = "flights"
ERROR_INDEX = "eta_errors"
POLL_INTERVAL = 30

es = Elasticsearch(ES_HOST)

def ensure_error_index():
    if not es.indices.exists(index=ERROR_INDEX):
        mapping = {
            "mappings": {
                "properties": {
                    "icao24": {"type": "keyword"},
                    "callsign": {"type": "text"},
                    "destination": {"type": "keyword"},
                    "prediction_method": {"type": "keyword"},
                    "eta_method": {"type": "keyword"},
                    "predicted_eta_sec": {"type": "integer"},
                    "actual_duration_sec": {"type": "integer"},
                    "error_sec": {"type": "integer"},
                    "error_pct": {"type": "float"},
                    "recorded_at": {"type": "date"},
                }
            }
        }
        es.indices.create(index=ERROR_INDEX, body=mapping)
        print(f"[TRACKER] Index '{ERROR_INDEX}' created.")

def get_recent_landings(cursor):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"on_ground": True}},
                    {"range": {"timestamp": {"gt": cursor}}}
                ]
            }
        },
        "sort": [{"timestamp": {"order": "asc"}}],
        "size": 100
    }
    try:
        res = es.search(index=FLIGHT_INDEX, body=query)
        return [hit["_source"] for hit in res["hits"]["hits"]]
    except Exception as e:
        print(f"[TRACKER] Query error: {e}")
        return []

def get_prediction(icao24):
    try:
        res = es.get(index=PRED_INDEX, id=icao24)
        return res["_source"]
    except NotFoundError:
        return None
    except Exception:
        return None

def save_error(doc):
    try:
        es.index(index=ERROR_INDEX, body=doc)
        print(f"[TRACKER] Saved error for {doc.get('callsign', doc['icao24'])}: {doc['error_sec']}s error")
    except Exception as e:
        print(f"[TRACKER] Save error failed: {e}")

def main():
    print("[TRACKER] Starting ETA Performance Tracker...")
    try:
        es.info()
        print(f"[TRACKER] Connected to ES at {ES_HOST}")
    except Exception as e:
        print(f"[TRACKER] ES connection failed: {e}")
        exit(1)

    ensure_error_index()
    cursor = int(time.time()) - 3600
    last_report = time.time()
    processed = 0

    while True:
        try:
            landings = get_recent_landings(cursor)
            for flight in landings:
                icao24 = flight.get("icao24")
                if not icao24:
                    continue

                landing_ts = flight.get("timestamp")
                if landing_ts and landing_ts > cursor:
                    cursor = landing_ts

                pred = get_prediction(icao24)
                if not pred or pred.get("status") != "ok":
                    continue

                predicted_eta = pred.get("eta_seconds")
                if not predicted_eta:
                    continue

                predicted_at = pred.get("predicted_at")
                if predicted_at:
                    try:
                        cleaned = predicted_at.rstrip("Z")
                        dt_pred = datetime.fromisoformat(cleaned)
                        predicted_at_ts = dt_pred.timestamp()
                        actual_duration = landing_ts - predicted_at_ts
                        if actual_duration <= 0:
                            continue
                        error_sec = abs(actual_duration - predicted_eta)
                        error_pct = round((error_sec / predicted_eta) * 100, 2) if predicted_eta > 0 else 0

                        doc = {
                            "icao24": icao24,
                            "callsign": pred.get("callsign"),
                            "destination": pred.get("destination"),
                            "prediction_method": pred.get("prediction_method"),
                            "eta_method": pred.get("eta_method"),
                            "predicted_eta_sec": predicted_eta,
                            "actual_duration_sec": round(actual_duration),
                            "error_sec": round(error_sec),
                            "error_pct": error_pct,
                            "recorded_at": datetime.utcnow().isoformat() + "Z",
                        }
                        save_error(doc)
                        processed += 1
                    except (ValueError, TypeError):
                        continue

            if time.time() - last_report > 60:
                print(f"[TRACKER] Processed so far: {processed} landings tracked")
                last_report = time.time()

        except Exception as e:
            print(f"[TRACKER] Error: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()

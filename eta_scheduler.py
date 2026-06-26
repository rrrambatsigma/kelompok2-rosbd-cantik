import time
import json
import os
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from elasticsearch import Elasticsearch, NotFoundError
from eta_pipeline import predict, get_trajectory

# ES_HOST = os.getenv("ES_HOST", "http://100.99.130.69:9200")
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = "flights"
PRED_INDEX = "flight_predictions"
POLL_INTERVAL = 30
MIN_TRACK_POINTS = 1
MAX_FLIGHTS = 200
WORKERS = 4

es = Elasticsearch(ES_HOST)

def ensure_index():
    if not es.indices.exists(index=PRED_INDEX):
        mapping = {
            "mappings": {
                "properties": {
                    "icao24": {"type": "keyword"},
                    "callsign": {"type": "text"},
                    "destination": {"type": "keyword"},
                    "prediction_method": {"type": "keyword"},
                    "confidence": {"type": "float"},
                    "eta_seconds": {"type": "integer"},
                    "eta_minutes": {"type": "float"},
                    "eta_method": {"type": "keyword"},
                    "distance_km_to_dest": {"type": "float"},
                    "track_points": {"type": "integer"},
                    "status": {"type": "keyword"},
                    "current_position": {"type": "object"},
                    "predicted_at": {"type": "date"},
                    "last_contact": {"type": "long"},
                    "on_ground": {"type": "boolean"},
                }
            }
        }
        es.indices.create(index=PRED_INDEX, body=mapping)
        print(f"Index '{PRED_INDEX}' created.")

def get_active_flights(last_check):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"on_ground": False}},
                    {"range": {"last_contact": {"gte": last_check}}}
                ]
            }
        },
        "aggs": {
            "flights": {
                "terms": {
                    "field": "icao24.keyword",
                    "size": MAX_FLIGHTS,
                    "order": {"max_last_contact": "desc"}
                },
                "aggs": {
                    "max_last_contact": {"max": {"field": "last_contact"}},
                    "point_count": {"value_count": {"field": "last_contact"}}
                }
            }
        },
        "size": 0
    }
    try:
        res = es.search(index=ES_INDEX, body=query)
    except Exception as e:
        print(f"ES query error: {e}")
        return []

    flights = []
    for bucket in res["aggregations"]["flights"]["buckets"]:
        flights.append({
            "icao24": bucket["key"],
            "points": bucket["point_count"]["value"],
            "last_contact": int(bucket["max_last_contact"]["value"])
        })

    return flights

def get_existing_prediction(icao24):
    try:
        res = es.get(index=PRED_INDEX, id=icao24)
        return res["_source"]
    except NotFoundError:
        return None
    except Exception:
        return None

def save_prediction(data):
    icao24 = data["icao24"]
    try:
        es.index(index=PRED_INDEX, id=icao24, body=data)
    except Exception as e:
        print(f"  [FAIL] save {icao24}: {e}")

def run_once(last_check):
    flights = get_active_flights(last_check)
    total = len(flights)

    if total == 0:
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] 0 flights, nothing new", flush=True)
        return

    def process_flight(flight):
        icao24 = flight["icao24"]
        lc = flight["last_contact"]
        if flight["points"] < MIN_TRACK_POINTS:
            return ("skipped", icao24, None)
        existing = get_existing_prediction(icao24)
        if existing and existing.get("status") in ("landed", "failed"):
            return ("skipped", icao24, None)
        try:
            result = predict(icao24)
            result["predicted_at"] = datetime.now(timezone.utc).isoformat() + "Z"
            result["last_contact"] = lc
            result["on_ground"] = False
            if result.get("status") == "no_data":
                return ("no_data", icao24, None)
            save_prediction(result)
            return ("saved", icao24, result)
        except Exception:
            return ("failed", icao24, None)

    saved = skipped = no_data = failed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_flight, f): f for f in flights}
        for future in as_completed(futures):
            status, _, _ = future.result()
            if status == "saved":
                saved += 1
            elif status == "skipped":
                skipped += 1
            elif status == "no_data":
                no_data += 1
            elif status == "failed":
                failed += 1
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}] {saved} saved, {skipped} skipped, {no_data} no_data, {failed} failed", flush=True)

def main():
    print("=" * 30)
    print("  ETA SCHEDULER")
    print("=" * 30)

    ensure_index()
    last_check = int(time.time()) - 120

    while True:
        try:
            run_once(last_check)
            last_check = int(time.time())
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"  [ERROR] {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()

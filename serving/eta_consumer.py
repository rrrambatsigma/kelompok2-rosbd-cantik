import os
import time
import json
import sys
from datetime import datetime, timezone
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from elasticsearch import Elasticsearch, NotFoundError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eta_pipeline import predict

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = "flights"
GROUP_ID = "eta-consumer-group"
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch:9200")
PRED_INDEX = "flight_predictions"
COOLDOWN_SECONDS = 30

es = Elasticsearch(f"http://{ES_HOST}")

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
        print(f"[ETA] Index '{PRED_INDEX}' created.")

def connect_kafka():
    print(f"[ETA] Connecting to Kafka at {KAFKA_BOOTSTRAP}...")
    while True:
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=GROUP_ID,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
            )
            print("[ETA] Connected to Kafka")
            return consumer
        except NoBrokersAvailable:
            print("[ETA] Broker not ready, retry 5s")
            time.sleep(5)
        except Exception as e:
            print(f"[ETA] Error: {e}, retry 5s")
            time.sleep(5)

def save_prediction(data):
    icao24 = data["icao24"]
    try:
        es.index(index=PRED_INDEX, id=icao24, body=data)
    except Exception as e:
        print(f"[ETA] Save error {icao24}: {e}")

def main():
    try:
        es.info()
        print("[ETA] Elasticsearch connected")
    except Exception as e:
        print(f"[ETA] ES connection failed: {e}")
        exit(1)

    ensure_index()
    consumer = connect_kafka()
    print(f"[ETA] Listening on topic '{TOPIC}'")

    last_predicted = {}
    stats = {"total": 0, "predicted": 0, "skipped": 0, "errors": 0}
    last_report = time.time()

    for msg in consumer:
        flight = msg.value
        stats["total"] += 1

        icao24 = flight.get("icao24")
        if not icao24:
            stats["skipped"] += 1
            continue

        now = time.time()
        if icao24 in last_predicted and now - last_predicted[icao24] < COOLDOWN_SECONDS:
            stats["skipped"] += 1
            continue

        try:
            result = predict(icao24)
            result["predicted_at"] = datetime.now(timezone.utc).isoformat() + "Z"
            result["last_contact"] = flight.get("last_contact") or flight.get("timestamp")
            result["on_ground"] = flight.get("on_ground", False)

            if result.get("status") != "no_data":
                save_prediction(result)
                stats["predicted"] += 1
            else:
                stats["skipped"] += 1

            last_predicted[icao24] = now
        except Exception as e:
            print(f"[ETA] Predict error {icao24}: {e}")
            stats["errors"] += 1

        if time.time() - last_report > 30:
            print(f"[ETA] Stats: {stats['total']} msgs, {stats['predicted']} predicted, "
                  f"{stats['skipped']} skipped, {stats['errors']} errors")
            last_report = time.time()

if __name__ == "__main__":
    main()

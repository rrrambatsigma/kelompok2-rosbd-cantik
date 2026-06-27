import os
import json
import time
import requests
from datetime import datetime
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
from kafka.errors import NoBrokersAvailable

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "100.99.130.69:9093")
TOPIC = "flights"
GROUP_ID = "anomaly-detector-meiva"
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost:9200")
SERVING_URL = os.getenv("SERVING_URL", "http://localhost:8001")

es = Elasticsearch(f"http://{ES_HOST}")


def unwrap(val, default=None):
    if isinstance(val, list) and len(val) > 0:
        return val[0]
    return val if val is not None else default


def connect_kafka():
    print(f"[DETECTOR-MEIVA] Connecting to Kafka at {KAFKA_BOOTSTRAP}...")
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
            print("[DETECTOR-MEIVA] Connected to Kafka Meiva")
            return consumer
        except NoBrokersAvailable:
            print("[DETECTOR-MEIVA] Broker not ready, retry 5s")
            time.sleep(5)
        except Exception as e:
            print(f"[DETECTOR-MEIVA] Error: {e}, retry 5s")
            time.sleep(5)


def predict_anomaly(flight: dict) -> dict:
    try:
        resp = requests.post(
            f"{SERVING_URL}/predict/stream",
            json={
                "icao24": unwrap(flight.get("icao24")),
                "callsign": unwrap(flight.get("callsign"), ""),
                "latitude": unwrap(flight.get("latitude"), 0),
                "longitude": unwrap(flight.get("longitude"), 0),
                "velocity": unwrap(flight.get("velocity"), 0),
                "baro_altitude": unwrap(flight.get("baro_altitude"), 0),
                "true_track": unwrap(flight.get("true_track"), 0),
            },
            timeout=2,
        )
        if resp.status_code == 200:
            return resp.json()
        print(f"[DETECTOR-MEIVA] API error: {resp.status_code}")
        return None
    except requests.ConnectionError:
        return None
    except Exception as e:
        print(f"[DETECTOR-MEIVA] Prediction error: {e}")
        return None


def save_result(flight: dict, result: dict):
    if result is None:
        return

    doc = {
        "icao24": unwrap(flight.get("icao24")),
        "callsign": unwrap(flight.get("callsign")),
        "latitude": unwrap(flight.get("latitude")),
        "longitude": unwrap(flight.get("longitude")),
        "velocity": unwrap(flight.get("velocity")),
        "velocity_kmh": unwrap(flight.get("velocity_kmh")),
        "baro_altitude": unwrap(flight.get("baro_altitude")),
        "geo_altitude": unwrap(flight.get("geo_altitude")),
        "true_track": unwrap(flight.get("true_track")),
        "vertical_rate": unwrap(flight.get("vertical_rate")),
        "region": unwrap(flight.get("region")),
        "on_ground": unwrap(flight.get("on_ground")),
        "origin_country": unwrap(flight.get("origin_country")),
        "timestamp": unwrap(flight.get("timestamp"), time.time()),
        "is_anomaly": result.get("is_anomaly", False),
        "recon_error": result.get("recon_error", 0),
        "svdd_distance": result.get("svdd_distance", 0),
        "combined_score": result.get("combined_score", 0),
        "attack_type": result.get("attack_type", "normal"),
        "anomaly_type_detected": result.get("anomaly_type_detected"),
        "dominant_feature": result.get("dominant_feature", ""),
        "source": "meiva-kafka",
        "processed_at": datetime.utcnow().isoformat() + "Z",
    }

    try:
        es.index(index="anomaly-stream", document=doc)
    except Exception as e:
        print(f"[DETECTOR-MEIVA] ES save error: {e}")


def main():
    print(f"[DETECTOR-MEIVA] VAE-SVDD Anomaly Detector for Meiva Kafka")
    print(f"[DETECTOR-MEIVA] Kafka: {KAFKA_BOOTSTRAP}, Topic: {TOPIC}")
    print(f"[DETECTOR-MEIVA] Serving API: {SERVING_URL}")
    print(f"[DETECTOR-MEIVA] Elasticsearch: {ES_HOST}")

    consumer = connect_kafka()
    print(f"[DETECTOR-MEIVA] Listening on topic '{TOPIC}'")

    stats = {"total": 0, "anomalies": 0, "api_errors": 0, "kafka_empty": 0}
    last_report = time.time()

    for msg in consumer:
        flight = msg.value
        stats["total"] += 1

        result = predict_anomaly(flight)
        if result:
            if result.get("is_anomaly"):
                stats["anomalies"] += 1
                print(f"[DETECTOR-MEIVA] ANOMALY: {unwrap(flight.get('icao24'))} - {result.get('attack_type')}")
            save_result(flight, result)
        else:
            stats["api_errors"] += 1

        if time.time() - last_report > 30:
            rate = stats["total"] / 30
            print(f"[DETECTOR-MEIVA] Stats: {stats['total']} processed "
                  f"({rate:.1f}/s), {stats['anomalies']} anomalies, "
                  f"{stats['api_errors']} API errors")
            last_report = time.time()


if __name__ == "__main__":
    main()

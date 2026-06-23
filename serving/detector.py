import os
import json
import time
import requests
from datetime import datetime
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
from kafka.errors import NoBrokersAvailable

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = "flights"
GROUP_ID = "anomaly-detector-group"
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch:9200")
SERVING_URL = os.getenv("SERVING_URL", "http://vae-serving:8001")

es = Elasticsearch(f"http://{ES_HOST}")


def connect_kafka():
    print(f"[DETECTOR] Connecting to Kafka at {KAFKA_BOOTSTRAP}...")
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
            print("[DETECTOR] Connected to Kafka")
            return consumer
        except NoBrokersAvailable:
            print("[DETECTOR] Broker not ready, retry 5s")
            time.sleep(5)
        except Exception as e:
            print(f"[DETECTOR] Error: {e}, retry 5s")
            time.sleep(5)


def predict_anomaly(flight: dict) -> dict:
    try:
        resp = requests.post(
            f"{SERVING_URL}/predict/stream",
            json={
                "icao24": flight.get("icao24"),
                "callsign": flight.get("callsign"),
                "longitude": flight.get("longitude", 0),
                "latitude": flight.get("latitude", 0),
                "velocity": flight.get("velocity", 0),
                "geo_altitude": flight.get("geo_altitude", 0),
                "true_track": flight.get("true_track", 0),
                "vertical_rate": flight.get("vertical_rate", 0),
            },
            timeout=2,
        )
        if resp.status_code == 200:
            return resp.json()
        print(f"[DETECTOR] API error: {resp.status_code}")
        return None
    except requests.ConnectionError:
        return None
    except Exception as e:
        print(f"[DETECTOR] Prediction error: {e}")
        return None


def save_result(flight: dict, result: dict):
    if result is None:
        return

    doc = {
        "icao24": flight.get("icao24"),
        "callsign": flight.get("callsign"),
        "longitude": flight.get("longitude"),
        "latitude": flight.get("latitude"),
        "velocity": flight.get("velocity"),
        "geo_altitude": flight.get("geo_altitude"),
        "true_track": flight.get("true_track"),
        "vertical_rate": flight.get("vertical_rate"),
        "timestamp": flight.get("timestamp", time.time()),
        "is_anomaly": result.get("is_anomaly", False),
        "recon_error": result.get("recon_error", 0),
        "svdd_distance": result.get("svdd_distance", 0),
        "combined_score": result.get("combined_score", 0),
        "attack_type": result.get("attack_type", "normal"),
        "anomaly_type_detected": result.get("anomaly_type_detected"),
        "dominant_feature": result.get("dominant_feature", ""),
        "processed_at": datetime.utcnow().isoformat() + "Z",
    }

    try:
        es.index(index="anomaly-stream", document=doc)
    except Exception as e:
        print(f"[DETECTOR] ES save error: {e}")


def main():
    print(f"[DETECTOR] VAE-SVDD Anomaly Detector starting...")
    print(f"[DETECTOR] Serving API: {SERVING_URL}")

    consumer = connect_kafka()
    print(f"[DETECTOR] Listening on topic '{TOPIC}'")

    stats = {"total": 0, "anomalies": 0, "api_errors": 0}
    last_report = time.time()

    for msg in consumer:
        flight = msg.value
        stats["total"] += 1

        result = predict_anomaly(flight)
        if result:
            if result.get("is_anomaly"):
                stats["anomalies"] += 1
                print(f"[DETECTOR] Anomaly: {flight.get('icao24')} - {result.get('attack_type')}")
            save_result(flight, result)
        else:
            stats["api_errors"] += 1

        if time.time() - last_report > 30:
            print(f"[DETECTOR] Stats: {stats['total']} processed, "
                  f"{stats['anomalies']} anomalies, {stats['api_errors']} API errors")
            last_report = time.time()


if __name__ == "__main__":
    main()

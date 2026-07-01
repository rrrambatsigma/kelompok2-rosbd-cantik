"""
meiva_detector_if.py — Standalone IF detector
Kafka Meiva → Isolation Forest → ES lokal (anomaly-stream-if)

Berjalan paralel dengan VAE-LSTM detector.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import joblib
from datetime import datetime, timezone
from collections import deque, defaultdict
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from elasticsearch import Elasticsearch

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.telegram_notifier import notify_startup, notify_performance as tg_notify_performance, notify_anomaly as tg_notify_anomaly

KAFKA_HOST = "100.99.130.69:9093"
TOPIC = "flights"
GROUP_ID = "rambat-detector-if"

MODEL_DIR = os.path.join(BASE_DIR, "models", "isolation-forest")
ES_HOST = "http://localhost:9200"
ANOMALY_INDEX = "anomaly-stream-if"

WINDOW_SIZE = 10
FEATURES_BASE = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]
REPORT_INTERVAL = 30
THROTTLE_SECONDS = 3

buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
last_infer_time = {}
stats = {"total": 0, "anomalies": 0, "buffer_ready": 0, "es_errors": 0, "reconnects": 0,
         "attack_counts": defaultdict(int)}
last_report = time.time()
errors_history = deque(maxlen=1000)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def load_model():
    log(f"[LOAD] Loading model from {MODEL_DIR}...")
    if_path = os.path.join(MODEL_DIR, "if_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    config_path = os.path.join(MODEL_DIR, "config.json")

    if not os.path.exists(if_path):
        log("[ERROR] Model IF tidak ditemukan. Jalankan training dulu.")
        sys.exit(1)

    if_model = joblib.load(if_path)
    scaler = joblib.load(scaler_path)

    with open(config_path) as f:
        config = json.load(f)
    threshold = config.get("f1max_threshold", -0.0949)

    log(f"[LOAD] IF: {if_model.n_estimators} trees, contamination={if_model.contamination}")
    log(f"[LOAD] Scaler: {scaler.mean_.shape[0]} features")
    log(f"[LOAD] Threshold: {threshold:.4f} (F1-max)")

    return if_model, scaler, threshold


def connect_kafka():
    log(f"[KAFKA] Connecting to {KAFKA_HOST}...")
    while True:
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=KAFKA_HOST,
                group_id=GROUP_ID,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
            )
            log(f"[KAFKA] Connected to {KAFKA_HOST}, topic '{TOPIC}'")
            log(f"[KAFKA] Group ID: {GROUP_ID}")
            return consumer
        except NoBrokersAvailable:
            log("[KAFKA] Broker not ready, retry 5s...")
            time.sleep(5)
        except Exception as e:
            log(f"[KAFKA] Error: {e}, retry 5s...")
            time.sleep(5)


def connect_es():
    try:
        es = Elasticsearch(ES_HOST, request_timeout=5)
        if es.ping():
            log(f"[ES] Connected to {ES_HOST}")
            return es
        log(f"[ES] Cannot reach {ES_HOST}")
        return None
    except Exception as e:
        log(f"[ES] Connection failed: {e}")
        return None


def compute_derived_features(window_5f):
    deltas = np.diff(window_5f, axis=0, prepend=window_5f[0:1])
    return np.concatenate([window_5f, deltas], axis=1)


def classify_attack(per_feature_error):
    fe = per_feature_error[:5]
    total = fe.sum() + 1e-10
    lat_lon_error = fe[0] + fe[1]
    track_ratio = fe[4] / total
    vel_ratio = fe[2] / total
    lat_lon_ratio = lat_lon_error / total

    if track_ratio > 0.5:
        return "heading_manipulation"
    elif vel_ratio > 0.5:
        return "velocity_drift"
    elif lat_lon_ratio > 0.5 and (fe[0] / total) < 0.4:
        return "random_position"
    elif lat_lon_ratio > 0.5:
        return "constant_position"
    elif total > 2.0:
        return "flight_merge"
    else:
        return "dos_deletion"


def infer_window(if_model, scaler, threshold, window_5f):
    window_10f = compute_derived_features(window_5f)  # (10, 10)
    flat = window_10f.reshape(1, -1)                  # (1, 100) flatten dulu
    scaled = scaler.transform(flat)                   # (1, 100) baru scale
    score = -float(if_model.decision_function(scaled)[0])

    is_anomaly = score > threshold

    per_feature = np.mean((window_10f[:, :5]) ** 2, axis=0)
    attack_type = classify_attack(per_feature) if is_anomaly else "normal"

    return {
        "is_anomaly": is_anomaly,
        "score": round(score, 6),
        "attack_type": attack_type,
    }


def predict_and_save(icao24, flight_data, if_model, scaler, threshold, es):
    buf = buffers[icao24]
    if len(buf) < WINDOW_SIZE:
        return

    window_5f = np.array(buf, dtype=np.float32)
    result = infer_window(if_model, scaler, threshold, window_5f)

    stats["buffer_ready"] += 1
    if result["is_anomaly"]:
        stats["anomalies"] += 1
        stats["attack_counts"][result["attack_type"]] += 1
        tg_notify_anomaly({
            "icao24": icao24,
            "callsign": flight_data.get("callsign", "?"),
            "attack_type": result["attack_type"],
            "score": result["score"],
            "latitude": flight_data.get("latitude"),
            "longitude": flight_data.get("longitude"),
            "velocity": flight_data.get("velocity"),
            "baro_altitude": flight_data.get("baro_altitude"),
        }, model_name="Isolation Forest", threshold=-0.0949)
    errors_history.append(result["score"])

    callsign = flight_data.get("callsign", "?") or "?"
    if result["is_anomaly"]:
        log(f"[ANOMALI] {icao24} | {callsign} | score={result['score']:.4f} | {result['attack_type']}")
    elif stats["buffer_ready"] % 50 == 0:
        log(f"[NORMAL]  {stats['buffer_ready']} windows processed...")

    doc = {
        "icao24": icao24, "callsign": callsign,
        "latitude": flight_data.get("latitude"), "longitude": flight_data.get("longitude"),
        "velocity": flight_data.get("velocity"), "baro_altitude": flight_data.get("baro_altitude"),
        "true_track": flight_data.get("true_track"),
        "is_anomaly": result["is_anomaly"], "if_score": result["score"],
        "attack_type": result["attack_type"],
        "anomaly_type_detected": result["attack_type"] if result["is_anomaly"] else None,
        "timestamp": flight_data.get("timestamp", time.time()),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source": "meiva_kafka", "detector": "rambat-if", "window_size": WINDOW_SIZE,
    }

    if es is not None:
        try:
            es.options(request_timeout=2).index(index=ANOMALY_INDEX, document=doc)
        except Exception as e:
            stats["es_errors"] += 1
            if stats["es_errors"] <= 3:
                log(f"[ES] Save error: {e}")


def print_stats(current_threshold):
    elapsed = time.time() - last_report
    rate = stats["total"] / elapsed if elapsed > 0 else 0
    anom_pct = (stats["anomalies"] / max(stats["buffer_ready"], 1)) * 100
    log(f"[STATS] {stats['total']} received | "
        f"{stats['buffer_ready']} windows | "
        f"{stats['anomalies']} anomalies ({anom_pct:.1f}%) | "
        f"{rate:.1f} msg/s | ES errors: {stats['es_errors']}")

    p50 = p90 = p95 = max_err = None
    if len(errors_history) >= 10:
        err_arr = list(errors_history)
        p50 = float(np.percentile(err_arr, 50))
        p90 = float(np.percentile(err_arr, 90))
        p95 = float(np.percentile(err_arr, 95))
        max_err = float(max(err_arr))
        log(f"[DIST] score: p50={p50:.2f} p90={p90:.2f} p95={p95:.2f} | threshold={current_threshold:.4f}")

    uptime = time.time() - START_TIME
    tg_notify_performance(dict(stats), "Isolation Forest", p50, p90, p95, max_err, uptime)


def main():
    global last_report, START_TIME
    START_TIME = time.time()

    log("=" * 55)
    log("MEIVA DETECTOR IF — Isolation Forest")
    log("=" * 55)

    log("\n[1/3] Loading model...")
    if_model, scaler, threshold = load_model()
    notify_startup("Isolation Forest")

    log("\n[2/3] Connecting to services...")
    es = connect_es()
    consumer = connect_kafka()

    log("\n[3/3] Ready! Waiting for data from Kafka Meiva...")
    log(f"  Buffer: {WINDOW_SIZE} timesteps per flight")
    log(f"  Threshold: {threshold:.4f}")
    log(f"  Report interval: {REPORT_INTERVAL}s")
    log(f"  (Menunggu data - jangan ditutup)")
    log("=" * 55)

    last_report = time.time()

    while True:
        try:
            for msg in consumer:
                try:
                    flight = msg.value
                    stats["total"] += 1

                    lat = flight.get("latitude")
                    lon = flight.get("longitude")
                    if lat is None or lon is None:
                        continue

                    raw = np.array([
                        lat, lon,
                        flight.get("velocity", 0) or 0,
                        flight.get("baro_altitude", 0) or 0,
                        flight.get("true_track", 0) or 0,
                    ], dtype=np.float32)

                    icao24 = flight.get("icao24", "unknown")

                    now = time.time()
                    last_infer = last_infer_time.get(icao24, 0)
                    if now - last_infer < THROTTLE_SECONDS:
                        continue

                    buf = buffers[icao24]
                    buf.append(raw)

                    if len(buf) >= WINDOW_SIZE:
                        predict_and_save(icao24, flight, if_model, scaler, threshold, es)
                        last_infer_time[icao24] = time.time()

                    if time.time() - last_report > REPORT_INTERVAL:
                        print_stats(threshold)
                        last_report = time.time()

                except Exception as e:
                    log(f"[ERROR] {e}")
                    continue

        except Exception as e:
            stats["reconnects"] += 1
            log(f"[KAFKA] Connection lost: {e}, reconnecting in 5s... "
                f"(reconnect #{stats['reconnects']})")
            time.sleep(5)
            try:
                consumer = connect_kafka()
            except:
                continue


if __name__ == "__main__":
    main()

"""
meiva_detector.py
Standalone anomaly detector — Kafka Meiva → VAE-LSTM v3 → ES lokal

Alur:
  1. Load model VAE-LSTM v3 (tanpa API server)
  2. Connect Kafka Meiva (100.99.130.69:9093)
  3. Buffer 10 data per pesawat
  4. Setiap buffer penuh → infer VAE → hitung recon_error
  5. Flag anomali jika recon_error > threshold F1-max
  6. Simpan ke local ES → langsung muncul di Grafana
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import torch
import joblib
from datetime import datetime, timezone
from collections import deque, defaultdict
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from elasticsearch import Elasticsearch

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modelling.anomaly.vae_lstm import VAELSTM

# ─── KONFIGURASI ────────────────────────────────────────────
KAFKA_HOST = "100.99.130.69:9093"
TOPIC = "flights"
GROUP_ID = "rambat-detector"

MODEL_DIR = os.path.join(BASE_DIR, "models", "vae-svdd-trained")
ES_HOST = "http://localhost:9200"
ANOMALY_INDEX = "anomaly-stream"

WINDOW_SIZE = 10
FEATURES_BASE = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]
REPORT_INTERVAL = 30

# ─── STATE ──────────────────────────────────────────────────
buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
last_infer_time = {}
stats = {"total": 0, "anomalies": 0, "buffer_ready": 0, "es_errors": 0, "reconnects": 0}
last_report = time.time()
THROTTLE_SECONDS = 3
errors_history = deque(maxlen=1000)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def load_model():
    """Load VAE-LSTM + scaler + threshold dari model directory."""
    vae_path = os.path.join(MODEL_DIR, "vae_model.pt")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    config_path = os.path.join(MODEL_DIR, "config.json")

    if not os.path.exists(vae_path):
        log(f"[ERROR] Model tidak ditemukan di {vae_path}")
        log("[ERROR] Jalankan training dulu: python -m modelling.anomaly.train")
        sys.exit(1)

    # Load VAE
    checkpoint = torch.load(vae_path, map_location="cpu")
    vae = VAELSTM(
        n_features=checkpoint.get("input_dim", 10),
        window_size=checkpoint.get("window_size", 10),
        hidden_dim=checkpoint.get("hidden_dim", 64),
        latent_dim=checkpoint.get("latent_dim", 4),
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    log(f"[LOAD] VAE-LSTM: {checkpoint.get('input_dim')} features, "
        f"latent={checkpoint.get('latent_dim')}")

    # Load scaler
    scaler = joblib.load(scaler_path)
    log(f"[LOAD] Scaler loaded")

    # Load config + threshold
    with open(config_path) as f:
        config = json.load(f)
    # Threshold = P95 dari distribusi error real (berdasarkan [DIST] log)
    threshold = 2.0
    log(f"[LOAD] Threshold: {threshold:.1f} (manual)")

    return vae, scaler, threshold


def connect_kafka():
    """Connect ke Kafka Meiva."""
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
    """Connect ke local Elasticsearch."""
    try:
        es = Elasticsearch(ES_HOST, request_timeout=5)
        if es.ping():
            log(f"[ES] Connected to {ES_HOST}")
            return es
        log(f"[ES] Cannot reach {ES_HOST}, results only print to console")
        return None
    except Exception as e:
        log(f"[ES] Connection failed: {e}")
        return None


def compute_derived_features(window_5f):
    """
    Hitung derived features (delta antar-timestep).
    window_5f: (10, 5) → return: (10, 10)
    """
    deltas = np.diff(window_5f, axis=0, prepend=window_5f[0:1])
    return np.concatenate([window_5f, deltas], axis=1)


def classify_attack(per_feature_error):
    """
    Tentukan jenis anomali dari per-feature error.
    per_feature_error: array (10,) — error per feature (5 base + 5 derived)
    """
    # Pakai 5 base features untuk klasifikasi
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


def infer_window(vae, scaler, threshold, window_5f):
    """
    Infer 1 window (10 timesteps) → return result dict.
    window_5f: numpy array (10, 5)
    """
    # Hitung derived features
    window_10f = compute_derived_features(window_5f)

    # Scale — transform per timestep (10,10), bukan flatten
    scaled = scaler.transform(window_10f)  # (10, 10) → scale per timestep
    scaled = scaled.reshape(1, WINDOW_SIZE, 10)  # (1, 10, 10) → batch dim

    # VAE inference
    with torch.no_grad():
        recon, _, _, _ = vae(torch.FloatTensor(scaled))

    recon_np = recon.numpy()

    # Reconstruction error (semua 10 fitur)
    recon_error = float(np.mean((scaled - recon_np) ** 2))

    # Per-feature error untuk klasifikasi (5 base features)
    per_feature = np.mean((scaled[:, :, :5] - recon_np[:, :, :5]) ** 2, axis=1)[0]

    # Deteksi
    is_anomaly = recon_error > threshold

    # Klasifikasi attack type
    attack_type = classify_attack(per_feature) if is_anomaly else "normal"

    return {
        "is_anomaly": is_anomaly,
        "recon_error": round(recon_error, 6),
        "attack_type": attack_type,
    }


def predict_and_save(icao24, flight_data, vae, scaler, threshold, es):
    """
    Ambil buffer → infer → simpan ke ES.
    flight_data: record terakhir dari Kafka (untuk metadata)
    """
    buf = buffers[icao24]
    if len(buf) < WINDOW_SIZE:
        return

    window_5f = np.array(buf, dtype=np.float32)
    result = infer_window(vae, scaler, threshold, window_5f)

    stats["buffer_ready"] += 1
    if result["is_anomaly"]:
        stats["anomalies"] += 1
    errors_history.append(result["recon_error"])

    # Log ke console
    callsign = flight_data.get("callsign", "?") or "?"
    status = "⚠️ ANOMALI" if result["is_anomaly"] else "✓ normal"
    if result["is_anomaly"]:
        log(f"[ANOMALI] {icao24} | {callsign} | "
            f"error={result['recon_error']:.4f} | {result['attack_type']}")
    elif stats["buffer_ready"] % 50 == 0:
        log(f"[NORMAL]  {stats['buffer_ready']} windows processed...")

    # Simpan ke ES
    doc = {
        "icao24": icao24,
        "callsign": callsign,
        "latitude": flight_data.get("latitude"),
        "longitude": flight_data.get("longitude"),
        "velocity": flight_data.get("velocity"),
        "baro_altitude": flight_data.get("baro_altitude"),
        "true_track": flight_data.get("true_track"),
        "is_anomaly": result["is_anomaly"],
        "recon_error": result["recon_error"],
        "attack_type": result["attack_type"],
        "anomaly_type_detected": result["attack_type"] if result["is_anomaly"] else None,
        "timestamp": flight_data.get("timestamp", time.time()),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source": "meiva_kafka",
        "detector": "rambat",
        "window_size": WINDOW_SIZE,
    }

    if es is not None:
        try:
            es.options(request_timeout=2).index(index=ANOMALY_INDEX, document=doc)
        except Exception as e:
            stats["es_errors"] += 1
            if stats["es_errors"] <= 3:
                log(f"[ES] Save error: {e}")


def print_stats(current_threshold):
    """Cetak statistik periodik + distribusi error."""
    elapsed = time.time() - last_report
    rate = stats["total"] / elapsed if elapsed > 0 else 0
    anom_pct = (stats["anomalies"] / max(stats["buffer_ready"], 1)) * 100
    log(f"[STATS] {stats['total']} received | "
        f"{stats['buffer_ready']} windows | "
        f"{stats['anomalies']} anomalies ({anom_pct:.1f}%) | "
        f"{rate:.1f} msg/s | "
        f"ES errors: {stats['es_errors']}")
    if len(errors_history) >= 10:
        err_arr = list(errors_history)
        p50 = float(np.percentile(err_arr, 50))
        p90 = float(np.percentile(err_arr, 90))
        p95 = float(np.percentile(err_arr, 95))
        log(f"[DIST] error: p50={p50:.2f} p90={p90:.2f} p95={p95:.2f} | "
            f"threshold={current_threshold:.1f}")


def main():
    global last_report

    log("=" * 55)
    log("MEIVA DETECTOR — VAE-LSTM v3")
    log("=" * 55)

    # Load model
    log("\n[1/3] Loading model...")
    vae, scaler, threshold = load_model()

    # Connect ES (opsional, fallback ke console-only)
    log("\n[2/3] Connecting to services...")
    es = connect_es()

    # Connect Kafka
    consumer = connect_kafka()
    log("\n[3/3] Ready! Waiting for data...")
    log(f"  Buffer: {WINDOW_SIZE} timesteps per flight")
    log(f"  Threshold: {threshold:.4f}")
    log(f"  Report interval: {REPORT_INTERVAL}s")
    log(f"  (Menunggu data dari Kafka Meiva - jangan ditutup)")
    log("=" * 55)

    last_report = time.time()

    # ── Main Loop (auto-reconnect) ──
    while True:
        try:
            for msg in consumer:
                try:
                    flight = msg.value
                    stats["total"] += 1

                    # Validasi: butuh latitude & longitude
                    lat = flight.get("latitude")
                    lon = flight.get("longitude")
                    if lat is None or lon is None:
                        continue

                    # Ambil 5 fitur base
                    raw = np.array([
                        lat,
                        lon,
                        flight.get("velocity", 0) or 0,
                        flight.get("baro_altitude", 0) or 0,
                        flight.get("true_track", 0) or 0,
                    ], dtype=np.float32)

                    icao24 = flight.get("icao24", "unknown")

                    # Throttle: minimal 3 detik antar infer per flight
                    now = time.time()
                    last_infer = last_infer_time.get(icao24, 0)
                    if now - last_infer < THROTTLE_SECONDS:
                        continue

                    # Buffer per pesawat
                    buf = buffers[icao24]
                    buf.append(raw)

                    # Infer jika buffer sudah penuh
                    if len(buf) >= WINDOW_SIZE:
                        predict_and_save(icao24, flight, vae, scaler, threshold, es)
                        last_infer_time[icao24] = time.time()

                    # Report periodik
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

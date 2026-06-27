"""
batch_es.py — Batch processing data dari ES Meiva (v3)
VAE-LSTM only (tanpa SVDD). Threshold F1-max dari training.

Alur:
  1. Fetch semua data dari ES Meiva via scroll API
  2. Cleaning & segmentasi flight
  3. Sliding window (size=10, stride=5)
  4. Load VAE-LSTM v3 + Scaler
  5. Infer semua window → recon_error > threshold
  6. Simpan hasil ke ES anomaly-stream
  7. Buka Grafana → data langsung muncul
"""
import os
import sys
import json
import time
import numpy as np
import torch
import joblib
from datetime import datetime
from collections import defaultdict
from elasticsearch import Elasticsearch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

REMOTE_ES = os.getenv("REMOTE_ES", "http://100.99.130.69:9200")
REMOTE_INDEX = os.getenv("REMOTE_INDEX", "flights")
LOCAL_ES = os.getenv("LOCAL_ES", "http://localhost:9200")
ANOMALY_INDEX = "anomaly-stream"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "vae-svdd-trained")

WINDOW_SIZE = 10
STRIDE = 5
SEGMENT_GAP = 1800
MIN_FLIGHT_LEN = 10
SCROLL_SIZE = 10000
BATCH_INFER = 256

FEATURES = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]
DERIVED = ["dlat", "dlon", "dvel", "dalt", "dtrack"]
ALL_FEATURES = FEATURES + DERIVED

es_remote = None
es_local = None
stats = {"fetched": 0, "after_clean": 0, "flight_segments": 0, "windows": 0, "anomalies": 0, "errors": 0}
t_start = None


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def load_model():
    log(f"Loading model from {MODEL_DIR}...")
    config_path = os.path.join(MODEL_DIR, "config.json")
    vae_path = os.path.join(MODEL_DIR, "vae_model.pt")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    with open(config_path) as f:
        config = json.load(f)

    from modelling.anomaly.vae_lstm import VAELSTM
    checkpoint = torch.load(vae_path, map_location="cpu")
    vae = VAELSTM(
        n_features=checkpoint["input_dim"],
        window_size=checkpoint["window_size"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()

    scaler = joblib.load(scaler_path)
    threshold = config.get("f1max_threshold", config.get("global_threshold", 0.032))

    log(f"  VAE: {checkpoint['input_dim']} features, latent={checkpoint['latent_dim']}")
    log(f"  Threshold F1-max: {threshold:.4f}")

    return vae, scaler, threshold, config


def fetch_all_data():
    log(f"Fetching data from {REMOTE_ES}/{REMOTE_INDEX}...")
    all_docs = []
    result = es_remote.search(
        index=REMOTE_INDEX,
        scroll="10m",
        size=SCROLL_SIZE,
        body={
            "query": {"match_all": {}},
            "sort": [{"timestamp": {"order": "asc"}}],
            "_source": ["icao24", "callsign", "latitude", "longitude",
                        "velocity", "baro_altitude", "true_track", "timestamp", "on_ground"]
        }
    )
    scroll_id = result.get("_scroll_id")
    hits = result["hits"]["hits"]
    all_docs.extend([h["_source"] for h in hits])
    batch = 1
    while len(hits) > 0:
        result = es_remote.scroll(scroll_id=scroll_id, scroll="10m")
        scroll_id = result.get("_scroll_id")
        hits = result["hits"]["hits"]
        all_docs.extend([h["_source"] for h in hits])
        batch += 1
        if batch % 10 == 0:
            log(f"  Fetched {len(all_docs):,} records...")
    es_remote.clear_scroll(scroll_id=scroll_id)
    log(f"  TOTAL: {len(all_docs):,} records fetched")
    stats["fetched"] = len(all_docs)
    return all_docs


def clean_data(docs):
    log("Cleaning data...")
    n0 = len(docs)
    df_records = []
    for d in docs:
        try:
            if d.get("on_ground"):
                continue
            lat = d.get("latitude")
            lon = d.get("longitude")
            vel = d.get("velocity")
            alt = d.get("baro_altitude")
            track = d.get("true_track")
            ts = d.get("timestamp")
            if lat is None or lon is None:
                continue
            if vel is not None and not (0 <= vel <= 350):
                continue
            if alt is not None and not (-500 <= alt <= 20000):
                continue
            if track is not None and not (0 <= track <= 360):
                continue
            if ts is None or vel is None or alt is None or track is None:
                continue
            df_records.append({
                "icao24": d.get("icao24"),
                "callsign": d.get("callsign"),
                "timestamp": float(ts),
                "latitude": float(lat),
                "longitude": float(lon),
                "velocity": float(vel),
                "baro_altitude": float(alt),
                "true_track": float(track),
            })
        except (TypeError, ValueError):
            continue

    # Hapus duplikat (icao24 + timestamp)
    seen = set()
    unique = []
    for r in df_records:
        key = (r["icao24"], r["timestamp"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    df_records = unique
    stats["after_clean"] = len(df_records)
    log(f"  Cleaned: {len(df_records):,} / {n0:,} records ({100*len(df_records)/max(n0,1):.1f}%)")
    return df_records


def segment_flights(records):
    log("Segmentasi flight (gap > 30 menit)...")
    records.sort(key=lambda r: (r["icao24"], r["timestamp"]))

    flights = []
    current = []

    for r in records:
        if not current:
            current = [r]
        elif r["icao24"] == current[-1]["icao24"]:
            if r["timestamp"] - current[-1]["timestamp"] > SEGMENT_GAP:
                flights.append(current)
                current = [r]
            else:
                current.append(r)
        else:
            flights.append(current)
            current = [r]
    if current:
        flights.append(current)

    # Filter flight pendek
    long_flights = [f for f in flights if len(f) >= MIN_FLIGHT_LEN]
    stats["flight_segments"] = len(long_flights)
    log(f"  Segments: {len(flights):,} total, {len(long_flights):,} with >= {MIN_FLIGHT_LEN} records")
    return long_flights


def compute_derived(window_5f):
    deltas = np.diff(window_5f, axis=0, prepend=window_5f[0:1])
    return np.concatenate([window_5f, deltas], axis=1)


def classify_attack(per_feature_error):
    total = per_feature_error.sum() + 1e-10
    lat_lon = (per_feature_error[0] + per_feature_error[1]) / total
    vel_r = per_feature_error[2] / total
    track_r = per_feature_error[4] / total
    if track_r > 0.5:
        return "heading_manipulation"
    elif vel_r > 0.5:
        return "velocity_drift"
    elif lat_lon > 0.5 and (per_feature_error[0] / total) < 0.4:
        return "random_position"
    elif lat_lon > 0.5:
        return "constant_position"
    elif total > 0.8:
        return "flight_merge"
    else:
        return "dos_deletion"


def process_flights(flights, vae, scaler, threshold, config):
    log("Processing windows...")
    all_windows = []
    all_flight_ids = []
    all_timestamps = []

    for flight in flights:
        values = np.array([[r[f] for f in FEATURES] for r in flight], dtype=np.float32)
        fid = flight[0]["icao24"]
        for i in range(0, len(values) - WINDOW_SIZE + 1, STRIDE):
            w5 = values[i:i + WINDOW_SIZE]
            w10 = compute_derived(w5)
            all_windows.append(w10)
            all_flight_ids.append(fid)
            all_timestamps.append(flight[i + WINDOW_SIZE - 1]["timestamp"])

    stats["windows"] = len(all_windows)
    log(f"  Total windows: {len(all_windows):,}")

    if not all_windows:
        return

    all_windows = np.array(all_windows, dtype=np.float32)
    n_windows = len(all_windows)
    n_features = all_windows.shape[2]

    # Scale
    flat = all_windows.reshape(-1, n_features)
    scaled = scaler.transform(flat).reshape(n_windows, WINDOW_SIZE, n_features)

    # Infer
    log("  Running VAE-LSTM inference (v3, no SVDD)...")
    all_results = []
    for start in range(0, n_windows, BATCH_INFER):
        end = min(start + BATCH_INFER, n_windows)
        batch = scaled[start:end]
        with torch.no_grad():
            recon, mu, logvar, z = vae(torch.FloatTensor(batch))

        recon_np = recon.numpy()
        batch_np = batch

        for j in range(len(batch)):
            recon_error = float(np.mean((batch_np[j] - recon_np[j]) ** 2))
            is_anomaly = recon_error > threshold

            attack_type = "normal"
            if is_anomaly:
                fe = np.mean((batch_np[j] - recon_np[j]) ** 2, axis=0)
                attack_type = classify_attack(fe)
                stats["anomalies"] += 1

            all_results.append({
                "icao24": all_flight_ids[start + j],
                "is_anomaly": is_anomaly,
                "recon_error": recon_error,
                "weighted_score": recon_error,
                "attack_type": attack_type,
                "anomaly_type_detected": attack_type if is_anomaly else None,
                "dominant_feature": "",
                "timestamp": datetime.utcfromtimestamp(all_timestamps[start + j]).isoformat() + "Z",
                "window_size": WINDOW_SIZE,
                "batch_source": "batch_es_v3.py",
            })

        if (start // BATCH_INFER + 1) % 10 == 0:
            log(f"    Infer: {end:,}/{n_windows:,} windows, "
                f"{stats['anomalies']} anomalies detected")

    # Save to ES
    log(f"  Saving {len(all_results):,} results to ES anomaly-stream...")
    saved = 0
    for i, res in enumerate(all_results):
        try:
            es_local.index(index=ANOMALY_INDEX, document=res)
            saved += 1
            if saved % 5000 == 0:
                log(f"    Saved {saved:,}/{len(all_results):,} to ES...")
                es_local.indices.refresh(index=ANOMALY_INDEX)
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] < 5:
                log(f"    ES save error: {e}")

    es_local.indices.refresh(index=ANOMALY_INDEX)
    log(f"  Saved {saved:,} documents to anomaly-stream")
    stats["saved"] = saved


def main():
    global es_remote, es_local, t_start
    t_start = time.time()

    log("=" * 55)
    log("BATCH-ES v3: Batch Detection 14,2 Juta Data dari Meiva")
    log("=" * 55)

    # Connect
    es_remote = Elasticsearch(REMOTE_ES, request_timeout=30)
    es_local = Elasticsearch(LOCAL_ES, request_timeout=30)
    remote_info = es_remote.info()
    total_remote = es_remote.count(index=REMOTE_INDEX)["count"]
    log(f"ES teman: {remote_info['name']} ({remote_info['version']['number']})")
    log(f"  Index '{REMOTE_INDEX}': {total_remote:,} docs")

    # Load model (v3, no SVDD)
    vae, scaler, threshold, config = load_model()

    # Fetch data
    docs = fetch_all_data()

    # Clean
    cleaned = clean_data(docs)
    del docs

    # Segment flights
    flights = segment_flights(cleaned)
    del cleaned

    # Process all windows
    process_flights(flights, vae, scaler, threshold, config)

    # Done
    elapsed = time.time() - t_start
    log("=" * 55)
    log(f"SELESAI! Waktu: {elapsed/60:.1f} menit")
    log(f"  Data fetched:     {stats['fetched']:,}")
    log(f"  After cleaning:   {stats['after_clean']:,}")
    log(f"  Flight segments:  {stats['flight_segments']:,}")
    log(f"  Windows diproses: {stats['windows']:,}")
    log(f"  Anomali ditemukan:{stats['anomalies']:,}")
    log(f"  Disimpan ke ES:   {stats.get('saved', 0):,}")
    log(f"  Errors:           {stats['errors']}")
    log("=" * 55)
    log("Buka Grafana: http://localhost:3000 -> Last 7 days")
    log("=" * 55)


if __name__ == "__main__":
    main()

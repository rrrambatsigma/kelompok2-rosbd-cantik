import os
import sys
import json
import time
import numpy as np
import torch
import joblib
import requests
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modelling.anomaly.vae_lstm import VAELSTM

REMOTE_ES = "http://100.99.130.69:9200"
REMOTE_INDEX = "flights"
LOCAL_ES = "http://localhost:9200"
ANOMALY_INDEX = "anomaly-stream"
MODEL_DIR = os.path.join(BASE_DIR, "models", "vae-svdd")

WINDOW_SIZE = 10
STRIDE = 5
FEATURES = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]
ALL_FEATURES = FEATURES + ["dlat", "dlon", "dvel", "dalt", "dtrack"]

TOP_FLIGHTS = [
    "4cae7d", "4cac86", "40666d", "4bc883", "738061",
    "48c2b7", "4bcda9", "406ac3", "461faa", "4d245c"
]

stats = {"fetched": 0, "cleaned": 0, "windows": 0, "anomalies": 0, "saved": 0}
t_start = time.time()

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def load_model():
    log("Loading model...")
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)

    ckpt = torch.load(os.path.join(MODEL_DIR, "vae_model.pt"), map_location="cpu")
    vae = VAELSTM(
        ckpt["input_dim"], ckpt["window_size"],
        ckpt["hidden_dim"], ckpt["latent_dim"]
    )
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()

    svdd = joblib.load(os.path.join(MODEL_DIR, "svdd_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    threshold = config.get("best_threshold", config.get("threshold", 0.228))

    log(f"VAE: {ckpt['input_dim']}feat, latent={ckpt['latent_dim']}")
    log(f"SVDD: {len(svdd.support_)} vectors")
    log(f"Threshold: {threshold:.4f}")
    return vae, svdd, scaler, threshold

def fetch_flight(icao24):
    all_docs = []
    after = None
    max_retries = 3

    while len(all_docs) < 3000:
        body = {
            "size": 10000,
            "query": {"term": {"icao24.keyword": icao24}},
            "sort": [{"timestamp": {"order": "asc"}}],
            "_source": ["icao24", "latitude", "longitude", "velocity",
                        "baro_altitude", "true_track", "timestamp"]
        }
        if after:
            body["search_after"] = after

        for attempt in range(max_retries):
            try:
                r = requests.get(f"{REMOTE_ES}/{REMOTE_INDEX}/_search", json=body, timeout=30)
                if r.status_code != 200:
                    log(f"  ES fetch returned {r.status_code}, attempt {attempt+1}")
                    if attempt < max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                        continue
                    return all_docs if all_docs else []
                break
            except requests.exceptions.Timeout:
                log(f"  Timeout fetch {icao24} (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    log(f"  Gagal fetch {icao24}: timeout setelah {max_retries} percobaan")
                    return all_docs if all_docs else []
            except requests.exceptions.ConnectionError as e:
                log(f"  Connection error fetch {icao24}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    return all_docs if all_docs else []

        hits = r.json()["hits"]["hits"]
        if not hits:
            break

        all_docs.extend([h["_source"] for h in hits])
        after = hits[-1]["sort"]

        if len(hits) < 10000:
            break

    return all_docs

def clean(records):
    clean_records = []
    reasons = {"null_latlon": 0, "vel_range": 0, "alt_range": 0, "track_range": 0, "null_field": 0, "parse_error": 0}
    for r in records:
        try:
            lat = r.get("latitude")
            lon = r.get("longitude")
            vel = r.get("velocity")
            alt = r.get("baro_altitude")
            track = r.get("true_track")

            if lat is None or lon is None:
                reasons["null_latlon"] += 1
                continue
            if vel is not None and not (0 <= vel <= 500):
                reasons["vel_range"] += 1
                continue
            if alt is not None and not (-1000 <= alt <= 45000):
                reasons["alt_range"] += 1
                continue
            if track is not None and not (0 <= track <= 360):
                reasons["track_range"] += 1
                continue
            if vel is None or alt is None or track is None:
                reasons["null_field"] += 1
                continue

            clean_records.append({
                "icao24": r.get("icao24"),
                "timestamp": float(r.get("timestamp", 0)),
                "latitude": float(lat),
                "longitude": float(lon),
                "velocity": float(vel),
                "baro_altitude": float(alt),
                "true_track": float(track),
            })
        except (TypeError, ValueError):
            reasons["parse_error"] += 1
            continue

    filtered = len(records) - len(clean_records)
    if filtered > 0:
        parts = [f"{k}={v}" for k, v in reasons.items() if v > 0]
        log(f"  Cleaned: {len(clean_records)}/{len(records)} kept, filtered: {', '.join(parts)}")
    return clean_records

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

def process_flight(icao24, records, vae, svdd, scaler, threshold):
    values = np.array([[r[f] for f in FEATURES] for r in records], dtype=np.float32)
    n = len(values)

    windows_5f = []
    timestamps = []
    for i in range(0, n - WINDOW_SIZE + 1, STRIDE):
        windows_5f.append(values[i:i + WINDOW_SIZE])
        timestamps.append(records[i + WINDOW_SIZE - 1]["timestamp"])

    if not windows_5f:
        return 0

    windows_10f = np.array([compute_derived(w) for w in windows_5f], dtype=np.float32)
    n_windows = len(windows_10f)
    stats["windows"] += n_windows

    flat = windows_10f.reshape(-1, len(ALL_FEATURES))
    scaled = scaler.transform(flat).reshape(n_windows, WINDOW_SIZE, len(ALL_FEATURES))

    batch_size = 256
    results = []
    debug_recon = []

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = scaled[start:end]
        x = torch.FloatTensor(batch)

        with torch.no_grad():
            mu, logvar = vae.encode(x)
            recon = vae.decode(mu)

        recon_np = recon.numpy()
        batch_np = batch
        mu_np = mu.numpy()

        for j in range(len(batch)):
            recon_error = float(np.mean((batch_np[j] - recon_np[j]) ** 2))
            svdd_scores = svdd.decision_function(mu_np[j:j+1])
            svdd_dist = float(-svdd_scores[0])
            combined = recon_error + svdd_dist
            is_anomaly = recon_error > threshold

            attack_type = "normal"
            if is_anomaly:
                fe = np.mean((batch_np[j] - recon_np[j]) ** 2, axis=0)
                attack_type = classify_attack(fe)
                stats["anomalies"] += 1

            results.append({
                "icao24": icao24,
                "is_anomaly": is_anomaly,
                "recon_error": recon_error,
                "svdd_distance": svdd_dist,
                "combined_score": combined,
                "attack_type": attack_type,
                "anomaly_type_detected": attack_type if is_anomaly else None,
                "dominant_feature": "",
                "timestamp": datetime.utcfromtimestamp(timestamps[start + j]).isoformat() + "Z",
                "window_size": WINDOW_SIZE,
                "batch_source": "process_meiva",
            })

    # Save to local ES
    saved = 0
    for res in results:
        try:
            requests.post(f"{LOCAL_ES}/{ANOMALY_INDEX}/_doc", json=res, timeout=3)
            saved += 1
        except:
            pass

    stats["saved"] += saved
    return len(results)

def main():
    log("=" * 55)
    log("PROCESS MEIVA - VAE-LSTM Batch Detection")
    log(f"  ES Meiva: {REMOTE_ES}/{REMOTE_INDEX}")
    log(f"  ES Local: {LOCAL_ES}/{ANOMALY_INDEX}")
    log(f"  Flights:  {len(TOP_FLIGHTS)} top flights")
    log(f"  Window:   size={WINDOW_SIZE}, stride={STRIDE}")
    log("=" * 55)

    try:
        requests.get(f"{LOCAL_ES}", timeout=5)
        log("Local ES: OK")
    except:
        log("Local ES: NOT REACHABLE!")
        return

    vae, svdd, scaler, threshold = load_model()

    for idx, icao24 in enumerate(TOP_FLIGHTS):
        log(f"[{idx+1}/{len(TOP_FLIGHTS)}] Fetching {icao24}...")
        try:
            records = fetch_flight(icao24)
        except Exception as e:
            log(f"  Gagal fetch {icao24}: {e}, skip ke flight berikutnya")
            continue

        if not records:
            log(f"  Tidak ada data untuk {icao24}, skip")
            continue

        stats["fetched"] += len(records)

        cleaned = clean(records)
        stats["cleaned"] += len(cleaned)
        log(f"  Fetched {len(records)}, cleaned {len(cleaned)}")

        if len(cleaned) < WINDOW_SIZE:
            log(f"  Skipped (too few records)")
            continue

        try:
            n_results = process_flight(icao24, cleaned, vae, svdd, scaler, threshold)
            log(f"  Processed {n_results} windows, {stats['anomalies']} anomalies so far")
        except Exception as e:
            log(f"  Error processing {icao24}: {e}, skip")
            continue

        elapsed = time.time() - t_start
        log(f"  Elapsed: {elapsed:.0f}s")

    elapsed = time.time() - t_start
    log("=" * 55)
    log(f"SELESAI dalam {elapsed/60:.1f} menit!")
    log(f"  Fetched:  {stats['fetched']:,}")
    log(f"  Cleaned:  {stats['cleaned']:,}")
    log(f"  Windows:  {stats['windows']:,}")
    log(f"  Anomalies:{stats['anomalies']:,}")
    log(f"  Saved:    {stats['saved']:,} to ES anomaly-stream")
    log("=" * 55)
    log("Grafana: http://localhost:3000")

if __name__ == "__main__":
    main()

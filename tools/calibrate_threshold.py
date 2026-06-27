"""
calibrate_threshold.py — Kalibrasi threshold VAE-LSTM pakai data real dari ES Meiva.

Alur:
  1. Ambil 5000 record dari ES Meiva (flight dengan record terbanyak)
  2. Clean & sliding window
  3. Run VAE-LSTM inference
  4. Hitung P95 reconstruction error
  5. Update config.json dengan threshold baru
"""
import os, sys, json, time, requests
import numpy as np
import torch
import joblib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from modelling.anomaly.vae_lstm import VAELSTM

REMOTE_ES = "http://100.99.130.69:9200"
REMOTE_INDEX = "flights"
MODEL_DIR = os.path.join(BASE_DIR, "models", "vae-svdd")
WINDOW_SIZE = 10
STRIDE = 5
FEATURES = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

def load_model():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)
    ckpt = torch.load(os.path.join(MODEL_DIR, "vae_model.pt"), map_location="cpu")
    vae = VAELSTM(ckpt["input_dim"], ckpt["window_size"], ckpt["hidden_dim"], ckpt["latent_dim"])
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    svdd = joblib.load(os.path.join(MODEL_DIR, "svdd_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return vae, svdd, scaler, cfg

def fetch_top_flight():
    r = requests.get(f"{REMOTE_ES}/{REMOTE_INDEX}/_search", json={
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"range": {"velocity": {"gt": 50}}},
                    {"exists": {"field": "baro_altitude"}},
                    {"range": {"baro_altitude": {"gt": 1000}}}
                ]
            }
        },
        "aggs": {"top": {"terms": {"field": "icao24.keyword", "size": 5, "min_doc_count": 100}}}
    }, timeout=15)
    buckets = r.json()["aggregations"]["top"]["buckets"]
    for b in buckets:
        icao = b["key"]
        log(f"Fetching {icao} ({b['doc_count']} records)...")
        docs = []
        after = None
        while len(docs) < b["doc_count"]:
            body = {
                "size": 10000,
                "query": {"term": {"icao24.keyword": icao}},
                "sort": [{"timestamp": {"order": "asc"}}],
                "_source": ["icao24", "latitude", "longitude", "velocity",
                            "baro_altitude", "true_track", "timestamp"]
            }
            if after:
                body["search_after"] = after
            try:
                r2 = requests.get(f"{REMOTE_ES}/{REMOTE_INDEX}/_search", json=body, timeout=30)
                hits = r2.json()["hits"]["hits"]
                if not hits:
                    break
                docs.extend([h["_source"] for h in hits])
                after = hits[-1]["sort"]
                if len(hits) < 10000:
                    break
            except Exception as e:
                log(f"  Error: {e}, coba flight lain")
                break
        if len(docs) >= 100:
            log(f"  Got {len(docs)} records for {icao}")
            docs.sort(key=lambda r: r.get("timestamp", 0))
            return docs
    return []

def clean(records):
    clean_records = []
    for r in records:
        try:
            lat = r.get("latitude")
            lon = r.get("longitude")
            vel = r.get("velocity")
            alt = r.get("baro_altitude")
            track = r.get("true_track")
            if lat is None or lon is None:
                continue
            if vel is not None and not (0 <= vel <= 500):
                continue
            if alt is not None and not (-1000 <= alt <= 45000):
                continue
            if track is not None and not (0 <= track <= 360):
                continue
            if vel is None or alt is None or track is None:
                continue
            clean_records.append({
                "icao24": r.get("icao24"),
                "timestamp": float(r.get("timestamp", 0)),
                "latitude": float(lat), "longitude": float(lon),
                "velocity": float(vel), "baro_altitude": float(alt),
                "true_track": float(track),
            })
        except (TypeError, ValueError):
            continue
    return clean_records

def compute_derived(w5):
    deltas = np.diff(w5, axis=0, prepend=w5[0:1])
    return np.concatenate([w5, deltas], axis=1)

def main():
    log("=" * 55)
    log("CALIBRASI THRESHOLD - Data Real ADS-B")
    log("=" * 55)

    vae, svdd, scaler, cfg = load_model()
    old_threshold = cfg.get("best_threshold", cfg.get("threshold", 0.228))
    log(f"Threshold lama: {old_threshold:.4f}")

    records = fetch_top_flight()
    if not records:
        log("Tidak ada data! Cek koneksi ke ES Meiva.")
        return

    cleaned = clean(records)
    log(f"Records: {len(records)} fetched, {len(cleaned)} cleaned")

    values = np.array([[r[f] for f in FEATURES] for r in cleaned], dtype=np.float32)
    n = len(values)

    windows = []
    for i in range(0, n - WINDOW_SIZE + 1, STRIDE):
        windows.append(compute_derived(values[i:i + WINDOW_SIZE]))

    log(f"Windows: {len(windows)} (stride={STRIDE})")

    windows_np = np.array(windows, dtype=np.float32)
    n_w = len(windows_np)
    flat = windows_np.reshape(-1, len(FEATURES) * 2)
    scaled = scaler.transform(flat).reshape(n_w, WINDOW_SIZE, len(FEATURES) * 2)

    recon_errors = []
    svdd_dists = []
    batch_size = 256

    for start in range(0, n_w, batch_size):
        end = min(start + batch_size, n_w)
        batch = scaled[start:end]
        with torch.no_grad():
            recon, mu, logvar, z = vae(torch.FloatTensor(batch))
        for j in range(len(batch)):
            re = float(np.mean((batch[j] - recon[j].numpy()) ** 2))
            recon_errors.append(re)
            sd = float(-svdd.decision_function(z[j:j+1].numpy())[0])
            svdd_dists.append(sd)

    recon_errors = np.array(recon_errors)
    svdd_dists = np.array(svdd_dists)
    combined = recon_errors + svdd_dists

    p95_recon = np.percentile(recon_errors, 95)
    p95_combined = np.percentile(combined, 95)
    p99_recon = np.percentile(recon_errors, 99)

    log(f"\nReconstruction Error (data real):")
    log(f"  min={recon_errors.min():.4f}  max={recon_errors.max():.4f}")
    log(f"  mean={recon_errors.mean():.4f}  median={np.median(recon_errors):.4f}")
    log(f"  P90={np.percentile(recon_errors, 90):.4f}")
    log(f"  P95={p95_recon:.4f}")
    log(f"  P99={p99_recon:.4f}")
    log(f"\n  Threshold LAMA: {old_threshold:.4f}")
    log(f"  -> anomaly rate LAMA: {(recon_errors > old_threshold).mean()*100:.1f}%")
    log(f"\n  Threshold BARU (P95): {p95_recon:.4f}")
    log(f"  -> anomaly rate BARU: 5.0% (by definition)")

    log(f"\nCombine score:")
    log(f"  P95={p95_combined:.4f}")

    # Update config
    new_threshold = float(p95_recon)
    cfg["threshold"] = new_threshold
    cfg["best_threshold"] = new_threshold
    cfg["threshold_method"] = "recon_p95_real_data"
    cfg["threshold_youden_recon"] = new_threshold
    cfg["calibration_date"] = datetime.now().isoformat()
    cfg["calibration_notes"] = f"P95 dari {len(recon_errors)} windows data real ADS-B"

    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    log(f"\nConfig.json UPDATED!")
    log(f"  {old_threshold:.4f} -> {new_threshold:.4f}")
    log("=" * 55)

if __name__ == "__main__":
    main()

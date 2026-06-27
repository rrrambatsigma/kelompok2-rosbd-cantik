"""Compare recon_error for different data sizes from same flight"""
import os, sys, json, numpy as np, torch, joblib, requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from modelling.anomaly.vae_lstm import VAELSTM

MODEL_DIR = os.path.join(BASE_DIR, "models", "vae-svdd")
WINDOW_SIZE = 10; STRIDE = 5
FEATURES = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]
ALL_FEATURES = FEATURES + ["dlat", "dlon", "dvel", "dalt", "dtrack"]

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    cfg = json.load(f)
threshold = cfg.get("best_threshold", 0.228)

ckpt = torch.load(os.path.join(MODEL_DIR, "vae_model.pt"), map_location="cpu")
vae = VAELSTM(ckpt["input_dim"], ckpt["window_size"], ckpt["hidden_dim"], ckpt["latent_dim"])
vae.load_state_dict(ckpt["model_state_dict"])
vae.eval()
svdd = joblib.load(os.path.join(MODEL_DIR, "svdd_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# Fetch all data for flight 4cae7d using search_after
all_docs = []
after = None
while len(all_docs) < 2700:
    body = {
        "size": 10000,
        "query": {"term": {"icao24.keyword": "4cae7d"}},
        "sort": [{"timestamp": {"order": "asc"}}],
        "_source": ["icao24", "latitude", "longitude", "velocity", "baro_altitude", "true_track", "timestamp"]
    }
    if after:
        body["search_after"] = after
    r = requests.get('http://100.99.130.69:9200/flights/_search', json=body, timeout=30)
    hits = r.json()["hits"]["hits"]
    if not hits:
        break
    all_docs.extend([h["_source"] for h in hits])
    after = hits[-1]["sort"]
    if len(hits) < 10000:
        break

print(f"Total fetched: {len(all_docs)}")

# Clean
def compute_derived(w5):
    deltas = np.diff(w5, axis=0, prepend=w5[0:1])
    return np.concatenate([w5, deltas], axis=1)

# Test with different data sizes
for size_name, n_records in [("First 500", 500), ("First 1000", 1000), ("All", len(all_docs))]:
    records = all_docs[:n_records]
    cleaned = []
    for r in records:
        lat, lon, vel, alt, track = [r.get(f) for f in FEATURES]
        if lat is None or lon is None: continue
        if vel is None or alt is None or track is None: continue
        if not (0 <= vel <= 500): continue
        if not (-1000 <= alt <= 45000): continue
        if not (0 <= track <= 360): continue
        cleaned.append(r)

    values = np.array([[r[f] for f in FEATURES] for r in cleaned], dtype=np.float32)
    windows = []
    for i in range(0, len(values) - WINDOW_SIZE + 1, STRIDE):
        windows.append(compute_derived(values[i:i + WINDOW_SIZE]))

    if not windows:
        print(f"\n{size_name}: no windows")
        continue

    w_np = np.array(windows, dtype=np.float32)
    flat = w_np.reshape(-1, len(ALL_FEATURES))
    scaled = scaler.transform(flat).reshape(len(w_np), WINDOW_SIZE, len(ALL_FEATURES))

    with torch.no_grad():
        recon, mu, logvar, z = vae(torch.FloatTensor(scaled))

    recon_errors = [float(np.mean((scaled[j] - recon[j].numpy()) ** 2)) for j in range(len(scaled))]
    recon_errors = np.array(recon_errors)

    anom = (recon_errors > threshold).sum()
    print(f"\n{size_name} ({len(cleaned)} records, {len(windows)} windows):")
    print(f"  mean={recon_errors.mean():.4f} median={np.median(recon_errors):.4f}")
    print(f"  P90={np.percentile(recon_errors,90):.4f}  P95={np.percentile(recon_errors,95):.4f}")
    print(f"  threshold={threshold:.4f}  anomaly={anom}/{len(windows)} ({100*anom/len(windows):.1f}%)")

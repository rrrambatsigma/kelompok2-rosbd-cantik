"""Debug: compare recon_error computation between process_meiva and calibrate"""
import os, sys, json, numpy as np, torch, joblib, requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from modelling.anomaly.vae_lstm import VAELSTM

MODEL_DIR = os.path.join(BASE_DIR, "models", "vae-svdd")
WINDOW_SIZE = 10
STRIDE = 5
FEATURES = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]
ALL_FEATURES = FEATURES + ["dlat", "dlon", "dvel", "dalt", "dtrack"]

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    cfg = json.load(f)
threshold = cfg.get("best_threshold", 0.228)
print(f"Threshold from config: {threshold}")

ckpt = torch.load(os.path.join(MODEL_DIR, "vae_model.pt"), map_location="cpu")
vae = VAELSTM(ckpt["input_dim"], ckpt["window_size"], ckpt["hidden_dim"], ckpt["latent_dim"])
vae.load_state_dict(ckpt["model_state_dict"])
vae.eval()
svdd = joblib.load(os.path.join(MODEL_DIR, "svdd_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# Fetch one flight
r = requests.get('http://100.99.130.69:9200/flights/_search', json={
    'size': 500,
    'query': {'term': {'icao24.keyword': '4cae7d'}},
    'sort': [{'timestamp': {'order': 'asc'}}],
    '_source': ['icao24', 'latitude', 'longitude', 'velocity', 'baro_altitude', 'true_track', 'timestamp']
}, timeout=15)

records = [h['_source'] for h in r.json()['hits']['hits']]

# Clean
cleaned = []
for r in records:
    lat, lon, vel, alt, track = [r.get(f) for f in FEATURES]
    if lat is None or lon is None: continue
    if (vel is None) or (alt is None) or (track is None): continue
    if not (0 <= vel <= 500): continue
    if not (-1000 <= alt <= 45000): continue
    if not (0 <= track <= 360): continue
    cleaned.append(r)
print(f"Records: {len(records)} fetched, {len(cleaned)} cleaned")

def compute_derived(w5):
    deltas = np.diff(w5, axis=0, prepend=w5[0:1])
    return np.concatenate([w5, deltas], axis=1)

values = np.array([[r[f] for f in FEATURES] for r in cleaned], dtype=np.float32)
windows = []
for i in range(0, len(values) - WINDOW_SIZE + 1, STRIDE):
    windows.append(compute_derived(values[i:i + WINDOW_SIZE]))

windows_np = np.array(windows, dtype=np.float32)
flat = windows_np.reshape(-1, len(ALL_FEATURES))
scaled = scaler.transform(flat).reshape(len(windows_np), WINDOW_SIZE, len(ALL_FEATURES))

with torch.no_grad():
    recon, mu, logvar, z = vae(torch.FloatTensor(scaled))

recon_errors = []
for j in range(len(scaled)):
    re = float(np.mean((scaled[j] - recon[j].numpy()) ** 2))
    recon_errors.append(re)

recon_errors = np.array(recon_errors)
print(f"\nRecon Error stats (n={len(recon_errors)}):")
print(f"  min={recon_errors.min():.4f}  max={recon_errors.max():.4f}")
print(f"  mean={recon_errors.mean():.4f}  median={np.median(recon_errors):.4f}")
print(f"  P50={np.percentile(recon_errors,50):.4f}")
print(f"  P90={np.percentile(recon_errors,90):.4f}")
print(f"  P95={np.percentile(recon_errors,95):.4f}")
print(f"  P99={np.percentile(recon_errors,99):.4f}")
print(f"\nWith threshold {threshold:.4f}:")
anom_rate = (recon_errors > threshold).mean() * 100
print(f"  Anomaly rate: {anom_rate:.1f}%")
print(f"  Anomalies: {(recon_errors > threshold).sum()}/{len(recon_errors)}")

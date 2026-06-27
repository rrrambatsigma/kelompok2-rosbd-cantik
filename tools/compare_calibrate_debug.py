"""Compare the calibrate approach vs debug approach side-by-side"""
import os, sys, json, numpy as np, torch, joblib, requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from modelling.anomaly.vae_lstm import VAELSTM

MODEL_DIR = os.path.join(BASE_DIR, "models", "vae-svdd")
WINDOW_SIZE = 10; STRIDE = 5
FEATURES = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    cfg = json.load(f)
threshold = cfg.get("best_threshold", 0.228)

ckpt = torch.load(os.path.join(MODEL_DIR, "vae_model.pt"), map_location="cpu")
vae = VAELSTM(ckpt["input_dim"], ckpt["window_size"], ckpt["hidden_dim"], ckpt["latent_dim"])
vae.load_state_dict(ckpt["model_state_dict"])
vae.eval()
svdd = joblib.load(os.path.join(MODEL_DIR, "svdd_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# Fetch data like calibrate does
r = requests.get('http://100.99.130.69:9200/flights/_search', json={
    'size': 0,
    'query': {'bool': {'filter': [
        {'range': {'velocity': {'gt': 50}}},
        {'exists': {'field': 'baro_altitude'}},
        {'range': {'baro_altitude': {'gt': 1000}}}
    ]}},
    'aggs': {'top': {'terms': {'field': 'icao24.keyword', 'size': 1, 'min_doc_count': 100}}}
}, timeout=15)
buckets = r.json()['aggregations']['top']['buckets']
icao = buckets[0]['key']
print(f"Top flight: {icao} ({buckets[0]['doc_count']} records)")

# Fetch all
all_docs = []
after = None
while True:
    body = {
        'size': 10000,
        'query': {'term': {'icao24.keyword': icao}},
        'sort': [{'timestamp': {'order': 'asc'}}],
        '_source': ['icao24', 'latitude', 'longitude', 'velocity', 'baro_altitude', 'true_track', 'timestamp']
    }
    if after:
        body['search_after'] = after
    r2 = requests.get('http://100.99.130.69:9200/flights/_search', json=body, timeout=30)
    hits = r2.json()['hits']['hits']
    if not hits:
        break
    all_docs.extend([h['_source'] for h in hits])
    after = hits[-1]['sort']
    if len(hits) < 10000:
        break

print(f"Fetched {len(all_docs)} records")

# Clean like calibrate
cleaned = []
for r in all_docs:
    try:
        lat, lon, vel, alt, track = [r.get(f) for f in FEATURES]
        if lat is None or lon is None: continue
        if vel is not None and not (0 <= vel <= 500): continue
        if alt is not None and not (-1000 <= alt <= 45000): continue
        if track is not None and not (0 <= track <= 360): continue
        if vel is None or alt is None or track is None: continue
        cleaned.append(r)
    except: continue

print(f"Cleaned {len(cleaned)} records")

# Compute windows
def compute_derived(w5):
    deltas = np.diff(w5, axis=0, prepend=w5[0:1])
    return np.concatenate([w5, deltas], axis=1)

values = np.array([[r[f] for f in FEATURES] for r in cleaned], dtype=np.float32)

windows = []
for i in range(0, len(values) - WINDOW_SIZE + 1, STRIDE):
    windows.append(compute_derived(values[i:i + WINDOW_SIZE]))

print(f"Windows: {len(windows)}")

windows_np = np.array(windows, dtype=np.float32)
n_w = len(windows_np)
flat = windows_np.reshape(-1, len(FEATURES) * 2)
scaled = scaler.transform(flat).reshape(n_w, WINDOW_SIZE, len(FEATURES) * 2)

print(f"\nSample data stat (first 2 windows):")
print(f"  window_0 base: {values[0]}")
print(f"  window_0 derived: {windows[0][0]}")
print(f"  window_0 scaled: {scaled[0][0]}")

# Compute errors BOTH ways
recon_errors_batch = []
recon_errors_all = []

# Method 1: Batched (like calibrate)
for start in range(0, n_w, 256):
    end = min(start + 256, n_w)
    batch = scaled[start:end]
    with torch.no_grad():
        recon, mu, logvar, z = vae(torch.FloatTensor(batch))
    for j in range(len(batch)):
        re = float(np.mean((batch[j] - recon[j].numpy()) ** 2))
        recon_errors_batch.append(re)

# Method 2: All at once (like debug_recon2)
with torch.no_grad():
    recon, mu, logvar, z = vae(torch.FloatTensor(scaled))
for j in range(n_w):
    re = float(np.mean((scaled[j] - recon[j].numpy()) ** 2))
    recon_errors_all.append(re)

recon_errors_batch = np.array(recon_errors_batch)
recon_errors_all = np.array(recon_errors_all)

print(f"\nResults:")
print(f"  Len batch={len(recon_errors_batch)}  all={len(recon_errors_all)}")
print(f"  Match: {np.allclose(recon_errors_batch, recon_errors_all)}")
print(f"  Max diff: {np.abs(recon_errors_batch - recon_errors_all).max()}")

# Print first 10 values
print(f"\nFirst 10 recon_errors (batch): {np.round(recon_errors_batch[:10], 4)}")
print(f"First 10 recon_errors (all):   {np.round(recon_errors_all[:10], 4)}")

print(f"\nBatch method stats:")
print(f"  median={np.median(recon_errors_batch):.4f}  P95={np.percentile(recon_errors_batch, 95):.4f}")
print(f"  anomaly rate with threshold {threshold:.4f}: {(recon_errors_batch > threshold).mean()*100:.1f}%")

import os
import json
import time
import numpy as np
import torch
import joblib
from datetime import datetime, timezone
from typing import List, Optional
from collections import deque, defaultdict
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from elasticsearch import Elasticsearch

from modelling.anomaly.vae_lstm import VAELSTM

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "vae-svdd-trained")
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost:9200")
ANOMALY_INDEX = os.getenv("ANOMALY_INDEX", "anomaly-stream")

WINDOW_SIZE = 10
FEATURES_BASE = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]
FEATURES_ALL = FEATURES_BASE + ["dlat", "dlon", "dvel", "dalt", "dtrack"]
N_FEATURES_BASE = len(FEATURES_BASE)
N_FEATURES_ALL = len(FEATURES_ALL)
FEATURE_WEIGHTS = [0.15, 0.15, 0.25, 0.15, 0.30]
BUFFER_CLEANUP_INTERVAL = int(os.getenv("BUFFER_CLEANUP_INTERVAL", "120"))
BUFFER_TTL = int(os.getenv("BUFFER_TTL", "1800"))

# Validation ranges for ADS-B data
VALID_LAT = (-90, 90)
VALID_LON = (-180, 180)
VALID_VEL = (0, 500)
VALID_ALT = (-1000, 45000)
VALID_TRACK = (0, 360)

app = FastAPI(
    title="VAE-LSTM Anomaly Detection API",
    description="""
    Real-time anomaly detection untuk data ADS-B penerbangan.
    
    **Arsitektur:**
    - VAE-LSTM (Variational Autoencoder dengan LSTM) mendeteksi pola penerbangan normal
    - Feature-weighted reconstruction error + multi-threshold per dominant feature
    
    **Alur Deteksi:**
    1. Kirim data penerbangan via `/predict/stream` (1 record) atau `/predict` (batch)
    2. API buffer per `icao24` (10 timesteps)
    3. Setelah buffer penuh → VAE encode → decode → per-feature reconstruction error
    4. Weighted score: `lat*0.15 + lon*0.15 + vel*0.25 + alt*0.15 + track*0.30`
    5. Anomali jika error feature dominan > threshold-nya
    
    **6 Jenis Anomali:**
    - `constant_position` — Posisi diam tidak wajar
    - `random_position` — Posisi meloncat acak
    - `velocity_drift` — Kecepatan berubah drastis
    - `dos_deletion` — Data hilang (gap)
    - `flight_merge` — Data tercampur flight lain
    - `heading_manipulation` — Arah terbang tidak konsisten
    """,
    version="3.0.0",
    contact={
        "name": "ROSBD Kelompok 2",
        "url": "https://github.com/kelompok2-rosbd",
    },
    license_info={
        "name": "MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vae_model = None
scaler = None
config = None
global_threshold = None
es = None

buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
buffer_last_access = {}

# SSE stream buffer — nampung 100 anomaly terakhir untuk dashboard
anomaly_stream_buffer = deque(maxlen=100)


async def cleanup_stale_buffers():
    while True:
        await asyncio.sleep(BUFFER_CLEANUP_INTERVAL)
        now = time.time()
        stale = [k for k, t in buffer_last_access.items() if now - t > BUFFER_TTL]
        for k in stale:
            buffers.pop(k, None)
            buffer_last_access.pop(k, None)
        if stale:
            print(f"[SERVING] Cleaned {len(stale)} stale buffers")


class FlightData(BaseModel):
    icao24: Optional[str] = None
    callsign: Optional[str] = None
    latitude: float
    longitude: float
    velocity: Optional[float] = 0.0
    baro_altitude: Optional[float] = 0.0
    true_track: Optional[float] = 0.0


class BatchPredictRequest(BaseModel):
    flights: List[FlightData]


class AnomalyResult(BaseModel):
    is_anomaly: bool
    recon_error: float
    weighted_score: float
    attack_type: str
    dominant_feature: str
    anomaly_type_detected: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: List[AnomalyResult]
    model_version: str = "2.0.0"
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    es_connected: bool
    model_version: str


def compute_derived_features(window_5f):
    deltas = np.diff(window_5f, axis=0, prepend=window_5f[0:1])
    return np.concatenate([window_5f, deltas], axis=1)


def validate_flight_data(lat, lon, vel, alt, track):
    errors = []
    if lat is None or not (VALID_LAT[0] <= lat <= VALID_LAT[1]):
        errors.append(f"latitude={lat}")
    if lon is None or not (VALID_LON[0] <= lon <= VALID_LON[1]):
        errors.append(f"longitude={lon}")
    if vel is None or not (VALID_VEL[0] <= vel <= VALID_VEL[1]):
        errors.append(f"velocity={vel}")
    if alt is None or not (VALID_ALT[0] <= alt <= VALID_ALT[1]):
        errors.append(f"altitude={alt}")
    if track is None or not (VALID_TRACK[0] <= track <= VALID_TRACK[1]):
        errors.append(f"track={track}")
    return errors


def connect_es():
    global es
    try:
        es = Elasticsearch(f"http://{ES_HOST}", request_timeout=5)
        if es.ping():
            print(f"[SERVING] Connected to ES at {ES_HOST}")
            return True
        else:
            print(f"[SERVING] ES at {ES_HOST} not reachable")
            es = None
            return False
    except Exception as e:
        print(f"[SERVING] ES connection failed: {e}")
        es = None
        return False


def classify_attack(per_feature_error):
    total = per_feature_error.sum() + 1e-10
    lat_lon_error = per_feature_error[0] + per_feature_error[1]
    track_ratio = per_feature_error[4] / total
    vel_ratio = per_feature_error[2] / total
    lat_lon_ratio = lat_lon_error / total

    if track_ratio > 0.5:
        return "heading_manipulation"
    elif vel_ratio > 0.5:
        return "velocity_drift"
    elif lat_lon_ratio > 0.5 and (per_feature_error[0] / total) < 0.4:
        return "random_position"
    elif lat_lon_ratio > 0.5:
        return "constant_position"
    elif total > 0.8:
        return "flight_merge"
    else:
        return "dos_deletion"


@app.on_event("startup")
async def on_startup():
    global vae_model, scaler, config, global_threshold

    loop = asyncio.get_event_loop()
    loop.create_task(cleanup_stale_buffers())

    vae_path = os.path.join(MODEL_DIR, "vae_model.pt")
    if not os.path.exists(vae_path):
        print(f"[SERVING] No model at {vae_path}. Run train.py first.")
        return

    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    try:
        checkpoint = torch.load(vae_path, map_location="cpu")
        vae_model = VAELSTM(
            n_features=checkpoint.get("input_dim", N_FEATURES_ALL),
            window_size=checkpoint.get("window_size", WINDOW_SIZE),
            hidden_dim=checkpoint.get("hidden_dim", 64),
            latent_dim=checkpoint.get("latent_dim", 4),
        )
        vae_model.load_state_dict(checkpoint["model_state_dict"])
        vae_model.eval()
        print(f"[SERVING] VAE-LSTM: {checkpoint.get('input_dim')} features, "
              f"latent={checkpoint.get('latent_dim')}")
    except Exception as e:
        print(f"[SERVING] Failed to load VAE-LSTM: {e}")
        vae_model = None
        return

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[SERVING] Scaler loaded")
    else:
        print(f"[SERVING] No scaler at {scaler_path}")
        vae_model = None
        return

    # Load threshold from config
    if config:
        global_threshold = config.get("global_threshold", 0.5)
        print(f"[SERVING] Global threshold: {global_threshold}")
    else:
        global_threshold = 0.5
        print(f"[SERVING] Using default threshold: {global_threshold}")

    print(f"[SERVING] Loaded from {MODEL_DIR}")


@app.get("/health", response_model=HealthResponse, tags=["System"],
         summary="Health Check",
         description="Cek status service: model loading, koneksi Elasticsearch, dan versi.")
def health():
    model_version = (config.get("model_version", "?") if config else "?")
    return HealthResponse(
        status="ok" if vae_model is not None else "no_model",
        model_loaded=vae_model is not None,
        model_path=MODEL_DIR,
        es_connected=False,
        model_version=model_version,
    )


@app.get("/features", tags=["Model Info"],
         summary="Daftar Fitur Model",
         description="Menampilkan 5 fitur base (lat, lon, vel, alt, track) + 5 fitur derived (delta antar-timestep) yang digunakan model VAE-LSTM.")
def get_features():
    return {
        "base_features": FEATURES_BASE,
        "derived_features": ["dlat", "dlon", "dvel", "dalt", "dtrack"],
        "all_features": FEATURES_ALL,
        "window_size": WINDOW_SIZE,
        "model_loaded": vae_model is not None,
    }


@app.get("/model/info", tags=["Model Info"],
         summary="Konfigurasi Model",
         description="Menampilkan konfigurasi lengkap model: threshold, arsitektur VAE-LSTM, parameter SVDD, dan feature names.")
def model_info():
    if vae_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if config:
        return config
    return {"status": "no config"}


@app.get("/buffer/status", tags=["Debug"],
         summary="Status Buffer Per Pesawat",
         description="[DEBUG] Menampilkan jumlah buffer (timesteps) per icao24 yang sedang dikumpulkan sebelum inferensi.")
def buffer_status():
    return {
        "active_buffers": len(buffers),
        "buffers": {
            icao: len(buf)
            for icao, buf in sorted(buffers.items())
        },
    }


@app.get("/stream", tags=["Streaming"],
         summary="SSE Stream Anomali Real-time",
         description="""
    Server-Sent Events endpoint untuk streaming hasil deteksi anomali real-time ke Grafana.
    
    **Format Event:**
    ```
    data: {"icao24":"...","attack_type":"flight_merge","is_anomaly":true,"recon_error":4.33,...}
    ```
    
    **Cara pakai di Grafana:**
    Gunakan datasource "Simple JSON" atau "Infinity" yang connect ke endpoint ini.
    
    **Catatan:** Saat buffer belum penuh (< 10), akan muncul event `buffering` dengan status `is_anomaly: false`.
    """)
async def stream_anomalies():
    async def event_generator():
        last_len = 0
        while True:
            current_len = len(anomaly_stream_buffer)
            if current_len > last_len:
                for i in range(last_len, current_len):
                    yield "data: " + json.dumps(anomaly_stream_buffer[i]) + "\n\n"
                last_len = current_len
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"],
         summary="Batch Prediction",
         description="""
    Prediksi anomali untuk batch data penerbangan.
    
    **Requirements:**
    - Minimal 10 records untuk 1 window inferensi
    - Setiap record membutuhkan: latitude, longitude, velocity, baro_altitude, true_track
    - Semakin banyak records = semakin banyak window yang diproses
    
    **Contoh request (12 records → 3 windows):**
    ```json
    {
      "flights": [
        {"latitude": -6.2, "longitude": 106.8, "velocity": 250, "baro_altitude": 10000, "true_track": 90},
        ...
      ]
    }
    ```
    
    **Response:** Array of predictions dengan `is_anomaly`, `recon_error`, `svdd_distance`, `combined_score`, `attack_type`.
    """)
def predict(request: BatchPredictRequest):
    if vae_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    n = len(request.flights)
    if n < WINDOW_SIZE:
        raise HTTPException(status_code=400,
                            detail=f"Need >= {WINDOW_SIZE} records, got {n}")

    predictions = []
    for i in range(0, n - WINDOW_SIZE + 1):
        batch = request.flights[i:i + WINDOW_SIZE]
        window_5f = np.array([[
            f.latitude, f.longitude, f.velocity or 0.0,
            f.baro_altitude or 0.0, f.true_track or 0.0,
        ] for f in batch], dtype=np.float32)

        window_10f = compute_derived_features(window_5f)
        scaled_2d = scaler.transform(window_10f)
        scaled = scaled_2d.reshape(1, WINDOW_SIZE, N_FEATURES_ALL)

        with torch.no_grad():
            x = torch.FloatTensor(scaled)
            mu, logvar = vae_model.encode(x)
            recon = vae_model.decode(mu)

        recon_np = recon.numpy()
        data_np = scaled
        recon_error = float(np.mean((data_np - recon_np) ** 2))
        is_anomaly = recon_error > (config.get("f1max_threshold", global_threshold) if config else global_threshold)

        attack_type = "normal"
        if is_anomaly:
            per_feature = np.mean((data_np - recon_np) ** 2, axis=(0, 2))
            attack_type = classify_attack(per_feature)

        predictions.append(AnomalyResult(
            is_anomaly=is_anomaly,
            recon_error=recon_error,
            weighted_score=recon_error,
            attack_type=attack_type,
            dominant_feature="",
            anomaly_type_detected=attack_type if is_anomaly else None,
        ))

    return PredictResponse(
        predictions=predictions,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/stream", tags=["Prediction"],
         summary="Streaming Prediction (1 Record)",
         description="""
    Prediksi anomali untuk 1 record data penerbangan secara real-time.
    
    **Alur:**
    1. API menyimpan buffer per `icao24` (max 10 timesteps)
    2. Jika buffer < 10 → return `is_anomaly: false` + `buffered: "N/10"`
    3. Jika buffer = 10 → inferensi VAE-LSTM + SVDD → return hasil
    
    **Contoh request:**
    ```json
    {
      "icao24": "8a0b1c",
      "callsign": "GIA123",
      "latitude": -6.2,
      "longitude": 106.8,
      "velocity": 250.0,
      "baro_altitude": 10000.0,
      "true_track": 90.0
    }
    ```
    
    **Response saat buffer penuh:**
    ```json
    {
      "is_anomaly": true,
      "recon_error": 4.33,
      "svdd_distance": 261.89,
      "combined_score": 266.23,
      "attack_type": "flight_merge",
      "anomaly_type_detected": "flight_merge"
    }
    ```
    
    **Catatan:** Hasil juga otomatis dikirim ke SSE stream (`/stream`) dan disimpan ke Elasticsearch index `anomaly-stream`.
    """)
def predict_stream(flight: FlightData):
    global vae_model, scaler, config, feature_weights, thresholds, global_threshold
    if vae_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    icao24 = flight.icao24 or "unknown"
    lat, lon, vel, alt, track = (
        flight.latitude, flight.longitude,
        flight.velocity or 0.0, flight.baro_altitude or 0.0,
        flight.true_track or 0.0
    )

    errs = validate_flight_data(lat, lon, vel, alt, track)
    if errs:
        raise HTTPException(status_code=400, detail=f"Invalid data: {', '.join(errs)}")

    now = time.time()
    buf = buffers[icao24]
    buffer_last_access[icao24] = now

    raw = np.array([lat, lon, vel, alt, track], dtype=np.float32)
    buf.append(raw)

    if len(buf) < WINDOW_SIZE:
        anomaly_stream_buffer.append({
            "icao24": flight.icao24,
            "attack_type": "buffering",
            "is_anomaly": False,
            "recon_error": 0.0,
            "weighted_score": 0.0,
            "buffered": f"{len(buf)}/{WINDOW_SIZE}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return AnomalyResult(
            is_anomaly=False, recon_error=0.0, weighted_score=0.0,
            attack_type="normal",
            dominant_feature="none",
            anomaly_type_detected=None,
        )

    window_5f = np.array(buf, dtype=np.float32)
    window_10f = compute_derived_features(window_5f)
    scaled_2d = scaler.transform(window_10f)
    scaled = scaled_2d.reshape(1, WINDOW_SIZE, N_FEATURES_ALL)

    with torch.no_grad():
        x = torch.FloatTensor(scaled)
        mu, logvar = vae_model.encode(x)
        recon = vae_model.decode(mu)

    recon_np = recon.numpy()
    recon_error = float(np.mean((scaled - recon_np) ** 2))
    threshold = (config.get("f1max_threshold", global_threshold) if config else global_threshold)
    is_anomaly = recon_error > threshold

    attack_type = "normal"
    if is_anomaly:
        per_feature = np.mean((scaled - recon_np) ** 2, axis=(0, 2))
        attack_type = classify_attack(per_feature)

    result = AnomalyResult(
        is_anomaly=is_anomaly,
        recon_error=recon_error,
        weighted_score=recon_error,
        attack_type=attack_type,
        dominant_feature="",
        anomaly_type_detected=attack_type if is_anomaly else None,
    )

    ts = datetime.now(timezone.utc).isoformat()

    anomaly_stream_buffer.append({
        "icao24": flight.icao24,
        "attack_type": attack_type,
        "is_anomaly": is_anomaly,
        "recon_error": recon_error,
        "weighted_score": recon_error,
        "timestamp": ts,
    })

    return result


@app.get("/api/stats", tags=["API Lois"],
         summary="Statistik Anomali",
         description="Total data, jumlah anomali, normal, dan anomaly rate.")
def api_stats():
    try:
        es_local = Elasticsearch(f"http://{ES_HOST}", request_timeout=3)
        total = es_local.count(index=ANOMALY_INDEX)["count"]
        anom = es_local.count(index=ANOMALY_INDEX, body={
            "query": {"term": {"is_anomaly": True}}
        })["count"]
        normal = total - anom
        rate = (anom / total * 100) if total > 0 else 0
        return {"total": total, "anomalies": anom, "normal": normal, "rate": round(rate, 2)}
    except Exception as e:
        return {"error": str(e), "total": 0, "anomalies": 0, "normal": 0, "rate": 0}


@app.get("/api/results", tags=["API Lois"],
         summary="Hasil Prediksi Terbaru",
         description="Ambil N hasil prediksi terbaru dari ES. Param: limit=50")
def api_results(limit: int = 50):
    try:
        es_local = Elasticsearch(f"http://{ES_HOST}", request_timeout=3)
        data = es_local.search(
            index=ANOMALY_INDEX,
            body={
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": min(limit, 500),
                "_source": ["icao24", "callsign", "latitude", "longitude",
                            "velocity", "baro_altitude", "true_track",
                            "is_anomaly", "recon_error", "attack_type", "timestamp"]
            }
        )
        hits = data["hits"]["hits"]
        return [h["_source"] for h in hits]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/anomalies", tags=["API Lois"],
         summary="Filter Anomali by Type",
         description="Ambil anomali berdasarkan attack_type. Param: type=flight_merge, limit=50")
def api_anomalies(type: str = "", limit: int = 50):
    try:
        es_local = Elasticsearch(f"http://{ES_HOST}", request_timeout=3)
        if type:
            body = {
                "query": {"term": {"attack_type": type}},
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": min(limit, 500),
            }
        else:
            body = {
                "query": {"term": {"is_anomaly": True}},
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": min(limit, 500),
            }
        data = es_local.search(
            index=ANOMALY_INDEX,
            body=body,
            _source=["icao24", "callsign", "latitude", "longitude",
                     "velocity", "baro_altitude", "true_track",
                     "is_anomaly", "recon_error", "attack_type", "timestamp"]
        )
        hits = data["hits"]["hits"]
        return [h["_source"] for h in hits]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/stream", tags=["API Lois"],
         summary="SSE Stream Hasil Deteksi Real-time",
         description="Server-Sent Events endpoint. Streaming data real-time dari ES setiap 3 detik. "
                     "Semua data (normal + anomali) muncul tanpa duplikat.")
async def api_stream():
    async def event_generator():
        seen_ids = set()
        while True:
            try:
                es_local = Elasticsearch(f"http://{ES_HOST}", request_timeout=2)
                body = {
                    "sort": [{"timestamp": {"order": "desc"}}],
                    "size": 5,
                    "_source": ["icao24", "callsign", "latitude", "longitude",
                                "velocity", "baro_altitude", "true_track",
                                "is_anomaly", "recon_error", "attack_type", "timestamp"]
                }
                data = es_local.search(index=ANOMALY_INDEX, body=body)
                hits = data["hits"]["hits"]
                new_count = 0
                for hit in reversed(hits):
                    doc_id = hit["_id"]
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        yield f"data: {json.dumps(hit['_source'])}\n\n"
                        new_count += 1
                if len(seen_ids) > 1000:
                    seen_ids.clear()
            except:
                pass
            await asyncio.sleep(3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

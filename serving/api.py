import os
import json
import numpy as np
import torch
import joblib
from datetime import datetime
from typing import List, Optional
from collections import deque, defaultdict
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from elasticsearch import Elasticsearch

from modelling.vae_lstm import VAELSTM

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "vae-svdd")
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch:9200")

WINDOW_SIZE = 10
FEATURES_BASE = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]
FEATURES_ALL = FEATURES_BASE + ["dlat", "dlon", "dvel", "dalt", "dtrack"]
N_FEATURES_BASE = len(FEATURES_BASE)
N_FEATURES_ALL = len(FEATURES_ALL)

app = FastAPI(title="VAE-LSTM Anomaly Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vae_model = None
svdd_model = None
scaler = None
config = None
es = None

buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

# SSE stream buffer — nampung 100 anomaly terakhir untuk dashboard
anomaly_stream_buffer = deque(maxlen=100)


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
    svdd_distance: float
    combined_score: float
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


def compute_derived_features(window_5f):
    deltas = np.diff(window_5f, axis=0, prepend=window_5f[0:1])
    return np.concatenate([window_5f, deltas], axis=1)


def classify_attack(per_feature_error):
    total = per_feature_error.sum() + 1e-10
    lat_lon_error = per_feature_error[0] + per_feature_error[1]
    vel_error = per_feature_error[2]
    track_error = per_feature_error[4]

    lat_lon_ratio = lat_lon_error / total
    vel_ratio = vel_error / total
    track_ratio = track_error / total

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
def load_model():
    global vae_model, svdd_model, scaler, config, es
    try:
        es = Elasticsearch(f"http://{ES_HOST}", request_timeout=5)
        if es.ping():
            print(f"[SERVING] Connected to ES at {ES_HOST}")
        else:
            print(f"[SERVING] ES at {ES_HOST} not reachable")
            es = None
    except Exception as e:
        print(f"[SERVING] ES connection failed: {e}")
        es = None

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
            latent_dim=checkpoint.get("latent_dim", 16),
        )
        vae_model.load_state_dict(checkpoint["model_state_dict"])
        vae_model.eval()
        print(f"[SERVING] VAE-LSTM: {checkpoint.get('input_dim')} features, "
              f"latent={checkpoint.get('latent_dim')}")
    except Exception as e:
        print(f"[SERVING] Failed to load VAE-LSTM: {e}")
        vae_model = None
        return

    svdd_path = os.path.join(MODEL_DIR, "svdd_model.pkl")
    if os.path.exists(svdd_path):
        svdd_model = joblib.load(svdd_path)
        print(f"[SERVING] SVDD: {len(svdd_model.support_)} support vectors")
    else:
        print(f"[SERVING] No SVDD at {svdd_path}")
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

    print(f"[SERVING] Loaded from {MODEL_DIR}")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if vae_model is not None else "no_model",
        model_loaded=vae_model is not None,
        model_path=MODEL_DIR,
    )


@app.get("/features")
def get_features():
    return {
        "base_features": FEATURES_BASE,
        "derived_features": ["dlat", "dlon", "dvel", "dalt", "dtrack"],
        "all_features": FEATURES_ALL,
        "window_size": WINDOW_SIZE,
        "model_loaded": vae_model is not None,
    }


@app.get("/model/info")
def model_info():
    if vae_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if config:
        return config
    return {"status": "no config"}


@app.get("/buffer/status")
def buffer_status():
    return {
        "active_buffers": len(buffers),
        "buffers": {
            icao: len(buf)
            for icao, buf in sorted(buffers.items())
        },
    }


@app.get("/stream")
async def stream_anomalies():
    """SSE endpoint: stream anomaly detection results real-time ke Grafana."""
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


@app.post("/predict", response_model=PredictResponse)
def predict(request: BatchPredictRequest):
    if vae_model is None or svdd_model is None or scaler is None:
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
            recon, mu, logvar, z = vae_model(torch.FloatTensor(scaled))

        recon_error = float(np.mean((scaled - recon.numpy()) ** 2))
        z_np = z.numpy()
        svdd_score = svdd_model.decision_function(z_np)[0]
        svdd_dist = float(-svdd_score)
        combined = recon_error + svdd_dist

        threshold = (config.get("best_threshold",
                     config.get("threshold_youden_recon",
                     config.get("threshold", 0.706))) if config else 0.706)
        is_anomaly = recon_error > threshold

        attack_type = "normal"
        if is_anomaly:
            fe = np.mean((scaled - recon.numpy()) ** 2, axis=1)[0]
            attack_type = classify_attack(fe)

        predictions.append(AnomalyResult(
            is_anomaly=is_anomaly,
            recon_error=recon_error,
            svdd_distance=svdd_dist,
            combined_score=combined,
            attack_type=attack_type,
            dominant_feature="",
            anomaly_type_detected=attack_type if is_anomaly else None,
        ))

    return PredictResponse(
        predictions=predictions,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/predict/stream")
def predict_stream(flight: FlightData):
    global vae_model, svdd_model, scaler, config
    if vae_model is None or svdd_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    icao24 = flight.icao24 or "unknown"
    buf = buffers[icao24]

    raw = np.array([
        flight.latitude, flight.longitude, flight.velocity or 0.0,
        flight.baro_altitude or 0.0, flight.true_track or 0.0,
    ], dtype=np.float32)
    buf.append(raw)

    if len(buf) < WINDOW_SIZE:
        return AnomalyResult(
            is_anomaly=False, recon_error=0.0, svdd_distance=0.0,
            combined_score=0.0, attack_type="normal",
            dominant_feature="none",
            anomaly_type_detected=None,
        )

    window_5f = np.array(buf, dtype=np.float32)
    window_10f = compute_derived_features(window_5f)
    scaled_2d = scaler.transform(window_10f)
    scaled = scaled_2d.reshape(1, WINDOW_SIZE, N_FEATURES_ALL)

    with torch.no_grad():
        recon, mu, logvar, z = vae_model(torch.FloatTensor(scaled))

    recon_error = float(np.mean((scaled - recon.numpy()) ** 2))
    z_np = z.numpy()
    svdd_score = svdd_model.decision_function(z_np)[0]
    svdd_dist = float(-svdd_score)
    combined = recon_error + svdd_dist

    # Pakai recon_error (best method: recon_p95)
    threshold = config.get("best_threshold",
                config.get("threshold_youden_recon",
                config.get("threshold", 0.706))) if config else 0.706
    is_anomaly = recon_error > threshold

    attack_type = "normal"
    if is_anomaly:
        fe = np.mean((scaled - recon.numpy()) ** 2, axis=1)[0]
        attack_type = classify_attack(fe)

    result = AnomalyResult(
        is_anomaly=is_anomaly,
        recon_error=recon_error,
        svdd_distance=svdd_dist,
        combined_score=combined,
        attack_type=attack_type,
        dominant_feature="",
        anomaly_type_detected=attack_type if is_anomaly else None,
    )

    # Push ke SSE stream buffer
    anomaly_stream_buffer.append({
        "icao24": flight.icao24,
        "attack_type": attack_type,
        "is_anomaly": is_anomaly,
        "recon_error": recon_error,
        "combined_score": combined,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    if es is not None:
        try:
            doc = {
                "icao24": flight.icao24,
                "callsign": flight.callsign,
                "latitude": flight.latitude,
                "longitude": flight.longitude,
                "velocity": flight.velocity,
                "baro_altitude": flight.baro_altitude,
                "true_track": flight.true_track,
                "window_size": WINDOW_SIZE,
                "is_anomaly": result.is_anomaly,
                "recon_error": result.recon_error,
                "svdd_distance": result.svdd_distance,
                "combined_score": result.combined_score,
                "attack_type": result.attack_type,
                "dominant_feature": result.dominant_feature,
                "anomaly_type_detected": result.anomaly_type_detected,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            es.index(index="anomaly-stream", document=doc)
        except Exception as e:
            print(f"[SERVING] ES save error: {e}")

    return result

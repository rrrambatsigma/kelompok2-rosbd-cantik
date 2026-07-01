"""
api_if.py — FastAPI serving untuk Isolation Forest
Berjalan di port 8083, paralel dengan VAE API di port 8001.
"""

import os
import json
import time
import numpy as np
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "isolation-forest")
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost:9200")
ANOMALY_INDEX = "anomaly-stream-if"

WINDOW_SIZE = 10
FEATURES_BASE = ["latitude", "longitude", "velocity", "baro_altitude", "true_track"]
FEATURES_ALL = FEATURES_BASE + ["dlat", "dlon", "dvel", "dalt", "dtrack"]
N_FEATURES_ALL = len(FEATURES_ALL)
BUFFER_CLEANUP_INTERVAL = int(os.getenv("BUFFER_CLEANUP_INTERVAL", "120"))
BUFFER_TTL = int(os.getenv("BUFFER_TTL", "1800"))

VALID_LAT = (-90, 90)
VALID_LON = (-180, 180)
VALID_VEL = (0, 500)
VALID_ALT = (-1000, 45000)
VALID_TRACK = (0, 360)

app = FastAPI(
    title="Isolation Forest Anomaly Detection API",
    description="Real-time anomaly detection untuk data ADS-B menggunakan Isolation Forest.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if_model = None
scaler = None
config = None
threshold = None
es = None
eval_metrics = None

buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
buffer_last_access = {}
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
            print(f"[SERVING IF] Cleaned {len(stale)} stale buffers")


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
    if_score: float
    attack_type: str
    dominant_feature: str
    anomaly_type_detected: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: List[AnomalyResult]
    model_version: str = "if-1.0.0"
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
            print(f"[SERVING IF] Connected to ES at {ES_HOST}")
            return True
        else:
            print(f"[SERVING IF] ES at {ES_HOST} not reachable")
            es = None
            return False
    except Exception as e:
        print(f"[SERVING IF] ES connection failed: {e}")
        es = None
        return False


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


@app.on_event("startup")
async def on_startup():
    global if_model, scaler, config, threshold, es, eval_metrics

    connect_es()
    loop = asyncio.get_event_loop()
    loop.create_task(cleanup_stale_buffers())

    vae_path = os.path.join(MODEL_DIR, "if_model.pkl")
    if not os.path.exists(vae_path):
        print(f"[SERVING IF] No model at {vae_path}. Run training first.")
        return

    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        threshold = config.get("f1max_threshold", -0.0949)
        print(f"[SERVING IF] Threshold: {threshold:.4f}")

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            eval_metrics = json.load(f)
        print(f"[SERVING IF] Loaded eval metrics: F1={eval_metrics['f1']:.4f}, AUC={eval_metrics['auc']:.4f}")

    try:
        if_model = joblib.load(vae_path)
        print(f"[SERVING IF] IF: {if_model.n_estimators} trees, contamination={if_model.contamination}")
    except Exception as e:
        print(f"[SERVING IF] Failed to load IF: {e}")
        if_model = None
        return

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[SERVING IF] Scaler: {scaler.mean_.shape[0]} features")
    else:
        print(f"[SERVING IF] No scaler at {scaler_path}")
        return

    print(f"[SERVING IF] Loaded from {MODEL_DIR}")


@app.get("/health", response_model=HealthResponse)
def health():
    model_version = (config.get("f1max_threshold", "?") if config else "?")
    return HealthResponse(
        status="ok" if if_model is not None else "no_model",
        model_loaded=if_model is not None,
        model_path=MODEL_DIR,
        es_connected=es is not None and es.ping() if es else False,
        model_version=f"if-{model_version}",
    )


@app.get("/features")
def get_features():
    return {
        "base_features": FEATURES_BASE,
        "derived_features": ["dlat", "dlon", "dvel", "dalt", "dtrack"],
        "all_features": FEATURES_ALL,
        "window_size": WINDOW_SIZE,
        "model_loaded": if_model is not None,
    }


@app.get("/model/info")
def model_info():
    if if_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if config:
        return config
    return {"status": "no config"}


@app.get("/buffer/status")
def buffer_status():
    return {"active_buffers": len(buffers), "buffers": {icao: len(buf) for icao, buf in sorted(buffers.items())}}


@app.get("/stream")
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


@app.post("/predict", response_model=PredictResponse)
def predict(request: BatchPredictRequest):
    global if_model, scaler, threshold
    if if_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    n = len(request.flights)
    if n < WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"Need >= {WINDOW_SIZE} records, got {n}")

    predictions = []
    for i in range(0, n - WINDOW_SIZE + 1):
        batch = request.flights[i:i + WINDOW_SIZE]
        window_5f = np.array([[
            f.latitude, f.longitude, f.velocity or 0.0,
            f.baro_altitude or 0.0, f.true_track or 0.0,
        ] for f in batch], dtype=np.float32)

        window_10f = compute_derived_features(window_5f)
        scaled = scaler.transform(window_10f)
        flat = scaled.reshape(1, -1)
        if_score = -float(if_model.decision_function(flat)[0])
        is_anomaly = if_score > threshold

        attack_type = "normal"
        if is_anomaly:
            per_feature = np.mean((window_10f[:, :5]) ** 2, axis=0)
            attack_type = classify_attack(per_feature)

        predictions.append(AnomalyResult(
            is_anomaly=is_anomaly, if_score=if_score, attack_type=attack_type,
            dominant_feature="", anomaly_type_detected=attack_type if is_anomaly else None,
        ))

    return PredictResponse(predictions=predictions, timestamp=datetime.now(timezone.utc).isoformat())


@app.post("/predict/stream")
def predict_stream(flight: FlightData):
    global if_model, scaler, threshold
    if if_model is None or scaler is None:
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
            "icao24": flight.icao24, "attack_type": "buffering", "is_anomaly": False,
            "if_score": 0.0, "buffered": f"{len(buf)}/{WINDOW_SIZE}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return AnomalyResult(
            is_anomaly=False, if_score=0.0, attack_type="normal",
            dominant_feature="none", anomaly_type_detected=None,
        )

    window_5f = np.array(buf, dtype=np.float32)
    window_10f = compute_derived_features(window_5f)
    scaled = scaler.transform(window_10f)
    flat = scaled.reshape(1, -1)
    if_score = -float(if_model.decision_function(flat)[0])
    is_anomaly = if_score > threshold

    attack_type = "normal"
    if is_anomaly:
        per_feature = np.mean((window_10f[:, :5]) ** 2, axis=0)
        attack_type = classify_attack(per_feature)

    result = AnomalyResult(
        is_anomaly=is_anomaly, if_score=if_score, attack_type=attack_type,
        dominant_feature="", anomaly_type_detected=attack_type if is_anomaly else None,
    )

    anomaly_stream_buffer.append({
        "icao24": flight.icao24, "attack_type": attack_type, "is_anomaly": is_anomaly,
        "if_score": if_score, "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    if es is not None:
        try:
            doc = {
                "icao24": flight.icao24, "callsign": flight.callsign,
                "latitude": lat, "longitude": lon, "velocity": vel,
                "baro_altitude": alt, "true_track": track, "window_size": WINDOW_SIZE,
                "is_anomaly": result.is_anomaly, "if_score": result.if_score,
                "attack_type": result.attack_type, "dominant_feature": result.dominant_feature,
                "anomaly_type_detected": result.anomaly_type_detected, "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            es.index(index=ANOMALY_INDEX, document=doc, request_timeout=2)
        except Exception as e:
            print(f"[SERVING IF] ES save error: {e}")

    return result


# ─── API LEWAT UNTUK LOIS ───

@app.get("/api/stats")
def api_stats():
    try:
        es_local = Elasticsearch(f"http://{ES_HOST}", request_timeout=3)
        total = es_local.count(index=ANOMALY_INDEX)["count"]
        anom = es_local.count(index=ANOMALY_INDEX, body={"query": {"term": {"is_anomaly": True}}})["count"]
        normal = total - anom
        rate = (anom / total * 100) if total > 0 else 0
        return {"total": total, "anomalies": anom, "normal": normal, "rate": round(rate, 2)}
    except Exception as e:
        return {"error": str(e), "total": 0, "anomalies": 0, "normal": 0, "rate": 0}


@app.get("/api/results")
def api_results(limit: int = 50):
    try:
        es_local = Elasticsearch(f"http://{ES_HOST}", request_timeout=3)
        data = es_local.search(
            index=ANOMALY_INDEX,
            body={"sort": [{"timestamp": {"order": "desc"}}], "size": min(limit, 500),
                  "_source": ["icao24", "callsign", "latitude", "longitude",
                              "velocity", "baro_altitude", "true_track",
                              "is_anomaly", "if_score", "attack_type", "timestamp"]}
        )
        return [h["_source"] for h in data["hits"]["hits"]]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/anomalies")
def api_anomalies(type: str = "", limit: int = 50):
    try:
        es_local = Elasticsearch(f"http://{ES_HOST}", request_timeout=3)
        if type:
            body = {"query": {"term": {"attack_type": type}},
                    "sort": [{"timestamp": {"order": "desc"}}], "size": min(limit, 500)}
        else:
            body = {"query": {"term": {"is_anomaly": True}},
                    "sort": [{"timestamp": {"order": "desc"}}], "size": min(limit, 500)}
        data = es_local.search(
            index=ANOMALY_INDEX, body=body,
            _source=["icao24", "callsign", "latitude", "longitude",
                     "velocity", "baro_altitude", "true_track",
                     "is_anomaly", "if_score", "attack_type", "timestamp"]
        )
        return [h["_source"] for h in data["hits"]["hits"]]
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/stream")
async def api_stream():
    async def event_generator():
        seen_ids = set()
        while True:
            try:
                es_local = Elasticsearch(f"http://{ES_HOST}", request_timeout=2)
                body = {
                    "sort": [{"timestamp": {"order": "desc"}}], "size": 5,
                    "_source": ["icao24", "callsign", "latitude", "longitude",
                                "velocity", "baro_altitude", "true_track",
                                "is_anomaly", "if_score", "attack_type", "timestamp"]
                }
                data = es_local.search(index=ANOMALY_INDEX, body=body)
                hits = data["hits"]["hits"]
                for hit in reversed(hits):
                    doc_id = hit["_id"]
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        yield f"data: {json.dumps(hit['_source'])}\n\n"
                if len(seen_ids) > 1000:
                    seen_ids.clear()
            except:
                pass
            await asyncio.sleep(3)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/evaluation")
def api_evaluation():
    global eval_metrics
    if eval_metrics is None:
        raise HTTPException(status_code=404, detail="Evaluation metrics not found. Run training first.")
    cm = eval_metrics["confusion_matrix"]
    total = cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"]
    return {
        "accuracy": eval_metrics["accuracy"],
        "precision": eval_metrics["precision"],
        "recall": eval_metrics["recall"],
        "f1": eval_metrics["f1"],
        "auc": eval_metrics["auc"],
        "threshold": eval_metrics.get("f1max_threshold", eval_metrics.get("best_threshold")),
        "best_method": eval_metrics["best_method"],
        "total_test_windows": total,
        "confusion_matrix": cm,
        "per_attack_recall": eval_metrics["per_attack"],
    }

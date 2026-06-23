import os
import json
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from elasticsearch import Elasticsearch

from modelling.vae_svdd import VAESVDD, FEATURE_NAMES

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "vae-svdd")
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch:9200")

app = FastAPI(title="VAE-SVDD Anomaly Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
es = None


class FlightData(BaseModel):
    icao24: Optional[str] = None
    callsign: Optional[str] = None
    longitude: float
    latitude: float
    velocity: Optional[float] = 0.0
    geo_altitude: Optional[float] = 0.0
    true_track: Optional[float] = 0.0
    vertical_rate: Optional[float] = 0.0


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
    model_version: str = "1.0.0"
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


@app.on_event("startup")
def load_model():
    global model, es
    es = Elasticsearch(f"http://{ES_HOST}")

    vae_path = os.path.join(MODEL_DIR, "vae.pt")
    if os.path.exists(vae_path):
        model = VAESVDD()
        try:
            model.load(MODEL_DIR)
            print(f"[SERVING] Model loaded from {MODEL_DIR}")
        except Exception as e:
            print(f"[SERVING] Failed to load model: {e}")
            model = None
    else:
        print(f"[SERVING] No model found at {MODEL_DIR}. Run train.py first.")
        model = None


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if model is not None else "no_model",
        model_loaded=model is not None,
        model_path=MODEL_DIR,
    )


@app.get("/features")
def get_features():
    return {
        "features": FEATURE_NAMES,
        "model_loaded": model is not None,
    }


@app.get("/model/info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {"status": "no config"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array([[
        f.longitude, f.latitude, f.velocity or 0.0,
        f.geo_altitude or 0.0, f.true_track or 0.0, f.vertical_rate or 0.0,
    ] for f in request.flights], dtype=np.float32)

    results = model.predict(features)

    predictions = [
        AnomalyResult(
            is_anomaly=results["is_anomaly"][i],
            recon_error=results["recon_error"][i],
            svdd_distance=results["svdd_distance"][i],
            combined_score=results["combined_score"][i],
            attack_type=results["attack_type"][i],
            dominant_feature=results["dominant_feature"][i],
            anomaly_type_detected=results["attack_type"][i] if results["is_anomaly"][i] else None,
        )
        for i in range(len(request.flights))
    ]

    return PredictResponse(
        predictions=predictions,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/predict/stream")
def predict_stream(flight: FlightData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array([[
        flight.longitude, flight.latitude, flight.velocity or 0.0,
        flight.geo_altitude or 0.0, flight.true_track or 0.0, flight.vertical_rate or 0.0,
    ]], dtype=np.float32)

    results = model.predict(features)

    result = AnomalyResult(
        is_anomaly=results["is_anomaly"][0],
        recon_error=results["recon_error"][0],
        svdd_distance=results["svdd_distance"][0],
        combined_score=results["combined_score"][0],
        attack_type=results["attack_type"][0],
        dominant_feature=results["dominant_feature"][0],
        anomaly_type_detected=results["attack_type"][0] if results["is_anomaly"][0] else None,
    )

    try:
        doc = {
            "icao24": flight.icao24,
            "callsign": flight.callsign,
            "longitude": flight.longitude,
            "latitude": flight.latitude,
            "velocity": flight.velocity,
            "is_anomaly": result.is_anomaly,
            "recon_error": result.recon_error,
            "svdd_distance": result.svdd_distance,
            "combined_score": result.combined_score,
            "attack_type": result.attack_type,
            "anomaly_type_detected": result.anomaly_type_detected,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        es.index(index="anomaly-stream", document=doc)
    except Exception as e:
        print(f"[SERVING] Failed to save stream result: {e}")

    return result

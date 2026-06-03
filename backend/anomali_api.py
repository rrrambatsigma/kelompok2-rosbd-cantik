import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="SVDD Anomali Detection API")

class FlightData(BaseModel):
    features: List[float]

model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "/app/models/svdd_model.pkl")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except:
        model = None

@app.post("/detect")
async def detect(data: FlightData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = np.array(data.features).reshape(1, -1)
    # Asumsikan model SVDD memiliki method predict yang mengembalikan 1 (normal) atau -1 (anomali)
    result = model.predict(features)[0]
    is_anomaly = result == -1
    return {"is_anomaly": is_anomaly, "score": float(result)}
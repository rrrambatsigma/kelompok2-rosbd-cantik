import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="XGBoost Optimasi Rute API")

class RouteData(BaseModel):
    features: List[float]

model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "/app/models/optimasi_model.pkl")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except:
        model = None

@app.post("/optimize")
async def optimize(data: RouteData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"optimal_route_score": float(prediction)}
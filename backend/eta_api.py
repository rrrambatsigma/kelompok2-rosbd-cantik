import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="XGBoost ETA API", description="Prediksi Estimated Time of Arrival")

class FeatureData(BaseModel):
    features: List[float]

model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "/app/models/eta_model.pkl")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.post("/predict")
async def predict(data: FeatureData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return {"eta_prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
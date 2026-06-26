import os
from datetime import datetime
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from elasticsearch import Elasticsearch, NotFoundError

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# ES_HOST = os.getenv("ES_HOST", "http://100.99.130.69:9200")
ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")
ES_INDEX = "flight_predictions"

es = None

app = FastAPI(title="ETA Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PYDANTIC MODELS
# =========================

class CurrentPosition(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    altitude: Optional[float] = None
    heading: Optional[float] = None
    speed_kmh: Optional[float] = None


class PredictionResult(BaseModel):
    icao24: str
    callsign: Optional[str] = None
    destination: Optional[str] = None
    prediction_method: Optional[str] = None
    confidence: Optional[float] = None
    eta_seconds: Optional[int] = None
    eta_minutes: Optional[float] = None
    eta_method: Optional[str] = None
    distance_km_to_dest: Optional[float] = None
    track_points: Optional[int] = None
    current_position: Optional[CurrentPosition] = None
    status: str
    predicted_at: Optional[str] = None
    last_contact: Optional[int] = None
    on_ground: Optional[bool] = None


class PredictRequest(BaseModel):
    icao24: str


class PredictResponse(BaseModel):
    prediction: PredictionResult
    timestamp: str


class PredictionsListResponse(BaseModel):
    total: int
    predictions: List[PredictionResult]


class StatsResponse(BaseModel):
    total_predictions: int
    active_flights: int
    landed_flights: int
    failed_predictions: int
    top_destinations: List[Dict]
    top_methods: List[Dict]
    avg_confidence: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    es_connected: bool
    es_index: str
    total_predictions: int
    timestamp: str


# =========================
# STARTUP
# =========================

@app.on_event("startup")
def startup():
    global es
    try:
        es = Elasticsearch(ES_HOST)
        es.info()
    except Exception as e:
        print(f"[ETA API] ES connection failed: {e}")
        es = None


# =========================
# ENDPOINTS
# =========================

@app.get("/health", response_model=HealthResponse)
def health():
    total = 0
    es_ok = es is not None
    if es_ok:
        try:
            total = es.count(index=ES_INDEX)["count"]
        except Exception:
            es_ok = False

    return HealthResponse(
        status="ok" if es_ok else "es_down",
        es_connected=es_ok,
        es_index=ES_INDEX,
        total_predictions=total,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/eta/predictions", response_model=PredictionsListResponse)
def list_predictions(
    status: Optional[str] = Query(None, description="Filter by status (ok, failed, no_data)"),
    destination: Optional[str] = Query(None, description="Filter by destination airport ICAO"),
    method: Optional[str] = Query(None, description="Filter by prediction method (ml_classifier, callsign, heading_scoring)"),
    limit: int = Query(20, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Result offset for pagination"),
):
    if es is None:
        raise HTTPException(status_code=503, detail="Elasticsearch not connected")

    must = []
    if status:
        must.append({"term": {"status": status}})
    if destination:
        must.append({"term": {"destination": destination}})
    if method:
        must.append({"term": {"prediction_method": method}})

    body = {
        "query": {"bool": {"must": must}} if must else {"match_all": {}},
        "sort": [{"predicted_at": {"order": "desc"}}],
        "from": offset,
        "size": limit,
    }

    try:
        total = es.count(index=ES_INDEX, body={"query": body["query"]})["count"]
        res = es.search(index=ES_INDEX, body=body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    predictions = [format_hit(h) for h in res["hits"]["hits"]]

    return PredictionsListResponse(total=total, predictions=predictions)


@app.get("/eta/predictions.geojson")
def list_predictions_geojson(
    status: Optional[str] = Query(None, description="Filter by status"),
    destination: Optional[str] = Query(None, description="Filter by destination airport ICAO"),
    method: Optional[str] = Query(None, description="Filter by prediction method"),
    limit: int = Query(200, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Result offset"),
):
    if es is None:
        raise HTTPException(status_code=503, detail="Elasticsearch not connected")

    must = []
    if status:
        must.append({"term": {"status": status}})
    if destination:
        must.append({"term": {"destination": destination}})
    if method:
        must.append({"term": {"prediction_method": method}})

    body = {
        "query": {"bool": {"must": must}} if must else {"match_all": {}},
        "sort": [{"predicted_at": {"order": "desc"}}],
        "from": offset,
        "size": limit,
    }

    try:
        res = es.search(index=ES_INDEX, body=body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    features = []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        cp = src.get("current_position")
        if cp and isinstance(cp, dict) and cp.get("lon") is not None and cp.get("lat") is not None:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [cp["lon"], cp["lat"]],
                },
                "properties": {
                    "icao24": src.get("icao24"),
                    "callsign": src.get("callsign"),
                    "destination": src.get("destination"),
                    "prediction_method": src.get("prediction_method"),
                    "confidence": src.get("confidence"),
                    "eta_seconds": src.get("eta_seconds"),
                    "eta_minutes": src.get("eta_minutes"),
                    "eta_method": src.get("eta_method"),
                    "distance_km_to_dest": src.get("distance_km_to_dest"),
                    "altitude": cp.get("altitude"),
                    "heading": cp.get("heading"),
                    "speed_kmh": cp.get("speed_kmh"),
                    "track_points": src.get("track_points"),
                    "status": src.get("status"),
                    "predicted_at": src.get("predicted_at"),
                },
            }
            features.append(feature)

    return {"type": "FeatureCollection", "features": features}


@app.get("/eta/predictions/{icao24}", response_model=PredictionResult)
def get_prediction(icao24: str):
    if es is None:
        raise HTTPException(status_code=503, detail="Elasticsearch not connected")

    try:
        res = es.get(index=ES_INDEX, id=icao24)
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"No prediction found for icao24={icao24}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return format_hit(res)


@app.post("/eta/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    try:
        from eta_pipeline import predict
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Pipeline not available: {e}")

    try:
        result = predict(req.icao24)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    result["predicted_at"] = datetime.utcnow().isoformat() + "Z"

    return PredictResponse(
        prediction=format_raw(result),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/eta/stats", response_model=StatsResponse)
def stats():
    if es is None:
        raise HTTPException(status_code=503, detail="Elasticsearch not connected")

    try:
        total = es.count(index=ES_INDEX)["count"]

        active = es.count(index=ES_INDEX, body={"query": {"term": {"status": "ok"}}})["count"]
        landed = es.count(index=ES_INDEX, body={"query": {"term": {"status": "landed"}}})["count"]
        failed = es.count(index=ES_INDEX, body={"query": {"term": {"status": "failed"}}})["count"]

        dest_agg = es.search(index=ES_INDEX, body={
            "size": 0,
            "aggs": {
                "top_dest": {
                    "terms": {"field": "destination", "size": 10, "order": {"_count": "desc"}}
                }
            }
        })
        top_dest = [
            {"destination": b["key"], "count": b["doc_count"]}
            for b in dest_agg["aggregations"]["top_dest"]["buckets"]
        ]

        method_agg = es.search(index=ES_INDEX, body={
            "size": 0,
            "aggs": {
                "top_method": {
                    "terms": {"field": "prediction_method", "size": 10, "order": {"_count": "desc"}}
                }
            }
        })
        top_method = [
            {"method": b["key"], "count": b["doc_count"]}
            for b in method_agg["aggregations"]["top_method"]["buckets"]
        ]

        conf_agg = es.search(index=ES_INDEX, body={
            "size": 0,
            "query": {"term": {"status": "ok"}},
            "aggs": {
                "avg_conf": {"avg": {"field": "confidence"}}
            }
        })
        avg_conf = conf_agg["aggregations"]["avg_conf"]["value"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StatsResponse(
        total_predictions=total,
        active_flights=active,
        landed_flights=landed,
        failed_predictions=failed,
        top_destinations=top_dest,
        top_methods=top_method,
        avg_confidence=round(avg_conf, 4) if avg_conf else None,
    )


# =========================
# HELPERS
# =========================

def format_hit(hit) -> PredictionResult:
    src = hit["_source"]
    return format_raw(src)


def format_raw(src: dict) -> PredictionResult:
    cp = src.get("current_position")
    current_position = None
    if cp and isinstance(cp, dict):
        current_position = CurrentPosition(
            lat=cp.get("lat"),
            lon=cp.get("lon"),
            altitude=cp.get("altitude"),
            heading=cp.get("heading"),
            speed_kmh=cp.get("speed_kmh"),
        )

    return PredictionResult(
        icao24=src.get("icao24", ""),
        callsign=src.get("callsign"),
        destination=src.get("destination"),
        prediction_method=src.get("prediction_method"),
        confidence=src.get("confidence"),
        eta_seconds=src.get("eta_seconds"),
        eta_minutes=src.get("eta_minutes"),
        eta_method=src.get("eta_method"),
        distance_km_to_dest=src.get("distance_km_to_dest"),
        track_points=src.get("track_points"),
        current_position=current_position,
        status=src.get("status", "unknown"),
        predicted_at=src.get("predicted_at"),
        last_contact=src.get("last_contact"),
        on_ground=src.get("on_ground"),
    )

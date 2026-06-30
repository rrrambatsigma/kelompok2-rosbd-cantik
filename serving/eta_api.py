import os
import math
from datetime import datetime
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from elasticsearch import Elasticsearch, NotFoundError

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# ES_HOST = os.getenv("ES_HOST", "http://100.99.130.69:9200")
ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")
ES_INDEX = "flight_predictions"
HISTORY_INDEX = "flight_predictions_history"

es = None
route_durations = {}
airport_coords = {}

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
    route: Optional[str] = None
    origin: Optional[str] = None
    origin_method: Optional[str] = None
    origin_confidence: Optional[float] = None


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


class AircraftSummary(BaseModel):
    icao24: str
    callsign: Optional[str] = None
    destination: Optional[str] = None
    last_seen: Optional[float] = None
    last_eta: Optional[float] = None
    last_distance: Optional[float] = None
    last_altitude: Optional[float] = None
    last_speed: Optional[float] = None
    last_status: Optional[str] = None
    total_entries: int = 0


class AircraftListResponse(BaseModel):
    total: int
    hours: int
    aircraft: List[AircraftSummary]


class LandingEvent(BaseModel):
    icao24: str
    callsign: Optional[str] = None
    destination: Optional[str] = None
    landed_at: Optional[float] = None
    last_eta: Optional[float] = None
    route: Optional[str] = None
    method: str = "missing_from_active"


class LandedResponse(BaseModel):
    total: int
    landings: List[LandingEvent]


class DelayInfo(BaseModel):
    icao24: str
    callsign: Optional[str] = None
    delay_minutes: Optional[float] = None
    eta_actual: Optional[float] = None
    eta_ideal: Optional[float] = None
    status: str = "unknown"


class DelaysResponse(BaseModel):
    total: int
    delays: List[DelayInfo]


# =========================
# STARTUP
# =========================

@app.on_event("startup")
def startup():
    global es, route_durations, airport_coords
    try:
        es = Elasticsearch(ES_HOST)
        es.info()
    except Exception as e:
        print(f"[ETA API] ES connection failed: {e}")
        es = None

    try:
        route_path = os.path.join(BASE_DIR, "data", "final", "route_avg_duration_clean.csv")
        airport_path = os.path.join(BASE_DIR, "data", "final", "airport_lookup.csv")
        route_df = pd.read_csv(route_path)
        route_durations = dict(zip(route_df["route"], route_df["avg_duration_sec"]))
        ap_df = pd.read_csv(airport_path)
        airport_coords = dict(zip(ap_df["icao"], list(zip(ap_df["lat"], ap_df["lon"]))))
        print(f"[ETA API] Loaded {len(route_durations)} routes, {len(airport_coords)} airports")
    except Exception as e:
        print(f"[ETA API] Route data load failed: {e}")
        route_durations = {}
        airport_coords = {}


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
                    "last_contact": src.get("last_contact"),
                    "route": src.get("route"),
                    "origin": src.get("origin"),
                    "origin_method": src.get("origin_method"),
                    "origin_confidence": src.get("origin_confidence"),
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
                    "terms": {"field": "destination.keyword", "size": 10, "order": {"_count": "desc"}}
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
                    "terms": {"field": "prediction_method.keyword", "size": 10, "order": {"_count": "desc"}}
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


@app.get("/eta/history/aircraft", response_model=AircraftListResponse)
def list_aircraft_history(
    hours: int = Query(24, ge=1, le=168, description="Lookback hours"),
    limit: int = Query(200, ge=1, le=1000, description="Max aircraft"),
):
    if es is None:
        raise HTTPException(status_code=503, detail="Elasticsearch not connected")

    cutoff = datetime.utcnow().timestamp() - hours * 3600
    body = {
        "size": 0,
        "query": {"range": {"recorded_at": {"gte": cutoff}}},
        "aggs": {
            "aircraft": {
                "terms": {
                    "field": "icao24.keyword",
                    "size": limit,
                    "order": {"max_seen": "desc"}
                },
                "aggs": {
                    "max_seen": {"max": {"field": "recorded_at"}},
                    "entry_count": {"value_count": {"field": "recorded_at"}},
                    "latest": {
                        "top_hits": {
                            "size": 1,
                            "sort": [{"recorded_at": {"order": "desc"}}],
                            "_source": ["callsign", "destination", "eta_minutes",
                                        "distance_km_to_dest", "current_position", "status", "route"]
                        }
                    }
                }
            }
        }
    }

    try:
        res = es.search(index=HISTORY_INDEX, body=body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    aircraft = []
    for bucket in res["aggregations"]["aircraft"]["buckets"]:
        latest_src = bucket["latest"]["hits"]["hits"][0]["_source"]
        cp = latest_src.get("current_position", {})
        alt = cp.get("altitude") if isinstance(cp, dict) else None
        spd = cp.get("speed_kmh") if isinstance(cp, dict) else None
        aircraft.append(AircraftSummary(
            icao24=bucket["key"],
            callsign=latest_src.get("callsign"),
            destination=latest_src.get("destination"),
            last_seen=bucket["max_seen"]["value"],
            last_eta=latest_src.get("eta_minutes"),
            last_distance=latest_src.get("distance_km_to_dest"),
            last_altitude=alt,
            last_speed=spd,
            last_status=latest_src.get("status"),
            total_entries=int(bucket["entry_count"]["value"]),
        ))

    return AircraftListResponse(total=len(aircraft), hours=hours, aircraft=aircraft)


@app.get("/eta/history/landed", response_model=LandedResponse)
def list_landed(
    hours: int = Query(24, ge=1, le=168, description="Lookback hours"),
    min_altitude_ft: int = Query(500, description="Max altitude (ft) to consider landed"),
    max_speed_kmh: int = Query(100, description="Max speed (kmh) to consider landed"),
    max_distance_km: int = Query(10, description="Max distance (km) to dest to consider landed"),
):
    if es is None:
        raise HTTPException(status_code=503, detail="Elasticsearch not connected")

    cutoff = datetime.utcnow().timestamp() - hours * 3600

    try:
        active_res = es.search(index=ES_INDEX, body={
            "size": 0,
            "query": {"term": {"status": "ok"}},
            "aggs": {
                "active_icaos": {
                    "terms": {"field": "icao24.keyword", "size": 1000}
                }
            }
        })
        active_icaos = {b["key"] for b in active_res["aggregations"]["active_icaos"]["buckets"]}

        hist_res = es.search(index=HISTORY_INDEX, body={
            "size": 0,
            "query": {"range": {"recorded_at": {"gte": cutoff}}},
            "aggs": {
                "aircraft": {
                    "terms": {"field": "icao24.keyword", "size": 1000},
                    "aggs": {
                        "max_seen": {"max": {"field": "recorded_at"}},
                        "latest": {
                            "top_hits": {
                                "size": 1,
                                "sort": [{"recorded_at": {"order": "desc"}}],
                                "_source": True
                            }
                        }
                    }
                }
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    landings = []
    for bucket in hist_res["aggregations"]["aircraft"]["buckets"]:
        icao = bucket["key"]
        latest = bucket["latest"]["hits"]["hits"][0]["_source"]
        cp = latest.get("current_position", {})
        alt = cp.get("altitude", 9999) if isinstance(cp, dict) else 9999
        spd = cp.get("speed_kmh", 999) if isinstance(cp, dict) else 999
        dist = latest.get("distance_km_to_dest", 999)

        landed_at = bucket["max_seen"]["value"]

        if icao not in active_icaos:
            landings.append(LandingEvent(
                icao24=icao,
                callsign=latest.get("callsign"),
                destination=latest.get("destination"),
                landed_at=landed_at,
                last_eta=latest.get("eta_minutes"),
                route=latest.get("route"),
                method="missing_from_active",
            ))
            continue

        if (alt is not None and alt < min_altitude_ft and
            spd is not None and spd < max_speed_kmh and
            dist is not None and dist < max_distance_km):
            landings.append(LandingEvent(
                icao24=icao,
                callsign=latest.get("callsign"),
                destination=latest.get("destination"),
                landed_at=landed_at,
                last_eta=latest.get("eta_minutes"),
                route=latest.get("route"),
                method="low_alt_speed_distance",
            ))

    return LandedResponse(total=len(landings), landings=landings)


@app.get("/eta/history/delays", response_model=DelaysResponse)
def list_delays(
    hours: int = Query(24, ge=1, le=168, description="Lookback hours"),
    min_entries: int = Query(2, ge=1, le=100, description="Min history entries for delay calc"),
):
    if es is None:
        raise HTTPException(status_code=503, detail="Elasticsearch not connected")

    cutoff = datetime.utcnow().timestamp() - hours * 3600

    try:
        body = {
            "size": 0,
            "query": {"range": {"recorded_at": {"gte": cutoff}}},
            "aggs": {
                "aircraft": {
                    "terms": {"field": "icao24.keyword", "size": 500},
                    "aggs": {
                        "entry_count": {"value_count": {"field": "recorded_at"}},
                        "entries_by_time": {
                            "date_histogram": {
                                "field": "recorded_at",
                                "interval": "5m",
                                "order": {"_key": "desc"},
                                "min_doc_count": 1
                            },
                            "aggs": {
                                "latest_entry": {
                                    "top_hits": {
                                        "size": 1,
                                        "sort": [{"recorded_at": {"order": "desc"}}],
                                        "_source": ["callsign", "destination", "eta_minutes",
                                                    "distance_km_to_dest", "route", "recorded_at",
                                                    "current_position", "origin"]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        res = es.search(index=HISTORY_INDEX, body=body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    delays = []
    for bucket in res["aggregations"]["aircraft"]["buckets"]:
        entry_count = int(bucket["entry_count"]["value"])
        if entry_count < min_entries:
            continue

        time_buckets = bucket["entries_by_time"]["buckets"]
        if len(time_buckets) < 2:
            continue

        latest = time_buckets[0]["latest_entry"]["hits"]["hits"][0]["_source"]
        previous = time_buckets[1]["latest_entry"]["hits"]["hits"][0]["_source"]

        icao = bucket["key"]
        callsign = latest.get("callsign")
        eta_real = latest.get("eta_minutes")
        distance = latest.get("distance_km_to_dest")
        route = latest.get("route") or previous.get("route")

        eta_ideal = None
        if route and route in route_durations:
            avg_dur_sec = route_durations[route]
            avg_dur_min = avg_dur_sec / 60

            origin = latest.get("origin") or previous.get("origin")
            dest = latest.get("destination")
            total_dist = None
            if origin and dest and origin in airport_coords and dest in airport_coords:
                olat, olon = airport_coords[origin]
                dlat, dlon = airport_coords[dest]
                total_dist = haversine(olat, olon, dlat, dlon)

            if total_dist and total_dist > 0 and distance is not None and distance > 0:
                progress = 1 - (distance / total_dist)
                progress = max(0, min(0.99, progress))
                eta_ideal = avg_dur_min * (1 - progress)
            else:
                eta_ideal = round(avg_dur_min * 0.3, 1)

        delay_val = None
        status = "unknown"
        if eta_real is not None and eta_ideal is not None:
            delay_val = round(eta_real - eta_ideal, 1)
            if delay_val > 5:
                status = "delay"
            elif delay_val < -5:
                status = "ahead"
            else:
                status = "on_time"

        delays.append(DelayInfo(
            icao24=icao,
            callsign=callsign,
            delay_minutes=delay_val,
            eta_actual=eta_real,
            eta_ideal=round(eta_ideal, 1) if eta_ideal else None,
            status=status,
        ))

    delays.sort(key=lambda d: -(d.delay_minutes or 0))
    return DelaysResponse(total=len(delays), delays=delays)


@app.get("/eta/history/{icao24}")
def get_history(icao24: str, hours: int = Query(3, ge=1, le=72, description="Lookback hours"),
                limit: int = Query(500, ge=1, le=5000, description="Max results")):
    if es is None:
        raise HTTPException(status_code=503, detail="Elasticsearch not connected")

    cutoff = datetime.utcnow().timestamp() - hours * 3600
    body = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"icao24": icao24}},
                    {"range": {"recorded_at": {"gte": cutoff}}}
                ]
            }
        },
        "sort": [{"recorded_at": {"order": "asc"}}],
        "size": limit,
    }

    try:
        total = es.count(index="flight_predictions_history", body={"query": body["query"]})["count"]
        res = es.search(index="flight_predictions_history", body=body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history = []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        history.append({
            "recorded_at": src.get("recorded_at"),
            "predicted_at": src.get("predicted_at"),
            "destination": src.get("destination"),
            "eta_minutes": src.get("eta_minutes"),
            "eta_seconds": src.get("eta_seconds"),
            "confidence": src.get("confidence"),
            "prediction_method": src.get("prediction_method"),
            "eta_method": src.get("eta_method"),
            "distance_km_to_dest": src.get("distance_km_to_dest"),
            "track_points": src.get("track_points"),
            "status": src.get("status"),
            "current_position": src.get("current_position"),
        })

    return {"icao24": icao24, "total": total, "hours": hours, "history": history}


# =========================
# HELPERS
# =========================

R_EARTH = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R_EARTH * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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
        route=src.get("route"),
        origin=src.get("origin"),
        origin_method=src.get("origin_method"),
        origin_confidence=src.get("origin_confidence"),
    )

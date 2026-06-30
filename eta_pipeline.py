import pandas as pd
import numpy as np
import math
import joblib
import os
import sys
from datetime import datetime

from elasticsearch import Elasticsearch

# =========================
# CONFIG
# =========================
# ES_HOST = os.getenv("ES_HOST", "http://100.99.130.69:9200")
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = "flights"

AIRPORT_FILE = "data/final/airport_lookup.csv"
CALLSIGN_CONFIDENCE_FILE = "data/final/callsign_route_confidence.csv"
ROUTE_AVG_DURATION_FILE = "data/final/route_avg_duration_clean.csv"
CLASSIFIER_MODEL_FILE = "models/destination_classifier.pkl"
DEST_ENCODER_FILE = "models/destination_encoder.pkl"
ETA_MODEL_FILE = "models/eta_xgboost_balanced.pkl"
ROUTE_ENCODER_FILE = "models/route_encoder.pkl"

R = 6371.0

# =========================
# LOAD DATA
# =========================
print("Loading pipeline resources...")

airports = pd.read_csv(AIRPORT_FILE)
airport_dict = dict(zip(airports["icao"], list(zip(airports["lat"], airports["lon"]))))
airport_icaos = list(airport_dict.keys())
airport_lats = np.array([airport_dict[ap][0] for ap in airport_icaos])
airport_lons = np.array([airport_dict[ap][1] for ap in airport_icaos])

callsign_lookup = pd.read_csv(CALLSIGN_CONFIDENCE_FILE)
route_avg_dur = pd.read_csv(ROUTE_AVG_DURATION_FILE)
route_dur_dict = dict(zip(route_avg_dur["route"], route_avg_dur["avg_duration_sec"]))

clf = joblib.load(CLASSIFIER_MODEL_FILE)
dest_encoder = joblib.load(DEST_ENCODER_FILE)

eta_model = joblib.load(ETA_MODEL_FILE)
route_encoder = joblib.load(ROUTE_ENCODER_FILE)

known_routes = set(route_encoder.classes_)

try:
    es = Elasticsearch(ES_HOST)
    es.info()
    print("ES connected:", ES_HOST)
    es_available = True
except Exception:
    print("ES not available")
    es = None
    es_available = False

print("Pipeline ready.")

# =========================
# HELPERS
# =========================

def haversine(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing(lat1, lon1, lat2, lon2):
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)
    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))
    b = math.degrees(math.atan2(x, y))
    return (b + 360) % 360

def heading_diff(h1, h2):
    d = abs(h1 - h2)
    return min(d, 360 - d)

def haversine_vec(lat1, lon1, lat2, lon2):
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# =========================
# 1. GET TRAJECTORY FROM ES
# =========================

def get_trajectory(icao24, size=100):
    if not es_available or es is None:
        return pd.DataFrame()

    query = {
        "query": {"term": {"icao24.keyword": icao24}},
        "sort": [{"timestamp": {"order": "desc"}}],
        "size": size
    }
    try:
        result = es.search(index=INDEX_NAME, body=query)
    except Exception:
        return pd.DataFrame()

    rows = []
    for hit in result["hits"]["hits"]:
        src = hit["_source"]
        rows.append({
            "timestamp": src.get("timestamp"),
            "latitude": src.get("latitude"),
            "longitude": src.get("longitude"),
            "altitude": src.get("baro_altitude") or src.get("geo_altitude"),
            "velocity_kmh": src.get("velocity_kmh"),
            "heading": src.get("true_track"),
            "callsign": src.get("callsign"),
            "on_ground": src.get("on_ground")
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


# =========================
# 2. CALLSIGN LOOKUP
# =========================

def predict_by_callsign(callsign, min_confidence=0.95):
    if not callsign:
        return None
    match = callsign_lookup[callsign_lookup["callsign"] == callsign.strip()]
    if len(match) == 0:
        return None
    best = match.iloc[0]
    if best["confidence"] < min_confidence:
        return None
    route = best["route"]
    destination = route.split("_")[-1]
    return {
        "destination": destination,
        "route": route,
        "confidence": best["confidence"],
        "method": "callsign"
    }


# =========================
# 3. ML CLASSIFIER
# =========================

def predict_by_ml(lat, lon, alt, hdg):
    dists = [haversine(lat, lon, airport_lats[i], airport_lons[i])
             for i in range(len(airport_lats))]
    nearest_5_idx = np.argsort(dists)[:5]

    features = {"latitude": lat, "longitude": lon, "altitude": alt, "heading": hdg}
    for i, idx in enumerate(nearest_5_idx):
        features[f"dist_ap_{i}"] = dists[idx]
        features[f"bearing_ap_{i}"] = bearing(lat, lon, airport_lats[idx], airport_lons[idx])

    hd = heading_diff(hdg, features["bearing_ap_0"])
    features["heading_diff_0"] = hd

    feat_order = [
        "latitude", "longitude", "altitude", "heading",
        "dist_ap_0", "bearing_ap_0", "dist_ap_1", "bearing_ap_1",
        "dist_ap_2", "bearing_ap_2", "dist_ap_3", "bearing_ap_3",
        "dist_ap_4", "bearing_ap_4", "heading_diff_0"
    ]
    feat_df = pd.DataFrame([features])[feat_order]

    proba = clf.predict_proba(feat_df.values)[0]
    max_prob = proba.max()
    if max_prob < 0.3:
        return None
    best_idx = np.argmax(proba)
    destination = dest_encoder.classes_[best_idx]
    return {
        "destination": destination,
        "confidence": max_prob,
        "method": "ml_classifier"
    }


# =========================
# 4. HEADING-BASED SCORING
# =========================

def predict_by_heading(lat, lon, hdg, alt=None):
    results = []
    for ap_icao in airport_dict:
        ap_lat, ap_lon = airport_dict[ap_icao]
        d = haversine(lat, lon, ap_lat, ap_lon)
        b = bearing(lat, lon, ap_lat, ap_lon)
        hd = heading_diff(hdg, b)

        heading_score = max(0, 1 - hd / 180)
        distance_score = 1 / (1 + d / 500)
        alt_score = min(1, (alt or 5000) / 10000)

        score = 0.5 * heading_score + 0.3 * distance_score + 0.2 * alt_score

        results.append({
            "airport": ap_icao,
            "distance_km": round(d, 1),
            "heading_diff": round(hd, 1),
            "score": round(score, 4)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]


# =========================
# 5. ETA CALCULATION
# =========================

def predict_eta(lat, lon, destination, speed_kmh, alt, hdg, elapsed_seconds=None):
    ap_lat, ap_lon = airport_dict[destination]
    dist_km = haversine(lat, lon, ap_lat, ap_lon)

    # Method A: distance / speed (primary for real-time)
    eta_simple = (dist_km / speed_kmh) * 3600 if speed_kmh and speed_kmh > 0 else None

    # Method B: route avg duration (for long-range flights)
    eta_route = None
    route_found = None
    for route, avg_dur in route_dur_dict.items():
        if route.endswith(f"_{destination}"):
            route_found = route
            if elapsed_seconds is not None and elapsed_seconds > 120:
                remaining = avg_dur - elapsed_seconds
                if 600 < remaining < avg_dur:
                    eta_route = remaining
            break

    # Method C: XGBoost (only when we can estimate progress)
    eta_xgb = None
    if route_found and route_found in known_routes and speed_kmh > 100:
        total_dur = route_dur_dict.get(route_found, 3600)
        if elapsed_seconds and elapsed_seconds > 120:
            elap = elapsed_seconds
            prog = min(0.95, elap / total_dur)
        else:
            elap = eta_simple or 1800
            prog = 0.5
        route_enc = route_encoder.transform([route_found])[0]
        feat = np.array([[lat, lon, alt, hdg, elap, prog, route_enc]])
        try:
            pred = eta_model.predict(feat)[0]
            if 120 < pred < total_dur:
                eta_xgb = pred
        except Exception:
            pass

    # Pick best method (prefer simpler when close)
    if dist_km < 200 and eta_simple:
        best_eta = eta_simple
        method = "distance_speed"
    elif eta_xgb:
        best_eta = eta_xgb
        method = "xgboost"
    elif eta_route:
        best_eta = eta_route
        method = "route_avg"
    elif eta_simple:
        best_eta = eta_simple
        method = "distance_speed"
    else:
        best_eta = None
        method = "none"

    return {
        "distance_km_to_dest": round(dist_km, 1),
        "eta_seconds": round(best_eta) if best_eta else None,
        "eta_minutes": round(best_eta / 60, 1) if best_eta else None,
        "eta_method": method,
    }


# =========================
# MAIN PREDICT
# =========================

def to_native(obj):
    if obj is None:
        return None
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def deep_convert(d):
    if isinstance(d, dict):
        return {k: deep_convert(v) for k, v in d.items()}
    if isinstance(d, list):
        return [deep_convert(v) for v in d]
    return to_native(d)

def predict(icao24, track=None):
    result = {"icao24": icao24, "status": "ok"}

    # Step 1: Get trajectory
    if track is not None:
        traj = track
    else:
        traj = get_trajectory(icao24)

    if len(traj) == 0:
        result["status"] = "no_data"
        result["error"] = "No trajectory data available"
        return result

    latest = traj.iloc[-1]
    result["track_points"] = len(traj)
    result["callsign"] = latest.get("callsign")

    lat = latest["latitude"]
    lon = latest["longitude"]
    alt = latest.get("altitude") or 0
    hdg = latest.get("heading") or 0
    speed = latest.get("velocity_kmh") or 0

    result["current_position"] = {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "altitude": round(alt, 0),
        "heading": round(hdg, 0),
        "speed_kmh": round(speed, 0)
    }

    destination = None
    method = None
    confidence = 0

    # Step 2: Callsign lookup
    callsign = result.get("callsign")
    if callsign:
        cs_result = predict_by_callsign(callsign)
        if cs_result:
            destination = cs_result["destination"]
            method = cs_result["method"]
            confidence = cs_result["confidence"]
            result["route"] = cs_result["route"]

    # Step 3: ML classifier
    if not destination and alt > 100:
        ml_result = predict_by_ml(lat, lon, alt, hdg)
        if ml_result:
            destination = ml_result["destination"]
            method = ml_result["method"]
            confidence = ml_result["confidence"]

    # Step 4: Heading scoring fallback
    if not destination:
        heading_results = predict_by_heading(lat, lon, hdg, alt)
        if heading_results:
            destination = heading_results[0]["airport"]
            method = "heading_scoring"
            confidence = heading_results[0]["score"]

    result["destination"] = destination
    result["prediction_method"] = method
    result["confidence"] = round(confidence, 4)

    if not destination:
        result["status"] = "failed"
        result["error"] = "Could not predict destination"
        return result

    # Step 5: ETA
    elapsed = None
    if len(traj) >= 2:
        t0 = traj["timestamp"].iloc[0]
        t1 = traj["timestamp"].iloc[-1]
        elapsed = t1 - t0
    # if we have a long track history, use it as elapsed estimate
    if elapsed is None or elapsed < 60:
        if len(traj) >= 5:
            elapsed = max(elapsed or 0, 300)

    eta_result = predict_eta(lat, lon, destination, speed, alt, hdg, elapsed)
    result.update(eta_result)

    # Simpan history ke ES untuk dashboard riwayat
    try:
        pos = result.get("current_position", {})
        history_doc = {
            "icao24": icao24,
            "callsign": result.get("callsign"),
            "destination": result.get("destination"),
            "prediction_method": result.get("prediction_method"),
            "confidence": result.get("confidence"),
            "eta_seconds": result.get("eta_seconds"),
            "eta_minutes": result.get("eta_minutes"),
            "eta_method": result.get("eta_method"),
            "distance_km_to_dest": result.get("distance_km_to_dest"),
            "track_points": result.get("track_points"),
            "current_position": {
                "lat": pos.get("lat"),
                "lon": pos.get("lon"),
                "altitude": pos.get("altitude"),
                "heading": pos.get("heading"),
                "speed_kmh": pos.get("speed_kmh"),
            },
            "status": result.get("status"),
            "route": result.get("route"),
            "recorded_at": datetime.utcnow().timestamp(),
        }
        if es is not None:
            es.index(index="flight_predictions_history", body=history_doc)
    except Exception:
        pass

    if "heading_top5" not in result:
        heading_results = predict_by_heading(lat, lon, hdg, alt)
        result["heading_top5"] = [
            {"airport": r["airport"], "score": r["score"]}
            for r in heading_results[:5]
        ]

    return deep_convert(result)


# =========================
# SYNTHETIC TEST (no ES needed)
# =========================

def make_track(lat, lon, alt, hdg, speed, callsign=None, n=10):
    rows = []
    for i in range(n):
        rows.append({
            "timestamp": 1000000 + i * 5,
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "velocity_kmh": speed,
            "heading": hdg,
            "callsign": callsign,
            "on_ground": 0
        })
    return pd.DataFrame(rows)


# =========================
# CLI / DEMO
# =========================

def demo():
    test_cases = [
        {"name": "Approaching London (EGLL)", "lat": 51.5, "lon": -0.1, "alt": 3000, "hdg": 270, "speed": 400, "callsign": "BAW105"},
        {"name": "Approaching Frankfurt (EDDF)", "lat": 50.0, "lon": 8.6, "alt": 3500, "hdg": 180, "speed": 380, "callsign": "DLH123"},
        {"name": "Approaching Amsterdam (EHAM)", "lat": 52.3, "lon": 4.7, "alt": 2500, "hdg": 45, "speed": 350, "callsign": "KLM456"},
        {"name": "Approaching Paris (LFPG)", "lat": 49.0, "lon": 2.5, "alt": 2800, "hdg": 90, "speed": 360, "callsign": "AFR789"},
        {"name": "Approaching Barcelona (LEBL)", "lat": 41.3, "lon": 2.1, "alt": 2000, "hdg": 250, "speed": 320, "callsign": "VLG321"},
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {tc['name']}")
        print(f"{'='*60}")
        track = make_track(tc["lat"], tc["lon"], tc["alt"], tc["hdg"], tc["speed"], tc["callsign"])
        result = predict(icao24="test", track=track)
        for key, value in result.items():
            if key == "heading_top5":
                continue
            print(f"  {key}: {value}")
        if "heading_top5" in result:
            print("  heading_top5:")
            for r in result["heading_top5"][:3]:
                print(f"    {r['airport']}: {r['score']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        icao24 = sys.argv[1]
        result = predict(icao24)
        print("\n===== PREDICTION RESULT =====")
        for key, value in result.items():
            if key == "heading_top5":
                continue
            print(f"{key}: {value}")
        if "heading_top5" in result:
            print("\nTop 5 heading-scored airports:")
            for r in result["heading_top5"]:
                print(f"  {r['airport']}: {r['score']:.4f}")
    else:
        demo()
